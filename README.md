# llm-serving-from-scratch

This project builds a resource-constrained LLM serving prototype inspired by
[vLLM](https://github.com/vllm-project/vllm). The goal is to understand and
implement the core serving mechanics behind high-throughput LLM inference on a
single local GPU, including request batching, manual autoregressive decoding,
KV-cache reuse, and throughput/latency benchmarking.

> **Status:** Phase 2 finished.

---

## Why this exists

`model.generate()` hides almost everything that makes LLM serving interesting.
It can produce text, but it does not expose the scheduling, batching, KV-cache,
and latency tradeoffs that determine whether an inference server can handle
concurrent users efficiently.

This project rebuilds the serving loop from lower-level model forward calls.
It starts with the simplest serial generation path, then adds static batching,
manual autoregressive decoding with KV-cache reuse, continuous batching with
dynamic request membership, and PagedAttention-style KV-cache memory
management.

Each phase starts from a concrete bottleneck, implements the next serving
mechanism, and then measures what improved and what limitation remains. The
project is intentionally small, but the path follows the same sequence of
problems that real LLM serving systems must solve: how to batch requests, how
to reuse KV cache, how to avoid head-of-line blocking, and how to manage KV
memory efficiently as requests join and leave.

---

## Phase evolution

### Phase 0: Serial generation

The simplest server runs one request at a time. A request enters the server,
the model generates its answer, and only then can the next request use the GPU.

This is easy to understand, but it fails under concurrency. If eight users send
requests at the same time, seven of them are effectively waiting in line while
the first one generates. GPU utilization is also poor because the server does
not combine independent requests into larger tensor operations.

This gives the first bottleneck:

> Serial generation is simple, but concurrent requests become a queue.

### Phase 1: Static batching

Static batching addresses this by collecting multiple waiting requests into a
single batch. Instead of running eight separate decode loops, the server pads
the prompts into one tensor, runs prefill for the batch, and then decodes tokens
for all rows together.

This improves throughput by making the GPU to see larger batched operations. It is
much better than serial generation when many requests arrive at the same time.

However, static batching has a structural limitation: once a batch starts, its
membership is fixed.

If one request in the batch needs a long answer and another needs only a short
answer, the short request is still tied to the batch. In a simple static decode
loop, the batch keeps stepping until the batch-level stopping condition is met.
Even if a short request is logically finished, it cannot leave the batch
and free its row. New requests also cannot join until the current batch is done.

This creates head-of-line blocking:

> Static batching improves throughput, but short requests can be delayed by
> long requests in the same or earlier batch.

### Phase 2: Continuous batching

Continuous batching changes the batch from a fixed group into a dynamic active
set.

Requests still enter a queue. The scheduler prefills new requests into KV cache,
merges them into the active batch, decodes one token step for all active
requests, and removes requests as soon as they finish. While older requests are
still decoding, newer requests can join.

This directly targets the static batching problem. A short request no longer
has to wait for a long request's entire generation to finish before it can even
start. Finished requests can leave the active batch, and newly arrived requests
can be admitted between decode steps.

This phase introduces more scheduler complexity. The server must maintain
per-request state, keep KV-cache rows aligned with active requests, merge new
KV caches into the existing batch, and remove finished rows safely. But the
result is much better latency under mixed workloads.

> Continuous batching reduces head-of-line blocking by allowing requests to
> join and leave the active decode batch dynamically.

### Phase 3: PagedAttention-style KV-cache management

Continuous batching improves scheduling, but it does not fully solve memory
layout.

The current implementation still represents the active KV cache as padded
rectangular tensors. When requests have different prompt lengths or join at
different times, shorter rows may need padding so that all rows share a common
physical sequence length. This makes the implementation simple, but it can waste
GPU memory.

PagedAttention addresses this at the KV-cache level. Instead of storing each
request's KV cache as one large contiguous padded region, the cache is split
into fixed-size blocks. Each request owns a sequence of blocks. This makes it
possible to reuse memory more flexibly, reduce padding waste, and support more
dynamic scheduling at larger scale.

> Continuous batching fixes the scheduling problem. PagedAttention targets the
> KV-cache memory management problem.

---

## Serving methods

The server exposes multiple endpoints so each serving strategy can be compared
under the same model and hardware.

> **Benchmarking note:** These endpoints are colocated in one FastAPI app for
> convenience, not because they are intended to be mixed in one workload. For a
> clean experiment, run one endpoint at a time. The endpoints share the same
> GPU-resident model, so sending traffic to multiple generation endpoints at once
> would mix scheduling policies and make the measurements hard to interpret.

### `/generate_serial`

This is the baseline path. It uses the model's standard generation flow for one
request at a time.

Design:

1. One HTTP request enters the server.
2. The model generates the full response.
3. The response is returned.
4. The next request can then use the GPU.

This path is useful as a correctness and performance baseline, but it is not a
good serving design for concurrent traffic.

### `/generate`

This is the static batching path.

Design:

1. Requests enter an `asyncio.Queue`.
2. The batcher waits briefly for more requests, up to a maximum batch size or
   timeout.
3. The collected requests are padded into a single batch.
4. The model runs prefill once for the batch.
5. The batch enters a manual greedy decode loop.
6. The entire batch finishes together, and each request's Future is resolved.

Static batching is effective when requests arrive close together and have
similar output lengths. It improves throughput by turning multiple small model
calls into fewer batched model calls.

Its weakness is fixed batch membership. A request that finishes early is still
part of the batch until the batch-level decode loop ends. New requests cannot
join an already-running batch. This is why short requests can suffer high tail
latency when mixed with long requests.

### `/generate_continuous`

This is the continuous batching path.

Design:

1. Requests enter an `asyncio.Queue`.
2. If there are no active requests, the scheduler collects an initial prefill
   batch.
3. If there are already active requests, the scheduler drains any waiting
   requests without blocking.
4. New requests are prefetched into KV cache.
5. Their KV cache is merged into the existing active batch.
6. The scheduler decodes one token for every active request.
7. Finished requests are removed immediately.
8. The loop continues until there are no active requests.

This makes the active batch dynamic. A request can arrive after decoding has
already started, join the active batch, generate a few tokens, and leave before
older long requests finish.

This is closer to how real LLM serving systems handle continuous request
arrival.

---

## Results

All numbers come from this repo's benchmark scripts on a single local laptop GPU.

### Single-request baseline

Prompt-length sweep, output fixed at 128 tokens, 5 trials per cell:

| prompt_len | TTFT mean (ms) | decode (tok/s) | peak mem (MB) |
| ---------: | -------------: | -------------: | ------------: |
|         16 |           31.4 |           39.6 |          2961 |
|         64 |           29.9 |           39.1 |          2962 |
|        256 |           41.3 |           41.5 |          2976 |
|       1024 |          139.5 |           41.8 |          3046 |
|       2048 |          262.3 |           40.2 |          3141 |
|       4096 |          552.6 |           36.1 |          3325 |

This establishes the baseline behavior of one request at a time: prefill cost
grows with prompt length, while decode throughput is relatively stable.

### Static batching vs serial

| path               | concurrency=1 | concurrency=8 | mean batch size @ c=8 |
| ------------------ | ------------: | ------------: | --------------------: |
| `/generate_serial` |    ~0.5 req/s |    ~0.5 req/s |                     — |
| `/generate`        |    ~0.5 req/s |    ~3.9 req/s |                     8 |

Static batching significantly improves throughput under concurrent load because
multiple requests share batched model forward passes.

### Static batching vs continuous batching

The main benchmark uses open-loop request arrivals. Requests are scheduled over
a fixed time window independent of server completion time, which better
simulates real concurrent traffic.

#### Experiment A: uniform-length control

96 requests arrive over a 10 second window. All requests use similar prompts and
the same generation budget.

| endpoint   | class   |  n | e2e p50 (ms) | e2e p99 (ms) | qwait p50 (ms) | qwait p99 (ms) | req/s | out tok/s |
| ---------- | ------- | -: | -----------: | -----------: | -------------: | -------------: | ----: | --------: |
| static     | uniform | 96 |         9389 |        16385 |           7557 |          14555 |  4.02 |     257.5 |
| continuous | uniform | 96 |        10725 |        18160 |           8654 |          16083 |  3.74 |     239.5 |

In the uniform-length control workload, static batching is slightly faster.
This is expected: when requests are similar, static batching already works well,
while continuous batching pays extra overhead for dynamic admission, KV-cache
merging, finished-row cleanup, and active-set management.

This is an important result because it shows that continuous batching is not a
free optimization. It is a tradeoff.

#### Experiment B: heterogeneous-length workload

96 requests arrive over a 10 second window. About one quarter are long requests
with a larger generation budget; the rest are short requests with a small
generation budget.

| endpoint   | class |  n | e2e p50 (ms) | e2e p99 (ms) | qwait p50 (ms) | qwait p99 (ms) | req/s | out tok/s |
| ---------- | ----- | -: | -----------: | -----------: | -------------: | -------------: | ----: | --------: |
| static     | long  | 23 |        27616 |        42588 |          23157 |          38171 |  0.46 |      72.9 |
| static     | short | 73 |        27297 |        39001 |          22837 |          38476 |  1.45 |      23.1 |
| static     | all   | 96 |        27336 |        42540 |          22877 |          38476 |  1.90 |      96.1 |
| continuous | long  | 23 |         9982 |        16740 |           4720 |          12349 |  0.93 |     149.5 |
| continuous | short | 73 |         7740 |        13075 |           7277 |          12424 |  2.97 |      47.5 |
| continuous | all   | 96 |         9056 |        14796 |           4738 |          12424 |  3.90 |     197.0 |

This is the headline result.

Under static batching, short requests have almost the same latency as long
requests. The short requests are not slow because they generate many tokens;
they are slow because they wait behind long generations. This is head-of-line
blocking.

Continuous batching reduces this effect:

* short-request e2e p99 improves from 39.0s to 13.1s
* overall e2e p99 improves from 42.5s to 14.8s
* overall output throughput improves from 96.1 tok/s to 197.0 tok/s

In this benchmark, continuous batching cuts short-request tail latency by about
3x and roughly doubles output-token throughput under heterogeneous generation
lengths.

---

## Roadmap

* [x] Phase 0: local setup, model loading, single-request baseline benchmark
* [x] Phase 1: FastAPI server, serial endpoint, static batching, manual KV-cache decode
* [x] Phase 2: continuous batching with dynamic request joining/leaving
* [ ] Phase 3: PagedAttention-style KV-cache memory management
* [ ] Phase 4: streaming output, prefix caching, speculative decoding, polish

---

## Known limitations

* **Static head-of-line blocking.** Static batches have fixed membership. Short
  requests can be delayed by long requests because the batch cannot easily admit
  or remove rows dynamically.

* **Continuous batching overhead.** Continuous batching adds scheduling,
  KV-cache merge, cleanup, and active-set management overhead. In uniform-length
  workloads, static batching can be slightly faster.

* **Rectangular KV-cache layout.** The current continuous batcher still stores
  KV cache as padded tensors. This keeps the implementation inspectable, but it
  can waste memory when prompt lengths differ significantly.

* **No streaming yet.** The server returns a full response after completion
  rather than streaming tokens to the client.

* **Single-GPU only.** The project intentionally focuses on one local GPU and
  does not implement tensor parallelism, pipeline parallelism, or distributed
  serving.

---

## Project layout

```text
src/server/
  __init__.py
  app.py                  FastAPI app, lifespan, routes, model loading
  batcher.py              Phase 1 static batcher
  continuous_batcher.py   Phase 2 continuous batcher

scripts/
  interactive_chat.py     Single-request multi-turn chat
  benchmark_baseline.py   Single-request prompt-length sweep
  benchmark_server.py     Concurrent-load serial vs static batching benchmark
  benchmark_continuous_vs_static.py
                          Open-loop static vs continuous benchmark

benchmarks/
  *.jsonl                 Raw benchmark records
  *.json                  Benchmark summaries

docs/
  PROGRESS.md             Detailed daily engineering log

main.py                   Entry point
pyproject.toml            Dependencies
uv.lock                   Locked environment
```

---

## Environment

| Component    | Version                               |
| ------------ | ------------------------------------- |
| OS           | WSL2 + Ubuntu 22.04 LTS on Windows 11 |
| GPU          | NVIDIA RTX 3060 Laptop, 6 GB GDDR6    |
| CUDA         | 12.4                                  |
| Python       | 3.11, uv-managed                      |
| PyTorch      | 2.6.0+cu124                           |
| transformers | 5.8.0                                 |
| Model        | `Qwen/Qwen2.5-1.5B-Instruct` in FP16  |

---

## Running the server

```bash
uv run uvicorn src.server.app:app --host 0.0.0.0 --port 8000
```


Available endpoints:

| endpoint                    | purpose             |
| --------------------------- | ------------------- |
| `GET /`                     | health check        |
| `POST /generate_serial`     | serial baseline     |
| `POST /generate`            | static batching     |
| `POST /generate_continuous` | continuous batching |

---

## Running benchmarks

Single-request baseline:

```bash
uv run python scripts/benchmark_baseline.py
```

Serial vs static batching:

```bash
uv run python scripts/benchmark_server.py
```

Static vs continuous batching:

```bash
uv run python benchmarks/benchmark_continuous_vs_static.py
```

---

## Engineering log

[`docs/PROGRESS.md`](./docs/PROGRESS.md) is the detailed engineering log for
this project. It records what was built each session, what broke, the root
cause, and the implementation decisions made along the way.
