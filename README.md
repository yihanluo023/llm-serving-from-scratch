# llm-serving-from-scratch

This project builds a resource-constrained LLM serving prototype inspired by
[vLLM](https://github.com/vllm-project/vllm). The goal is to understand and
implement the core serving mechanics behind high-throughput LLM inference on a
single local GPU, including request batching, manual autoregressive decoding,
KV-cache reuse, and throughput/latency benchmarking.

This is not intended to replace production serving systems such as vLLM.
Instead, it focuses on implementing the core mechanisms in a small, inspectable
codebase so their performance tradeoffs can be measured directly.

> **Status:** Phase 1 (static batching) is complete and benchmarked. Phase 2
> (continuous batching) is in progress — the scheduler design, request
> admission, prefill path, and KV-cache merge strategy are being implemented;
> the per-step decode loop and request cleanup are next. See [Roadmap](#roadmap)
> for the exact state.

---

## Why this exists

`model.generate()` hides everything interesting. This project deliberately
removes that abstraction and rebuilds the inference loop from direct model
calls + `past_key_values`, so that every mechanism a real serving stack relies
on — left-padded batches, attention masks, position ids, KV-cache propagation,
request fan-out, dynamic batch membership — is understood rather than configured.

Each phase introduces one new problem and resolves it before moving on.
The known limitations of each phase are kept so the next phase has a concrete
symptom to fix.

---

## Architecture

The server keeps the model resident on the GPU for its entire lifetime. HTTP
handlers never touch the model directly. Instead, a single background coroutine 
owns the GPU, which serializes all GPU access without locks.

```
                  ┌─────────────────────────────────────────┐
  HTTP client ──> │  FastAPI  (app.py)                      │
                  │   POST /generate          (batched path)│
                  │   POST /generate_serial   (A/B baseline)│
                  └───────────────┬─────────────────────────┘
                                  │  submit(prompt, max_tokens)
                                  │  + per-request asyncio.Future
                                  ▼
                  ┌─────────────────────────────────────────┐
                  │  asyncio.Queue  (pending requests)      │
                  └───────────────┬─────────────────────────┘
                                  │
                                  ▼
                  ┌─────────────────────────────────────────┐
                  │  Background batcher coroutine           │
                  │   - collects a batch (size or timeout)  │
                  │   - manual prefill + greedy decode loop │
                  │   - propagates past_key_values          │
                  │   - resolves each request's Future      │
                  │                                         │
                  │   only this coroutine calls the model   │
                  └───────────────┬─────────────────────────┘
                                  │  model(...) call
                                  ▼
                          GPU-resident model
                       (Qwen2.5-1.5B-Instruct, FP16)
```

## Results

All numbers come from this repo's benchmark scripts.

### Single-request baseline (no batching)

Prompt-length sweep, output fixed at 128 tokens, 5 trials per cell, per-length
warmup:

| prompt_len | TTFT mean (ms) | decode (tok/s) | peak mem (MB) |
|-----------:|---------------:|---------------:|--------------:|
|         16 |           31.4 |           39.6 |          2961 |
|         64 |           29.9 |           39.1 |          2962 |
|        256 |           41.3 |           41.5 |          2976 |
|       1024 |          139.5 |           41.8 |          3046 |
|       2048 |          262.3 |           40.2 |          3141 |
|       4096 |          552.6 |           36.1 |          3325 |

### Static batching vs serial (concurrent load)

| path                | concurrency=1 | concurrency=8 | mean batch size @ c=8 |
|---------------------|--------------:|--------------:|----------------------:|
| `/generate_serial`  |    ~0.5 req/s |    ~0.5 req/s |                     — |
| `/generate`         |    ~0.5 req/s |    ~3.9 req/s |                     8 |

Throughput scales with concurrency until `MAX_BATCH=8`, then plateaus while
queue wait rises — the expected static-batching saturation behavior. 

---

## Roadmap

- [x] Phase 0: local setup, model loading, baseline benchmark
- [x] Phase 1: FastAPI server, serial endpoint, static batching, manual KV-cache decode
- [ ] Phase 2: continuous batching with dynamic request joining/leaving
- [ ] Phase 3: PagedAttention-style KV cache memory management
- [ ] Phase 4: streaming, prefix caching, speculative decoding, polish

---

## Known limitations

- **Head-of-line blocking (Phase 1).** The entire static batch must finish
  `max(max_tokens)` decode steps before *any* request returns, even if one hits
  EOS on step 1. Finished rows keep feeding EOS until the batch completes. 
- **Budget rounding (Phase 1).** Mixed `max_tokens` in a batch is rounded up to
  the max; short-budget requests pay for the longest one's decode tail.
- **Monotonic KV padding (Phase 2).** When a new request joins the active batch,
  the shared KV cache is left-padded to match the current active KV length.
  Without later compaction/head-cutting, this physical cache length can grow
  during a continuous batch window.

---

## Project layout

```
src/server/
  __init__.py
  app.py                  FastAPI app, lifespan, routes, model loading
  batcher.py              Phase 1 static batcher (complete)
  continuous_batcher.py   Phase 2 continuous batcher (in progress)
scripts/
  interactive_chat.py     Single-request multi-turn chat
  benchmark_baseline.py   Single-request prompt-length sweep
  benchmark_server.py     Concurrent-load serial vs batched benchmark
benchmarks/               Benchmark outputs (JSONL trials + summaries)
docs/
  PROGRESS.md             Detailed daily engineering log
main.py                   Entry point
pyproject.toml            Dependencies (uv-managed, uv.lock pinned)
```

---

## Environment

| Component | Version |
|-----------|---------|
| OS        | WSL2 + Ubuntu 22.04 LTS on Windows 11 |
| GPU       | NVIDIA RTX 3060 Laptop, 6 GB GDDR6 |
| CUDA      | 12.4 |
| Python    | 3.11 (uv-managed) |
| PyTorch   | 2.6.0+cu124 |
| transformers | 5.8.0 |
| Model     | `Qwen/Qwen2.5-1.5B-Instruct` (FP16, ~1.54B params, ~3.1 GB) |

## Engineering log

[`docs/PROGRESS.md`](./docs/PROGRESS.md) is a dated log of what was built each
session, what broke, the root cause, and the decision made. It is the honest
version of this README — including the bugs (negative decode times from
cold/warm asymmetry, suspected laptop-GPU thermal throttling at long prompts)
and what fixed them.