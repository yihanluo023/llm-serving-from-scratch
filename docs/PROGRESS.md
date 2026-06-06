# Project Progress Log

## Project Overview

**Name**: llm-serving-from-scratch
**Started**: 2026-05-09

### Goal

A from-scratch LLM inference serving system inspired by vLLM. Target capabilities:
continuous batching, PagedAttention-style KV cache management, full
metrics/observability, reproducible benchmarks.

## Roadmap

### Phase 0: Setup & Baseline — COMPLETE
- [x] WSL2 + Ubuntu 22.04
- [x] CUDA 12.4 + PyTorch 2.6 + GPU verified
- [x] uv + Python 3.11 environment
- [x] Project structure + git + GitHub repo
- [x] SSH key + first push to GitHub
- [x] VS Code + WSL extension
- [x] HuggingFace setup + Qwen2.5-1.5B downloaded and verified loadable
- [x] First interactive chat script (`scripts/interactive_chat.py`)
- [x] First baseline benchmark (`scripts/benchmark_baseline.py`)
- [x] Prompt-length sweep with per-length warmup
- [x] JSONL benchmark output + rich table summary

### Phase 1: Naive Serving & Static Batching — COMPLETE
- [x] FastAPI server with model loaded once on startup
- [x] Stateless `/generate_serial` baseline endpoint
- [x] Tokenizer integration with Qwen chat template
- [x] Single-request generation path with TTFT / decode / total latency metrics
- [x] Background batcher with `asyncio.Queue` + per-request `Future`
- [x] Size-or-timeout static batching policy
- [x] Manual forward-based generation loop
- [x] Batched prefill
- [x] Greedy decode from `logits[:, -1, :]`
- [x] KV cache propagation through `past_key_values`
- [x] Attention mask and position id handling for left-padded prompts
- [x] Per-request result fan-out after batch completion
- [x] Static batching benchmark across concurrency levels
- [x] Verified throughput improvement over serial baseline

### Phase 2: Continuous Batching — COMPLETE
- [x] Designed continuous batcher skeleton
- [x] Defined `_PendingItem` and `_RunningRequest`
- [x] Designed scheduler loop for active vs idle states
- [x] Decided active requests should share one batched KV cache
- [x] Established invariant: `active[i] <=> past_key_values batch row i`
- [x] Clarified one-token-lag generation state
- [x] Decided to use left padding for KV cache merge
- [x] Planned `real_seq_len` + `kv_pad_len` request metadata
- [x] Implement `_collect_initial_prefill_batch`
- [x] Implement `_drain_waiting_queue`
- [x] Implement `_prefill_new_requests`
- [x] Implement left-padding `_merge_new_requests`
- [x] Implement `_decode_one_step`
- [x] Implement finished-request cleanup
- [x] Implement result construction and metrics
- [x] Implement KV row filtering
- [x] Implement common-left-padding head-cut
- [x] Implement scheduler failure handling
- [x] Add HuggingFace `DynamicCache` <-> legacy tuple conversion helpers
- [x] Move blocking model forwards off the FastAPI event loop with `run_in_executor`
- [x] Write continuous batcher smoke test
- [x] Validate single-request generation
- [x] Validate concurrent requests
- [x] Validate mixed output lengths
- [x] Validate dynamic request joining/leaving
- [x] Validate final cleanup of `active` and `past_key_values`
- [x] Add `/generate_continuous` endpoint
- [x] Verify `/generate_continuous` through HTTP smoke tests
- [x] Benchmark static vs continuous batching
- [x] Confirm expected tradeoff:
  - static is slightly faster on uniform-length workloads
  - continuous strongly reduces head-of-line blocking on heterogeneous workloads

### Phase 3: KV Cache Memory Management / PagedAttention-style Design — IN PROGRESS
- [ ] Study vLLM block allocator and block table design
- [ ] Design per-request block metadata
- [ ] Replace naive batched KV padding with block/page-based KV management
- [ ] Integrate or prototype paged attention mechanism
- [ ] Measure memory utilization improvement
- [ ] Benchmark max concurrent requests under memory pressure

### Phase 4: Long-term polish — PLANNED
- [ ] Streaming responses with SSE
- [ ] Prefix caching
- [ ] Speculative decoding
- [ ] Multi-LoRA serving
- [ ] Better observability dashboard
- [ ] Technical writeup / blog post
---

## Environment

- OS: WSL2 + Ubuntu 22.04 LTS on Windows 11
- GPU: NVIDIA RTX 3060 Laptop, 6GB GDDR6
- CUDA Toolkit: 12.4
- Python: 3.11.15 (uv-managed)
- PyTorch: 2.6.0+cu124
- transformers: 5.8.0
- uv: 0.11.12

**Model under test**: `Qwen/Qwen2.5-1.5B-Instruct` (FP16, ~1.54B params, ~3.1GB)

---

## Daily Log

### 2026-05-09 (Day 1)

**Achievements**:
- Full dev environment from scratch (WSL2, CUDA, Python, PyTorch, GPU verified)
- Project repo created and pushed to GitHub
- Qwen2.5-1.5B-Instruct downloaded and verified loadable on GPU
  (3.09GB GPU memory, matches theoretical 1.54B × 2 bytes for FP16)

**Bugs/issues encountered**:
- `transformers` required `accelerate` for `device_map="cuda"` (fixed by adding it)
- `torch_dtype` deprecated in favor of `dtype`

---

### 2026-05-10 (Day 2)

**Achievements**:
- Wrote `scripts/interactive_chat.py` — single-request interactive chat
  with Qwen2.5-1.5B-Instruct, multi-turn history, command support
  (quit/reset)
- Captured first real baseline timing using `TextIteratorStreamer`:
  TTFT ≈ 28ms (cold prefill, ~25 tok prompt), decode ≈ 35-40 tok/s
- Verified prefill is ~20-25× faster per token than decode (compute-bound
  vs memory-bound difference made concrete in numbers, not just theory)
- Observed prompt accumulation across multi-turn dialogue: each turn's
  prompt grows by previous turn's full content + chat template overhead

**Bugs/issues encountered**:
- Initial timing approach (calling `generate()` twice, once for prefill
  with max_new_tokens=1, once for total) produced negative decode times.
  Root cause: cold-vs-warm cost asymmetry — second call benefits from
  CUDA kernel cache and weight residency. Replaced with single-pass
  streamer-based timing.
- Cold-start (first generate() in process) took ~600ms even for tiny
  prompts. Root cause: CUDA kernel JIT, cuBLAS autotuning, allocator
  init — one-time setup costs unrelated to model inference. Fixed by
  adding explicit warmup call before chat loop. After warmup, cold
  prefill drops from 645ms → 28ms (23× difference).

**Next session**:
- Write `scripts/benchmark_baseline.py` for systematic
  single-turn baseline across prompt lengths

---

### 2026-05-10 (Day 2, afternoon)

**Achievements**:
- Wrote `scripts/benchmark_baseline.py` — single-request prompt-length
  sweep with per-length warmup (cuBLAS autotunes per-shape),
  `min_new_tokens` to force fixed decode count, JSONL trial dump,
  rich.table summary. 5 trials per cell, ~3 minutes end-to-end.
- First systematic baseline across prompt lengths (output fixed at
  128 tok), single request, no batching:

  | prompt_len | TTFT mean (ms) | decode (tok/s) | peak mem (MB) |
  |-----------:|---------------:|---------------:|--------------:|
  |         16 |           31.4 |           39.6 |          2961 |
  |         64 |           29.9 |           39.1 |          2962 |
  |        256 |           41.3 |           41.5 |          2976 |
  |       1024 |          139.5 |           41.8 |          3046 |
  |       2048 |          262.3 |           40.2 |          3141 |
  |       4096 |          552.6 |           36.1 |          3325 |

- Per-length warmup gave excellent stability — TTFT std at long prompts
  is 0.4–0.8 ms (cuBLAS autotune fully cached after the warmup call).

**Bugs/issues encountered**:
- p=4096 decode tok/s degrades across the 5 trials
  (39.2, 39.4, 37.9, 30.9, 32.9 — std=4.0 vs <1.5 at every other
  prompt length). The one-way decline after sustained GPU load suggests
  possible laptop GPU thermal or power throttling. Need to confirm with
  clock/temperature snapshots before longer Phase 2 throughput benchmarks.

**Next session**:
- Start Phase 1: naive serving.
  - Implement a FastAPI `/generate` endpoint.
  - Keep the model resident on GPU across requests.
  - Return generated text, TTFT, total latency, and decode tok/s.
- Measure the overhead of HTTP serving compared with direct script-based
  inference.
- Use this naive server as the control baseline before implementing request
  queues, static batching, and continuous batching.

  ### 2026-05-11 (Day 3)

**Achievements**:
- Phase 1 begins. Built `src/server/app.py`: FastAPI server exposing
`POST /generate`. Model loaded once on startup via lifespan context
  manager, warmed up before serving, freed on shutdown.
- Single-request, stateless, synchronous design — each request runs
  `model.generate()` to completion. Intentionally the un-batched
  baseline against which subsequent batching work will be measured.
- Verified end-to-end: model resident in GPU (~3GB, only present while
  server process is alive), per-request decode tok/s matches Day 2
  baseline within noise (~36 tok/s at p≈30, o=128) — confirming the
  HTTP layer adds no measurable inference overhead.

**Design notes**:
- Stateless API: server holds no conversation history; clients send
  full message context per request.
- Lifespan-managed model loading: clean separation of startup/shutdown,
  fail-fast on load errors, explicit GPU cleanup on shutdown.
- Handler is `async def` but calls a synchronous `run_generate`.
  This will block the event loop under concurrent load — deliberately
  left as-is so the symptom is visible when batching is introduced.

**Next session**:
- Static batching with size-or-timeout admission (N=8, T=50ms initial).
- Background batcher task coordinating an `asyncio.Queue` of pending
  requests with per-request `asyncio.Future` for fan-out.
- Concurrent-load test script to demonstrate the throughput delta
  from naive serial → batched serving.

### 2026-05-13 (Day 4 & 5)

**Achievements**:
- Implemented Phase 1 static batching checkpoint in `src/server/batcher.py`.
- Kept the existing FastAPI server and queue-based batcher structure:
  requests enter an `asyncio.Queue`, `_collect_batch()` groups them by
  size-or-timeout, and each request is resolved through its own Future.
- Replaced the batched `model.generate()` path with an explicit
  forward-based inference loop:
  prefill full prompt batch → read `logits[:, -1, :]` → greedy decode →
  update `past_key_values` → repeat.
- Preserved request/result ordering: `results[i]` corresponds to `items[i]`.

**Design notes**:
- Static batching runs the whole batch for the shared
  `max(item.max_tokens)` decode length.
- EOS is tracked per row, but does not stop the batch early.
  Finished rows keep feeding EOS until the batch completes.
- Generated tokens are stored separately from prompt tokens, so final
  decoding returns only the model response.
- Manual `forward()` path exposes the internals hidden by
  HuggingFace `generate()`: logits, token selection, attention mask
  extension, and KV cache propagation.

**Next session**:
- Run server startup test.
- Test `/generate_serial` and `/generate`.
- Verify batched requests produce reasonable outputs.
- Debug any `position_ids`, `attention_mask`, or KV-cache issues.
- Run concurrent-load benchmark and compare serial vs static batching.

### 2026-05-14 (Day 5)

**Achievements**:
- Added HTTP server benchmark for `/generate_serial` and `/generate`.
- Measured throughput, scheduled token throughput, client-side latency,
  queue wait, and observed batch size across concurrency levels.
- Verified manual static batching works end-to-end with zero failed requests.
- Results matched the expected static batching behavior:
  throughput scaled with concurrency until `MAX_BATCH=8`, then plateaued,
  while queue wait increased once the batcher was saturated.

**Key result**:
- `/generate_serial` stayed around ~0.5 req/s across concurrency levels.
- `/generate` improved from ~0.5 req/s at concurrency=1 to ~3.9 req/s
  at concurrency=8.
- Mean batch size reached exactly 8 at concurrency=8 and 16.

**Next session**:
- Decide whether to clean up benchmark docs/README first or begin Phase 2:
  continuous batching.

### 2026-05-19 (Day 6-8)

**Achievements**:
- Started Phase 2 continuous batching design.
- Created the first skeleton for `src/server/continuous_batcher.py`.
- Planned the main components:
  - `_PendingItem` for requests waiting in the queue
  - `_RunningRequest` for requests already admitted into the active decode batch
  - `ContinuousBatcher` with lifecycle, submit, scheduler loop, prefill, merge, decode, cleanup, and KV helper functions
- Kept the same high-level serving pattern from static batching.

**Scheduler design notes**:
- If there are no active requests, the scheduler blocks until the first queued request arrives, then briefly collects more requests up to `max_batch` or `max_wait_ms`.
- If active requests already exist, the scheduler drains only currently queued requests without waiting.
- This avoids starving existing active requests while still allowing new requests to join between decode steps.
- Main loop order:
  1. clean up finished requests
  2. collect or drain new queued requests
  3. prefill new requests
  4. merge new requests into the active batch
  5. clean up requests that finished immediately during prefill
  6. decode one token step for all active requests

**Important invariant**:
- The batcher owns one batched KV cache.
- `_RunningRequest` objects do not each own their own KV cache.
- Instead:
  - self.active[i] <=> self.past_key_values batch row i

**Key insight**:
- Manual generation has a one-token lag:
  - past_key_values contains tokens that were actually passed into the model.
  - logits are used to select the next token, but that token is not yet in KV cache.
- For each active request:
  - KV cache = prompt + generated_ids[:-1]
  - last_token_id = generated_ids[-1]
  - Each decode step feeds last_token_id, updates KV cache, and selects the next token.
- Use left padding for KC cache merge:
  - Left padding keeps real cached tokens as a continuous suffix.
  - May allow a future “head cut” optimization. 

**Next session**:
- Implement _decode_one_step() using last_token_id, real_seq_len,
  kv_pad_len, attention mask, and position ids.
- Then implement cleanup/result handling.

### 2026-06-01

**Achievements**:
- Completed the main Phase 2 continuous batching helpers:
  `_decode_one_step`, `_cleanup_finished`, `_filter_past_key_values`,
  `_set_result`, `_head_cut_past_key_values`, and `_fail_all`.
- Implemented one-step batched decode with `past_key_values`, temporary
  `attention_mask` / `position_ids`, greedy token selection, and per-request
  state updates.
- Added finished-request cleanup, KV batch-row filtering, result fan-out via
  per-request `Future`, and failure handling to avoid hanging requests.
- Added common-left-padding head-cut to remove shared padding from KV cache
  after cleanup.

**Design notes**:
- Kept the core invariant:
  `active[i] <=> past_key_values batch row i`.
- Preserved the one-token-lag model:
  KV cache contains `prompt + generated_ids[:-1]`, while `last_token_id`
  is fed on the next decode step.
- Kept `attention_mask` and `position_ids` as temporary decode-step tensors
  instead of persistent scheduler state.

**Next session**:
- Write smoke tests for single request, concurrent requests, and dynamic
  request joining/leaving before connecting the FastAPI endpoint.

### 2026-06-04

**Achievements**:
* Added `scripts/smoke_test_continuous_batcher.py` to test the Phase 2
  `ContinuousBatcher` directly before relying on the FastAPI layer.
* Verified the continuous batcher with:
  * single-request generation
  * concurrent requests
  * mixed output lengths
  * dynamic request joining/leaving
  * final cleanup of `active` and `past_key_values`
* Added `/generate_continuous` to `src/server/app.py` while preserving the
  existing `/generate_serial` and `/generate` paths for future comparison.
* Verified `/generate_continuous` through single-request, concurrent, and
  staggered-arrival HTTP smoke tests.

**Bugs/issues encountered**:
* Smoke testing exposed a HuggingFace cache format mismatch:
  `outputs.past_key_values` now returns a `DynamicCache`, while the batcher
  internally expects the legacy tuple layout:
  `(key, value)` per transformer layer.
* Fixed this with boundary conversion helpers:
  `_to_legacy_past_key_values()` and `_from_legacy_past_key_values()`.
  The scheduler keeps using the legacy tuple format internally so KV tensor
  operations such as padding, concatenation, filtering, and head-cut remain
  easy to inspect.
* This also isolates HuggingFace cache API changes behind small helper
  functions, preserving the core scheduler logic.
* Staggered HTTP testing revealed an event-loop blocking issue: synchronous
  `model.forward()` calls inside `_run()` prevented FastAPI from promptly
  admitting late-arriving requests into the queue.
* Fixed this by dispatching blocking prefill/decode forwards through
  `run_in_executor()`. The event loop can now continue accepting HTTP requests
  while the current GPU forward is running, while scheduler state mutations
  remain serialized because `_run()` still awaits each forward before
  continuing.

**Design notes**:
* `ContinuousBatcher` now separates three concerns:
  * scheduler logic: active batch, waiting queue, joining/leaving
  * KV representation: legacy tuple internally, `DynamicCache` at model boundary
  * server responsiveness: blocking model forwards moved off the event loop
* The current `batch_size` metric is still a rough cleanup-time snapshot, not a
  precise lifecycle batch-size metric. Future benchmarks should track clearer
  metrics such as max active batch size or per-request admitted batch size.

**Next session**:
* Write a mixed-length internal benchmark comparing static batching against
  continuous batching.

### 2026-06-05 (benchmark)

**Achievements**:
* Added and ran `benchmarks/benchmark_continuous_vs_static.py` to compare
  `/generate` static batching against `/generate_continuous`.
* Used an open-loop benchmark with the same seeded arrival schedule for both
  endpoints:
  * 96 requests
  * 10s arrival window
  * uniform-length control workload
  * heterogeneous workload with mixed short and long generations
* Verified the expected tradeoff:
  * static batching is slightly faster in the uniform-length control workload
  * continuous batching strongly outperforms static batching in the
    heterogeneous workload

**Key result**:
* Uniform control workload:
  * static: e2e p99 = 16.4s, output throughput = 257.5 tok/s
  * continuous: e2e p99 = 18.2s, output throughput = 239.5 tok/s
  * Interpretation: continuous batching has some overhead when request lengths
    are similar and static batching already works well.
* Heterogeneous workload:
  * short-request e2e p99 improved from 39.0s to 13.1s
  * overall e2e p99 improved from 42.5s to 14.8s
  * overall output throughput improved from 96.1 tok/s to 197.0 tok/s
  * Interpretation: continuous batching reduces head-of-line blocking under
    mixed generation lengths by allowing requests to join and leave the active
    decode batch dynamically.

**Next session**:
* Begin Phase 3 planning around PagedAttention-style KV-cache memory management.
