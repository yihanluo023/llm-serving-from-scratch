# Project Progress Log

## Project Overview

**Name**: llm-serving-from-scratch
**Started**: 2026-05-09

### Goal

A from-scratch LLM inference serving system inspired by vLLM. Target capabilities:
continuous batching, PagedAttention-style KV cache management, full
metrics/observability, reproducible benchmarks.

## Roadmap (4 phases)
### Phase 0: Setup & Baseline (Week 1) — IN PROGRESS
- [x] WSL2 + Ubuntu 22.04
- [x] CUDA 12.4 + PyTorch 2.6 + GPU verified
- [x] uv + Python 3.11 environment
- [x] Project structure + git + GitHub repo
- [x] SSH key + first push to GitHub
- [x] VS Code + WSL extension
- [x] HuggingFace setup + Qwen2.5-1.5B downloaded and verified loadable
- [x] First interactive chat script (`scripts/interactive_chat.py`)
- [x] First baseline benchmark (`scripts/benchmark_baseline.py`) — prompt-length sweep, per-length warmup, JSONL+rich
- [ ] Read vLLM paper (Section 3-4) + browse vLLM source
- [ ] Note-taking: `docs/notes-vllm-reading.md`
### Phase 1: Naive Serving (Weeks 2-3)
- FastAPI server with `/generate` endpoint
- Tokenizer integration
- Single-request prefill + decode loop
- Static batching (intentionally suboptimal—becomes baseline for comparison)
- Streaming responses (SSE)
- Basic metrics: per-request latency
- **Milestone**: 10 concurrent curl requests all complete (even if slow)
### Phase 2: Continuous Batching (Weeks 4-5)
- Iteration-level scheduler
- Request queue, dynamic batch joining/leaving
- Naive KV cache (per-request contiguous allocation)
- Benchmark harness comparing static vs continuous
- **Milestone**: First proper benchmark report showing throughput delta
### Phase 3: PagedAttention (Weeks 6-9)
- Block allocator + block table
- Paged attention kernel integration (vllm-flash-attn or custom Triton)
- Memory utilization optimization (target: 30% → 80%+)
- Preemption mechanism
- **Milestone**: Concurrent request count significantly improved
### Phase 4: Long-term polish (post job-start, ongoing)
- Prefix caching
- Speculative decoding
- Multi-LoRA serving
- Visualization dashboard
- Technical blog posts
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