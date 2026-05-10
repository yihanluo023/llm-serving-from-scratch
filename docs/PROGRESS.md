# Project Progress Log

## Project Overview

**Name**: llm-serving-from-scratch
**Started**: 2026-05-09

### Goal

A from-scratch LLM inference serving system inspired by vLLM. Target capabilities:
continuous batching, PagedAttention-style KV cache management, full
metrics/observability, reproducible benchmarks.

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