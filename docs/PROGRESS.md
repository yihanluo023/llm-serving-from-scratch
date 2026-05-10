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
- transformers: 4.46.x
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
- Either: write `scripts/benchmark_baseline.py` for systematic
  single-turn baseline across prompt lengths
- Or: read vLLM paper Section 3-4, start `docs/notes-vllm-reading.md`