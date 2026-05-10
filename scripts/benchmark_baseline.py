"""
Single-request baseline benchmark for Qwen2.5-1.5B-Instruct.

Sweeps prompt length with fixed output length, N_TRIALS per config,
dumps raw per-trial JSONL + prints a rich.table summary.

Design notes:
- Single-turn only (multi-turn is contaminated by HF generate()'s internal
  KV reuse across calls — see docs/PRIVATE.md, Day 2).
- Per-length warmup: cuBLAS autotunes per (M,N,K) shape, so a single
  short warmup does NOT cover the 4096-token prefill path. Each
  prompt_len gets its own warmup call before the N_TRIALS timed runs.
- min_new_tokens == max_new_tokens forces exact decode length so
  decode_s is comparable across configs (no early EOS).
- torch.cuda.max_memory_allocated tracks peak; reset per prompt_len.
"""
import time
import json
import statistics
from datetime import datetime
from pathlib import Path
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from rich.console import Console
from rich.table import Table

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT_LENGTHS = [16, 64, 256, 1024, 2048, 4096]
OUTPUT_TOKENS = 128
N_TRIALS = 5
OUTPUT_DIR = Path("benchmarks")

# Public-domain-style filler. Tile to comfortably exceed PROMPT_LENGTHS[-1]
# after tokenization (~95 tok/copy * 80 = ~7600 tok).
SEED_TEXT = (
    "The study of inference systems for large language models brings "
    "together compiler engineering, GPU programming, distributed systems, "
    "and the theory of memory hierarchies. A modern serving stack must "
    "balance throughput against latency under highly variable load. "
    "Sequences arrive at unpredictable times, run for unpredictable "
    "durations, and exit in an order unrelated to the order in which "
    "they entered. Static batching wastes hardware on padding and "
    "stalls; continuous batching demands a scheduler that can change "
    "the active set every iteration without disturbing in-flight work. "
    "The KV cache, which holds attention state for every running request, "
    "dominates memory and motivates the paging analogy. "
)
TILED_TEXT = SEED_TEXT * 80


def load_model():
    print(f"Loading {MODEL_NAME}...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    print(f"Loaded in {time.perf_counter()-t0:.1f}s. "
          f"Device={model.device} dtype={model.dtype}")
    return tokenizer, model


def build_prompt_of_length(tokenizer, target_len):
    """
    Return (input_ids, attention_mask) of exactly target_len tokens on GPU.
    Slices the tiled filler text — deterministic, real English, exact length.
    """
    full_ids = tokenizer(TILED_TEXT, return_tensors="pt").input_ids
    if full_ids.shape[1] < target_len:
        raise ValueError(
            f"TILED_TEXT only produces {full_ids.shape[1]} tokens; "
            f"need {target_len}. Increase the tile multiplier."
        )
    sliced = full_ids[:, :target_len].to("cuda")
    attn_mask = torch.ones_like(sliced)
    return sliced, attn_mask


def run_trial(model, tokenizer, input_ids, attention_mask, max_new_tokens):
    """One timed generate() call. Returns a dict of timings + memory."""
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        min_new_tokens=max_new_tokens,  # force exact decode length
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    t_start = time.perf_counter()
    thread.start()

    t_first = None
    for _chunk in streamer:
        if t_first is None:
            t_first = time.perf_counter()

    thread.join()
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    ttft = t_first - t_start
    decode_s = t_end - t_first
    # min_new_tokens forces the count, so we know it exactly.
    num_new = max_new_tokens
    decode_tok_per_s = (num_new - 1) / decode_s if decode_s > 0 else 0.0

    return {
        "ttft_s": ttft,
        "decode_s": decode_s,
        "total_s": t_end - t_start,
        "num_new_tokens": num_new,
        "decode_tok_per_s": decode_tok_per_s,
        "gpu_mem_peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    console = Console()

    tokenizer, model = load_model()
    all_records = []

    with out_path.open("w") as f:
        for prompt_len in PROMPT_LENGTHS:
            input_ids, attn_mask = build_prompt_of_length(tokenizer, prompt_len)

            # Per-length warmup. cuBLAS autotune is shape-specific; the first
            # call at this prompt_len pays a one-time setup cost we don't
            # want polluting the measured trials.
            console.print(f"[dim]warmup p={prompt_len}...[/dim]")
            torch.cuda.reset_peak_memory_stats()
            _ = run_trial(model, tokenizer, input_ids, attn_mask,
                          max_new_tokens=8)

            # Reset peak again so the recorded mem reflects only the timed runs.
            torch.cuda.reset_peak_memory_stats()

            for trial_idx in range(N_TRIALS):
                r = run_trial(model, tokenizer, input_ids, attn_mask,
                              OUTPUT_TOKENS)
                rec = {
                    "config_label": f"p{prompt_len}_o{OUTPUT_TOKENS}",
                    "prompt_len": prompt_len,
                    "max_new_tokens": OUTPUT_TOKENS,
                    "trial_idx": trial_idx,
                    "timestamp": datetime.now().isoformat(),
                    **r,
                }
                f.write(json.dumps(rec) + "\n")
                f.flush()
                all_records.append(rec)
                console.print(
                    f"  p={prompt_len:>4} trial={trial_idx} "
                    f"TTFT={r['ttft_s']*1000:7.1f}ms  "
                    f"decode={r['decode_s']*1000:7.1f}ms "
                    f"({r['decode_tok_per_s']:5.1f} tok/s)  "
                    f"peak={r['gpu_mem_peak_mb']:.0f}MB"
                )

    # Summary
    table = Table(
        title=f"Baseline summary (n={N_TRIALS} per row, output={OUTPUT_TOKENS} tok)"
    )
    table.add_column("prompt_len", justify="right")
    table.add_column("TTFT mean (ms)", justify="right")
    table.add_column("TTFT std (ms)", justify="right")
    table.add_column("decode mean (tok/s)", justify="right")
    table.add_column("decode std (tok/s)", justify="right")
    table.add_column("peak mem (MB)", justify="right")

    for prompt_len in PROMPT_LENGTHS:
        rows = [r for r in all_records if r["prompt_len"] == prompt_len]
        ttfts_ms = [r["ttft_s"] * 1000 for r in rows]
        decs = [r["decode_tok_per_s"] for r in rows]
        peak = max(r["gpu_mem_peak_mb"] for r in rows)
        ttft_std = f"{statistics.stdev(ttfts_ms):.1f}" if len(ttfts_ms) > 1 else "-"
        dec_std = f"{statistics.stdev(decs):.2f}" if len(decs) > 1 else "-"
        table.add_row(
            str(prompt_len),
            f"{statistics.mean(ttfts_ms):.1f}",
            ttft_std,
            f"{statistics.mean(decs):.1f}",
            dec_std,
            f"{peak:.0f}",
        )

    console.print()
    console.print(table)
    console.print(f"\nRaw trials → {out_path}")


if __name__ == "__main__":
    main()
