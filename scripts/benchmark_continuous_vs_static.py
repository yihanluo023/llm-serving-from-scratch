"""
Continuous vs static batching benchmark (Phase 2 headline).

Goal: isolate and quantify what continuous batching buys over static batching.
The win is NOT raw throughput on a uniform workload — it's the elimination of
head-of-line blocking when request lengths are heterogeneous and arrivals are
spread over time. So we run two experiments against the SAME fixed-seed
workload, one endpoint at a time:

  Experiment A — CONTROL (uniform output length, same arrival schedule)
      Expectation: continuous ≈ static. Proves continuous doesn't regress and
      anchors the claim that B's gap comes specifically from heterogeneity.

  Experiment B — HEADLINE (heterogeneous length: a few long + many short)
      Open-loop arrivals (Poisson), so requests keep arriving while a batch is
      in flight. Static makes short requests wait for (a) the longest request
      in their batch and (b) the whole in-flight batch to drain before they are
      even admitted. Continuous lets short rows leave at EOS and new rows join
      mid-decode. The short-request p99 end-to-end latency is the headline.

Load model is OPEN-LOOP: every request fires at its scheduled arrival time
regardless of whether earlier ones have returned (real serving traffic, not a
closed-loop client that only sends the next request after one completes).

Metrics per request class (short / long):
  - e2e total_ms (server-side, enqueue → resolve): p50 / p99
  - client_ms (HTTP round trip): p50 / p99
  - queue_wait_ms (server-side): p50 / p99. Means different things per endpoint:
      continuous → ≈ TTFT (first token produced at prefill); static → pre-decode
      batch wait (excludes prefill). Kept under the honest generic name.
  - throughput: completed req/s, returned output tok/s

Usage:
    # Terminal 1 (NO --reload for clean numbers)
    uv run uvicorn src.server.app:app --host 0.0.0.0 --port 8000

    # Terminal 2
    uv run python scripts/benchmark_continuous_vs_static.py
"""

import asyncio
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.table import Table


BASE_URL = "http://localhost:8000"

STATIC_ENDPOINT = "/generate"
CONTINUOUS_ENDPOINT = "/generate_continuous"

# Reproducibility: same arrivals + class assignment for every endpoint/experiment.
SEED = 1234

# Workload size and burst shape. Keep NUM/WINDOW (arrival rate) fixed when
# scaling N — that preserves the load regime and only sharpens the percentiles.
NUM_REQUESTS = 96
# Poisson arrivals over this window (seconds). NUM/WINDOW = mean arrival rate.
ARRIVAL_WINDOW_S = 10.0

# Experiment B (heterogeneous) length mix.
LONG_FRACTION = 0.25
SHORT_MAX_TOKENS = 16
LONG_MAX_TOKENS = 160

# Experiment A (control) uniform length.
UNIFORM_MAX_TOKENS = 64

REQUEST_TIMEOUT_S = 600.0
# Let the server fully drain between runs so back-to-back runs don't bleed.
COOLDOWN_S = 2.0

OUTPUT_DIR = Path("benchmarks")

TOPICS = [
    "continuous batching", "KV cache", "prefill", "decode latency",
    "GPU memory bandwidth", "tokenization", "attention", "throughput",
    "left padding", "the transformer block", "sampling temperature",
    "speculative decoding", "PagedAttention", "model quantization",
    "tensor parallelism", "the softmax function", "positional embeddings",
    "batch scheduling", "warmup", "FP16 vs BF16", "the EOS token",
    "prompt length", "request queueing", "head-of-line blocking",
]


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile, p in [0, 100]."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = round((p / 100) * (len(s) - 1))
    return s[idx]


def generate_workload(seed: int) -> list[dict[str, Any]]:
    """
    Build the shared base workload: arrival times (Poisson) + class assignment.

    Returned items are length-agnostic; the per-experiment views below decide
    the prompt and max_tokens. This keeps arrivals and class labels identical
    across experiments and endpoints, so any latency difference is attributable
    to the scheduler, not to luck.
    """
    rng = random.Random(seed)
    rate = NUM_REQUESTS / ARRIVAL_WINDOW_S

    items: list[dict[str, Any]] = []
    t = 0.0
    for i in range(NUM_REQUESTS):
        t += rng.expovariate(rate)
        cls = "long" if rng.random() < LONG_FRACTION else "short"
        topic = TOPICS[i % len(TOPICS)]
        items.append({
            "id": i,
            "arrival_s": t,
            "cls": cls,
            "topic": topic,
        })
    return items


def view_control(item: dict[str, Any]) -> dict[str, Any]:
    """Experiment A: every request same length, prompts uniform-ish."""
    return {
        **item,
        "cls": "uniform",
        "prompt": f"Explain {item['topic']} in a short paragraph.",
        "max_tokens": UNIFORM_MAX_TOKENS,
    }


def view_headline(item: dict[str, Any]) -> dict[str, Any]:
    """Experiment B: long class gets a big budget, short class a tiny one."""
    if item["cls"] == "long":
        prompt = f"Write a detailed, multi-paragraph explanation of {item['topic']}."
        max_tokens = LONG_MAX_TOKENS
    else:
        prompt = f"Define {item['topic']} in one short sentence."
        max_tokens = SHORT_MAX_TOKENS
    return {**item, "prompt": prompt, "max_tokens": max_tokens}


async def fire_one(
    client: httpx.AsyncClient,
    item: dict[str, Any],
    t0: float,
    endpoint: str,
) -> dict[str, Any]:
    """
    Sleep until this request's scheduled arrival, then send it. Open-loop:
    the send time depends only on the schedule, never on other requests.
    """
    target = t0 + item["arrival_s"]
    delay = target - time.perf_counter()
    if delay > 0:
        await asyncio.sleep(delay)

    send = time.perf_counter()
    try:
        resp = await client.post(
            f"{BASE_URL}{endpoint}",
            json={"prompt": item["prompt"], "max_tokens": item["max_tokens"]},
        )
        recv = time.perf_counter()
        resp.raise_for_status()
        d = resp.json()
        return {
            "ok": True,
            "endpoint": endpoint,
            "id": item["id"],
            "cls": item["cls"],
            "arrival_s": item["arrival_s"],
            "send_s": send - t0,
            "recv_s": recv - t0,
            "client_ms": (recv - send) * 1000,
            "total_ms": d["total_ms"],
            # Server-side queue wait. NOTE: means different things per endpoint —
            # for continuous it ≈ TTFT (first token is produced at prefill), for
            # static it's the pre-decode batch wait (excludes prefill). So we
            # keep the honest generic name rather than calling it ttft.
            "queue_wait_ms": d["queue_wait_ms"],
            "output_tokens": d["output_tokens"],
            "batch_size": d["batch_size"],
            "error": None,
        }
    except Exception as e:
        recv = time.perf_counter()
        return {
            "ok": False,
            "endpoint": endpoint,
            "id": item["id"],
            "cls": item["cls"],
            "arrival_s": item["arrival_s"],
            "send_s": send - t0,
            "recv_s": recv - t0,
            "client_ms": (recv - send) * 1000,
            "total_ms": None,
            "queue_wait_ms": None,
            "output_tokens": 0,
            "batch_size": None,
            "error": repr(e),
        }


async def run_endpoint(
    client: httpx.AsyncClient,
    requests: list[dict[str, Any]],
    endpoint: str,
) -> tuple[list[dict[str, Any]], float]:
    """Drive one endpoint with the open-loop schedule. Returns (records, wall_s)."""
    t0 = time.perf_counter()
    tasks = [
        asyncio.create_task(fire_one(client, item, t0, endpoint))
        for item in requests
    ]
    records = await asyncio.gather(*tasks)
    wall_s = max(r["recv_s"] for r in records) - min(r["send_s"] for r in records)
    return records, wall_s


def summarize(
    records: list[dict[str, Any]],
    wall_s: float,
) -> dict[str, dict[str, Any]]:
    """Summarize one endpoint run, broken down by request class plus 'all'."""
    ok = [r for r in records if r["ok"]]
    classes = sorted({r["cls"] for r in records})

    out: dict[str, dict[str, Any]] = {}
    for cls in [*classes, "all"]:
        rows = ok if cls == "all" else [r for r in ok if r["cls"] == cls]
        if not rows:
            continue
        e2e = [r["total_ms"] for r in rows]
        client = [r["client_ms"] for r in rows]
        queue_wait = [r["queue_wait_ms"] for r in rows]
        out_tok = sum(r["output_tokens"] for r in rows)
        out[cls] = {
            "count": len(rows),
            "e2e_p50": percentile(e2e, 50),
            "e2e_p99": percentile(e2e, 99),
            "client_p50": percentile(client, 50),
            "client_p99": percentile(client, 99),
            "queue_wait_p50": percentile(queue_wait, 50),
            "queue_wait_p99": percentile(queue_wait, 99),
            "req_per_s": len(rows) / wall_s if wall_s > 0 else 0.0,
            "out_tok_per_s": out_tok / wall_s if wall_s > 0 else 0.0,
        }
    out["_meta"] = {"wall_s": wall_s, "failed": len(records) - len(ok)}
    return out


def print_experiment(
    console: Console,
    title: str,
    static_summary: dict[str, dict[str, Any]],
    cont_summary: dict[str, dict[str, Any]],
) -> None:
    table = Table(title=title)
    table.add_column("endpoint", justify="left")
    table.add_column("class", justify="left")
    table.add_column("n", justify="right")
    table.add_column("e2e p50", justify="right")
    table.add_column("e2e p99", justify="right")
    table.add_column("qwait p50", justify="right")
    table.add_column("qwait p99", justify="right")
    table.add_column("req/s", justify="right")
    table.add_column("out tok/s", justify="right")

    def add_rows(label: str, summary: dict[str, dict[str, Any]]) -> None:
        classes = [c for c in summary if c != "_meta"]
        # Put 'all' last.
        classes = sorted(classes, key=lambda c: (c == "all", c))
        for cls in classes:
            s = summary[cls]
            table.add_row(
                label,
                cls,
                str(s["count"]),
                f"{s['e2e_p50']:.0f}",
                f"{s['e2e_p99']:.0f}",
                f"{s['queue_wait_p50']:.0f}",
                f"{s['queue_wait_p99']:.0f}",
                f"{s['req_per_s']:.2f}",
                f"{s['out_tok_per_s']:.1f}",
            )

    add_rows("static", static_summary)
    table.add_section()
    add_rows("continuous", cont_summary)

    console.print()
    console.print(table)

    # Headline one-liner: focus-class e2e p99, stated in the actual direction
    # (whichever endpoint is faster gets reported as an N× speedup — never
    # assume continuous wins; in the control experiment static usually does).
    focus = "short" if "short" in static_summary else "uniform"
    if focus in static_summary and focus in cont_summary:
        s99 = static_summary[focus]["e2e_p99"]
        c99 = cont_summary[focus]["e2e_p99"]
        if s99 > 0 and c99 > 0:
            if c99 < s99:
                verdict = f"[bold green]continuous {s99 / c99:.2f}× faster[/bold green]"
            elif s99 < c99:
                verdict = f"[bold yellow]static {c99 / s99:.2f}× faster[/bold yellow]"
            else:
                verdict = "[bold]tied[/bold]"
            console.print(
                f"  → [bold]{focus}[/bold] e2e p99: "
                f"static {s99:.0f} ms vs continuous {c99:.0f} ms  →  {verdict}"
            )


async def main() -> None:
    console = Console()
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = OUTPUT_DIR / f"cont_vs_static_raw_{timestamp}.jsonl"
    summary_path = OUTPUT_DIR / f"cont_vs_static_summary_{timestamp}.json"

    base = generate_workload(SEED)
    experiments = [
        ("Experiment A — CONTROL (uniform length)", [view_control(i) for i in base]),
        ("Experiment B — HEADLINE (heterogeneous length)", [view_headline(i) for i in base]),
    ]

    all_records: list[dict[str, Any]] = []
    all_summaries: dict[str, Any] = {}

    timeout = httpx.Timeout(REQUEST_TIMEOUT_S)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            health = await client.get(f"{BASE_URL}/")
            health.raise_for_status()
            console.print(f"[green]Server OK:[/green] {health.json()}")
        except Exception as e:
            console.print(f"[red]Server health check failed:[/red] {repr(e)}")
            console.print("Start it with: uvicorn src.server.app:app --port 8000")
            return

        for title, requests in experiments:
            console.print(f"\n[bold]{title}[/bold]  "
                          f"(n={NUM_REQUESTS}, window={ARRIVAL_WINDOW_S}s, seed={SEED})")

            console.print("  running static ...")
            static_records, static_wall = await run_endpoint(
                client, requests, STATIC_ENDPOINT
            )
            await asyncio.sleep(COOLDOWN_S)

            console.print("  running continuous ...")
            cont_records, cont_wall = await run_endpoint(
                client, requests, CONTINUOUS_ENDPOINT
            )
            await asyncio.sleep(COOLDOWN_S)

            static_summary = summarize(static_records, static_wall)
            cont_summary = summarize(cont_records, cont_wall)

            print_experiment(console, title, static_summary, cont_summary)

            for r in (*static_records, *cont_records):
                r["experiment"] = title
            all_records.extend(static_records)
            all_records.extend(cont_records)
            all_summaries[title] = {
                "static": static_summary,
                "continuous": cont_summary,
            }

    with raw_path.open("w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    with summary_path.open("w") as f:
        json.dump(all_summaries, f, indent=2)

    console.print(f"\nRaw  → {raw_path}")
    console.print(f"Summary → {summary_path}")
    console.print(
        "\n[dim]How to read: in A (uniform) the two are close — continuous may "
        "even be a touch slower, since its per-step scheduling is pure overhead "
        "when there's no head-of-line blocking to fix. In B (heterogeneous) "
        "continuous should be sharply faster on the short class; that gap is the "
        "head-of-line blocking static cannot avoid.[/dim]"
    )


if __name__ == "__main__":
    asyncio.run(main())
