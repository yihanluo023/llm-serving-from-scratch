"""
HTTP benchmark for the FastAPI LLM serving server.

This script benchmarks server endpoints, not raw model execution.

Usage:
    # Terminal 1
    uv run uvicorn src.server.app:app --host 0.0.0.0 --port 8000

    # Terminal 2
    uv run python scripts/benchmark_server.py

What it measures:
- /generate_serial: naive per-request generate() baseline
- /generate: queue-based static batching path

Main metrics:
- req/s: completed requests per second
- scheduled_tok/s: requested decode token slots per second
- returned_tok/s: actual returned output tokens per second
- p50/p95 latency: client-side end-to-end latency
- queue wait: server-side batching queue wait, only for /generate
- batch size: observed batch size, only for /generate
"""

import asyncio
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.table import Table


BASE_URL = "http://localhost:8000"

ENDPOINTS = [
    "/generate_serial",
    "/generate",
]

CONCURRENCIES = [1, 2, 4, 8, 16]

NUM_REQUESTS = 32
MAX_TOKENS = 64

REQUEST_TIMEOUT_S = 120.0
OUTPUT_DIR = Path("benchmarks")

PROMPT = (
    "Explain what request batching means in one concise paragraph. "
    "Focus on the tradeoff between throughput and latency."
)


def percentile(values: list[float], p: float) -> float:
    """
    Return percentile using nearest-rank style indexing.

    values:
        List of numeric values.
    p:
        Percentile in [0, 100], e.g. 50 or 95.
    """
    if not values:
        return 0.0

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    idx = round((p / 100) * (len(sorted_values) - 1))
    return sorted_values[idx]


async def send_one_request(
    client: httpx.AsyncClient,
    endpoint: str,
    request_idx: int,
    concurrency: int,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    """
    Send one HTTP request and return a per-request record.

    The record includes both client-side timing and server-returned metrics.
    """
    url = f"{BASE_URL}{endpoint}"
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
    }

    t0 = time.perf_counter()

    try:
        response = await client.post(url, json=payload)
        t1 = time.perf_counter()

        client_latency_ms = (t1 - t0) * 1000

        response.raise_for_status()
        data = response.json()

        return {
            "ok": True,
            "endpoint": endpoint,
            "concurrency": concurrency,
            "request_idx": request_idx,
            "client_latency_ms": client_latency_ms,

            # Common response fields.
            "output_tokens": data.get("output_tokens", 0),
            "prompt_tokens": data.get("prompt_tokens", 0),
            "text": data.get("text", ""),

            # Present on /generate.
            # Missing on /generate_serial, so keep None.
            "queue_wait_ms": data.get("queue_wait_ms"),
            "batch_decode_ms": data.get("batch_decode_ms"),
            "batch_size": data.get("batch_size"),

            # Present on /generate_serial.
            "ttft_ms": data.get("ttft_ms"),
            "decode_ms": data.get("decode_ms"),

            "error": None,
        }

    except Exception as e:
        t1 = time.perf_counter()
        return {
            "ok": False,
            "endpoint": endpoint,
            "concurrency": concurrency,
            "request_idx": request_idx,
            "client_latency_ms": (t1 - t0) * 1000,
            "output_tokens": 0,
            "prompt_tokens": 0,
            "text": "",
            "queue_wait_ms": None,
            "batch_decode_ms": None,
            "batch_size": None,
            "ttft_ms": None,
            "decode_ms": None,
            "error": repr(e),
        }


async def run_one_config(
    client: httpx.AsyncClient,
    endpoint: str,
    concurrency: int,
    num_requests: int,
    prompt: str,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], float]:
    """
    Run one benchmark config.

    Config = one endpoint + one concurrency level.

    Returns:
        records:
            One record per request.
        wall_s:
            Wall-clock time from launching the first task to receiving
            the final response.
    """
    sem = asyncio.Semaphore(concurrency)

    async def bounded_request(i: int) -> dict[str, Any]:
        async with sem:
            return await send_one_request(
                client=client,
                endpoint=endpoint,
                request_idx=i,
                concurrency=concurrency,
                prompt=prompt,
                max_tokens=max_tokens,
            )

    tasks = [bounded_request(i) for i in range(num_requests)]

    t_start = time.perf_counter()
    records = await asyncio.gather(*tasks)
    t_end = time.perf_counter()

    wall_s = t_end - t_start
    return records, wall_s


def summarize_config(
    records: list[dict[str, Any]],
    wall_s: float,
    max_tokens: int,
) -> dict[str, Any]:
    """
    Summarize one config into one table row.
    """
    if not records:
        raise ValueError("Cannot summarize empty records.")

    endpoint = records[0]["endpoint"]
    concurrency = records[0]["concurrency"]

    ok_records = [r for r in records if r["ok"]]
    failed = len(records) - len(ok_records)

    latencies = [r["client_latency_ms"] for r in ok_records]
    output_tokens = [r["output_tokens"] for r in ok_records]

    queue_waits = [
        r["queue_wait_ms"]
        for r in ok_records
        if r["queue_wait_ms"] is not None
    ]

    batch_sizes = [
        r["batch_size"]
        for r in ok_records
        if r["batch_size"] is not None
    ]

    completed = len(ok_records)

    req_per_s = completed / wall_s if wall_s > 0 else 0.0

    # Scheduled token slots are based on requested decode budget.
    # This is stable even if the model emits EOS early.
    scheduled_tok_per_s = (
        completed * max_tokens / wall_s
        if wall_s > 0 else 0.0
    )

    returned_tok_per_s = (
        sum(output_tokens) / wall_s
        if wall_s > 0 else 0.0
    )

    return {
        "endpoint": endpoint,
        "concurrency": concurrency,
        "num_requests": len(records),
        "completed": completed,
        "failed": failed,
        "wall_s": wall_s,
        "req_per_s": req_per_s,
        "scheduled_tok_per_s": scheduled_tok_per_s,
        "returned_tok_per_s": returned_tok_per_s,
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "mean_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "mean_queue_wait_ms": (
            statistics.mean(queue_waits) if queue_waits else None
        ),
        "mean_batch_size": (
            statistics.mean(batch_sizes) if batch_sizes else None
        ),
        "max_batch_size": max(batch_sizes) if batch_sizes else None,
    }


def print_summary_table(console: Console, summaries: list[dict[str, Any]]) -> None:
    table = Table(
        title=(
            f"Server benchmark "
            f"(num_requests={NUM_REQUESTS}, max_tokens={MAX_TOKENS})"
        )
    )

    table.add_column("endpoint", justify="left")
    table.add_column("conc", justify="right")
    table.add_column("req/s", justify="right")
    table.add_column("scheduled tok/s", justify="right")
    table.add_column("returned tok/s", justify="right")
    table.add_column("p50 lat ms", justify="right")
    table.add_column("p95 lat ms", justify="right")
    table.add_column("mean queue ms", justify="right")
    table.add_column("mean batch", justify="right")
    table.add_column("max batch", justify="right")
    table.add_column("fail", justify="right")

    for s in summaries:
        mean_queue = (
            f"{s['mean_queue_wait_ms']:.1f}"
            if s["mean_queue_wait_ms"] is not None
            else "-"
        )

        mean_batch = (
            f"{s['mean_batch_size']:.2f}"
            if s["mean_batch_size"] is not None
            else "-"
        )

        max_batch = (
            str(s["max_batch_size"])
            if s["max_batch_size"] is not None
            else "-"
        )

        table.add_row(
            s["endpoint"],
            str(s["concurrency"]),
            f"{s['req_per_s']:.2f}",
            f"{s['scheduled_tok_per_s']:.1f}",
            f"{s['returned_tok_per_s']:.1f}",
            f"{s['p50_latency_ms']:.1f}",
            f"{s['p95_latency_ms']:.1f}",
            mean_queue,
            mean_batch,
            max_batch,
            str(s["failed"]),
        )

    console.print()
    console.print(table)


async def main() -> None:
    console = Console()
    OUTPUT_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = OUTPUT_DIR / f"server_benchmark_raw_{timestamp}.jsonl"
    summary_path = OUTPUT_DIR / f"server_benchmark_summary_{timestamp}.json"

    timeout = httpx.Timeout(REQUEST_TIMEOUT_S)

    all_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=timeout) as client:
        # Lightweight server health check before running benchmark.
        try:
            health = await client.get(f"{BASE_URL}/")
            health.raise_for_status()
            console.print(f"[green]Server health check OK:[/green] {health.json()}")
        except Exception as e:
            console.print(f"[red]Server health check failed:[/red] {repr(e)}")
            console.print("Make sure the server is running:")
            console.print("  uvicorn src.server.app:app --host 0.0.0.0 --port 8000")
            return

        for endpoint in ENDPOINTS:
            for concurrency in CONCURRENCIES:
                console.print(
                    f"\n[bold]Running[/bold] "
                    f"endpoint={endpoint} concurrency={concurrency} "
                    f"num_requests={NUM_REQUESTS} max_tokens={MAX_TOKENS}"
                )

                records, wall_s = await run_one_config(
                    client=client,
                    endpoint=endpoint,
                    concurrency=concurrency,
                    num_requests=NUM_REQUESTS,
                    prompt=PROMPT,
                    max_tokens=MAX_TOKENS,
                )

                summary = summarize_config(
                    records=records,
                    wall_s=wall_s,
                    max_tokens=MAX_TOKENS,
                )

                all_records.extend(records)
                summaries.append(summary)

                console.print(
                    f"  wall={wall_s:.2f}s "
                    f"req/s={summary['req_per_s']:.2f} "
                    f"p50={summary['p50_latency_ms']:.1f}ms "
                    f"p95={summary['p95_latency_ms']:.1f}ms "
                    f"failed={summary['failed']}"
                )

                # If many requests fail, stop early so we do not spam a broken server.
                if summary["failed"] > 0:
                    console.print(
                        "[yellow]Some requests failed. "
                        "Continuing, but check raw JSONL for errors.[/yellow]"
                    )

    # Save raw per-request records.
    with raw_path.open("w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    # Save summarized rows.
    with summary_path.open("w") as f:
        json.dump(summaries, f, indent=2)

    print_summary_table(console, summaries)

    console.print(f"\nRaw records → {raw_path}")
    console.print(f"Summary → {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())