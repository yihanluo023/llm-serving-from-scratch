import asyncio
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.server.continuous_batcher import ContinuousBatcher


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def assert_result_ok(name: str, result: dict, max_tokens: int) -> None:
    print(f"\n[{name}] result:")
    print(f"text: {result['text']!r}")
    print(f"prompt_tokens: {result['prompt_tokens']}")
    print(f"output_tokens: {result['output_tokens']}")
    print(f"queue_wait_ms: {result['queue_wait_ms']:.2f}")
    print(f"batch_decode_ms: {result['batch_decode_ms']:.2f}")
    print(f"total_ms: {result['total_ms']:.2f}")
    print(f"batch_size: {result['batch_size']}")
    print(f"batch_decode_tok_per_s: {result['batch_decode_tok_per_s']:.2f}")

    assert isinstance(result["text"], str)
    assert result["prompt_tokens"] > 0
    assert 1 <= result["output_tokens"] <= max_tokens
    assert result["total_ms"] >= 0


async def test_single_request(batcher: ContinuousBatcher) -> None:
    print("\n=== test_single_request ===")

    result = await batcher.submit(
        prompt="Answer with one short sentence: what is 2 + 2?",
        max_tokens=16,
    )

    assert_result_ok("single", result, max_tokens=16)


async def test_concurrent_requests(batcher: ContinuousBatcher) -> None:
    print("\n=== test_concurrent_requests ===")

    prompts = [
        "Answer briefly: what is Python?",
        "Answer briefly: what is a GPU?",
        "Answer briefly: what is batching?",
        "Answer briefly: what is a token?",
    ]

    tasks = [
        asyncio.create_task(batcher.submit(prompt, max_tokens=24))
        for prompt in prompts
    ]

    results = await asyncio.gather(*tasks)

    for i, result in enumerate(results):
        assert_result_ok(f"concurrent-{i}", result, max_tokens=24)


async def test_different_lengths_and_cleanup(batcher: ContinuousBatcher) -> None:
    print("\n=== test_different_lengths_and_cleanup ===")

    tasks = [
        asyncio.create_task(
            batcher.submit("Say exactly one word: cat.", max_tokens=4)
        ),
        asyncio.create_task(
            batcher.submit("Write a very short explanation of KV cache.", max_tokens=32)
        ),
        asyncio.create_task(
            batcher.submit("Answer briefly: what is CUDA?", max_tokens=16)
        ),
    ]

    results = await asyncio.gather(*tasks)

    max_tokens_list = [4, 32, 16]
    for i, (result, max_tokens) in enumerate(zip(results, max_tokens_list)):
        assert_result_ok(f"different-length-{i}", result, max_tokens=max_tokens)

    # After all requests finish, scheduler should have no active state left.
    # Give the background loop one scheduling tick to run cleanup.
    await asyncio.sleep(0.05)

    assert len(batcher.active) == 0
    assert batcher.past_key_values is None

    print("\n[cleanup] active is empty and past_key_values is None")


async def test_dynamic_joining(batcher: ContinuousBatcher) -> None:
    print("\n=== test_dynamic_joining ===")

    # Start one longer request first.
    long_task = asyncio.create_task(
        batcher.submit(
            "Write a short paragraph explaining why batching improves GPU throughput.",
            max_tokens=48,
        )
    )

    # Let the long request enter active decoding.
    await asyncio.sleep(0.08)

    # These requests should join while the first request is still running.
    late_tasks = [
        asyncio.create_task(
            batcher.submit("Answer briefly: what is latency?", max_tokens=16)
        ),
        asyncio.create_task(
            batcher.submit("Answer briefly: what is throughput?", max_tokens=16)
        ),
    ]

    results = await asyncio.gather(long_task, *late_tasks)

    max_tokens_list = [48, 16, 16]
    for i, (result, max_tokens) in enumerate(zip(results, max_tokens_list)):
        assert_result_ok(f"dynamic-join-{i}", result, max_tokens=max_tokens)

    await asyncio.sleep(0.05)

    assert len(batcher.active) == 0
    assert batcher.past_key_values is None

    print("\n[dynamic cleanup] active is empty and past_key_values is None")


async def main() -> None:
    print(f"Loading {MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()

    print(f"Loaded. device={model.device}, dtype={model.dtype}")

    batcher = ContinuousBatcher(
        model=model,
        tokenizer=tokenizer,
        max_batch=8,
        max_wait_ms=50.0,
    )

    await batcher.start()

    try:
        t0 = time.perf_counter()

        await test_single_request(batcher)
        await test_concurrent_requests(batcher)
        await test_different_lengths_and_cleanup(batcher)
        await test_dynamic_joining(batcher)

        elapsed = time.perf_counter() - t0
        print(f"\nALL CONTINUOUS BATCHER SMOKE TESTS PASSED in {elapsed:.2f}s")

    finally:
        await batcher.stop()
        del model
        del tokenizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    asyncio.run(main())