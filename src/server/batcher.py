"""
Static batcher for Phase 1.

One background coroutine collects pending requests from an asyncio.Queue
until it gathers `max_batch` items or `max_wait_ms` elapses since the first
arrival. It then runs model.generate() once on the padded batch and resolves
each item's Future with its slice of the output.

Properties this design preserves:
- Only this batcher coroutine ever calls model.generate(). HTTP handlers
  never touch the model directly. This serializes GPU access without locks.
- model.generate() is synchronous-blocking; it's dispatched via
  loop.run_in_executor() so it doesn't freeze the event loop while other
  handlers await their Futures, the HTTP server accepts new connections,
  and the queue keeps filling.
- Left padding (padding_side="left") so the rightmost position of every
  row is the "next token to generate" slot. With right padding, the new
  tokens would land after the pad block.

Known warts preserved by design (this is Phase 1, not Phase 2):
- The entire batch must finish max_new_tokens before *any* request returns,
  even if one hits EOS early. This is the chair-shuffling problem we want
  the benchmark to expose; Phase 2 (continuous batching) will fix it.
- Mixed max_tokens across a batch is rounded up to max(); shorter-budget
  requests still pay for the longest one's decode tail. Same reasoning.
"""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class _PendingItem:
    prompt: str
    max_tokens: int
    future: asyncio.Future
    t_enqueue: float = field(default_factory=time.perf_counter)


class Batcher:
    def __init__(
        self,
        model,
        tokenizer,
        max_batch: int = 8,
        max_wait_ms: float = 50.0,
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch = max_batch
        self.max_wait_s = max_wait_ms / 1000.0
        self.system_prompt = system_prompt

        # Tokenizers without an explicit pad token need one for batched
        # padding. Qwen2.5's tokenizer ships with pad_token set, but be
        # defensive — falling back to eos is the standard recipe and is
        # safe as long as attention_mask masks the padding positions out
        # (which we always do below).
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.queue: asyncio.Queue[_PendingItem] = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._stopping = False

    # ---------- lifecycle ----------

    async def start(self) -> None:
        assert self._task is None, "Batcher already started"
        self._task = asyncio.create_task(self._run(), name="batcher")

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # ---------- public submit ----------

    async def submit(self, prompt: str, max_tokens: int) -> dict:
        """
        Submit one request and await its batched result.

        Returns a dict matching the GenerateResponse schema in app.py.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        item = _PendingItem(prompt=prompt, max_tokens=max_tokens, future=future)
        await self.queue.put(item)
        return await future

    # ---------- core loop ----------

    async def _run(self) -> None:
        """The single coroutine that owns the GPU."""
        while not self._stopping:
            try:
                items = await self._collect_batch()
            except asyncio.CancelledError:
                raise
            if not items:
                continue

            try:
                results = await self._run_batch(items)
            except Exception as e:
                # Don't kill the loop; fail just this batch's callers.
                for item in items:
                    if not item.future.done():
                        item.future.set_exception(e)
                continue

            for item, result in zip(items, results):
                if not item.future.done():
                    item.future.set_result(result)

    async def _collect_batch(self) -> list[_PendingItem]:
        """
        Block on the first arrival, then opportunistically gather up to
        max_batch-1 more within max_wait_s of that first arrival.

        Timing convention: the deadline anchors to the moment the batcher
        first SEES the head-of-queue item (i.e. now, after `await get()`),
        NOT to that item's t_enqueue. Under steady-state load these are
        the same. Under overload they diverge: the head item may have
        been waiting in the queue for seconds while a previous batch was
        on the GPU, so its t_enqueue is already far in the past. Anchoring
        to t_enqueue would make `remaining` instantly negative and we'd
        ship single-item batches, collapsing throughput. We pick the
        first-pop anchor instead — under overload we trade the per-item
        queue-wait bound for keeping batches full, which is the policy
        every production serving stack converges to.
        """
        first: _PendingItem = await self.queue.get()
        items = [first]

        deadline = time.perf_counter() + self.max_wait_s

        while len(items) < self.max_batch:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                more = await asyncio.wait_for(self.queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            items.append(more)
        return items

    async def _run_batch(self, items: list[_PendingItem]) -> list[dict]:
        """
        Run model.generate() once on the padded batch. The synchronous
        work is dispatched to the default thread executor so this
        coroutine doesn't stall the event loop while the GPU is busy.
        """
        loop = asyncio.get_running_loop()
        t_batch_start = time.perf_counter()
        outputs = await loop.run_in_executor(
            None, self._generate_batch_sync, items
        )
        t_batch_end = time.perf_counter()

        results = []
        decode_s = t_batch_end - t_batch_start
        for item, (text, prompt_len, output_len) in zip(items, outputs):
            queue_wait_s = t_batch_start - item.t_enqueue
            results.append({
                "text": text,
                "prompt_tokens": prompt_len,
                "output_tokens": output_len,
                "queue_wait_ms": queue_wait_s * 1000,
                "batch_decode_ms": decode_s * 1000,
                "total_ms": (t_batch_end - item.t_enqueue) * 1000,
                "batch_size": len(items),
                "batch_decode_tok_per_s": (
                    (output_len - 1) / decode_s
                    if decode_s > 0 and output_len > 1 else 0.0
                ),
            })
        return results

    def _generate_batch_sync(self, items: list[_PendingItem]) -> list[tuple[str, int, int]]:
        """
        Manual static batching with forward() + KV cache.

        Returns:
            list of (text, true_prompt_len, output_len), same order as items.
        """
        tok = self.tokenizer
        device = self.model.device

        # 1. Build chat-formatted prompts.
        prompt_texts = [
            tok.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": item.prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for item in items
        ]

        # 2. Tokenize as a left-padded batch.
        old_side = tok.padding_side
        tok.padding_side = "left"
        try:
            enc = tok(
                prompt_texts,
                return_tensors="pt",
                padding=True,
            ).to(device)
        finally:
            tok.padding_side = old_side

        input_ids = enc.input_ids                 # [B, T]
        attention_mask = enc.attention_mask       # [B, T]

        batch_size = input_ids.shape[0]
        batch_max_new = max(item.max_tokens for item in items)

        eos_id = tok.eos_token_id
        if eos_id is None:
            raise ValueError("Tokenizer must have eos_token_id for this implementation.")

        eos_token = torch.full(
            (batch_size,),
            eos_id,
            dtype=input_ids.dtype,
            device=device,
        )

        # Position ids:
        # For left padding, real tokens should get positions 0,1,2,...
        # Pad tokens get 0, but they are masked out by attention_mask.
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)

        # 3. Prefill
        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )

        past_key_values = outputs.past_key_values

        next_token_logits = outputs.logits[:, -1, :]   # [B, vocab_size]

        generated_tokens: list[torch.Tensor] = []

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        current_len = input_ids.shape[1]

        # 4. Decode loop: generate one token per row per step.
        for step in range(batch_max_new):
            raw_next_token = torch.argmax(next_token_logits, dim=-1)  # [B]

            # If a row is already finished, keep feeding EOS for that row.
            next_token = torch.where(finished, eos_token, raw_next_token)

            generated_tokens.append(next_token)

            # Mark newly finished rows.
            finished = finished | (next_token == eos_id)

            if step == batch_max_new - 1:
                break

            # Extend attention mask by one real token for every row.
            new_mask_col = torch.ones(
                (batch_size, 1),
                dtype=attention_mask.dtype,
                device=device,
            )
            attention_mask = torch.cat([attention_mask, new_mask_col], dim=1)

            # Position id for the new token.
            # The next generated token is at position current_len for padded layout,
            # but for left-padded rows the semantic position should be
            # number of real tokens so far - 1.
            decode_position_ids = attention_mask.long().sum(dim=-1, keepdim=True) - 1

            with torch.inference_mode():
                outputs = self.model(
                    input_ids=next_token[:, None],       # [B, 1]
                    attention_mask=attention_mask,       # [B, T + generated_so_far]
                    position_ids=decode_position_ids,    # [B, 1]
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]  # [B, vocab_size]
            current_len += 1

        # 5. Stack generated tokens into pure answer ids.
        generated_ids = torch.stack(generated_tokens, dim=1)  # [B, batch_max_new]

        # 6. Decode each row back to text.
        results: list[tuple[str, int, int]] = []

        for i, item in enumerate(items):
            true_prompt_len = int(enc.attention_mask[i].sum().item())

            new_token_ids = generated_ids[i, : item.max_tokens]

            # Trim at first EOS.
            eos_positions = (new_token_ids == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                first_eos = eos_positions[0].item()
                new_token_ids = new_token_ids[:first_eos]

            text = tok.decode(new_token_ids, skip_special_tokens=True)
            output_len = int(new_token_ids.shape[0])

            results.append((text, true_prompt_len, output_len))

        return results