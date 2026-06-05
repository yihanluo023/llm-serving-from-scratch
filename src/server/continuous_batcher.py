"""
Continuous batcher for Phase 2.

One background scheduler owns the model. Requests enter a waiting queue,
get prefetched into KV cache, join the active batch, decode one token step
at a time, and leave as soon as they hit EOS or max_tokens.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, Any

import torch
from transformers import DynamicCache


@dataclass
class _PendingItem:
    prompt: str
    max_tokens: int
    future: asyncio.Future
    t_enqueue: float = field(default_factory=time.perf_counter)


@dataclass
class _RunningRequest:
    """Request that has joined the active decode batch."""

    item: _PendingItem
    prompt_tokens: int
    generated_ids: list[int]
    last_token_id: int
    real_seq_len: int
    kv_pad_len: int
    t_prefill_done: float
    t_first_token: Optional[float] = None
    finished: bool = False


class ContinuousBatcher:
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

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.queue: asyncio.Queue[_PendingItem] = asyncio.Queue()
        self.active: list[_RunningRequest] = []
        self.past_key_values: Optional[Any] = None

        self._task: Optional[asyncio.Task] = None
        self._stopping = False

    # ---------- lifecycle ----------

    async def start(self) -> None:
        assert self._task is None, "ContinuousBatcher already started"
        self._task = asyncio.create_task(
            self._run(),
            name="continuous-batcher",
        )

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

    # ---------- core scheduler ----------

    async def _run(self) -> None:
        """
        Main continuous batching loop.

        When idle, wait for an initial prefill batch. While active requests are
        decoding, drain already queued requests without waiting, prefill/merge
        them, then decode one token for the active batch.

        Model forwards run in the default executor so they do not block the
        FastAPI event loop from admitting new requests. This coroutine still
        awaits each forward before mutating scheduler state, so active/KV state
        stays serialized.
        """
        loop = asyncio.get_running_loop()
        while not self._stopping:
            try:
                self._cleanup_finished()
                self._head_cut_past_key_values()

                if not self.active:
                    items = await self._collect_initial_prefill_batch()
                else:
                    items = self._drain_waiting_queue()

                if items:
                    new_running, new_past = await loop.run_in_executor(
                        None, self._prefill_new_requests, items
                    )
                    self._merge_new_requests(new_running, new_past)
                    # Some new requests may finish immediately during prefill:
                    # first token is EOS, or max_tokens == 1.
                    self._cleanup_finished()

                if self.active:
                    await loop.run_in_executor(None, self._decode_one_step)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._fail_all(e)

    # ---------- queue admission ----------

    async def _collect_initial_prefill_batch(self) -> list[_PendingItem]:
        """
        Wait for the first request, then collect more until max_batch or
        max_wait_s. Used only when the GPU has no active decode batch.
        """
        first: _PendingItem = await self.queue.get()
        items = [first]

        deadline = time.perf_counter() + self.max_wait_s

        while len(items) < self.max_batch:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break

            try:
                more = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                break

            items.append(more)

        return items

    def _drain_waiting_queue(self) -> list[_PendingItem]:
        """
        Take already-waiting requests without blocking.

        Used while active decoding is in progress, so existing requests do
        not pause while the scheduler waits for more arrivals.
        """
        available_slots = self.max_batch - len(self.active)
        if available_slots <= 0:
            return []

        items: list[_PendingItem] = []

        while len(items) < available_slots:
            try:
                item = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            items.append(item)

        return items

    # ---------- prefill and merge ----------

    def _prefill_new_requests(
        self,
        items: list[_PendingItem],
    ) -> tuple[list[_RunningRequest], Any]:
        """
        Tokenize pending prompts, run prefill, sample each first token, and
        return running request metadata plus the new batch KV cache.
        """
        if not items:
            raise ValueError("_prefill_new_requests called with empty items")

        tok = self.tokenizer
        device = self.model.device

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

        input_ids = enc.input_ids
        attention_mask = enc.attention_mask

        eos_id = tok.eos_token_id
        if eos_id is None:
            raise ValueError("Tokenizer must have eos_token_id for this implementation.")

        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )

        new_past_key_values = self._to_legacy_past_key_values(outputs.past_key_values)
        next_token_logits = outputs.logits[:, -1, :]
        first_tokens = torch.argmax(next_token_logits, dim=-1)

        t_prefill_done = time.perf_counter()

        new_running: list[_RunningRequest] = []

        for i, item in enumerate(items):
            first_token_id = int(first_tokens[i].item())
            prompt_tokens = int(attention_mask[i].sum().item())
            padded_prompt_len = int(input_ids.shape[1])
            kv_pad_len = padded_prompt_len - prompt_tokens

            req = _RunningRequest(
                item=item,
                prompt_tokens=prompt_tokens,
                generated_ids=[first_token_id],
                last_token_id=first_token_id,
                real_seq_len=prompt_tokens,
                kv_pad_len=kv_pad_len,
                t_prefill_done=t_prefill_done,
                t_first_token=t_prefill_done,
                finished=False,
            )

            # If prefill already produced EOS, this request is logically done.
            # Cleanup will resolve it before the next real decode step.
            if first_token_id == eos_id or len(req.generated_ids) >= item.max_tokens:
                req.finished = True

            new_running.append(req)

        return new_running, new_past_key_values

    def _merge_new_requests(
        self,
        new_running: list[_RunningRequest],
        new_past_key_values: Any,
    ) -> None:
        """
        Merge newly prefetched requests into the active batch.

        KV layout invariant per row:
        [pad] * kv_pad_len + real cached tokens.

        Maintains:
        active[i] matches batch row i in every KV cache tensor.
        """
        if not new_running:
            return

        if self.past_key_values is None:
            assert not self.active, "KV cache is None but active is not empty"
            self.active = list(new_running)
            self.past_key_values = new_past_key_values
            return

        old_len = self._get_past_seq_len(self.past_key_values)
        new_len = self._get_past_seq_len(new_past_key_values)
        target_len = max(old_len, new_len)

        old_extra_pad = target_len - old_len
        new_extra_pad = target_len - new_len

        old_past = self._left_pad_past_key_values_to_len(
            self.past_key_values,
            target_len,
        )
        new_past = self._left_pad_past_key_values_to_len(
            new_past_key_values,
            target_len,
        )

        if old_extra_pad > 0:
            for req in self.active:
                req.kv_pad_len += old_extra_pad

        if new_extra_pad > 0:
            for req in new_running:
                req.kv_pad_len += new_extra_pad

        merged_past = []

        for old_layer, new_layer in zip(old_past, new_past):
            old_k, old_v = old_layer
            new_k, new_v = new_layer

            merged_k = torch.cat([old_k, new_k], dim=0)
            merged_v = torch.cat([old_v, new_v], dim=0)

            merged_past.append((merged_k, merged_v))

        self.active.extend(new_running)
        self.past_key_values = tuple(merged_past)

    def _to_legacy_past_key_values(self, past_key_values: Any) -> Any:
        """
        Convert HuggingFace Cache objects such as DynamicCache into the older
        tuple-of-layers format used by this educational implementation.

        Legacy layout:
            tuple(
                (key, value),  # layer 0
                (key, value),  # layer 1
                ...
            )
        """
        # Older transformers: Cache objects expose to_legacy_cache().
        if hasattr(past_key_values, "to_legacy_cache"):
            return past_key_values.to_legacy_cache()

        # Mid-era transformers: DynamicCache stored parallel key/value lists.
        if hasattr(past_key_values, "key_cache") and hasattr(
            past_key_values, "value_cache"
        ):
            return tuple(
                (k, v)
                for k, v in zip(
                    past_key_values.key_cache,
                    past_key_values.value_cache,
                )
            )

        # transformers >= 5: DynamicCache holds a list of layer objects, each
        # exposing .keys / .values tensors.
        if hasattr(past_key_values, "layers"):
            return tuple(
                (layer.keys, layer.values) for layer in past_key_values.layers
            )

        return past_key_values

    def _from_legacy_past_key_values(self, past_key_values: Any) -> Any:
        """
        Convert a legacy tuple-of-layers KV cache back into the Cache object
        the model expects as input.

        transformers >= 5 dropped support for feeding the legacy tuple format
        into model.forward(past_key_values=...); it now requires a Cache
        instance. This is the inverse of _to_legacy_past_key_values so the
        scheduler can keep operating on plain tuples internally.
        """
        if past_key_values is None:
            return None

        # Already a Cache object (not a legacy tuple/list) -> pass through.
        if not isinstance(past_key_values, (tuple, list)):
            return past_key_values

        # Older transformers exposed an explicit constructor.
        if hasattr(DynamicCache, "from_legacy_cache"):
            return DynamicCache.from_legacy_cache(past_key_values)

        # transformers >= 5: the constructor accepts the per-layer (k, v) tuples.
        return DynamicCache(ddp_cache_data=past_key_values)

    def _get_past_seq_len(self, past_key_values: Any) -> int:
        """
        Return the physical sequence length stored in a past_key_values object.
        """
        first_layer = past_key_values[0]
        first_k = first_layer[0]
        return int(first_k.shape[2])

    def _left_pad_past_key_values_to_len(self, past_key_values: Any, target_len: int,) -> Any:
        """
        Left-pad every K/V tensor on the sequence-length dimension.
        """
        current_len = self._get_past_seq_len(past_key_values)

        if current_len == target_len:
            return past_key_values

        if current_len > target_len:
            raise ValueError(
                f"Cannot pad KV cache from length {current_len} down to {target_len}"
            )

        pad_len = target_len - current_len
        padded_layers = []

        for layer in past_key_values:
            k, v = layer

            k_pad_shape = list(k.shape)
            v_pad_shape = list(v.shape)

            k_pad_shape[2] = pad_len
            v_pad_shape[2] = pad_len

            k_pad = torch.zeros(
                k_pad_shape,
                dtype=k.dtype,
                device=k.device,
            )
            v_pad = torch.zeros(
                v_pad_shape,
                dtype=v.dtype,
                device=v.device,
            )

            padded_k = torch.cat([k_pad, k], dim=2)
            padded_v = torch.cat([v_pad, v], dim=2)

            padded_layers.append((padded_k, padded_v))

        return tuple(padded_layers)

    # ---------- decode and completion ----------

    def _decode_one_step(self) -> None:
        """
        Decode one token step for all active requests.

        Current invariant before this function:
        - self.active[i] matches batch row i in self.past_key_values
        - each KV row layout is:
            [pad] * kv_pad_len + [real cached tokens] * real_seq_len
        - KV cache contains prompt + generated_ids[:-1]
        - last_token_id == generated_ids[-1]
        - last_token_id has not yet been inserted into KV cache

        This function feeds last_token_id for every active request.
        After model forward:
        - old last_token_id is inserted into the returned KV cache
        - logits select the next token
        - next token becomes the new last_token_id
        """
        if not self.active:
            return

        if self.past_key_values is None:
            raise ValueError("Cannot decode with active requests but no KV cache.")

        tok = self.tokenizer
        device = self.model.device

        eos_id = tok.eos_token_id
        if eos_id is None:
            raise ValueError("Tokenizer must have eos_token_id for this implementation.")

        batch_size = len(self.active)
        past_len = self._get_past_seq_len(self.past_key_values)

        # Shape: [batch_size, 1]
        # Each row feeds the token that was selected in the previous step
        # but is not yet present in KV cache.
        input_ids = torch.tensor(
            [[req.last_token_id] for req in self.active],
            dtype=torch.long,
            device=device,
        )

        # Shape: [batch_size, past_len + 1]
        #
        # For each row:
        #   past part: [0] * kv_pad_len + [1] * real_seq_len
        #   new token: final [1]
        attention_mask = torch.zeros(
            (batch_size, past_len + 1),
            dtype=torch.long,
            device=device,
        )

        for i, req in enumerate(self.active):
            real_start = req.kv_pad_len
            real_end = req.kv_pad_len + req.real_seq_len

            # Real cached tokens in past_key_values.
            attention_mask[i, real_start:real_end] = 1

            # Current input token, i.e. req.last_token_id.
            attention_mask[i, past_len] = 1

        # Shape: [batch_size, 1]
        #
        # The current input token's true position is real_seq_len.
        # Do not use past_len, because past_len includes left padding.
        position_ids = torch.tensor(
            [[req.real_seq_len] for req in self.active],
            dtype=torch.long,
            device=device,
        )

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=self._from_legacy_past_key_values(
                    self.past_key_values
                ),
                use_cache=True,
            )

        # Now the old last_token_id has been inserted into KV cache.
        self.past_key_values = self._to_legacy_past_key_values(outputs.past_key_values)

        # Select the next token for each request.
        next_token_logits = outputs.logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)

        t_now = time.perf_counter()

        for i, req in enumerate(self.active):
            next_token_id = int(next_tokens[i].item())

            req.generated_ids.append(next_token_id)
            req.last_token_id = next_token_id

            # The token that just entered KV cache is the previous last_token_id,
            # so real cached length grows by exactly one.
            req.real_seq_len += 1

            if req.t_first_token is None:
                req.t_first_token = t_now

            if next_token_id == eos_id or len(req.generated_ids) >= req.item.max_tokens:
                req.finished = True

    def _cleanup_finished(self) -> None:
        """
        Resolve completed futures and remove finished rows from active state
        and KV cache.
        """
        if not self.active:
            self.past_key_values = None
            return

        finished_reqs: list[_RunningRequest] = []
        keep_indices_list: list[int] = []
        kept_active: list[_RunningRequest] = []

        for i, req in enumerate(self.active):
            if req.finished:
                finished_reqs.append(req)
            else:
                keep_indices_list.append(i)
                kept_active.append(req)

        if not finished_reqs:
            return

        for req in finished_reqs:
            self._set_result(req)

        self.active = kept_active

        if not self.active:
            self.past_key_values = None
            return

        device = self.model.device
        keep_indices = torch.tensor(
            keep_indices_list,
            dtype=torch.long,
            device=device,
        )

        self._filter_past_key_values(keep_indices)

    def _set_result(self, req: _RunningRequest) -> None:
        """
        Decode generated_ids and set the response dict on req.item.future.
        """
        if req.item.future.done():
            return

        t_done = time.perf_counter()

        text = self.tokenizer.decode(
            req.generated_ids,
            skip_special_tokens=True,
        )

        total_s = t_done - req.item.t_enqueue
        queue_wait_s = req.t_prefill_done - req.item.t_enqueue

        # In this implementation, first token is selected during prefill.
        # t_first_token should normally equal t_prefill_done.
        if req.t_first_token is not None:
            batch_decode_s = t_done - req.t_first_token
        else:
            batch_decode_s = 0.0

        output_tokens = len(req.generated_ids)

        # first token belongs to prefill/TTFT, remaining tokens belong to decode.
        if batch_decode_s > 0 and output_tokens > 1:
            batch_decode_tok_per_s = (output_tokens - 1) / batch_decode_s
        else:
            batch_decode_tok_per_s = 0.0

        result = {
            "text": text,
            "prompt_tokens": req.prompt_tokens,
            "output_tokens": output_tokens,
            "queue_wait_ms": queue_wait_s * 1000,
            "batch_decode_ms": batch_decode_s * 1000,
            "total_ms": total_s * 1000,
            "batch_size": len(self.active),
            "batch_decode_tok_per_s": batch_decode_tok_per_s,
        }

        req.item.future.set_result(result)

    # ---------- KV helpers ----------

    def _filter_past_key_values(self, keep_indices: torch.Tensor) -> None:
        """
        Keep only selected batch rows in every K/V tensor.

        keep_indices indexes the batch dimension, dim=0.
        """
        if self.past_key_values is None:
            return

        filtered_layers = []

        for layer in self.past_key_values:
            k, v = layer

            filtered_k = torch.index_select(k, dim=0, index=keep_indices).contiguous()
            filtered_v = torch.index_select(v, dim=0, index=keep_indices).contiguous()

            filtered_layers.append((filtered_k, filtered_v))

        self.past_key_values = tuple(filtered_layers)
    
    def _head_cut_past_key_values(self) -> None:
        """
        Remove common left-padding prefix shared by all active KV rows.

        If every active row has at least N left-pad positions, those N positions
        are useless for all rows and can be removed from every K/V tensor.
        """
        if not self.active or self.past_key_values is None:
            return

        cut_len = min(req.kv_pad_len for req in self.active)

        if cut_len <= 0:
            return

        cut_layers = []

        for layer in self.past_key_values:
            k, v = layer
            cut_k = k[:, :, cut_len:, :].contiguous()
            cut_v = v[:, :, cut_len:, :].contiguous()
            cut_layers.append((cut_k, cut_v))

        self.past_key_values = tuple(cut_layers)

        for req in self.active:
            req.kv_pad_len -= cut_len

    # ---------- error handling ----------

    def _fail_all(self, exc: Exception) -> None:
        """
        Fail queued and active requests, then reset scheduler state.
        """
        print(f"[continuous-batcher] error: {type(exc).__name__}: {exc}")

        # Fail all active requests
        for req in self.active:
            if not req.item.future.done():
                req.item.future.set_exception(exc)

        # Fail all queued-but-not-yet-prefilled requests
        while True:
            try:
                item = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            if not item.future.done():
                item.future.set_exception(exc)

        # Reset scheduler state
        self.active = []
        self.past_key_values = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
