"""
LLM serving — Phase 1, Step 2.

FastAPI server with a single background batcher coroutine. Each /generate
request hands its prompt to the batcher and awaits the batched result.
Naive single-request path is still selectable via /generate_serial for
A/B comparison from the benchmark script.
"""
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.server.batcher import Batcher
from src.server.continuous_batcher import ContinuousBatcher


# Config
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_BATCH = 8
MAX_WAIT_MS = 50.0


# Pydantic schemas
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    output_tokens: int
    queue_wait_ms: float
    batch_decode_ms: float
    total_ms: float
    batch_size: int
    batch_decode_tok_per_s: float


class SerialGenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float
    decode_ms: float
    total_ms: float
    decode_tok_per_s: float


# Model loading
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
    elapsed = time.perf_counter() - t0
    print(f"Loaded in {elapsed:.1f}s. Device: {model.device}, dtype: {model.dtype}")
    return tokenizer, model


# Serial path (kept for A/B): a synchronous one-request-at-a-time
# generate(), no batching, no padding. Mirrors the Phase 1 Step 1 baseline.
def run_generate_serial(model, tokenizer, prompt: str, max_new_tokens: int):
    from threading import Thread
    from transformers import TextIteratorStreamer

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    prompt_len = input_ids.shape[1]

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    t_start = time.perf_counter()
    thread.start()

    chunks = []
    t_first = None
    for chunk in streamer:
        if t_first is None:
            t_first = time.perf_counter()
        chunks.append(chunk)

    thread.join()
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    response_text = "".join(chunks)
    num_new = len(tokenizer.encode(response_text, add_special_tokens=False))
    ttft_s = (t_first - t_start) if t_first else 0.0
    decode_s = (t_end - t_first) if t_first else 0.0
    decode_tok_per_s = (
        (num_new - 1) / decode_s if decode_s > 0 and num_new > 1 else 0.0
    )
    return response_text, {
        "prompt_tokens": prompt_len,
        "output_tokens": num_new,
        "ttft_ms": ttft_s * 1000,
        "decode_ms": decode_s * 1000,
        "total_ms": (t_end - t_start) * 1000,
        "decode_tok_per_s": decode_tok_per_s,
    }


# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[lifespan] startup: loading model")
    tokenizer, model = load_model()

    print("[lifespan] warmup ...")
    # Warm the serial path first; this compiles the shared model kernels.
    # The batcher paths are warmed below, once started, so their own shapes
    # (padded prefill / merge / decode loop) get autotuned too.
    _ = run_generate_serial(model, tokenizer, "Hi", max_new_tokens=10)

    batcher = Batcher(
        model=model,
        tokenizer=tokenizer,
        max_batch=MAX_BATCH,
        max_wait_ms=MAX_WAIT_MS,
    )
    await batcher.start()
    print(f"[lifespan] static batcher started (max_batch={MAX_BATCH}, max_wait_ms={MAX_WAIT_MS})")

    continuous_batcher = ContinuousBatcher(
        model=model,
        tokenizer=tokenizer,
        max_batch=MAX_BATCH,
        max_wait_ms=MAX_WAIT_MS,
    )
    await continuous_batcher.start()
    print(f"[lifespan] continuous batcher started (max_batch={MAX_BATCH}, max_wait_ms={MAX_WAIT_MS})")

    # Warm the batcher-specific code paths so the first real request to either
    # endpoint doesn't pay autotune cost (matters for a clean first-request TTFT).
    _ = await batcher.submit("Hi", 10)
    _ = await continuous_batcher.submit("Hi", 10)
    print("[lifespan] warmup done")

    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.batcher = batcher
    app.state.continuous_batcher = continuous_batcher

    yield

    print("[lifespan] shutdown: stopping batchers")
    await app.state.batcher.stop()
    await app.state.continuous_batcher.stop()

    print("[lifespan] shutdown: freeing GPU memory")
    del app.state.model
    del app.state.tokenizer
    del app.state.batcher
    del app.state.continuous_batcher
    torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)


# Routes
@app.get("/")
async def health():
    return {
        "status": "alive",
        "model": MODEL_NAME,
        "max_batch": MAX_BATCH,
        "max_wait_ms": MAX_WAIT_MS,
    }

@app.post("/generate_serial", response_model=SerialGenerateResponse)
async def generate_serial(req: GenerateRequest, request: Request):
    """
    Serial path. Kept so concurrent benchmarks can A/B against the
    naive Step-1 behavior without restarting the server.
    """
    model = request.app.state.model
    tokenizer = request.app.state.tokenizer
    text, timing = run_generate_serial(model, tokenizer, req.prompt, req.max_tokens)
    return SerialGenerateResponse(text=text, **timing)


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, request: Request):
    """Batched path. Default endpoint for Phase 1 Step 2."""
    batcher: Batcher = request.app.state.batcher
    result = await batcher.submit(req.prompt, req.max_tokens)
    return GenerateResponse(**result)


@app.post("/generate_continuous", response_model=GenerateResponse)
async def generate_continuous(req: GenerateRequest, request: Request):
    """Continuous batching path for Phase 2."""
    batcher: ContinuousBatcher = request.app.state.continuous_batcher
    result = await batcher.submit(req.prompt, req.max_tokens)
    return GenerateResponse(**result)
