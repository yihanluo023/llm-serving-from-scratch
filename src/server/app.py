"""
LLM serving — Phase 1, Step 1.

Single-request synchronous server.
Each /generate request runs model.generate() to completion before
returning. Concurrent requests will queue (blocked by the GIL +
single GPU). This is intentional in this phase.
"""
import time
from contextlib import asynccontextmanager
from threading import Thread

import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)


# Config
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Pydantic schemas
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256

class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float
    decode_ms: float
    total_ms: float
    decode_tok_per_s: float


# Model loading
def load_model():
    """Load tokenizer and model onto GPU in FP16."""
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


# Inference
def run_generate(model, tokenizer, prompt: str, max_new_tokens: int):
    """
    Run one generate() call with timing. Returns (text, timing_dict).

    Takes raw prompt string, applies chat template inside.
    """
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

    timing = {
        "prompt_tokens": prompt_len,
        "output_tokens": num_new,
        "ttft_ms": ttft_s * 1000,
        "decode_ms": decode_s * 1000,
        "total_ms": (t_end - t_start) * 1000,
        "decode_tok_per_s": decode_tok_per_s,
    }
    return response_text, timing


# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[lifespan] startup: loading model")
    tokenizer, model = load_model()

    # Warmup
    print("[lifespan] warmup ...")
    _ = run_generate(model, tokenizer, "Hi", max_new_tokens=10)
    print("[lifespan] warmup done")

    app.state.model = model
    app.state.tokenizer = tokenizer

    yield

    print("[lifespan] shutdown: freeing GPU memory")
    del app.state.model
    del app.state.tokenizer
    torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)


# Routes
@app.get("/")
async def health():
    return {"status": "alive", "model": MODEL_NAME}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, request: Request):
    model = request.app.state.model
    tokenizer = request.app.state.tokenizer

    text, timing = run_generate(model, tokenizer, req.prompt, req.max_tokens)
    return GenerateResponse(text=text, **timing)
