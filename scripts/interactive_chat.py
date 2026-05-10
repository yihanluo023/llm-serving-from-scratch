"""
Interactive chat with Qwen2.5-1.5B-Instruct.
"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

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

def build_prompt(tokenizer, history):
    """
    Apply Qwen2.5's chat template to a conversation history.

    history: list of {"role": "user"|"assistant", "content": str}
    Returns: input_ids tensor on GPU, ready for model.generate().
    """
    text = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    return inputs.input_ids, inputs.attention_mask

def generate_response(model, tokenizer, input_ids, attention_mask, max_new_tokens=256):
    """
    Run generation and return
    """
    prompt_len = input_ids.shape[1]

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    # generate() blocks until done. Run it in a background thread so the
    # main thread can iterate over the streamer as tokens come out.
    thread = Thread(target=model.generate, kwargs=gen_kwargs)

    t_start = time.perf_counter()
    thread.start()

    # Capture time of first token
    chunks = []
    t_first_token = None
    for text_chunk in streamer:
        if t_first_token is None:
            t_first_token = time.perf_counter()
        chunks.append(text_chunk)

    thread.join()
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    response = "".join(chunks)

    # Re-tokenize the response just to count tokens accurately
    num_new = len(tokenizer.encode(response, add_special_tokens=False))

    ttft = t_first_token - t_start          # Time To First Token (≈ prefill)
    decode_s = t_end - t_first_token        # Time generating remaining tokens
    decode_tok_per_s = (num_new - 1) / decode_s if decode_s > 0 and num_new > 1 else 0

    timing = {
        "ttft_s": ttft,
        "decode_s": decode_s,
        "total_s": t_end - t_start,
        "num_new_tokens": num_new,
        "decode_tok_per_s": decode_tok_per_s,
    }
    return response, timing

def main():
    tokenizer, model = load_model()

    history = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    
    # Warmup: trigger CUDA kernel JIT, cuBLAS autotuning, allocator init.
    # The first generate() call carries ~hundreds of ms of one-time setup
    # cost that has nothing to do with model inference. Burning that here
    # means subsequent timings reflect real prefill/decode cost.
    print("Warming up...")
    warmup_input, warmup_mask = build_prompt(tokenizer, [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi"},
    ])
    _ = generate_response(model, tokenizer, warmup_input, warmup_mask, max_new_tokens=10)
    print("Warmup done.\n")

    print("\n=== Interactive chat. Type 'quit' to exit, 'reset' to clear history. ===\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            history = [{"role": "system", "content": "You are a helpful assistant."}]
            print("[history cleared]\n")
            continue

        history.append({"role": "user", "content": user_input})
        input_ids, attention_mask = build_prompt(tokenizer, history)
        prompt_len = input_ids.shape[1]

        response, timing = generate_response(model, tokenizer, input_ids, attention_mask)
        history.append({"role": "assistant", "content": response})

        print(f"\nAssistant: {response}\n")
        print(
            f"  [prompt={prompt_len} tok | new={timing['num_new_tokens']} tok | "
            f"TTFT={timing['ttft_s']*1000:.0f}ms | "
            f"decode={timing['decode_s']*1000:.0f}ms | "
            f"{timing['decode_tok_per_s']:.1f} tok/s]\n"
        )


if __name__ == "__main__":
    main()