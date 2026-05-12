"""
LLM compression: GPT-2 next-token probabilities + arithmetic coding.

for each token in the input, ask GPT-2 "given everything that came before,
what's your probability distribution over the next token?" The arithmetic coder
then spends bits proportional to -log2(P(actual token)) -- if GPT-2 was 90%
sure about the next token, encoding costs ~0.15 bits.

compressor and decompressor must use the exact same model with deterministic
inference. send zero model information in the output, the decompressor
recomputes the same probabilities and recovers the tokens.

    python llm_compress.py compress   <text_file>       <compressed_file>
    python llm_compress.py decompress <compressed_file> <text_file>
"""

import argparse
import os
import struct
import sys
import time

import constriction
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_NAME = "gpt2"  # 117M params, ~500MB on disk. larger models give better compression but are slower to run.


def load_model():
    print(f"loading {MODEL_NAME}... (first run downloads ~500MB)", flush=True)
    t0 = time.time()
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.eval()
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)
    return model, tokenizer


def logits_to_probs(logits):
    """convert a logits tensor to a normalized numpy float32 distribution.
    We clamp tiny probabilities so the arithmetic coder always has a valid range."""
    probs = torch.softmax(logits, dim=-1).numpy().astype(np.float32)
    probs = np.maximum(probs, 1e-7)
    probs = probs / probs.sum()
    return probs.astype(np.float32)


def compress(input_path, output_path):
    model, tokenizer = load_model()

    with open(input_path, "rb") as f:
        text = f.read().decode("utf-8", errors="replace")

    tokens = tokenizer.encode(text)
    if not tokens:
        print("empty input")
        sys.exit(1)
    print(f"input:  {len(text):,} chars  ->  {len(tokens):,} GPT-2 tokens")

    # prepend BOS so the model has a starting context for predicting tokens[0].
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id

    # encoder must use the exact same forward-pass code path the
    # decoder will use (one token at a time, with KV cache). Using a single
    # bulk forward pass here gives slightly different logits in some attention
    # kernels, which desyncs the arithmetic coder and produces garbage on
    # decompression. Slower but deterministic.
    encoder = constriction.stream.queue.RangeEncoder()
    past = None
    last_token = bos
    t0 = time.time()

    for i, token in enumerate(tokens):
        input_ids = torch.tensor([[last_token]])
        with torch.no_grad():
            out = model(input_ids, past_key_values=past, use_cache=True)
            logits = out.logits[0, -1]
            past = out.past_key_values

        probs = logits_to_probs(logits)
        dist = constriction.stream.model.Categorical(probs, perfect=False)
        encoder.encode(token, dist)
        last_token = token

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1:,}/{len(tokens):,} tokens   ({(i+1)/elapsed:.1f} tok/s)",
                  flush=True)

    compressed = encoder.get_compressed()
    print(f"encoded in {time.time() - t0:.1f}s", flush=True)

    payload = compressed.tobytes()
    with open(output_path, "wb") as f:
        # header: 4 bytes for token count.
        f.write(struct.pack(">I", len(tokens)))
        f.write(payload)

    orig = len(text.encode("utf-8"))
    comp = os.path.getsize(output_path)
    print()
    print(f"original:    {orig:,} bytes")
    print(f"compressed:  {comp:,} bytes  ({100*comp/orig:.1f}% of original)")
    print(f"effective:   {8*comp/len(tokens):.2f} bits/token  ({8*comp/orig:.2f} bits/byte)")


def decompress(input_path, output_path):
    model, tokenizer = load_model()

    with open(input_path, "rb") as f:
        header = f.read(4)
        num_tokens = struct.unpack(">I", header)[0]
        payload = f.read()
    print(f"decoding {num_tokens:,} tokens from {len(payload):,} compressed bytes")

    compressed = np.frombuffer(payload, dtype=np.uint32).copy()
    decoder = constriction.stream.queue.RangeDecoder(compressed)

    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id

    decoded = []
    past = None
    last_token = bos

    t0 = time.time()
    for i in range(num_tokens):
        # first call seeds the model with BOS, then we use the KV cache and
        # only feed the most recently decoded token. Each step is one fast
        # forward over a single token instead of recomputing the full prefix.
        input_ids = torch.tensor([[last_token]])
        with torch.no_grad():
            out = model(input_ids, past_key_values=past, use_cache=True)
            logits = out.logits[0, -1]
            past = out.past_key_values

        probs = logits_to_probs(logits)
        dist = constriction.stream.model.Categorical(probs, perfect=False)
        token = int(decoder.decode(dist))

        decoded.append(token)
        last_token = token

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1:,}/{num_tokens:,} tokens   ({(i+1)/elapsed:.1f} tok/s)",
                  flush=True)

    text = tokenizer.decode(decoded)
    with open(output_path, "wb") as f:
        f.write(text.encode("utf-8"))

    print(f"\nrecovered {num_tokens:,} tokens, {len(text):,} chars in {time.time() - t0:.1f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["compress", "decompress"])
    p.add_argument("input")
    p.add_argument("output")
    args = p.parse_args()
    if args.mode == "compress":
        compress(args.input, args.output)
    else:
        decompress(args.input, args.output)


if __name__ == "__main__":
    main()
