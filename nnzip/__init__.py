"""
nnzip: text compression that uses a local GPT-2 as the probability model.

For each token, GPT-2 produces a probability distribution over the next token.
Arithmetic coding spends -log2(P) bits per token, so tokens the model is
confident about cost almost nothing. English text typically lands at ~13-20%
of original size -- usually 3-5x better than gzip.

Both encoder and decoder must use the exact same model. The compressed file
contains zero model information -- the decoder re-runs the same forward passes
and decodes the bits.

CLI:
    nnzip compress file.txt          # produces file.txt.nnz
    nnzip decompress file.txt.nnz   # produces file.txt
    compress file.txt               # same as `nnzip compress`
    decompress file.txt.nnz        # same as `nnzip decompress`
"""

import argparse
import os
import struct
import sys
import time

MODEL_NAME = "gpt2"  # 117M params, ~500MB on disk. Override with $NNZIP_MODEL.
MAGIC = b"NNZP"
VERSION = 1


def _load_deps():
    """Heavy imports happen here so `nnzip --help` is fast."""
    global torch, np, constriction, GPT2LMHeadModel, AutoTokenizer
    import torch as _torch
    import numpy as _np
    import constriction as _constriction
    from transformers import GPT2LMHeadModel as _GPT2LMHeadModel
    from transformers import AutoTokenizer as _AutoTokenizer
    torch = _torch
    np = _np
    constriction = _constriction
    GPT2LMHeadModel = _GPT2LMHeadModel
    AutoTokenizer = _AutoTokenizer


def _load_model():
    model_name = os.environ.get("NNZIP_MODEL", MODEL_NAME)
    print(f"loading {model_name}... (first run downloads ~500MB)", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)
    return model, tokenizer, model_name


def _logits_to_probs(logits):
    probs = torch.softmax(logits, dim=-1).numpy().astype(np.float32)
    probs = np.maximum(probs, 1e-7)
    return (probs / probs.sum()).astype(np.float32)


def compress_text(text):
    """Compress UTF-8 text. Returns bytes of the .nnz format."""
    _load_deps()
    model, tokenizer, model_name = _load_model()

    tokens = tokenizer.encode(text)
    if not tokens:
        raise ValueError("empty input")

    bos = (tokenizer.bos_token_id
           if tokenizer.bos_token_id is not None
           else tokenizer.eos_token_id)

    enc = constriction.stream.queue.RangeEncoder()
    past = None
    last_token = bos
    t0 = time.time()

    for i, token in enumerate(tokens):
        input_ids = torch.tensor([[last_token]])
        with torch.no_grad():
            out = model(input_ids, past_key_values=past, use_cache=True)
            logits = out.logits[0, -1]
            past = out.past_key_values

        probs = _logits_to_probs(logits)
        dist = constriction.stream.model.Categorical(probs, perfect=False)
        enc.encode(token, dist)
        last_token = token

        if (i + 1) % 50 == 0 or i == len(tokens) - 1:
            elapsed = time.time() - t0
            print(f"  encoded {i+1}/{len(tokens)} tokens "
                  f"({(i+1)/max(elapsed,1e-9):.1f} tok/s)", flush=True)

    payload = enc.get_compressed().tobytes()
    model_bytes = model_name.encode("utf-8")
    # Header: magic (4) + version (1) + model_name_len (1) + model_name + token_count (4)
    header = (MAGIC
              + bytes([VERSION, len(model_bytes)])
              + model_bytes
              + struct.pack(">I", len(tokens)))
    return header + payload


def decompress_bytes(data):
    """Inverse of compress_text. Returns the original UTF-8 text."""
    _load_deps()

    if not data.startswith(MAGIC):
        raise ValueError("not an nnzip file (missing magic bytes)")
    pos = 4
    version = data[pos]; pos += 1
    if version != VERSION:
        raise ValueError(f"unsupported nnzip version {version}")
    name_len = data[pos]; pos += 1
    model_name = data[pos:pos + name_len].decode("utf-8"); pos += name_len
    num_tokens = struct.unpack(">I", data[pos:pos + 4])[0]; pos += 4
    payload = data[pos:]

    if model_name != os.environ.get("NNZIP_MODEL", MODEL_NAME):
        os.environ["NNZIP_MODEL"] = model_name  # ensure we load the right one

    model, tokenizer, _ = _load_model()
    bos = (tokenizer.bos_token_id
           if tokenizer.bos_token_id is not None
           else tokenizer.eos_token_id)

    compressed = np.frombuffer(payload, dtype=np.uint32).copy()
    dec = constriction.stream.queue.RangeDecoder(compressed)

    decoded = []
    past = None
    last_token = bos
    t0 = time.time()

    for i in range(num_tokens):
        input_ids = torch.tensor([[last_token]])
        with torch.no_grad():
            out = model(input_ids, past_key_values=past, use_cache=True)
            logits = out.logits[0, -1]
            past = out.past_key_values

        probs = _logits_to_probs(logits)
        dist = constriction.stream.model.Categorical(probs, perfect=False)
        token = int(dec.decode(dist))
        decoded.append(token)
        last_token = token

        if (i + 1) % 50 == 0 or i == num_tokens - 1:
            elapsed = time.time() - t0
            print(f"  decoded {i+1}/{num_tokens} tokens "
                  f"({(i+1)/max(elapsed,1e-9):.1f} tok/s)", flush=True)

    return tokenizer.decode(decoded)


EXTENSION = ".nnz"


def _resolve_output_for_compress(input_path, output_path):
    return output_path or (input_path + EXTENSION)


def _resolve_output_for_decompress(input_path, output_path):
    if output_path:
        return output_path
    if input_path.lower().endswith(EXTENSION):
        return input_path[: -len(EXTENSION)]
    return input_path + ".decompressed"


def compress_file(input_path, output_path=None):
    output_path = _resolve_output_for_compress(input_path, output_path)
    with open(input_path, "rb") as f:
        text = f.read().decode("utf-8")
    print(f"compressing {input_path} -> {output_path}")
    blob = compress_text(text)
    with open(output_path, "wb") as f:
        f.write(blob)
    orig = len(text.encode("utf-8"))
    comp = os.path.getsize(output_path)
    print()
    print(f"original:    {orig:,} bytes")
    print(f"compressed:  {comp:,} bytes "
          f"({100*comp/max(orig,1):.1f}% of original)")
    return output_path


def decompress_file(input_path, output_path=None):
    output_path = _resolve_output_for_decompress(input_path, output_path)
    with open(input_path, "rb") as f:
        data = f.read()
    print(f"decompressing {input_path} -> {output_path}")
    text = decompress_bytes(data)
    with open(output_path, "wb") as f:
        f.write(text.encode("utf-8"))
    print(f"\nrecovered {len(text):,} chars -> {output_path}")
    return output_path


# ----- CLI entry points -----

def main(argv=None):
    """The `nnzip` command with subcommands."""
    parser = argparse.ArgumentParser(
        prog="nnzip",
        description="LLM-based text compression (uses local GPT-2).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("compress", help="compress a text file")
    p1.add_argument("input")
    p1.add_argument("output", nargs="?", default=None,
                    help="default: <input>.nnz")

    p2 = sub.add_parser("decompress", help="decompress an .nnz file")
    p2.add_argument("input")
    p2.add_argument("output", nargs="?", default=None,
                    help="default: strip .nnz from input")

    args = parser.parse_args(argv)
    if args.cmd == "compress":
        compress_file(args.input, args.output)
    else:
        decompress_file(args.input, args.output)


def compress_main(argv=None):
    """The bare `compress` command."""
    parser = argparse.ArgumentParser(
        prog="compress",
        description="Compress a text file with nnzip (local GPT-2).",
    )
    parser.add_argument("input")
    parser.add_argument("output", nargs="?", default=None,
                        help="default: <input>.nnz")
    args = parser.parse_args(argv)
    compress_file(args.input, args.output)


def decompress_main(argv=None):
    """The bare `decompress` command."""
    parser = argparse.ArgumentParser(
        prog="decompress",
        description="Decompress an .nnz file with nnzip (local GPT-2).",
    )
    parser.add_argument("input")
    parser.add_argument("output", nargs="?", default=None,
                        help="default: strip .nnz from input")
    args = parser.parse_args(argv)
    decompress_file(args.input, args.output)


if __name__ == "__main__":
    main()
