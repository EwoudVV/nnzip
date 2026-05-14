"""
nnzip: text compression that uses a local GPT-2 (via llama.cpp) as the
probability model.

For each token, GPT-2 produces a probability distribution over the next token.
Arithmetic coding spends -log2(P) bits per token, so tokens the model is
confident about cost almost nothing. English text typically lands at ~15-25%
of original size -- usually 3-5x better than gzip.

llama.cpp gives us cross-platform native inference (Mac/Linux/Windows, optional
Metal/CUDA) with a much smaller dependency footprint than torch+transformers.

CLI:
    nnzip compress file.txt          # produces file.txt.nnz
    nnzip decompress file.txt.nnz    # produces file.txt
    compress file.txt                # same as `nnzip compress`
    decompress file.txt.nnz          # same as `nnzip decompress`

File format (v2):
    4 bytes : magic "NNZP"
    1 byte  : version (= 2)
    4 bytes : token_count (uint32 BE)
    rest    : arithmetic-coded payload (uint32 words)
"""

import argparse
import os
import struct
import sys
import time

MAGIC = b"NNZP"
VERSION = 2  # v1 used torch+transformers; v2 uses llama.cpp

# Pre-converted FP16 GGUF of OpenAI GPT-2 (124M params, ~252MB) on Hugging Face.
# Both encoder and decoder must use the same model file; pinning the repo and
# filename guarantees that.
HF_REPO = "sjfalken/openai-gpt2-124M-F16-gguf"
HF_FILE = "openai-gpt2-124M-F16.gguf"

# GPT-2 supports positions 0..1023. When the context hits MAX_CACHE we reset
# and re-feed the last KEEP_AFTER tokens. Encoder and decoder do this at the
# same iteration, so they stay in sync.
N_CTX = 1024
MAX_CACHE = 1023
KEEP_AFTER = 512


def _load_deps():
    """Lazy imports so `nnzip --help` doesn't pull in llama_cpp."""
    global Llama, np, constriction, hf_hub_download
    from llama_cpp import Llama as _Llama
    import numpy as _np
    import constriction as _constriction
    from huggingface_hub import hf_hub_download as _hf_hub_download
    Llama = _Llama
    np = _np
    constriction = _constriction
    hf_hub_download = _hf_hub_download


def _load_model():
    override = os.environ.get("NNZIP_MODEL_PATH")
    if override:
        model_path = override
    else:
        print(f"resolving {HF_REPO}/{HF_FILE}... "
              f"(first run downloads ~250MB)", flush=True)
        model_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)

    print(f"loading {os.path.basename(model_path)}...", flush=True)
    t0 = time.time()
    llm = Llama(
        model_path=model_path,
        n_ctx=N_CTX,
        n_threads=max(1, (os.cpu_count() or 4) - 1),
        verbose=False,
        logits_all=True,  # we need logits at every position
    )
    print(f"loaded in {time.time() - t0:.1f}s "
          f"(vocab {llm.n_vocab()}, threads {llm.n_threads})", flush=True)
    return llm


def _logits_to_probs(logits):
    """Convert a logits row to a normalized float32 probability array."""
    # numerical-stable softmax
    m = logits.max()
    e = np.exp(logits - m, dtype=np.float64)
    probs = e / e.sum()
    probs = np.maximum(probs.astype(np.float32), 1e-7)
    return (probs / probs.sum()).astype(np.float32)


def compress_text(text):
    """Compress UTF-8 text. Returns bytes of the v2 .nnz format."""
    _load_deps()
    llm = _load_model()

    tokens = llm.tokenize(text.encode("utf-8"))
    if not tokens:
        raise ValueError("empty input")

    bos = llm.token_bos() if llm.token_bos() != -1 else llm.token_eos()

    enc = constriction.stream.queue.RangeEncoder()
    llm.reset()
    llm.eval([bos])
    ctx_len = 1  # how many tokens we've evaluated, including bos

    t0 = time.time()
    for i, token in enumerate(tokens):
        # logits at position ctx_len-1 = prediction for the next token
        logits = np.asarray(llm.scores[ctx_len - 1], dtype=np.float32).copy()
        probs = _logits_to_probs(logits)
        dist = constriction.stream.model.Categorical(probs, perfect=False)
        enc.encode(token, dist)

        # Add this token to the context (it becomes the "previous" for the next prediction)
        if ctx_len >= MAX_CACHE:
            # sliding window: reset, re-feed last KEEP_AFTER tokens then add current
            tail = tokens[max(0, i - KEEP_AFTER + 1) : i + 1]
            llm.reset()
            llm.eval([bos] + list(tail))
            ctx_len = 1 + len(tail)
        else:
            llm.eval([token])
            ctx_len += 1

        if (i + 1) % 50 == 0 or i == len(tokens) - 1:
            elapsed = time.time() - t0
            print(f"  encoded {i+1}/{len(tokens)} tokens "
                  f"({(i+1)/max(elapsed,1e-9):.1f} tok/s)", flush=True)

    payload = enc.get_compressed().tobytes()
    header = MAGIC + bytes([VERSION]) + struct.pack(">I", len(tokens))
    return header + payload


def decompress_bytes(data):
    """Inverse of compress_text. Returns UTF-8 text."""
    _load_deps()

    if not data.startswith(MAGIC):
        raise ValueError("not an nnzip file (missing magic)")
    pos = 4
    version = data[pos]; pos += 1
    if version != VERSION:
        raise ValueError(
            f"unsupported nnzip file version {version} "
            f"(this build handles v{VERSION}); "
            f"reinstall the matching nnzip version to read this file"
        )
    num_tokens = struct.unpack(">I", data[pos:pos + 4])[0]; pos += 4
    payload = data[pos:]

    llm = _load_model()
    bos = llm.token_bos() if llm.token_bos() != -1 else llm.token_eos()

    compressed = np.frombuffer(payload, dtype=np.uint32).copy()
    dec = constriction.stream.queue.RangeDecoder(compressed)

    decoded = []
    llm.reset()
    llm.eval([bos])
    ctx_len = 1

    t0 = time.time()
    for i in range(num_tokens):
        logits = np.asarray(llm.scores[ctx_len - 1], dtype=np.float32).copy()
        probs = _logits_to_probs(logits)
        dist = constriction.stream.model.Categorical(probs, perfect=False)
        token = int(dec.decode(dist))
        decoded.append(token)

        if ctx_len >= MAX_CACHE:
            tail = decoded[-KEEP_AFTER:]
            llm.reset()
            llm.eval([bos] + list(tail))
            ctx_len = 1 + len(tail)
        else:
            llm.eval([token])
            ctx_len += 1

        if (i + 1) % 50 == 0 or i == num_tokens - 1:
            elapsed = time.time() - t0
            print(f"  decoded {i+1}/{num_tokens} tokens "
                  f"({(i+1)/max(elapsed,1e-9):.1f} tok/s)", flush=True)

    return llm.detokenize(decoded).decode("utf-8", errors="replace")


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
    parser = argparse.ArgumentParser(
        prog="nnzip",
        description="LLM-based text compression (local GPT-2 via llama.cpp).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("compress", help="compress a text file")
    p1.add_argument("input")
    p1.add_argument("output", nargs="?", default=None)
    p2 = sub.add_parser("decompress", help="decompress an .nnz file")
    p2.add_argument("input")
    p2.add_argument("output", nargs="?", default=None)
    args = parser.parse_args(argv)
    if args.cmd == "compress":
        compress_file(args.input, args.output)
    else:
        decompress_file(args.input, args.output)


def compress_main(argv=None):
    parser = argparse.ArgumentParser(
        prog="compress",
        description="Compress a text file with nnzip (local GPT-2 via llama.cpp).",
    )
    parser.add_argument("input")
    parser.add_argument("output", nargs="?", default=None)
    args = parser.parse_args(argv)
    compress_file(args.input, args.output)


def decompress_main(argv=None):
    parser = argparse.ArgumentParser(
        prog="decompress",
        description="Decompress an .nnz file with nnzip (local GPT-2 via llama.cpp).",
    )
    parser.add_argument("input")
    parser.add_argument("output", nargs="?", default=None)
    args = parser.parse_args(argv)
    decompress_file(args.input, args.output)


if __name__ == "__main__":
    main()
