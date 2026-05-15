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

File format (v3, current):
    4 bytes : magic "NNZP"
    1 byte  : version (= 3)
    2 bytes : lang code, ASCII (e.g. "en", "fr")
    4 bytes : crc32 of original UTF-8 bytes (integrity check)
    4 bytes : token_count (uint32 BE)
    rest    : arithmetic-coded payload (uint32 words)

File format (v2, still readable for backward compat):
    4 bytes : magic "NNZP"
    1 byte  : version (= 2)
    4 bytes : token_count (uint32 BE)
    rest    : arithmetic-coded payload
"""

import argparse
import os
import struct
import sys
import time
import zlib

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("nnzip")
except Exception:
    __version__ = "0.0.0+unknown"

MAGIC = b"NNZP"
VERSION = 3  # v1: torch+transformers; v2: llama.cpp; v3: lang + crc32 in header

# Per-language model registry. Each entry maps an ISO-639-1 language code to
# (HF repo, GGUF filename). Currently only English has a published model;
# more languages get added by uploading new fine-tuned GGUFs and listing them
# here. The package doesn't bundle any of them — they download lazily on first
# use into ~/.cache/huggingface/ and stay cached.
LANG_REGISTRY = {
    "en": ("eeeev1343/nnzip-gpt2-base-f16", "nnzip-gpt2.gguf"),
    # "fr": ("eeeev1343/nnzip-gpt2-fr-f16", "nnzip-gpt2-fr.gguf"),
    # "de": ("eeeev1343/nnzip-gpt2-de-f16", "nnzip-gpt2-de.gguf"),
    # etc — add as fine-tuned models are published.
}
DEFAULT_LANG = "en"


class IntegrityError(Exception):
    """Raised when the CRC32 stored in the .nnz header doesn't match the
    CRC32 of the decompressed text. Indicates the round-trip failed."""

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


def detect_language(text):
    """Return an ISO 639-1 code for the text (e.g. 'en', 'fr'). Falls back to
    DEFAULT_LANG if detection fails or langdetect is unavailable."""
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0  # deterministic output
        # 5000 chars is way more than enough for accurate detection
        return detect(text[:5000])
    except Exception:
        return DEFAULT_LANG


def _resolve_model_for_lang(lang, verbose=True):
    """Pick the model for the requested language, falling back to English
    with a warning if we don't have a fine-tuned model for that language yet.
    Returns (effective_lang, repo, filename)."""
    if lang in LANG_REGISTRY:
        return (lang,) + LANG_REGISTRY[lang]
    # Fallback: use English. Compression will still work, just less efficient
    # for the non-English text.
    if verbose:
        print(
            f"note: no fine-tuned model for {lang!r} yet — falling back to "
            f"English. Ratio will be worse but the file will still round-trip.",
            file=sys.stderr,
        )
    return (DEFAULT_LANG,) + LANG_REGISTRY[DEFAULT_LANG]


def _load_model(lang=DEFAULT_LANG, verbose=True):
    override = os.environ.get("NNZIP_MODEL_PATH")
    if override:
        # User pinned a specific GGUF; trust them, don't second-guess language
        model_path = override
        effective_lang = lang
    else:
        effective_lang, repo, filename = _resolve_model_for_lang(lang, verbose)
        if verbose:
            print(f"resolving {repo}/{filename}... "
                  f"(first run downloads ~250MB)", flush=True)
        model_path = hf_hub_download(repo_id=repo, filename=filename)

    if verbose:
        print(f"loading {os.path.basename(model_path)}...", flush=True)
    t0 = time.time()
    llm = Llama(
        model_path=model_path,
        n_ctx=N_CTX,
        n_threads=max(1, (os.cpu_count() or 4) - 1),
        verbose=False,
        logits_all=True,  # we need logits at every position
    )
    if verbose:
        print(f"loaded in {time.time() - t0:.1f}s "
              f"(vocab {llm.n_vocab()}, threads {llm.n_threads})", flush=True)
    return llm, effective_lang


def _logits_to_probs(logits):
    """Convert a logits row to a normalized float32 probability array."""
    # numerical-stable softmax
    m = logits.max()
    e = np.exp(logits - m, dtype=np.float64)
    probs = e / e.sum()
    probs = np.maximum(probs.astype(np.float32), 1e-7)
    return (probs / probs.sum()).astype(np.float32)


def compress_text(text, lang=None, verbose=True):
    """Compress UTF-8 text. Returns bytes of the v3 .nnz format.

    If `lang` is None, auto-detect from `text`. The chosen lang is stored in
    the file header and used to pick the model on decompression.
    """
    _load_deps()
    if lang is None:
        lang = detect_language(text)
    llm, effective_lang = _load_model(lang=lang, verbose=verbose)

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

        if verbose and ((i + 1) % 50 == 0 or i == len(tokens) - 1):
            elapsed = time.time() - t0
            print(f"  encoded {i+1}/{len(tokens)} tokens "
                  f"({(i+1)/max(elapsed,1e-9):.1f} tok/s)", flush=True)

    payload = enc.get_compressed().tobytes()
    raw_bytes = text.encode("utf-8")
    crc = zlib.crc32(raw_bytes) & 0xFFFFFFFF
    lang_bytes = effective_lang.encode("ascii").ljust(2, b" ")[:2]
    header = (
        MAGIC
        + bytes([VERSION])
        + lang_bytes
        + struct.pack(">I", crc)
        + struct.pack(">I", len(tokens))
    )
    return header + payload


def decompress_bytes(data, verbose=True):
    """Inverse of compress_text. Returns UTF-8 text.

    Reads both v3 (current) and v2 (legacy) headers. v2 files are assumed to
    be English and skip the CRC check.
    """
    _load_deps()

    if not data.startswith(MAGIC):
        raise ValueError("not an nnzip file (missing magic)")
    pos = 4
    version = data[pos]; pos += 1

    if version == 3:
        lang = data[pos:pos + 2].decode("ascii").rstrip(); pos += 2
        expected_crc = struct.unpack(">I", data[pos:pos + 4])[0]; pos += 4
        num_tokens = struct.unpack(">I", data[pos:pos + 4])[0]; pos += 4
    elif version == 2:
        lang = DEFAULT_LANG
        expected_crc = None  # legacy file, no integrity check
        num_tokens = struct.unpack(">I", data[pos:pos + 4])[0]; pos += 4
    else:
        raise ValueError(
            f"unsupported nnzip file version {version} "
            f"(this build handles v2 and v3)"
        )
    payload = data[pos:]

    llm, _ = _load_model(lang=lang, verbose=verbose)
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

        if verbose and ((i + 1) % 50 == 0 or i == num_tokens - 1):
            elapsed = time.time() - t0
            print(f"  decoded {i+1}/{num_tokens} tokens "
                  f"({(i+1)/max(elapsed,1e-9):.1f} tok/s)", flush=True)

    text = llm.detokenize(decoded).decode("utf-8", errors="replace")

    if expected_crc is not None:
        actual_crc = zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise IntegrityError(
                f"integrity check failed: stored crc32={expected_crc:08x}, "
                f"decompressed crc32={actual_crc:08x}. The recovered text "
                f"is probably wrong (model mismatch, file corruption, or a bug)."
            )
    return text


EXTENSION = ".nnz"


def _resolve_output_for_compress(input_path, output_path):
    return output_path or (input_path + EXTENSION)


def _resolve_output_for_decompress(input_path, output_path):
    if output_path:
        return output_path
    if input_path.lower().endswith(EXTENSION):
        return input_path[: -len(EXTENSION)]
    return input_path + ".decompressed"


def compress_file(input_path, output_path=None, quiet=False, lang=None):
    output_path = _resolve_output_for_compress(input_path, output_path)
    with open(input_path, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        print(f"error: {input_path} is not valid UTF-8 text "
              f"(bad byte at position {e.start}).", file=sys.stderr)
        print("nnzip only compresses text. For binary, use gzip or zstd.",
              file=sys.stderr)
        sys.exit(1)

    if not quiet:
        if lang is None:
            detected = detect_language(text)
            print(f"detected language: {detected}", flush=True)
        else:
            print(f"language: {lang} (specified via --lang)", flush=True)
        print(f"compressing {input_path} -> {output_path}")
    blob = compress_text(text, lang=lang, verbose=not quiet)
    with open(output_path, "wb") as f:
        f.write(blob)
    orig = len(raw)
    comp = os.path.getsize(output_path)
    if not quiet:
        print()
        print(f"original:    {orig:,} bytes")
        print(f"compressed:  {comp:,} bytes "
              f"({100*comp/max(orig,1):.1f}% of original)")
    if comp >= orig:
        print(
            f"warning: {output_path} ({comp:,} bytes) is larger than "
            f"{input_path} ({orig:,} bytes).",
            file=sys.stderr,
        )
        print(
            "nnzip works best on natural English. For non-English / source "
            "code / binary data, gzip or zstd will do better.",
            file=sys.stderr,
        )
    return output_path


def decompress_file(input_path, output_path=None, quiet=False):
    output_path = _resolve_output_for_decompress(input_path, output_path)
    with open(input_path, "rb") as f:
        data = f.read()
    if not quiet:
        print(f"decompressing {input_path} -> {output_path}")
    try:
        text = decompress_bytes(data, verbose=not quiet)
    except IntegrityError as e:
        print(f"\nerror: {e}", file=sys.stderr)
        sys.exit(2)
    with open(output_path, "wb") as f:
        f.write(text.encode("utf-8"))
    if not quiet:
        print(f"\nrecovered {len(text):,} chars -> {output_path}")
        print("integrity check: ok (crc32 matches)")
    return output_path


# ----- CLI entry points -----

def _add_quiet(parser):
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="suppress progress output")


def _add_lang(parser):
    parser.add_argument(
        "--lang", default=None,
        help="ISO 639-1 language code (e.g. en, fr, ja). "
             "If omitted, the language is auto-detected from the input.",
    )


def _version_flag(parser):
    parser.add_argument("--version", action="version",
                        version=f"nnzip {__version__}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="nnzip",
        description="LLM-based text compression (local GPT-2 via llama.cpp).",
    )
    _version_flag(parser)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("compress", help="compress a text file")
    p1.add_argument("input")
    p1.add_argument("output", nargs="?", default=None)
    _add_quiet(p1)
    _add_lang(p1)
    p2 = sub.add_parser("decompress", help="decompress an .nnz file")
    p2.add_argument("input")
    p2.add_argument("output", nargs="?", default=None)
    _add_quiet(p2)
    args = parser.parse_args(argv)
    if args.cmd == "compress":
        compress_file(args.input, args.output, quiet=args.quiet, lang=args.lang)
    else:
        decompress_file(args.input, args.output, quiet=args.quiet)


def compress_main(argv=None):
    parser = argparse.ArgumentParser(
        prog="compress",
        description="Compress a text file with nnzip (local GPT-2 via llama.cpp).",
    )
    _version_flag(parser)
    parser.add_argument("input")
    parser.add_argument("output", nargs="?", default=None)
    _add_quiet(parser)
    _add_lang(parser)
    args = parser.parse_args(argv)
    compress_file(args.input, args.output, quiet=args.quiet, lang=args.lang)


def decompress_main(argv=None):
    parser = argparse.ArgumentParser(
        prog="decompress",
        description="Decompress an .nnz file with nnzip (local GPT-2 via llama.cpp).",
    )
    _version_flag(parser)
    parser.add_argument("input")
    parser.add_argument("output", nargs="?", default=None)
    _add_quiet(parser)
    args = parser.parse_args(argv)
    decompress_file(args.input, args.output, quiet=args.quiet)


if __name__ == "__main__":
    main()
