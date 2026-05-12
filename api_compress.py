"""
LLM compression using OpenAI's API as the probability source.

For each token after position 0 ask GPT-3.5-turbo-instruct for its top-5
predictions given the previous tokens. The rank of the actual token (0-4
for in-top-5, or escape if not) gets arithmetic-coded with a fixed rank
distribution that's the same on both sides, robust to 4th-decimal logprob
jitter that is see across identical API calls.

Position 0 and escapes are stored as literals. To avoid needing a tokenizer
in the decoder, literals go into a small in-file table of unique strings, and
the stream just contains table indices.

File format:
    u32 BE  token_count                   total number of tokens (N)
    u16 BE  literal_table_size            number of unique literal strings (L)
    u16 BE  literal_seq_count             # of literal emissions (T)
    L times: u8 length + UTF-8 bytes      the literal table
    u32 BE  ac_bit_count
    ceil(ac_bit_count/8) bytes            arithmetic-coded stream

Usage:
    python api_compress.py compress   <text>       <compressed>
    python api_compress.py decompress <compressed> <text>
"""

import argparse
import asyncio
import base64
import json
import os
import struct
import sys
import time
from pathlib import Path

import tiktoken
from openai import AsyncOpenAI

from arithmetic_coder import (
    ArithmeticEncoder,
    ArithmeticDecoder,
    cdf_from_quantized,
    quantize_probs,
)

MODEL = "gpt-3.5-turbo-instruct"
TOP_K = 5
CDF_TOTAL = 16384  # 2^14; AC headroom

encoding = tiktoken.encoding_for_model(MODEL)

# FIXED rank distribution (probabilities sum to 1).
# Last entry is the escape mass. Both sides MUST use this exact array.
FIXED_RANK_PROBS = [0.55, 0.15, 0.08, 0.04, 0.03, 0.15]
FIXED_RANK_QUANT = quantize_probs(FIXED_RANK_PROBS, CDF_TOTAL)
FIXED_RANK_CDF = cdf_from_quantized(FIXED_RANK_QUANT)
ESCAPE_RANK = TOP_K


def load_api_key():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("OPENAI_API_KEY")


def quantize_logprob(lp):
    """Round to 2 decimal places for stable sort despite 4th-decimal jitter."""
    return round(lp, 2)


def sorted_top_strings(top_lp_dict):
    """Return list of token strings sorted by quantized logprob DESC, tie-break
    lexically ASC. This sort must be reproducible across encoder/decoder
    calls, quantization handles tiny logprob jitter."""
    items = [(s, quantize_logprob(lp)) for s, lp in top_lp_dict.items()]
    items.sort(key=lambda x: (-x[1], x[0]))
    return [s for s, _ in items]


async def _fetch_logprobs_at(aclient, prefix_text, sem):
    """Send a string prompt (not token ids) so behavior matches what an
    in browser JS decoder would do."""
    async with sem:
        resp = await aclient.completions.create(
            model=MODEL,
            prompt=prefix_text,
            max_tokens=1,
            logprobs=TOP_K,
            temperature=0,
            seed=42,
            echo=False,
        )
        return resp.choices[0].logprobs.top_logprobs[0]


async def _fetch_all_logprobs(api_key, token_ids, concurrency=20):
    """Get top-k logprobs for each position i in 1..N-1 using the decoded
    prefix text as the prompt."""
    aclient = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(concurrency)

    # build prefix text up to (but not including) each position
    prefixes = []
    running = ""
    for tok_id in token_ids[:-1]:
        running += encoding.decode([tok_id])
        prefixes.append(running)
    # prefixes[i-1] is the text before position i, for i in 1..N-1

    tasks = [_fetch_logprobs_at(aclient, prefixes[i - 1], sem)
             for i in range(1, len(token_ids))]
    return await asyncio.gather(*tasks)


def token_string(tok_id):
    """decode a single token id to its string. For normal English text this
    is well-defined, for partial-byte BPE tokens results may be a replacement
    char, but the same string will always be produced for the same id."""
    return encoding.decode([tok_id])


def _encode_text_to_payload(text):
    """run the full encode pipeline. Returns (payload_bytes, stats_dict)."""
    api_key = load_api_key()
    if not api_key:
        print("OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    token_ids = encoding.encode(text)
    print(f"input: {len(text)} chars -> {len(token_ids)} tokens")

    if len(token_ids) > 65535:
        print("too many tokens (max 65535 for this file format)", file=sys.stderr)
        sys.exit(1)

    n_calls = max(0, len(token_ids) - 1)
    if n_calls > 0:
        print(f"firing {n_calls} parallel API calls...", flush=True)
        t0 = time.time()
        all_top_lp = asyncio.run(_fetch_all_logprobs(api_key, token_ids))
        print(f"  done in {time.time()-t0:.1f}s", flush=True)
    else:
        all_top_lp = []

    rank_or_escape = []
    literal_seq = []
    literal_seq.append(token_string(token_ids[0]))

    for i in range(1, len(token_ids)):
        top_lp = all_top_lp[i - 1]
        ordered = sorted_top_strings(top_lp)
        actual_str = token_string(token_ids[i])

        rank = None
        for r, s in enumerate(ordered):
            if s == actual_str and r < TOP_K:
                rank = r
                break

        if rank is None:
            rank_or_escape.append(ESCAPE_RANK)
            literal_seq.append(actual_str)
        else:
            rank_or_escape.append(rank)

    # dedupe literals into a table
    seen = {}
    literal_table = []
    for s in literal_seq:
        if s not in seen:
            seen[s] = len(literal_table)
            literal_table.append(s)
    literal_seq_idx = [seen[s] for s in literal_seq]

    if len(literal_table) > 65535:
        print("too many unique literals", file=sys.stderr)
        sys.exit(1)

    enc = ArithmeticEncoder()
    table_size = max(2, len(literal_table))

    first_idx = literal_seq_idx[0]
    enc.encode(first_idx, first_idx + 1, table_size)

    seq_pos = 1
    for r in rank_or_escape:
        enc.encode(FIXED_RANK_CDF[r], FIXED_RANK_CDF[r + 1], CDF_TOTAL)
        if r == ESCAPE_RANK:
            idx = literal_seq_idx[seq_pos]
            enc.encode(idx, idx + 1, table_size)
            seq_pos += 1

    ac_bytes, bit_count = enc.finish()

    # serialize to the on-disk format
    parts = []
    parts.append(struct.pack(">I", len(token_ids)))
    parts.append(struct.pack(">H", len(literal_table)))
    parts.append(struct.pack(">H", len(literal_seq)))
    for s in literal_table:
        b = s.encode("utf-8")
        if len(b) > 255:
            print(f"literal too long ({len(b)} bytes): {s!r}", file=sys.stderr)
            sys.exit(1)
        parts.append(bytes([len(b)]))
        parts.append(b)
    parts.append(struct.pack(">I", bit_count))
    parts.append(ac_bytes)

    payload = b"".join(parts)
    stats = {
        "orig_bytes": len(text.encode("utf-8")),
        "comp_bytes": len(payload),
        "token_count": len(token_ids),
        "n_escape": sum(1 for r in rank_or_escape if r == ESCAPE_RANK),
        "literal_count": len(literal_table),
        "literal_table_bytes": sum(len(s.encode("utf-8")) for s in literal_table),
        "ac_bytes": len(ac_bytes),
        "ac_bits": bit_count,
    }
    return payload, stats


def _print_stats(stats):
    o, c = stats["orig_bytes"], stats["comp_bytes"]
    print()
    print(f"original:    {o:,} bytes")
    print(f"compressed:  {c:,} bytes ({100*c/o:.1f}%)")
    print(f"tokens:      {stats['token_count']}  escapes: {stats['n_escape']} "
          f"({100*stats['n_escape']/max(1,stats['token_count']-1):.1f}%)")
    print(f"literal table: {stats['literal_count']} unique strings, "
          f"{stats['literal_table_bytes']} bytes")
    print(f"ac stream:   {stats['ac_bytes']} bytes ({stats['ac_bits']} bits)")


def compress(input_path, output_path):
    with open(input_path, "rb") as f:
        text = f.read().decode("utf-8")
    payload, stats = _encode_text_to_payload(text)
    with open(output_path, "wb") as f:
        f.write(payload)
    _print_stats(stats)


def compress_html(input_path, output_path):
    with open(input_path, "rb") as f:
        text = f.read().decode("utf-8")
    payload, stats = _encode_text_to_payload(text)

    template_path = Path(__file__).parent / "template.html"
    template = template_path.read_text()

    b64 = base64.b64encode(payload).decode("ascii")
    n_tokens = stats["token_count"]
    approx_cost = n_tokens * 0.002  # ~$0.002 per API call (decode side)
    approx_time = n_tokens * 1.0    # ~1 sec per token

    html = (template
        .replace("__PAYLOAD_B64__", b64)
        .replace("__FIXED_RANK_CDF__", json.dumps(FIXED_RANK_CDF))
        .replace("__CDF_TOTAL__", str(CDF_TOTAL))
        .replace("__TOP_K__", str(TOP_K))
        .replace("__MODEL__", MODEL)
        .replace("__APPROX_COST__", f"{approx_cost:.2f}")
        .replace("__APPROX_TIME__", f"{approx_time:.0f}"))

    with open(output_path, "w") as f:
        f.write(html)
    _print_stats(stats)
    html_size = os.path.getsize(output_path)
    print(f"html file:   {html_size:,} bytes "
          f"({html_size - stats['comp_bytes']:,} of which is the template + base64 overhead)")


async def _decompress_async(input_path, output_path):
    api_key = load_api_key()
    if not api_key:
        print("OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    aclient = AsyncOpenAI(api_key=api_key)

    with open(input_path, "rb") as f:
        token_count = struct.unpack(">I", f.read(4))[0]
        literal_count = struct.unpack(">H", f.read(2))[0]
        literal_seq_count = struct.unpack(">H", f.read(2))[0]
        literal_table = []
        for _ in range(literal_count):
            length = f.read(1)[0]
            literal_table.append(f.read(length).decode("utf-8"))
        ac_bit_count = struct.unpack(">I", f.read(4))[0]
        ac_bytes = f.read()

    print(f"tokens: {token_count}, literals (unique): {literal_count}, "
          f"literals (total): {literal_seq_count}")

    dec = ArithmeticDecoder(ac_bytes, ac_bit_count)
    table_size = max(2, literal_count)

    decoded_strings = []

    # position 0: read literal index
    idx = dec.decode_uniform(table_size)
    decoded_strings.append(literal_table[idx])

    t0 = time.time()
    for i in range(1, token_count):
        # use string prompt so behavior matches an in-browser JS decoder.
        prefix_text = "".join(decoded_strings)

        resp = await aclient.completions.create(
            model=MODEL,
            prompt=prefix_text,
            max_tokens=1,
            logprobs=TOP_K,
            temperature=0,
            seed=42,
            echo=False,
        )
        top_lp = resp.choices[0].logprobs.top_logprobs[0]
        ordered = sorted_top_strings(top_lp)

        r = dec.decode_cdf(FIXED_RANK_CDF, CDF_TOTAL)
        if r < TOP_K:
            decoded_strings.append(ordered[r])
        else:
            # escape: read literal index
            idx = dec.decode_uniform(table_size)
            decoded_strings.append(literal_table[idx])

        if (i % 10 == 0) or i == token_count - 1:
            elapsed = time.time() - t0
            print(f"  {i}/{token_count - 1} ({i/max(elapsed,1e-9):.1f} tok/s)",
                  flush=True)

    text = "".join(decoded_strings)
    with open(output_path, "wb") as f:
        f.write(text.encode("utf-8"))
    print(f"\nrecovered {token_count} tokens, {len(text)} chars")


def decompress(input_path, output_path):
    asyncio.run(_decompress_async(input_path, output_path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["compress", "decompress", "compress-html"])
    p.add_argument("input")
    p.add_argument("output")
    args = p.parse_args()
    if args.mode == "compress":
        compress(args.input, args.output)
    elif args.mode == "compress-html":
        compress_html(args.input, args.output)
    else:
        decompress(args.input, args.output)


if __name__ == "__main__":
    main()
