"""Benchmark nnzip vs gzip / bzip2 / xz / zstd on a small corpus of clean
ASCII files. Prints a Markdown table that can be pasted into README.md.

Run from the repo root:
    python3 benchmarks/run_benchmarks.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# Import nnzip from the repo source (not whatever's installed system-wide)
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from nnzip import compress_text, decompress_bytes  # noqa: E402

CORPUS = HERE / "corpus"
FILES = [
    ("prose_modern.txt", "Modern English prose"),
    ("wiki_factual.txt", "Wikipedia-style factual"),
    ("dialogue.txt", "Fiction dialogue"),
    ("markdown_docs.md", "Markdown documentation"),
    ("code_python.py", "Python source code"),
    ("json_data.json", "JSON data"),
    ("repetitive.txt", "Highly repetitive text"),
]


def run_external(cmd: list[str], data: bytes) -> tuple[int, float]:
    """Pipe data through cmd, return (output_size_bytes, seconds)."""
    t0 = time.monotonic()
    proc = subprocess.run(cmd, input=data, capture_output=True, check=True)
    dt = time.monotonic() - t0
    return len(proc.stdout), dt


def run_nnzip(text: str) -> tuple[int, float, float, bool]:
    """Returns (output_size, encode_seconds, decode_seconds, ok)."""
    t0 = time.monotonic()
    blob = compress_text(text, verbose=False)
    enc = time.monotonic() - t0

    t0 = time.monotonic()
    recovered = decompress_bytes(blob, verbose=False)
    dec = time.monotonic() - t0

    return len(blob), enc, dec, recovered == text


def fmt_pct(num: int, denom: int) -> str:
    return f"{100.0 * num / denom:.1f}%"


def fmt_bpb(num: int, denom: int) -> str:
    return f"{8.0 * num / denom:.2f}"


def main() -> None:
    print("Loading nnzip (downloads model on first run)...", file=sys.stderr)
    # Warm up so the first file isn't penalized for model load
    _ = compress_text("warm-up.", verbose=False)
    print("Model loaded. Running benchmarks.\n", file=sys.stderr)

    rows = []
    for fname, label in FILES:
        path = CORPUS / fname
        data = path.read_bytes()
        text = data.decode("utf-8")
        orig = len(data)

        gz, gz_t = run_external(["gzip", "-9", "-c"], data)
        bz, bz_t = run_external(["bzip2", "-9", "-c"], data)
        xz, xz_t = run_external(["xz", "-9", "-e", "-c"], data)
        zs, zs_t = run_external(["zstd", "--ultra", "-22", "-q", "-c"], data)
        nn, nn_enc, nn_dec, ok = run_nnzip(text)

        rows.append({
            "fname": fname,
            "label": label,
            "orig": orig,
            "gz": gz, "bz": bz, "xz": xz, "zs": zs, "nn": nn,
            "nn_enc_s": nn_enc, "nn_dec_s": nn_dec,
            "ok": ok,
        })

        status = "OK" if ok else "ROUND-TRIP FAILED"
        print(
            f"{label:30s} orig={orig:5d}B  gzip={gz:5d}  bzip2={bz:5d}  "
            f"xz={xz:5d}  zstd={zs:5d}  nnzip={nn:5d}  [{status}]",
            file=sys.stderr,
        )

    # ---- Markdown table ----
    print()
    print("## Benchmark: nnzip vs classical compressors")
    print()
    print("All sizes in bytes. Compression ratio = compressed / original "
          "(lower is better). Bits-per-byte (bpb) = 8 × ratio.")
    print()
    print("| File | Type | Original | gzip -9 | bzip2 -9 | xz -9e | zstd -22 | "
          "**nnzip** |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| `{r['fname']}` | {r['label']} | {r['orig']:,} | "
            f"{r['gz']:,} ({fmt_pct(r['gz'], r['orig'])}) | "
            f"{r['bz']:,} ({fmt_pct(r['bz'], r['orig'])}) | "
            f"{r['xz']:,} ({fmt_pct(r['xz'], r['orig'])}) | "
            f"{r['zs']:,} ({fmt_pct(r['zs'], r['orig'])}) | "
            f"**{r['nn']:,} ({fmt_pct(r['nn'], r['orig'])})** |"
        )

    # Totals row
    total_orig = sum(r["orig"] for r in rows)
    total_gz = sum(r["gz"] for r in rows)
    total_bz = sum(r["bz"] for r in rows)
    total_xz = sum(r["xz"] for r in rows)
    total_zs = sum(r["zs"] for r in rows)
    total_nn = sum(r["nn"] for r in rows)
    print(
        f"| **Total** | — | **{total_orig:,}** | "
        f"**{total_gz:,} ({fmt_pct(total_gz, total_orig)})** | "
        f"**{total_bz:,} ({fmt_pct(total_bz, total_orig)})** | "
        f"**{total_xz:,} ({fmt_pct(total_xz, total_orig)})** | "
        f"**{total_zs:,} ({fmt_pct(total_zs, total_orig)})** | "
        f"**{total_nn:,} ({fmt_pct(total_nn, total_orig)})** |"
    )

    # ---- Bits-per-byte table ----
    print()
    print("### Bits per byte (lower is better)")
    print()
    print("| File | gzip -9 | bzip2 -9 | xz -9e | zstd -22 | **nnzip** |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| `{r['fname']}` | "
            f"{fmt_bpb(r['gz'], r['orig'])} | "
            f"{fmt_bpb(r['bz'], r['orig'])} | "
            f"{fmt_bpb(r['xz'], r['orig'])} | "
            f"{fmt_bpb(r['zs'], r['orig'])} | "
            f"**{fmt_bpb(r['nn'], r['orig'])}** |"
        )


if __name__ == "__main__":
    main()
