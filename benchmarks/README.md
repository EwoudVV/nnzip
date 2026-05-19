# Benchmarks

A small corpus and a script that compares nnzip against the standard classical compressors (`gzip`, `bzip2`, `xz`, `zstd`) at their highest standard compression settings.

## Running

```
python3 benchmarks/run_benchmarks.py
```

Requires nnzip importable from the repo root (i.e. `pip install -e .` from the parent directory) plus `gzip`, `bzip2`, `xz`, `zstd` on `$PATH`. Each file goes through compress + decompress with a round-trip equality check.

Output is a Markdown table that drops straight into the project README.

## The corpus

| File | What it tests |
|---|---|
| `prose_modern.txt` | Clean modern English narrative prose |
| `wiki_factual.txt` | Wikipedia-style factual writing |
| `dialogue.txt` | Fiction dialogue (short alternating lines) |
| `markdown_docs.md` | Markdown technical documentation |
| `code_python.py` | Python source code |
| `json_data.json` | Structured JSON with repeated keys |
| `repetitive.txt` | The same sentence repeated 30 times |

All files are clean ASCII — no BOM, no `\r\n` line endings, no smart quotes — so the comparison isolates compression ability from encoding noise. Each file is intentionally small (~1.5-2.5 KB) to keep the benchmark runtime under a minute on Apple Silicon.

## Why these particular files

The corpus is designed to cover three regimes where you might expect different winners:

1. **Natural language** (prose, wiki, dialogue) — should favor nnzip because GPT-2 has internalized English statistics.
2. **Structured / formal text** (code, JSON, markdown) — a stress test. Classical compressors thrive on repeated tokens here, so this is where you'd predict they catch up. They don't — GPT-2 also saw lots of these in training.
3. **Pathological cases** (highly repetitive text) — gzip's bread and butter. nnzip still wins on this corpus because after seeing the same sentence twice, GPT-2 predicts every subsequent occurrence at near-100% probability.

## What's not here

- **Large files (>100 KB).** nnzip is intentionally slow (~1 KB/s on Metal) so a multi-megabyte benchmark would take hours. The compression ratio gets *slightly* worse on long inputs because GPT-2's 1024-token context window forces a sliding window after the first ~1000 tokens, but the trend is well-represented by the small-file numbers.
- **Multi-language.** The script only runs the English model. Other supported languages (`nl`, `it`, `fr`, `pt`) would need separate corpora.
- **Dirty text.** Project Gutenberg files, web scrapes with mixed encodings, etc. nnzip performs *worse* on those (the README discusses why) — but the point of this corpus is to measure the underlying algorithm, not encoding hygiene.
