# compression-experiments

## install
```
pip install nnzip
```

| Stage | Idea | Best result |
|---|---|---|
| 1 | Brute-force hash inversion (Python) | works for ≤3-byte files; ~0.6 M hashes/s |
| 2 | Same, in C with CommonCrypto + threads | ~45 M H/s |
| 3 | NEON SHA-256 hardware intrinsics + threads | ~380 M H/s |
| 4 | Apple Metal GPU compute shader | ~1.0 GH/s |
| 5 | CPU + GPU concurrent | **~1.4 GH/s** (~2300× the Python version) |
| 6 | "Deterministic index" version of the hash idea | proved the pigeonhole wall |
| 7 | Local GPT-2 + arithmetic coding | English text → **13%** of original (vs gzip's ~57%) |
| 8 | OpenAI API + self-extracting HTML page | same idea, no local model needed |

Stages 1-6 is one idea, proving why arbitrary lossless compression below input size is impossible. Stages 7-8 is another idea, that proved to be much better than well known compression algorithms for english text.

## What's in here

### The brute-force hash side (stages 1-5)

| File | What it does |
|---|---|
| `compress.py` / `decompress.py` | The original idea: store a file as just SHA-256 + length; "decompress" by trying every possible file until the hash matches. |
| `compress_index.py` / `decompress_index.py` | The "deterministic generator" refinement — turns out it's just storing the file as a big integer. Shows the size wall. |
| `brute.c` | C version of decompression, CommonCrypto, pthreads. ~75× Python. |
| `brute_neon.c` | C with ARMv8 SHA-2 hardware instructions. ~635× Python. |
| `brute_mb.c` | 4-way multi-buffer SIMD SHA-256 — interesting failure: slower than hardware SHA on M1. |
| `brute_metal.m` | Metal compute shader, all 4096 ALU lanes of the M1 Max GPU. ~1.7×10³ × Python. |
| `brute_combined.m` | CPU NEON-HW + GPU in parallel. ~2.3×10³ × Python. |

### The actual-compression side (stages 6-8)

| File | What it does |
|---|---|
| `nnzip/` | Python package: local GPT-2 + arithmetic coding. Installs as `compress`, `decompress`, and `nnzip` CLI commands. |
| `pyproject.toml` | Packaging config: declares the CLI entry points and dependencies. |
| `arithmetic_coder.py` | Portable bit-level arithmetic coder. Identical output in Python and the JS port (see `template.html`). |
| `api_compress.py` | OpenAI API compression (no local model, but slow and pay-per-use). Three modes: `compress` produces a binary `.api` file, `decompress` reverses it, `compress-html` bakes the payload into a portable HTML self-extractor. |
| `template.html` | The HTML self-extractor template with the JS arithmetic decoder inline. The Python compressor fills in the `__PAYLOAD_B64__` etc. placeholders. |

### Test data

| File | Use |
|---|---|
| `sample.txt` | ~860-byte English paragraph for compression demos. |

## How to use

### Build the C/Metal brute forcers (the hardware showcase)

```
clang -O3 -Wall -Wno-deprecated-declarations -o brute brute.c
clang -O3 -Wall -Wno-deprecated-declarations -o brute_neon brute_neon.c
clang -O3 -march=native -Wno-deprecated-declarations -o brute_mb brute_mb.c
clang -O3 -fobjc-arc -framework Foundation -framework Metal -o brute_metal brute_metal.m
clang -O3 -fobjc-arc -framework Foundation -framework Metal -o brute_combined brute_combined.m
```

Then:

```
printf 'word' > test4.txt
python3 compress.py test4.txt test4.compressed
./brute_combined test4.compressed test4.recovered 8   # ~2 seconds on M1 Max
```

The full 4-byte search space is 4.3 billion candidates. The Python decompressor would take ~38 minutes. The combined CPU+GPU version finds the right one in ~2 seconds.

### Local LLM compression — the `nnzip` CLI (works on Mac and Windows)

Install the package (one-time, downloads PyTorch + transformers ~600 MB):

```
python3.12 -m venv venv
./venv/bin/pip install -e .
```

On Windows, use `python -m venv venv` and `venv\Scripts\pip install -e .` — same package, same commands afterward.

Then `compress` and `decompress` are available as commands:

```
./venv/bin/compress sample.txt           # produces sample.txt.nnz
./venv/bin/decompress sample.txt.nnz     # restores sample.txt
```

Or use the namespaced command:

```
./venv/bin/nnzip compress sample.txt
./venv/bin/nnzip decompress sample.txt.nnz
```

First run downloads GPT-2 (~500 MB) and caches it in `~/.cache/huggingface`. Expect ~13-20% of original size on English text. Non-English / source code / random binary may not compress (or may grow).

To use a larger model with better compression, set the environment variable:

```
NNZIP_MODEL=gpt2-medium ./venv/bin/compress sample.txt   # ~1.5 GB download, better ratio
```

The model name is stored in the `.nnz` file so the decompressor automatically loads the matching one.

### OpenAI API compression (small payloads, portable HTML)

Requires an OpenAI API key. Put it in `.env`:

```
OPENAI_API_KEY=sk-...
```

(`.env` is in `.gitignore` — never gets committed.)

```
./venv/bin/pip install openai tiktoken
./venv/bin/python api_compress.py compress sample.txt sample.api
./venv/bin/python api_compress.py decompress sample.api sample.out

# or, produce a self-extracting HTML page:
./venv/bin/python api_compress.py compress-html sample.txt sample.html
# then open sample.html in any browser, paste your own API key, click Decompress
```

The HTML file is ~10 KB of template + the compressed payload (tens of bytes for small texts, scaling with input). Best on files at least a few KB; the template overhead is paid once.

## Known limitations

- **API determinism risk.** OpenAI's logprobs jitter at ~4th decimal across identical calls. mitigate by (a) quantizing logprobs to 2 decimals before sorting and (b) using a fixed rank distribution rather than one derived from per-call probabilities. Natural English text works in testing; pathological inputs could still desync.
- **Speed.** API-based decompression is ~1 token/sec (one network round-trip per token). A 1 KB file takes ~6 minutes to extract. This is a demo, not a tool.
- **Cost.** Each decompression costs the recipient ~$0.002 per token via the OpenAI API. A 1 KB file costs about 50¢ per extraction.
- **Only English text compresses well.** Random binary, source code, non-English get poor ratios (sometimes worse than the original) because GPT-3.5's predictions are weak there.
- **The compressed payload is not encrypted.** Anyone with the same OpenAI API access can decompress. If you want privacy too, encrypt before compressing.

## Influences and prior art

- Witten, Neal, Cleary, ["Arithmetic Coding for Data Compression"](https://web.stanford.edu/class/ee398a/handouts/papers/WittenACM87ArithmCoding.pdf) (1987) — the arithmetic coder used here.
- DeepMind, ["Language Modeling Is Compression"](https://arxiv.org/abs/2309.10668) (2024) — showed big LLMs beat gzip 3-4× via this technique.
- Fabrice Bellard's [`ts_zip`](https://bellard.org/ts_zip/) (2023) — production LLM compression with a local model.

## License

MIT.
