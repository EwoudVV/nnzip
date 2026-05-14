# nnzip — neural-network text compression

[![PyPI](https://img.shields.io/pypi/v/nnzip)](https://pypi.org/project/nnzip/)
[![tests](https://github.com/EwoudVV/nnzip/actions/workflows/tests.yml/badge.svg)](https://github.com/EwoudVV/nnzip/actions/workflows/tests.yml)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/nnzip)](https://pypi.org/project/nnzip/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A cross-platform CLI that compresses English text using a local GPT-2 as a probability model. On natural prose it gets around **15-25% of the original size** — typically 3-5× better than `gzip`.

```
pip install nnzip
```

```
compress book.txt              # produces book.txt.nnz
decompress book.txt.nnz        # restores book.txt
```

Works on **macOS, Linux, and Windows**. No GPU required; uses [llama.cpp](https://github.com/ggerganov/llama.cpp) under the hood so it picks up Metal on Apple Silicon and AVX/CUDA elsewhere automatically.

## What it actually does

When you compress a file, nnzip walks through it one token at a time. At each position it asks GPT-2: *given everything before, what's your probability distribution over the next token?* Then it spends `-log₂(P(actual token))` bits encoding it with arithmetic coding.

If GPT-2 is 90% sure about the next token (the very common case in fluent English), encoding costs about 0.15 bits. If GPT-2 is totally surprised (1-in-50,000), it costs ~16 bits. The average across natural English ends up around 4-5 bits per token instead of the ~32 bits each token would take if stored naively.

Decompression runs the same forward passes in the same order. Because GPT-2 is deterministic with greedy inference, both sides see identical probability distributions and the arithmetic coder unwinds back to the exact original token stream. The decompressed file is **bit-identical** to the original.

The compressed `.nnz` file contains zero model weights — just `MAGIC + version + token_count + arithmetic-coded payload`. Both ends rely on the same pinned GGUF model from Hugging Face, downloaded once to `~/.cache/huggingface/` on first use (~252 MB).

## Quick demo

```
$ printf 'The morning rain pattered against the windows of the small cottage.' > demo.txt
$ wc -c demo.txt
67 demo.txt

$ compress demo.txt
compressing demo.txt -> demo.txt.nnz
loading openai-gpt2-124M-F16.gguf...
loaded in 0.3s (vocab 50257, threads 9)
  encoded 13/13 tokens (47.6 tok/s)

original:    67 bytes
compressed:  21 bytes (31.3% of original)

$ decompress demo.txt.nnz
recovered 67 chars -> demo.txt
```

A 50 KB chunk of *Pride and Prejudice* lands at about 23% of the original (~11.5 KB). For comparison, `gzip -9` on the same input gets ~57%.

## Performance and limits, plainly

- **Speed.** Around 50 tokens/sec on an Apple M1 Max CPU. A 100 KB English file takes a couple of minutes to compress and another couple to decompress. This is slower than `gzip` by orders of magnitude. It is not a tool you'd use to compress your downloads folder.
- **English-only is its sweet spot.** GPT-2 was trained on English internet text. Random binary, source code, non-English, base64-encoded blobs — these typically compress to 100%+ of the original (nnzip gives up and emits its escape encoding for everything). Don't use this on data that doesn't look like English prose.
- **Lossless.** Provably. Arithmetic coding with a deterministic probability source round-trips bit-for-bit.
- **GPT-2 has a 1024-token context window.** Past that, nnzip uses a sliding window of the last 512 tokens to predict the next one. Long-range compression suffers a little after the first ~1000 tokens, but it still works on arbitrarily large files.
- **Cross-platform install, same-machine round-trip.** llama.cpp uses platform-specific acceleration (Metal on Mac, AVX on Linux/Windows, CUDA if available), and these can produce floating-point logits that differ in the last few bits across machines. Same machine that compressed should decompress. Same OS / CPU family is usually fine.
- **No encryption.** Anyone with the same nnzip version can decompress a `.nnz` file. Use a real encryption tool on top if you need privacy.

## Why GPT-2 is a great compressor for English

Shannon's source coding theorem says you can't compress data below its entropy — the average number of bits needed per symbol given perfect prediction. For English text, the entropy is somewhere around 1.0-1.3 bits per character. Most classical compressors (gzip, bzip2, xz) approximate the entropy using simple statistical models — adjacent character frequencies, run-length, Lempel-Ziv pattern matching. Their best on plain English is around 25-30% of original.

GPT-2 is a much smarter model. It's seen billions of words and learned what's plausible at a phrase, sentence, and paragraph level. So when it predicts the next token, its distribution is sharper — closer to the data's true entropy. Sharper predictions mean fewer bits per symbol via arithmetic coding. That's all the trick is.

Bigger models compress better still. DeepMind showed in [Language Modeling Is Compression](https://arxiv.org/abs/2309.10668) (2024) that Chinchilla 70B compresses Wikipedia to ~8% of original, beating every classical codec. The trade-off is obvious: bigger model, more compute. GPT-2 small (124M params, 252 MB) is a practical sweet spot — fast enough to actually use, small enough to ship via pip.

## Optional tunables

| Env var | Effect |
|---|---|
| `NNZIP_MODEL_PATH=/path/to/your.gguf` | Use a different GGUF model (any llama.cpp-compatible GPT-2 variant). Both sides need to use the same one. |

## What's in this repo

The `nnzip` CLI is the *current* thing in this project. The repo also includes a multi-stage experiment that led here — the kind of journey that goes from "wrong idea" to "right idea." If you only care about the tool, skip the rest.

### The actual tool (stages 7-8 of the journey)

| File | What it does |
|---|---|
| `nnzip/__init__.py` | The whole package: model loading, arithmetic coding, CLI entry points |
| `pyproject.toml` | Declares the `compress`, `decompress`, and `nnzip` CLI commands plus dependencies |
| `arithmetic_coder.py` | A standalone portable arithmetic coder (used by the HTML self-extractor below; nnzip itself uses `constriction`) |
| `api_compress.py` | An earlier OpenAI-API-based experiment: same idea but uses OpenAI's API as the probability model instead of a local one. Slower and pay-per-use; left in for reference. |
| `template.html` | A self-extracting HTML wrapper for the API version — the `.nnz` payload bakes into a single HTML file the recipient can open in any browser |

### The hash brute-forcing detour (stages 1-5)

Before landing on real compression, the project spent stage 1-5 trying to brute-force decompress files from just their SHA-256 hash + length. That doesn't actually work (the pigeonhole principle is a wall), but it's an entertaining way to learn why and to push hardware to its limits.

| File | Role | Best result |
|---|---|---|
| `compress.py` / `decompress.py` | Python brute forcer | ~0.6 M hashes/s |
| `compress_index.py` / `decompress_index.py` | "Deterministic ordering" variant that makes the failure visible | proves the size wall |
| `brute.c` | C version with CommonCrypto + pthreads | ~45 M H/s, ~75× Python |
| `brute_neon.c` | ARMv8 SHA-2 hardware intrinsics | ~380 M H/s, ~635× Python |
| `brute_mb.c` | 4-way multi-buffer SIMD SHA-256 — an instructive failure (slower than hardware SHA on M1) | ~80 M H/s |
| `brute_metal.m` | Metal compute shader on M1 Max's GPU (32 cores, 4096 ALU lanes) | ~1.0 GH/s |
| `brute_combined.m` | CPU NEON-HW and GPU running concurrently on different parts of the search space | **~1.4 GH/s** (~2300× Python) |

Build them with `clang -O3 -Wall -Wno-deprecated-declarations -o brute brute.c` etc. They're not part of the pip package — they're standalone executables for stress-testing.

## The journey, summarized

| Stage | Idea | Outcome |
|---|---|---|
| 1 | "Just send the SHA-256 hash and brute-force decompress" | Doesn't work — pigeonhole guarantees collisions |
| 2 | C + threads | Faster brute force, same impossibility |
| 3 | NEON hardware SHA | Faster still |
| 4 | M1 Max GPU compute shader | 1 GH/s |
| 5 | CPU + GPU concurrent | 1.4 GH/s |
| 6 | "Use a deterministic generator and send the index" | Mathematically equivalent to storing the file as a giant integer — the index is the same size as the file |
| 7 | **Local GPT-2 + arithmetic coding** | Actually compresses |
| 8 | API and HTML variants | Same idea, different deployment models |

The lesson behind 1-6 is the pigeonhole principle: there are more N-byte inputs than there are shorter outputs, so no scheme can compress every input. Real compression escapes by giving up on compressing *arbitrary* data and instead exploiting the patterns in the data we actually have. nnzip takes that to its modern extreme — the "pattern" is everything GPT-2 learned about English from billions of words of internet text.

## Inspirations and prior art

- Witten, Neal, Cleary, ["Arithmetic Coding for Data Compression"](https://web.stanford.edu/class/ee398a/handouts/papers/WittenACM87ArithmCoding.pdf) (1987) — the core algorithm.
- DeepMind, ["Language Modeling Is Compression"](https://arxiv.org/abs/2309.10668) (2024) — showed that big LLMs are state-of-the-art compressors.
- Fabrice Bellard's [`ts_zip`](https://bellard.org/ts_zip/) (2023) — production LLM compression with a custom model.

## License

MIT.
