# Third-party licenses

GigaAM-v3 Streaming is MIT-licensed (see `LICENSE`). It contains no code,
weights or audio from other projects; everything below is downloaded by the
user at install or run time and stays under its own license.

## Model

| Component | Where it comes from | License |
|---|---|---|
| GigaAM code (`gigaam` package) | https://github.com/salute-developers/GigaAM | MIT, (c) 2024 GigaChat Team |
| GigaAM v3 checkpoints and tokenizers (`v3_e2e_rnnt`, `v3_rnnt`) | downloaded by `gigaam.load_model` from the GigaAM CDN; mirrored at https://huggingface.co/ai-sage/GigaAM-v3 | MIT (model card: `license: mit`) |

## Python dependencies

| Package | Used for | License |
|---|---|---|
| numpy | core (required) | BSD-3-Clause |
| torch, torchaudio | model inference (`[model]`) | BSD-3-Clause, BSD-2-Clause |
| soundfile | reading WAV in examples and benchmarks | BSD-3-Clause (its wheels bundle libsndfile, LGPL-2.1, dynamically loaded) |
| aiohttp | WebSocket example server (`[server]`) | Apache-2.0 |
| pyarrow | reading Golos parquet files (`[bench]`) | Apache-2.0 |
| pytest, ruff | tests and lint (`[test]`) | MIT |

GigaAM's own dependencies (hydra-core MIT, omegaconf BSD-3-Clause, onnx
Apache-2.0, onnxruntime MIT, sentencepiece Apache-2.0, tqdm MPL-2.0 and MIT)
come with it. All of these are permissive and compatible with MIT
distribution of this package, which bundles none of them.

## Benchmark data (not redistributed)

`benchmarks/prepare_data.py` downloads test audio to a local directory of
your choice; no audio or transcript is committed here. Only aggregate numbers
are published in `benchmarks/results/`.

| Set | Source | License |
|---|---|---|
| Golos crowd / farfield test | https://github.com/salute-developers/golos, read from the `bond005/sberdevices_golos_*` mirrors | Golos license (custom, see `license/en_us.pdf` in the Golos repository) |
| FLEURS `ru_ru` test | https://huggingface.co/datasets/google/fleurs | CC BY 4.0 |

Common Voice and OpenSTT were left out: Common Voice requires accepting its
terms per download (no anonymous scripted fetch), and OpenSTT's
CC BY-NC license and size make it a poor default for a reproducible check.
