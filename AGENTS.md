# AGENTS.md

Guide for coding agents and contributors. What the library is and how to use
it: [README](README.md) and [`docs/`](docs/).

## Layout

| Path | What |
|---|---|
| `src/gigaam_v3_streaming/` | the package: `recognizer.py` (two lanes, endpointing, events), `session.py` (one lane), `rnnt.py` (greedy decoding over an encoder window), `ring.py` (audio buffer), `config.py`, `metrics.py` (WER/CER) |
| `tests/` | pytest; fake backends plus a tiny torch RNN-T, no GPU or download |
| `examples/` | file transcription, WebSocket server and client |
| `benchmarks/` | data preparation, benchmark runners, `results/` with summaries and per-clip numeric rows |
| `docs/` | [usage](docs/usage.md), [how it works](docs/how-it-works.md), [benchmarks](docs/benchmarks.md) |

## Setup

```bash
pip install "gigaam[torch] @ git+https://github.com/salute-developers/GigaAM"
pip install -e ".[test]"
```

## Tests

```bash
pytest          # fake backends; no GPU, no download (CI: .github/workflows/test.yml)
ruff check .
```

`tests/test_stock_backend.py` runs the real `StockGigaAM` path (dtype cast,
`model.forward`, central-frame map, `rnnt.py` with carried decoder state) on
a tiny torch RNN-T built like GigaAM, on CPU in about a second, and checks
that streaming reproduces an offline greedy decode token for token. It is
skipped when torch is not installed.

`tests/test_gigaam_model.py` also runs the real checkpoint when
`GIGAAM_TEST_WAV` names a 16 kHz mono speech WAV, for example GigaAM's
public sample:

```bash
wget -O example.wav https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/example.wav
GIGAAM_TEST_WAV=example.wav pytest tests/test_gigaam_model.py
```

Test audio is synthetic or public; benchmark audio and per-clip transcripts
stay in `data/` and `*.clips.jsonl`, which are git-ignored.

Before a change is done: `pytest`, `ruff check .` and `ruff format --check .`
pass (CI runs them on Python 3.10 and 3.12).

## Rules

* Every number in `README.md` and `docs/` comes from a file in
  `benchmarks/results/`; rerun or rebuild (`--from-rows`) before changing one.
* `README.md` has a Russian and an English section; change both together.
* Commit only numeric per-clip rows. Audio and transcripts stay in `data/`
  and `*.clips.jsonl`, which are git-ignored.
* Note each user-visible change in `CHANGELOG.md`.
