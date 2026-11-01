# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Three modes with measured trade-offs (`load_engine`, `MODES`,
  `RecognizerConfig.for_mode`): `gpu`, `cpu` and `cpu-preview`; `auto`
  picks `gpu` when CUDA is available. `from_pretrained` takes `mode=`.
- CPU modes (`[cpu]` extra): GigaAM's encoder in ONNX Runtime with INT8
  weights, built once from the stock checkpoint in a child process and
  cached; the PyTorch encoder is never built at run time. ONNX Runtime's
  optimized graph is cached per runtime version, thread count and CPU,
  with its weights memory-mapped.
- Size- and SHA-256-pinned, atomic download of the `v3_e2e_rnnt` and
  `v3_rnnt` checkpoints.
- `benchmarks/run_cpu.py --mode`, `benchmarks/load_check.py` (start time and
  peak memory in fresh processes) and `benchmarks/modes_table.py` (the
  README's mode tables, checked by a test); 898-clip results for the CPU
  modes and the comparison with gigastt on the same CPU.

## [0.1.0] - 2026-11-01

First public version.

### Added

- `StreamingRecognizer`: feed 16 kHz audio in any chunk size, get `Partial`
  and `Final` events; finals carry word start/end times.
- Two-lane decoding over one sample clock: a stable lane (240 ms steps, 8 s
  left / 800 ms right context) whose text never changes, and a faster preview
  lane (160 ms steps, 3.2 s / 320 ms) for the not-yet-settled tail.
- Model-native endpointing (`endpoint_ms`, default 600 ms of decoded silence
  after the last token) and a word-boundary cut for long utterances
  (`max_utterance_ms`); no separate VAD model. An utterance closes only
  before a new word: a word piece or punctuation that the model emits late
  joins its word instead of opening a final of its own.
- `finish()` decodes the last words with the same lookahead as the rest;
  `cancel()` drops a stream.
- Finals and their times do not depend on how the audio is chunked.
- Low-level `StreamingSession` and `StreamingConfig` for custom pipelines;
  a pluggable `Backend` (encoder call and greedy decoder) with
  `StockGigaAM` as the default.
- Examples: file transcription, and a WebSocket server (with a
  `--max-streams` limit) and client.
- Tests on fake backends plus the real stock path on a tiny torch RNN-T on
  CPU; an optional real-checkpoint test.
- Benchmarks: public-data preparation (Golos, FLEURS) and an offline vs
  streaming comparison with WER/CER, latency at 1x, partial stability, word
  timing agreement and GPU memory, with per-clip numeric rows (no
  transcripts) and summaries rebuilt from rows (`--from-rows`,
  `--only-ids`); a separate memory and compute check per mode on GPU or CPU.
- `benchmarks/run_gigastt.py`: the same clips, normalization and metrics for
  a running gigastt server (offline over HTTP, streaming over its
  WebSocket, latency at 1x), with the offline word times it measures
  against (`v3_e2e_rnnt_anchors.jsonl`) and its results.
- README in Russian and English; `docs/` with usage and API, design and
  benchmarks; `AGENTS.md` for contributors and coding agents.

[Unreleased]: https://github.com/Sprytin/GigaAM-v3-Streaming/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Sprytin/GigaAM-v3-Streaming/releases/tag/v0.1.0
