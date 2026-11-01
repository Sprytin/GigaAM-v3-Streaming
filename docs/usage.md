# Usage

Install, API, supported models and examples. The overview is in the
[README](../README.md).

## Install

```bash
pip install "gigaam[torch] @ git+https://github.com/salute-developers/GigaAM"
pip install -e ".[test]"      # in a clone; extras: cpu (onnxruntime, onnx), server (aiohttp), bench (soundfile, pyarrow)
```

* Python 3.10+. The package itself needs only numpy; the model comes from the
  `gigaam` package, which is not on PyPI in a current version (PyPI has 0.1.0;
  the measured version is 0.2.0 from the repository).
* GigaAM's `[torch]` extra asks for torch>=2.6. Everything here was measured
  with torch 2.5.1+cu124, which also works.
* Live use needs either a CUDA GPU (mode `gpu`) or the `[cpu]` extra (modes
  `cpu` and `cpu-preview`, see [how it works](how-it-works.md#cpu-modes)).
  The stock PyTorch model on CPU falls behind real time (3.6 s per audio
  second on 3 threads, see [benchmarks](benchmarks.md#setup-and-results)).
* Weights download on first `from_pretrained` / `load_model` / `load_engine`;
  pass `download_root=` to choose the directory (default `~/.cache/gigaam`,
  as GigaAM's). `v3_e2e_rnnt` and `v3_rnnt` are pinned by size and SHA-256
  and downloaded atomically: a cut or altered file is never left under the
  final name, and a damaged one is fetched again. Other models go through
  GigaAM's own download.

## API

The distribution is `gigaam-v3-streaming`; the import package is
`gigaam_v3_streaming`. Exports: `load_engine`, `Engine`, `MODES`,
`StreamingRecognizer`, `RecognizerConfig`, `Partial`, `Final`, `Word`,
`Event`, `load_model`, and the single-lane `StreamingSession`,
`StreamingConfig`, `StreamEvent`.

* `load_engine(mode="auto", name="v3_e2e_rnnt", *, device=None, threads=None, cache_dir=None, download_root=None, quantization=None, torch_threads=1, **load_kwargs) -> Engine`
  loads the model once in one of `MODES` (`"gpu"`, `"cpu"`, `"cpu-preview"`;
  `"auto"` is `gpu` when CUDA is available, else `cpu`). `gpu` without CUDA
  raises `RuntimeError`; an unknown mode raises `ValueError` naming the
  modes. `engine.recognizer(config=None)` makes a stream with the mode's
  schedule; `engine.mode`, `.model`, `.backend`, `.config`. `load_kwargs`
  go to `gigaam.load_model` in mode `gpu` only. CPU arguments:
  * `threads`: ONNX Runtime's intra-op thread count (default `min(3,
    cores)`); `torch_threads=1` sets PyTorch's process-wide thread count
    for the small frontend and head, as measured (`None` leaves it alone).
  * `cache_dir`: where the INT8 encoder and its optimized graph live;
    default `$GIGAAM_V3_STREAMING_CACHE`, else
    `$XDG_CACHE_HOME/gigaam-v3-streaming` (`~/.cache/...`).
  * `quantization`: `"int8"` (per tensor, the default) or
    `"int8-per-channel"`; they measured the same
    ([benchmarks](benchmarks.md#cpu-modes)).
* `StreamingRecognizer.from_pretrained(name="v3_e2e_rnnt", device=None, config=None, mode="auto", **load_kwargs)`
  is `load_engine(mode, name, device=device, **load_kwargs).recognizer(config)`;
  or `StreamingRecognizer(model, config=None, backend=None)`.
* `feed(audio, sample_offset=None) -> list[Partial | Final]`: 16 kHz mono,
  float32, int16 or s16le bytes, any length. With `sample_offset` (absolute
  sample index of the chunk) gaps and overlaps raise instead of shifting
  times.
* `finish() -> list[Final]`: decode the tail with full lookahead and close
  the last utterance. `cancel()`: drop the stream.
* `Partial(text, stable_text, start, end)`: current utterance; `stable_text`
  is the part that will not change. `Final(text, start, end, words)` with
  `Word(text, start, end)`. Times are seconds from the start of the stream.
* `RecognizerConfig.for_mode(mode, **overrides)` is the schedule of a mode:
  `gpu` the defaults below, `cpu` a stable lane of 960 ms steps with 4 s left
  and 800 ms right context and no preview, `cpu-preview` the same plus a
  preview lane of 480 ms steps, 2.4 s / 320 ms.
* `RecognizerConfig(stable=..., preview=..., endpoint_ms=600, max_utterance_ms=30000)`.
  `preview=None` halves the compute (`benchmarks/results/v3_e2e_rnnt_memory.json`:
  0.27 -> 0.15 s per s), but partials then trail the audio by the stable
  lane's 800 ms lookahead. `endpoint_ms=None` turns pause endpointing off;
  `max_utterance_ms` still cuts long speech, so set it to `None` too for
  one final per stream.
* One recognizer per audio stream. Recognizers keep their own state and never
  change the model, so many can share one loaded model; call into it from one
  thread, as `examples/ws_server.py` does with a single worker thread.
* `StreamingSession(model, config, backend=None)` is one lane without
  endpointing: `push(audio)` and `finish()` return `StreamEvent(start_sample,
  end_sample, token_ids, token_samples, is_final)` with raw token ids; text is
  rendered by the recognizer.
* `backend` (`gigaam_v3_streaming.session.Backend`) is how a session runs the
  model: `encode(window) -> [1, D, T]` features, one frame per 640-sample
  encoder hop, and `new_decoder(max_symbols_per_step)` returning a greedy
  decoder whose `decode(features)` gives `(token_id, frame)` pairs. The
  default `StockGigaAM(model)` calls `model.forward` (cast to the model's
  device and dtype) and the stock RNN-T head; the tests pass fakes here.

Models: built and measured for GigaAM v3. `v3_e2e_rnnt` (punctuation and
casing) is the default and the one measured streaming; `v3_rnnt` (lowercase,
no punctuation) has the same architecture and was measured offline only.
`load_model` accepts any checkpoint with an RNN-T head, so the older
`v1_rnnt` and `v2_rnnt` load too, but they are untested here. CTC checkpoints
are rejected: decoding is RNN-T only.

## Examples

```bash
python examples/transcribe_file.py speech.wav --realtime
python examples/ws_server.py --port 8765          # needs [server]
python examples/ws_client.py speech.wav --url ws://127.0.0.1:8765/
```

The WebSocket protocol is in the docstring of `examples/ws_server.py`.

Build the CPU modes' encoder ahead of time, for example while building an
image (`--rebuild` drops a model's cached files):

```bash
python -m gigaam_v3_streaming.cpu --threads 3
```
