# How it works

Design of the streaming decoder and its limits.

## Decoding

GigaAM's Conformer encoder is bidirectional and has no streaming cache, so
each step re-encodes a bounded window `left context | emit | right context`
placed on a fixed grid of the absolute sample clock, and runs the stock RNN-T
predictor and joint greedily over the emit part, carrying predictor state
between windows. Because windows follow the sample clock, finals do not
depend on how audio was chunked.

| Lane | Step | Left / right context | Role |
|---|---|---|---|
| stable | 240 ms | 8 s / 800 ms | text that never changes once shown |
| preview | 160 ms | 3.2 s / 320 ms | newest words, revised until the stable lane reaches them |

* A `Partial` is the stable text plus the preview tail.
* A `Final` comes once the stable lane has decoded `endpoint_ms` past the
  last token: the model's own silence detection, no VAD.
* An utterance closes only before a token that starts a new word. The model
  sometimes emits the rest of a word, or its punctuation, several hundred ms
  late; such a piece joins its word, and punctuation arriving after its
  utterance closed is dropped.
* Utterances longer than `max_utterance_ms` are cut at a word boundary.
* `finish()` pads silence so the last words get the same lookahead; no time
  past the end of the audio is reported.
* Word times use GigaAM's `word_timestamps=True` convention on the 40 ms
  encoder grid: start = frame of the first token, end = one frame after the
  last token.

Code: `recognizer.py` (two lanes, endpointing, events), `session.py`
(one lane), `rnnt.py` (greedy decoding over an encoder window), `ring.py`
(audio buffer), `config.py`, `metrics.py` (WER/CER used by the benchmark).

## CPU modes

`load_engine("cpu")` and `load_engine("cpu-preview")` run GigaAM's encoder,
almost all of the compute, in ONNX Runtime with INT8 weights; the log-mel
frontend and the RNN-T predictor and joint stay the checkpoint's own
PyTorch modules (`onnx_backend.OnnxEncoder`, a `Backend`). Code: `cpu.py`,
`onnx_backend.py`, `engine.py`, `download.py`.

* The schedule differs from the GPU's: the stable lane steps 960 ms with
  4 s left context, so each step re-encodes about 5.8 s once per 0.96 s
  instead of 9 s every 240 ms; `cpu-preview` adds a 480 ms preview lane
  with 2.4 s / 320 ms context.
* First start: `cpu.prepare` exports the encoder with GigaAM's own
  `to_onnx`, quantizes its MatMul weights with ONNX Runtime's
  `quantize_dynamic` (`QInt8`; per tensor by default,
  `quantization="int8-per-channel"` per output channel) and stores
  `<name>-<checkpoint sha256[:16]>-encoder-<quantization>.onnx` in the
  cache. The export needs the whole PyTorch model, so it runs in a child
  process (`python -m gigaam_v3_streaming.cpu`) that gives its memory back
  when it exits; it works in a temporary directory and renames the result
  into place, so an interrupted build leaves nothing under the final name.
* Every start: `cpu.CpuModel` memory-maps the checkpoint and builds only the
  frontend, the RNN-T head and the tokenizer from its config; the PyTorch
  encoder is never built and its weights are never read.
* ONNX Runtime spends most of a session's creation optimizing the graph.
  The optimized graph is saved once as ONNX with its weights in a separate
  file (`<encoder>-<sha>-ort<version>-t<threads>-<machine>-<flags
  hash>/encoder.onnx` and `encoder.data`), and later sessions load it with
  optimization off, so ONNX Runtime memory-maps the weights instead of
  copying them: less resident memory, and the weights are shared through
  the page cache by every process on the machine. (ONNX Runtime's own ORT
  format was measured too: it loaded no faster and peaked higher, as it
  copies the weights, so it is not used.) The graph is specific to the
  runtime version, the thread count and the CPU's instruction set, so all
  three are in the entry's name: a new ONNX Runtime, another thread count or
  the cache copied to another CPU makes a new entry instead of loading a
  wrong one. The idea, not code, is from gigastt's session cache. An entry
  is built in a temporary directory and renamed into place; one that does
  not load (a damaged graph or weights) is rebuilt with a warning; a cache
  directory that cannot be written costs only the saving.

## Limits

* Each stable step re-encodes about 9 s of audio and each preview step about
  3.7 s: the price of an unchanged offline encoder, 8-10 times the offline
  compute.
* The stock PyTorch model is too slow on CPU; the CPU modes trade
  first-text delay for compute instead (see the mode table in the README).
* No batching across streams; `examples/ws_server.py` serializes all streams
  on one worker thread and refuses connections past `--max-streams`
  (default 2) with 503.
* Greedy decoding only; no beam search or language model.
* 16 kHz mono input only; resample before feeding.
