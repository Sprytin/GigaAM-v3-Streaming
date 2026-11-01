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

## Limits

* Each stable step re-encodes about 9 s of audio and each preview step about
  3.7 s: the price of an unchanged offline encoder, 8-10 times the offline
  compute.
* CUDA only in practice; on CPU streaming is slower than real time.
* No batching across streams; `examples/ws_server.py` serializes all streams
  on one worker thread and refuses connections past `--max-streams`
  (default 2) with 503.
* Greedy decoding only; no beam search or language model.
* 16 kHz mono input only; resample before feeding.
