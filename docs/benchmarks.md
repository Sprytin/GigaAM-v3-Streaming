# Benchmarks

Accuracy, latency, compute and memory on public Russian audio, how to reproduce them, and a comparison with gigastt. Raw numbers are in [`benchmarks/results/`](../benchmarks/results/).

## Setup and results

898 public Russian clips (1.54 h): seeded samples of 300 clips each from the
Golos crowd and farfield test sets and FLEURS `ru_ru` test (one crowd and
one FLEURS clip over GigaAM's 25 s `transcribe` limit skipped). `v3_e2e_rnnt`,
default settings, 80 ms chunks, RTX 4070 SUPER, torch 2.5.1. Offline means
GigaAM's own `model.transcribe` on the whole clip. The GPU was shared with
another workload during the runs, so their timings are higher than on an
idle card (see compute below).

| Set | Clips | WER offline / streaming | CER offline / streaming | Same text as offline |
|---|---:|---:|---:|---:|
| Golos crowd | 299 | 14.64 / 14.90 | 12.84 / 12.20 | 89% |
| Golos farfield | 300 | 6.80 / 7.35 | 3.29 / 3.46 | 94% |
| FLEURS ru | 299 | 5.01 / 5.65 | 1.66 / 1.99 | 67% |
| All | 898 | 7.01 / 7.57 | 3.83 / 3.97 | 84% |

| At 1x real time | p50 | p95 |
|---|---:|---:|
| first partial, after the first word starts (offline word time) | 0.42 s | 0.55 s |
| final, after the last word ends (includes the 600 ms endpoint) | 1.60 s | 2.09 s |

* Partial stability: 12% of partial updates change a word already shown (the
  growing last word not counted).
* Word starts agree with offline `word_timestamps=True`: median 0 ms, 90% of
  8315 matched words within -40..+40 ms.
* Final lag is 800 ms right context + 600 ms endpoint + up to one 240 ms step
  + compute. On the first 100 clips per set, `endpoint_ms=400`
  (`v3_e2e_rnnt_endpoint400.json`) gives 1.40 s (p95 1.89 s) against 1.59 s
  (p95 2.15 s) at the default 600 ms (`v3_e2e_rnnt_first100.json`, the full
  run's rows for the same clips), at WER 7.39 against 7.31, but splits at
  shorter pauses: 42% of those clips gave more than one final, against 12%.
* The WER cost of streaming depends on the set: 0.6 points over all 898
  clips, 0.95 on those 300 clips (6.36 -> 7.31), and 1.8 on their Golos
  farfield part (7.35 -> 9.13 at either endpoint).
* GPU memory (`v3_e2e_rnnt_memory.json`, 30 clips, each mode alone, idle GPU):
  447 MB of weights (encoder fp16, head fp32), peak 470 MB offline and 469 MB
  for one stream, as `torch.cuda.max_memory_allocated`; the CUDA context and
  allocator cache come on top, so `nvidia-smi` shows more. The full run's
  473 MB peak covers both modes in one process.
* Compute, seconds per audio second (`rtf`, compute over the clip's own
  audio for both modes): 0.05 offline and 0.52 streaming in the full run on
  the shared GPU; 0.03 and 0.27 (0.15 without preview) on an idle GPU in the
  memory check. Streaming costs 8-10 times the compute; streams are not
  batched, so one RTX 4070 SUPER serves about 3 real-time streams when idle
  and about 2 when shared. `streaming_per_fed_second` (0.40) divides by
  the audio plus the 2 s silence tail actually fed. Per audio second the
  stable lane encodes about 37.7 s (a 9.04 s window every 0.24 s) and the
  preview lane 23 s (3.68 s every 0.16 s).
* CPU (`v3_e2e_rnnt_cpu.json`, 3 clips, 3 threads, the model in fp32):
  0.09 s per audio second offline, 3.6 streaming and 1.9 without preview,
  so streaming on CPU falls behind real time.

Method: WER and CER normalize hypothesis and reference the same way
(lowercase, `ё`->`е`, punctuation removed; `metrics.normalize_text`). Numbers
are not spelled out, so Golos crowd, whose references write numbers as words
("пятьсот рублей" against "500 ₽"), scores higher than GigaAM's
[evaluation table](https://github.com/salute-developers/GigaAM/blob/main/evaluation.md),
which normalizes numerals (9.1 for `e2e_rnnt`). As a harness check, `v3_rnnt`
offline scores 2.76 on crowd and 3.52 on farfield against 2.4 and 3.9 there.
Each streaming run feeds the clip plus 2 s of silence in 80 ms chunks, so
finals come from the endpointer, then calls `finish()`. Latency is simulated
from the measured compute time of every `feed` call with chunks arriving at
their audio time; reference times are the offline word times. Timings are
wall-clock, so each run records its conditions in `note`. Two clips hit by
a stall were timed again (same text) and are marked `"retimed": true` in
the rows.

Each run writes three files: the summary `.json`, `.rows.jsonl` with one
numeric row per clip (id, set, durations, error counts, timings; no
transcripts; committed), and `.clips.jsonl` with the transcripts
(git-ignored). `--from-rows` rebuilds a summary from rows without a GPU and
`--only-ids` restricts it to the clips of another rows file.

Reproduce (`pip install -e ".[bench]"`, a CUDA GPU):

```bash
python benchmarks/prepare_data.py --out data --per-set 300 --seed 0
python benchmarks/run_benchmark.py --data data --model v3_e2e_rnnt \
    --out benchmarks/results/v3_e2e_rnnt.json
python benchmarks/run_benchmark.py --data data --model v3_e2e_rnnt --endpoint-ms 400 \
    --limit 100 --out benchmarks/results/v3_e2e_rnnt_endpoint400.json
python benchmarks/run_benchmark.py --data data --model v3_rnnt --offline-only \
    --out benchmarks/results/v3_rnnt_offline.json
python benchmarks/memory_check.py --data data --per-set 10 \
    --out benchmarks/results/v3_e2e_rnnt_memory.json
python benchmarks/memory_check.py --data data --per-set 1 --device cpu \
    --out benchmarks/results/v3_e2e_rnnt_cpu.json
python benchmarks/run_benchmark.py --from-rows benchmarks/results/v3_e2e_rnnt.rows.jsonl \
    --only-ids benchmarks/results/v3_e2e_rnnt_endpoint400.rows.jsonl \
    --out benchmarks/results/v3_e2e_rnnt_first100.json
```

`--limit N` takes the first N clips of each set, so the endpoint-400 run
covers the same clips as `--limit 100` at the default endpoint. Common Voice and OpenSTT
are left out; `THIRD_PARTY_LICENSES.md` says why.

gigastt (see [the comparison](#comparison-with-gigastt)): write the offline word times
once on a GPU, start the server, then measure it from outside; `--server-pid`
adds its CPU time and peak memory, and a rerun with the same `--out` resumes.

```bash
python benchmarks/run_gigastt.py --write-anchors --data data \
    --out benchmarks/results/v3_e2e_rnnt_anchors.jsonl
gigastt serve --offline --model-variant e2e_rnnt --model-dir <models> \
    --pool-size 1 --encoder-intra-threads 3 --port 9876 &
python benchmarks/run_gigastt.py --data data --pace fast --server-pid $! \
    --out benchmarks/results/gigastt_e2e_rnnt_fast.json
python benchmarks/run_gigastt.py --data data --pace fast --tail-ms 0 --server-pid $! \
    --out benchmarks/results/gigastt_e2e_rnnt_fast_tail0.json
python benchmarks/run_gigastt.py --data data --pace realtime --server-pid $! \
    --anchors benchmarks/results/v3_e2e_rnnt_anchors.jsonl \
    --out benchmarks/results/gigastt_e2e_rnnt_realtime.json
```

## Comparison with gigastt

Upstream GigaAM has no streaming mode: `transcribe` takes clips up to 25 s,
`transcribe_longform` cuts long audio with a VAD, and the Triton server
serves whole requests.

[gigastt](https://github.com/ekhodzitsky/gigastt) (MIT) streams GigaAM v3 as
well. How the two differ:

| | gigastt 2.22.0 | this library |
|---|---|---|
| Form | Rust CLI and server (WebSocket, HTTP, OpenAI-compatible), SDKs, binaries | Python library on stock `gigaam` + PyTorch |
| Inference | ONNX Runtime, INT8 encoder, CPU by default (CUDA, CoreML builds) | PyTorch, encoder fp16 on CUDA |
| Windows | up to 2.5 s, 1.5 s left context, 0.8 s stride, stable-prefix commits | 8 s left / 0.8 s right, 240 ms stride, plus a 160 ms preview lane |
| Decoder state | fresh per window | carried across windows on an absolute sample clock |
| Endpointing | a run of blank decoder steps (VAD optional) | 600 ms of decoded silence, closes only before a new word |
| Extras | punctuation for the plain head, number normalization, diarization, telephony codecs, multilingual CTC | word times, finals invariant to chunking |

Measured here on the same 898 clips, normalization and metrics
(`benchmarks/run_gigastt.py`, `gigastt_e2e_rnnt_*.json`): the release
binary for Linux x86_64 with its `e2e_rnnt` model (INT8 encoder), defaults
(no punctuation or number normalization added, no VAD), one session, 3
encoder threads, a Ryzen 5 7500F shared with other workloads.

| Set | gigastt WER offline / streaming | this library WER offline / streaming |
|---|---:|---:|
| Golos crowd | 14.32 / 23.83 | 14.64 / 14.90 |
| Golos farfield | 6.57 / 16.97 | 6.80 / 7.35 |
| FLEURS ru | 5.20 / 8.34 | 5.01 / 5.65 |
| All | 7.05 / 12.41 | 7.01 / 7.57 |
| All, CER | 3.76 / 8.75 | 3.83 / 3.97 |

| At 1x real time, p50 / p95 | gigastt | this library |
|---|---:|---:|
| first partial, after the first word starts | 0.78 / 1.62 s | 0.42 / 0.55 s |
| final, after the last word ends | 1.22 / 1.95 s | 1.60 / 2.09 s |
| compute per audio second, streaming | 0.67 s CPU time (3 threads) | 0.52 s GPU (0.27 idle) |
| memory, peak | 516 MB resident (1x run) | 469 MB GPU |

* Offline accuracy is the same, so the INT8 encoder costs nothing here.
  Streaming adds 5.4 WER points. On 46% of clips a pause ends an utterance
  early and the next one starts from overlapping audio, so words repeat or
  break at the join (`ЛДП` / `ЛДПР ТВ.` / `ТВ.`).
* Stopping right after the clip, as gigastt's own client does, instead of
  feeding 2 s of silence first (`gigastt_e2e_rnnt_fast_tail0.json`): 10.94
  streaming WER, 7.32 CER, because the last utterance closes on `stop`
  (682 clips) instead of on a pause.
* Transcripts at real-time pace are identical, clip for clip (898 of 898),
  to those sent as fast as the server takes them, so the WER of the fast
  runs holds at 1x.
* Latency: frame k is sent at (k+1) x 100 ms after audio time 0, and times
  are wall-clock against the same offline word times as here; this library's
  are simulated from the measured compute of each `feed` call. gigastt's
  finals come sooner because its endpoint is a shorter pause; that is also
  why it splits more.
* This library's numbers are from the GPU runs above.

Pick gigastt for a ready-made server, many audio formats, or less compute
at a larger accuracy cost; pick this library when streaming accuracy
matters.
