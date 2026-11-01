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
  so the stock model streams on CPU slower than real time; the
  [CPU modes](#cpu-modes) do it in real time.

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

## CPU modes

The 898 clips through the library's CPU modes (`benchmarks/run_cpu.py
--mode`), 3 ONNX Runtime threads on a Ryzen 5 7500F shared with other
workloads. Latency is measured against the GPU run's offline word times
(`v3_e2e_rnnt_anchors.jsonl`), so it is on the same clock as above.

| Run | WER offline / streaming | CER streaming | First text p50 / p95 | Final p50 / p95 | CPU s per audio s | Peak memory |
|---|---:|---:|---:|---:|---:|---:|
| `cpu` (`v3_e2e_rnnt_mode_cpu.json`) | 7.01 / 7.52 | 4.05 | 1.53 / 1.82 s | 1.98 / 2.48 s | 0.50 | 928 MB |
| `cpu-preview` (`v3_e2e_rnnt_mode_cpu_preview.json`) | 7.01 / 7.50 | 4.05 | 0.65 / 0.85 s | 2.30 / 2.71 s | 0.98 | 952 MB |
| `cpu`, per-channel INT8 (`v3_e2e_rnnt_mode_cpu_int8pc.json`) | 7.06 / 7.51 | 3.98 | 1.53 / 1.82 s | 1.99 / 2.47 s | 0.50 | 1596 MB¹ |
| before: stock model + INT8 encoder (`v3_e2e_rnnt_cpu_int8.json`) | 7.01 / 7.52 | 4.05 | 1.52 / 1.81 s | 1.97 / 2.45 s | 0.48 | 2309 MB |
| before: the same with preview (`v3_e2e_rnnt_cpu_int8_preview.json`) | 7.01 / 7.50 | 4.05 | 0.64 / 0.84 s | 2.25 / 2.68 s | 1.11 | 2310 MB |
| before: fp32 ONNX encoder (`v3_e2e_rnnt_cpu_onnx_fp32.json`) | 7.00 / 7.45 | 4.04 | 1.59 / 1.90 s | 2.13 / 2.82 s | 1.12 | 3366 MB |

¹ This run also built its encoder and graph cache, which sets its peak.

* The library modes decode the same text as the stock model with an INT8
  encoder (same WER and CER) at 40% of its memory; the preview mode also
  costs 12% less CPU, the plain mode is within the shared CPU's noise.
* INT8 costs 0.07 WER points against fp32 and halves the CPU.
* Per-channel against per-tensor INT8, paired on the same 898 clips:
  streaming word errors 649 against 650 of 8640 (difference -0.01 points,
  95% bootstrap interval -0.21..+0.17), offline 610 against 606, CPU
  equal; better on 16 clips, worse on 15. No difference, so per-tensor
  stays the default: it builds faster (33 s against 43 s).
* Streams were not run in parallel; a machine with N cores has room for at
  most about N ÷ (CPU s per audio s) real-time streams.
* Peak memory is the whole benchmark process; 320 MB of it are the
  encoder's weights, memory-mapped from the cache and shared between
  processes.

Start-up (`benchmarks/load_check.py`, `v3_e2e_rnnt_cpu_load.json`): fresh
processes, 3 threads, files in the page cache, median of 5, time to the
first decoded 10 s window and peak resident memory:

| Start | Ready | Peak memory |
|---|---:|---:|
| Stock model on CPU + INT8 encoder, graph optimized on every start (before) | 4.6 s | 2329 MB |
| Lean model, graph optimized on every start | 2.1 s | 1024 MB |
| Lean model + cached graph, mapped weights (`load_engine("cpu")`) | 2.1 s | 799 MB |

The cached graph halves the session load (0.44 s against 0.89 s) but the
first window pays for mapping the weights in (0.70 s against 0.34 s), so
the time to the first result is the same; what it saves is memory, and the
mapped weights are shared between processes. Building the encoder once:
33 s (per channel 43 s) and a 4.9 GB peak in the export child.

Reproduce (`pip install -e ".[bench,cpu]"`, no GPU; the clips from
`prepare_data.py` as above):

```bash
python benchmarks/run_cpu.py --data data --mode cpu --threads 3 \
    --anchors benchmarks/results/v3_e2e_rnnt_anchors.jsonl \
    --out benchmarks/results/v3_e2e_rnnt_mode_cpu.json
python benchmarks/run_cpu.py --data data --mode cpu-preview --threads 3 \
    --anchors benchmarks/results/v3_e2e_rnnt_anchors.jsonl \
    --out benchmarks/results/v3_e2e_rnnt_mode_cpu_preview.json
python benchmarks/run_cpu.py --data data --mode cpu --quantization int8-per-channel \
    --threads 3 --anchors benchmarks/results/v3_e2e_rnnt_anchors.jsonl \
    --out benchmarks/results/v3_e2e_rnnt_mode_cpu_int8pc.json
python benchmarks/load_check.py --cache-dir /tmp/gigaam-cache --repeat 5 \
    --out benchmarks/results/v3_e2e_rnnt_cpu_load.json
python benchmarks/modes_table.py --write README.md   # the README's mode tables
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
| Inference | ONNX Runtime, INT8 encoder, CPU by default (CUDA, CoreML builds) | PyTorch, encoder fp16 on CUDA; or ONNX Runtime, INT8 encoder on CPU |
| Windows | up to 2.5 s, 1.5 s left context, 0.8 s stride, stable-prefix commits | 8 s left / 0.8 s right, 240 ms stride, plus a 160 ms preview lane (CPU: 960 ms stride, 4 s left, optional 480 ms preview) |
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
* This library's numbers are from the GPU runs above. On the same CPU as
  gigastt, its [CPU modes](#cpu-modes) give 7.52 streaming WER at 0.50 CPU s
  per audio s with first text at 1.53 s p50, or 7.50 at 0.98 with first
  text at 0.65 s; finals come at 1.98 and 2.30 s, later than gigastt's.

Pick gigastt for a ready-made server, many audio formats, or the soonest
finals at a larger accuracy cost; pick this library when streaming accuracy
matters, on a GPU or a CPU.
