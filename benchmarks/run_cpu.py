"""Streaming GigaAM on CPU: ONNX (INT8 or fp32) encoder, any lane schedule, same clips and metrics.

Like ``run_benchmark.py``, but everything runs on the CPU: a library mode
(``--mode cpu`` or ``cpu-preview``: ``load_engine`` as shipped, its schedule
unless ``--stable``/``--preview`` say otherwise), an ONNX encoder file
(``--encoder``, from ``export_onnx.py``) or the stock PyTorch encoder
(``--encoder torch``); the frontend and RNN-T head in PyTorch on one thread.
Per clip it records the offline decode with the same encoder (whole clip, one
window) and the streaming run, with wall time and process CPU time (all
threads) of every call. Latency anchors (offline PyTorch word times) come from
``--anchors`` (``v3_e2e_rnnt_anchors.jsonl``), so no GPU is needed.

Rows are appended as each clip finishes, and ``--resume`` skips the clips
already in ``<out>.rows.jsonl``, so a preempted run continues where it stopped.

    python benchmarks/run_cpu.py --data data --mode cpu --threads 3 \
        --anchors benchmarks/results/v3_e2e_rnnt_anchors.jsonl --out benchmarks/results/cpu.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
from run_benchmark import RATE, paced, public_row, revisions, write_report

from gigaam_v3_streaming import (
    Final,
    Partial,
    RecognizerConfig,
    StreamingConfig,
    StreamingRecognizer,
    load_engine,
    load_model,
)
from gigaam_v3_streaming.rnnt import RNNTGreedyState
from gigaam_v3_streaming.session import StockGigaAM


def lane(spec):
    if spec == "none":
        return None
    emit, left, right = (int(part) for part in spec.split("/"))
    return StreamingConfig(emit_ms=emit, left_context_ms=left, right_context_ms=right)


def spec(config):
    if config is None:
        return "none"
    return f"{config.emit_ms}/{config.left_context_ms}/{config.right_context_ms}"


def cpu_model():
    for line in open("/proc/cpuinfo", encoding="utf-8"):
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def clocks():
    return time.perf_counter(), time.process_time()


def offline(model, backend, audio):
    wall, cpu = clocks()
    encoded = backend.encode(audio)
    tokens = [token for token, _ in RNNTGreedyState(model.head, 10).decode(encoded)]
    text = model.decoding.tokenizer.decode(tokens)
    now_wall, now_cpu = clocks()
    return text, now_wall - wall, now_cpu - cpu


def stream(model, backend, audio, chunk, tail, config):
    padded = np.concatenate([audio, np.zeros(tail, dtype=np.float32)])
    recognizer = StreamingRecognizer(model, config, backend)
    timeline, cpu_total = [], 0.0
    steps = [
        (at, lambda at=at: recognizer.feed(padded[at : at + chunk]))
        for at in range(0, padded.size, chunk)
    ]
    steps.append((padded.size, recognizer.finish))
    for at, step in steps:
        wall, cpu = clocks()
        events = step()
        now_wall, now_cpu = clocks()
        cpu_total += now_cpu - cpu
        timeline.append((min(at + chunk, padded.size) / RATE, now_wall - wall, events))
    return timeline, cpu_total


def streaming_row(model, backend, audio, chunk, tail, config, anchor):
    timeline, cpu = stream(model, backend, audio, chunk, tail, config)
    events = list(paced(timeline))
    finals = [(wall, e) for wall, e in events if isinstance(e, Final)]
    first_partial = next((wall for wall, e in events if isinstance(e, Partial)), None)
    partial_words = [e.text.split() if isinstance(e, Partial) else None for _, e in events]
    return {
        "streaming": " ".join(e.text for _, e in finals),
        "stream_compute_s": sum(compute for _, compute, _ in timeline),
        "stream_cpu_s": cpu,
        "stream_audio_s": (audio.size + tail) / RATE,
        "first_partial_s": (
            first_partial - anchor["first_start"] if first_partial is not None and anchor else None
        ),
        "final_lag_s": finals[-1][0] - anchor["last_end"] if finals and anchor else None,
        "partials": sum(words is not None for words in partial_words),
        "revisions": revisions(partial_words),
        "word_offsets_s": [],
        "finals": len(finals),
    }


def cpu_summary(rows):
    rows = [row for row in rows if not row.get("skipped")]
    audio = sum(row["seconds"] for row in rows)
    return {
        "offline_wall": round(sum(row["offline_compute_s"] for row in rows) / audio, 4),
        "offline_cpu": round(sum(row["offline_cpu_s"] for row in rows) / audio, 4),
        "streaming_wall": round(sum(row["stream_compute_s"] for row in rows) / audio, 4),
        "streaming_cpu": round(sum(row["stream_cpu_s"] for row in rows) / audio, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="v3_e2e_rnnt")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mode", choices=("cpu", "cpu-preview"), help="a library CPU mode")
    source.add_argument("--encoder", help="ONNX encoder path, or 'torch' for the stock encoder")
    parser.add_argument("--quantization", help="with --mode: int8 or int8-per-channel")
    parser.add_argument("--cache-dir", type=Path, help="with --mode: the CPU modes' cache")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument(
        "--stable", help="emit/left/right ms (default: the mode's, else 240/8000/800)"
    )
    parser.add_argument(
        "--preview", help="emit/left/right ms or none (default: the mode's, else 160/3200/320)"
    )
    parser.add_argument("--endpoint-ms", type=int, default=600)
    parser.add_argument("--chunk-ms", type=int, default=80)
    parser.add_argument("--tail-ms", type=int, default=2000)
    parser.add_argument("--max-seconds", type=float, default=25.0)
    parser.add_argument("--limit", type=int, default=0, help="clips per set (0 = all)")
    parser.add_argument("--anchors", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--note", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import soundfile as sf
    import torch

    items = [json.loads(line) for line in (args.data / "manifest.jsonl").open(encoding="utf-8")]
    skipped = [
        {key: item[key] for key in ("set", "id", "seconds")} | {"skipped": True}
        for item in items
        if item["seconds"] > args.max_seconds
    ]
    items = [item for item in items if item["seconds"] <= args.max_seconds]
    if args.limit:
        items = [
            item
            for name in sorted({item["set"] for item in items})
            for item in [i for i in items if i["set"] == name][: args.limit]
        ]
    anchors = {}
    if args.anchors:
        for line in args.anchors.open(encoding="utf-8"):
            row = json.loads(line)
            anchors[(row["set"], row["id"])] = row

    torch.set_num_threads(args.threads if args.encoder == "torch" else 1)
    started = time.perf_counter()
    if args.mode:
        engine = load_engine(
            args.mode,
            args.model,
            threads=args.threads,
            cache_dir=args.cache_dir,
            quantization=args.quantization,
        )
        model, backend, config = engine.model, engine.backend, engine.config
        encoder = backend.encoder_path.name
    else:
        model = load_model(args.model, "cpu", fp16_encoder=False)
        config = RecognizerConfig(stable=lane("240/8000/800"), preview=lane("160/3200/320"))
        if args.encoder == "torch":
            backend, encoder = StockGigaAM(model), "torch"
        else:
            from gigaam_v3_streaming.onnx_backend import OnnxEncoder

            backend, encoder = (
                OnnxEncoder(model, args.encoder, args.threads),
                Path(args.encoder).name,
            )
            model.encoder.layers = torch.nn.ModuleList()  # only pre_encode geometry is still read
    load_s = time.perf_counter() - started
    config = RecognizerConfig(
        stable=lane(args.stable) if args.stable else config.stable,
        preview=lane(args.preview) if args.preview else config.preview,
        endpoint_ms=args.endpoint_ms,
    )
    chunk, tail = args.chunk_ms * RATE // 1000, args.tail_ms * RATE // 1000
    run = {
        "model": args.model,
        "mode": args.mode,
        "encoder": encoder,
        "load_s": round(load_s, 2),
        "settings": {
            "threads": args.threads,
            "chunk_ms": args.chunk_ms,
            "tail_ms": args.tail_ms,
            "endpoint_ms": args.endpoint_ms,
            "stable": spec(config.stable),
            "preview": spec(config.preview),
        },
        "environment": {
            "cpu": cpu_model(),
            "torch": torch.__version__,
            "onnxruntime": importlib.metadata.version("onnxruntime"),
            "python": platform.python_version(),
        },
        "compute": "wall = perf_counter, cpu = process_time (all threads), per audio second",
        "note": args.note,
    }

    rows_path = args.out.with_suffix(".rows.jsonl")
    clips_path = args.out.with_suffix(".clips.jsonl")
    done = {}
    if args.resume and rows_path.exists():
        for line in list(rows_path.open(encoding="utf-8"))[1:]:
            row = json.loads(line)
            done[(row["set"], row["id"])] = row
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not done:
        with rows_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"run": run}, ensure_ascii=False) + "\n")
            for row in skipped:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        clips_path.write_text("", encoding="utf-8")

    # Warm up the session on silence so the first clip is not charged for it.
    stream(model, backend, np.zeros(RATE * 3, dtype=np.float32), chunk, 0, config)
    todo = [item for item in items if (item["set"], item["id"]) not in done]
    for number, item in enumerate(todo, 1):
        audio, rate = sf.read(args.data / item["audio"], dtype="float32")
        assert rate == RATE
        text, offline_wall, offline_cpu = offline(model, backend, audio)
        row = {
            "set": item["set"],
            "id": item["id"],
            "seconds": item["seconds"],
            "reference": item["text"],
            "offline": text,
            "offline_compute_s": offline_wall,
        }
        row |= streaming_row(
            model, backend, audio, chunk, tail, config, anchors.get((item["set"], item["id"]))
        )
        public = public_row(row) | {
            "offline_cpu_s": round(offline_cpu, 4),
            "stream_cpu_s": round(row["stream_cpu_s"], 4),
        }
        with rows_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(public, ensure_ascii=False) + "\n")
        with clips_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if number % 25 == 0:
            print(f"{number}/{len(todo)}", flush=True)

    rows = [json.loads(line) for line in list(rows_path.open(encoding="utf-8"))[1:]]
    run["cpu_rtf"] = cpu_summary(rows)
    run["cpu_rtf_by_set"] = {
        name: cpu_summary([row for row in rows if row["set"] == name])
        for name in sorted({row["set"] for row in rows})
    }
    run["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    run["pid"] = os.getpid()
    write_report(args.out, run, rows)


if __name__ == "__main__":
    main()
