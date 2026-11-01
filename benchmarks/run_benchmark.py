"""Offline vs streaming GigaAM on a manifest from ``prepare_data.py``.

For every clip the script runs the stock offline ``model.transcribe`` and the
streaming recognizer on the same audio, then reports per set:

* WER/CER of both after the same normalization (lowercase, ``ё``->``е``,
  punctuation removed; see ``gigaam_v3_streaming.metrics.normalize_text``);
* compute: GPU seconds per audio second of both, on the same audio seconds;
* latency at 1x real time, simulated exactly from the measured compute time of
  every ``feed`` call (one worker, chunks arriving at their audio time): first
  partial after the offline model's first word emission, and final after the
  offline model's last word end, which includes the endpoint pause;
* how often a partial revises words it already showed, and how many clips
  came out as more than one final;
* GPU memory: weights and peak ``torch.cuda.max_memory_allocated`` (the CUDA
  context and allocator cache come on top; ``nvidia-smi`` shows more).

The streaming input is the clip followed by ``--tail-ms`` of silence, as a
live microphone would continue, then ``finish()``; finals are therefore
produced by the endpointer, not by the end of the file. Clips longer than
``--max-seconds`` (25 s, the limit of ``model.transcribe``) are skipped and
counted per set. ``--offline-only`` checks the harness against GigaAM's
published offline numbers without streaming.

Timings are wall-clock: anything else on the GPU makes them worse, never
better, and the report does not know whether the GPU was shared. Record that
with ``--note``.

Next to the report, ``<out>.rows.jsonl`` holds one numeric row per clip (ids,
error counts, timings; no transcripts) after a header line with the run
settings; ``--from-rows`` rebuilds the report from it, and ``--only-ids``
restricts it to the clips of another rows file. Transcripts go to the
git-ignored ``<out>.clips.jsonl``.

Usage::

    python benchmarks/run_benchmark.py --data data --model v3_e2e_rnnt \
        --out benchmarks/results/v3_e2e_rnnt.json
    python benchmarks/run_benchmark.py --from-rows benchmarks/results/v3_e2e_rnnt.rows.jsonl \
        --only-ids benchmarks/results/v3_e2e_rnnt_endpoint400.rows.jsonl \
        --out benchmarks/results/v3_e2e_rnnt_first100.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from gigaam_v3_streaming import Final, Partial, RecognizerConfig, StreamingRecognizer, load_model
from gigaam_v3_streaming.metrics import normalize_text, percentile, transcript_scores

RATE = 16_000


def offline(model, path):
    import torch

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    began = time.perf_counter()
    result = model.transcribe(str(path), word_timestamps=True)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return result, time.perf_counter() - began


def stream(model, audio, chunk, tail, config):
    """Feed ``audio`` + silence in ``chunk``-sample pieces; time every call."""
    import torch

    padded = np.concatenate([audio, np.zeros(tail, dtype=np.float32)])
    recognizer = StreamingRecognizer(model, config)
    timeline = []  # (audio_time_of_chunk_end, compute_seconds, events)
    for at in range(0, padded.size, chunk):
        began = time.perf_counter()
        events = recognizer.feed(padded[at : at + chunk])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timeline.append((min(at + chunk, padded.size) / RATE, time.perf_counter() - began, events))
    began = time.perf_counter()
    events = recognizer.finish()
    timeline.append((padded.size / RATE, time.perf_counter() - began, events))
    return timeline


def paced(timeline):
    """Yield (wall_time, event) as if chunks arrived in real time."""
    done = 0.0
    for arrival, compute, events in timeline:
        done = max(done, arrival) + compute
        for event in events:
            yield done, event


def revisions(partials):
    """Count partial updates that change words already shown (last word excluded)."""
    changed, previous = 0, []
    for words in partials:
        if words is None:
            previous = []
            continue
        settled = previous[:-1]
        if words[: len(settled)] != settled:
            changed += 1
        previous = words
    return changed


def word_offsets(stream_words, offline_words):
    a = [normalize_text(word.text) for word in stream_words]
    b = [normalize_text(word.text) for word in offline_words]
    offsets = []
    for block in SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks():
        for k in range(block.size):
            offsets.append(stream_words[block.a + k].start - offline_words[block.b + k].start)
    return offsets


TIMINGS = (
    "offline_compute_s",
    "stream_compute_s",
    "stream_audio_s",
    "first_partial_s",
    "final_lag_s",
    "partials",
    "revisions",
    "word_offsets_s",
    "finals",
)


def public_row(row):
    """The numeric part of a clip row: error counts instead of transcripts."""
    out = {key: row[key] for key in ("set", "id", "seconds")}
    for name in ("offline", "streaming"):
        if name in row:
            score = transcript_scores(row["reference"], row[name])
            out[name] = {
                metric: [score[mode][unit]["errors"], score[mode][unit]["reference_units"]]
                for metric, mode, unit in (
                    ("wer", "normalized", "wer"),
                    ("cer", "normalized", "cer"),
                    ("wer_raw", "raw", "wer"),
                )
            }
    if "streaming" in row:
        out["same_text"] = normalize_text(row["offline"]) == normalize_text(row["streaming"])
    for key in TIMINGS:
        if key in row:
            out[key] = _round(row[key])
    return out


def _round(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, list):
        return [_round(item) for item in value]
    return value


def summarize(rows):
    """Report one set from public rows; skipped clips are only counted."""

    def p(values, q):
        value = percentile(values, q)
        return None if value is None else round(value, 3)

    def rate(name, metric):
        errors = sum(row[name][metric][0] for row in rows)
        units = sum(row[name][metric][1] for row in rows)
        return round(100 * errors / max(1, units), 2)

    skipped = sum(bool(row.get("skipped")) for row in rows)
    rows = [row for row in rows if not row.get("skipped")]
    audio = sum(row["seconds"] for row in rows)
    systems = ("offline", "streaming") if "streaming" in rows[0] else ("offline",)
    summary = {
        "clips": len(rows),
        "skipped_longer_than_max_seconds": skipped,
        "audio_hours": round(audio / 3600, 3),
        "wer": {name: rate(name, "wer") for name in systems},
        "cer": {name: rate(name, "cer") for name in systems},
        "wer_raw": {name: rate(name, "wer_raw") for name in systems},
        "rtf": {"offline": round(sum(row["offline_compute_s"] for row in rows) / audio, 4)},
    }
    if "streaming" not in systems:
        return summary
    first = [row["first_partial_s"] for row in rows if row["first_partial_s"] is not None]
    final = [row["final_lag_s"] for row in rows if row["final_lag_s"] is not None]
    offsets = [value for row in rows for value in row["word_offsets_s"]]
    updates = sum(row["partials"] for row in rows)
    compute = sum(row["stream_compute_s"] for row in rows)
    # Same basis as offline: compute per second of the clip. The stream also
    # computed through the silent tail; per fed second it costs less.
    summary["rtf"]["streaming"] = round(compute / audio, 4)
    summary["rtf"]["streaming_per_fed_second"] = round(
        compute / sum(row["stream_audio_s"] for row in rows), 4
    )
    summary |= {
        "streaming_equals_offline_clips": sum(row["same_text"] for row in rows),
        "multi_final_clips_pct": round(100 * sum(row["finals"] > 1 for row in rows) / len(rows), 1),
        "first_partial_s": {"p50": p(first, 50), "p95": p(first, 95)},
        "final_lag_s": {"p50": p(final, 50), "p95": p(final, 95)},
        "partial_updates": updates,
        "partial_revision_pct": round(
            100 * sum(row["revisions"] for row in rows) / max(1, updates), 2
        ),
        "word_start_offset_s": {
            "median": p(offsets, 50),
            "p05": p(offsets, 5),
            "p95": p(offsets, 95),
            "words": len(offsets),
        },
    }
    return summary


def report(run, rows):
    sets = sorted({row["set"] for row in rows})
    return run | {
        "sets": {name: summarize([r for r in rows if r["set"] == name]) for name in sets},
        "all": summarize(rows),
    }


def write_report(out, run, rows):
    result = report(run, rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result["all"], ensure_ascii=False))


def read_rows(path):
    lines = [json.loads(line) for line in path.open(encoding="utf-8")]
    return lines[0]["run"], lines[1:]


def from_rows(args):
    run, rows = read_rows(args.from_rows)
    if args.only_ids:
        keep = {(row["set"], row["id"]) for row in read_rows(args.only_ids)[1]}
        rows = [row for row in rows if (row["set"], row["id"]) in keep]
        run = run | {"subset": f"clips of {args.only_ids.name}"}
    write_report(args.out, run, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--data", type=Path)
    parser.add_argument("--model", default="v3_e2e_rnnt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--download-root", default=None)
    parser.add_argument("--chunk-ms", type=int, default=80)
    parser.add_argument("--tail-ms", type=int, default=2000)
    parser.add_argument("--endpoint-ms", type=int, default=600)
    parser.add_argument("--max-seconds", type=float, default=25.0, help="offline transcribe limit")
    parser.add_argument("--limit", type=int, default=0, help="clips per set (0 = all)")
    parser.add_argument("--offline-only", action="store_true", help="skip streaming")
    parser.add_argument("--note", default="", help="run conditions, e.g. a shared GPU")
    parser.add_argument("--from-rows", type=Path, help="rebuild the report from a rows file")
    parser.add_argument("--only-ids", type=Path, help="with --from-rows: clips of this rows file")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.from_rows:
        from_rows(args)
        return
    if args.data is None:
        parser.error("--data is required unless --from-rows is given")

    import soundfile as sf
    import torch

    items = [json.loads(line) for line in (args.data / "manifest.jsonl").open(encoding="utf-8")]
    rows = [
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

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    kwargs = {"download_root": args.download_root} if args.download_root else {}
    model = load_model(args.model, args.device, **kwargs)
    cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    weights = torch.cuda.memory_allocated() if cuda else 0
    config = RecognizerConfig(endpoint_ms=args.endpoint_ms)
    chunk, tail = args.chunk_ms * RATE // 1000, args.tail_ms * RATE // 1000

    # Warm up kernels on silence so the first clip is not charged for it.
    stream(model, np.zeros(RATE * 3, dtype=np.float32), chunk, 0, config)
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    clips = []
    for number, item in enumerate(items, 1):
        path = args.data / item["audio"]
        audio, rate = sf.read(path, dtype="float32")
        assert rate == RATE
        result, offline_s = offline(model, path)
        row = {
            "set": item["set"],
            "id": item["id"],
            "seconds": item["seconds"],
            "reference": item["text"],
            "offline": result.text,
            "offline_compute_s": offline_s,
        }
        if not args.offline_only:
            row |= streaming_row(model, audio, chunk, tail, config, list(result.words or []))
        clips.append(row)
        rows.append(public_row(row))
        if number % 50 == 0:
            print(f"{number}/{len(items)}", flush=True)

    run = {
        "model": args.model,
        "settings": {
            "chunk_ms": args.chunk_ms,
            "tail_ms": args.tail_ms,
            "endpoint_ms": args.endpoint_ms,
            "max_seconds": args.max_seconds,
            "stable": vars_of(config.stable),
            "preview": vars_of(config.preview),
        },
        "environment": {
            "gpu": torch.cuda.get_device_name() if cuda else None,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "gpu_memory_mb": {
            "weights": round(weights / 2**20, 1),
            "peak_total": round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None,
        },
        "note": args.note,
    }
    write_report(args.out, run, rows)
    write_jsonl(args.out.with_suffix(".rows.jsonl"), [{"run": run}, *rows])
    write_jsonl(args.out.with_suffix(".clips.jsonl"), clips)


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def streaming_row(model, audio, chunk, tail, config, offline_words):
    timeline = stream(model, audio, chunk, tail, config)
    events = list(paced(timeline))
    finals = [(wall, e) for wall, e in events if isinstance(e, Final)]
    words = [word for _, e in finals for word in e.words]
    first_partial = next((wall for wall, e in events if isinstance(e, Partial)), None)
    partial_words = [e.text.split() if isinstance(e, Partial) else None for _, e in events]
    return {
        "streaming": " ".join(e.text for _, e in finals),
        "stream_compute_s": sum(compute for _, compute, _ in timeline),
        "stream_audio_s": (audio.size + tail) / RATE,
        "first_partial_s": (
            first_partial - offline_words[0].start
            if first_partial is not None and offline_words
            else None
        ),
        "final_lag_s": finals[-1][0] - offline_words[-1].end if finals and offline_words else None,
        "partials": sum(words is not None for words in partial_words),
        "revisions": revisions(partial_words),
        "word_offsets_s": word_offsets(words, offline_words),
        "finals": len(finals),
    }


def vars_of(config):
    if config is None:
        return None
    return {
        "emit_ms": config.emit_ms,
        "left_context_ms": config.left_context_ms,
        "right_context_ms": config.right_context_ms,
    }


if __name__ == "__main__":
    main()
