"""gigastt on the clips, normalization and metrics of ``run_benchmark.py``.

`gigastt <https://github.com/ekhodzitsky/gigastt>`_ is another streaming
server for GigaAM v3 (Rust, ONNX Runtime, INT8). This script measures a
running ``gigastt serve`` from the outside, through its public interfaces,
on the clips of a ``run_benchmark.py`` rows file:

* offline: ``POST /v1/transcribe`` with the WAV file;
* streaming: ``GET /v1/ws`` as its docs describe a client (wait for
  ``ready``, ``configure`` 16 kHz, PCM16 frames, ``stop``, keep every
  ``final`` until the socket ends). The input is the clip followed by
  ``--tail-ms`` of silence, as in ``run_benchmark.py``, so a final either
  comes from gigastt's own endpointing or from ``stop`` at the end.

``--pace fast`` sends frames as fast as the server takes them: WER/CER and
the wall time of the session, but no latency. ``--pace realtime`` sends each
frame when its audio has been "recorded" (frame k at (k+1) x chunk after
audio time 0) and measures latency on the same basis as ``run_benchmark.py``:
first non-empty partial after the offline GigaAM start of the first word,
and last non-empty final after the offline end of the last word. Those
offline word times come from ``--anchors``, written once with
``--write-anchors`` by the stock PyTorch model (needs CUDA and gigaam).

With ``--server-pid`` the server's CPU time (user + system, all threads) is
recorded per clip and its peak resident memory (``VmHWM``) at the end.

The rows file and report have the same format as ``run_benchmark.py``;
``word_offsets_s`` compares gigastt's streaming word starts with its own
offline ones. Transcripts go to the git-ignored ``<out>.clips.jsonl``, one
line per finished clip; a rerun with the same ``--out`` resumes after them.

Usage (the full set of runs is in docs/benchmarks.md)::

    python benchmarks/run_gigastt.py --write-anchors --data data \\
        --out benchmarks/results/v3_e2e_rnnt_anchors.jsonl
    gigastt serve --offline --model-variant e2e_rnnt --model-dir <models> \\
        --pool-size 1 --encoder-intra-threads 3 --port 9876 &
    python benchmarks/run_gigastt.py --data data --pace realtime --server-pid $! \\
        --anchors benchmarks/results/v3_e2e_rnnt_anchors.jsonl \\
        --out benchmarks/results/gigastt_e2e_rnnt_realtime.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from run_benchmark import public_row, read_rows, revisions, word_offsets, write_jsonl, write_report

RATE = 16_000
HERE = Path(__file__).resolve().parent


def clip_list(data, clips_rows):
    """Manifest items of the clips in ``clips_rows``, plus its skipped rows."""
    _, rows = read_rows(clips_rows)
    keep = {(row["set"], row["id"]) for row in rows if not row.get("skipped")}
    items = [json.loads(line) for line in (data / "manifest.jsonl").open(encoding="utf-8")]
    items = [item for item in items if (item["set"], item["id"]) in keep]
    return items, [row for row in rows if row.get("skipped")]


def write_anchors(args):
    import torch

    from gigaam_v3_streaming import load_model

    items, _ = clip_list(args.data, args.clips)
    model = load_model(args.model, args.device)
    with args.out.open("w", encoding="utf-8") as handle:
        for item in items:
            result = model.transcribe(str(args.data / item["audio"]), word_timestamps=True)
            words = list(result.words or [])
            anchor = {"set": item["set"], "id": item["id"]}
            if words:
                anchor |= {
                    "first_start": round(words[0].start, 4),
                    "last_end": round(words[-1].end, 4),
                }
            handle.write(json.dumps(anchor) + "\n")
    print(f"{len(items)} anchors, torch {torch.__version__}")


def cpu_seconds(pid):
    if pid is None:
        return None
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def peak_rss_mb(pid):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return round(int(line.split()[1]) / 1024, 1)
    return None


def pcm16(audio):
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def words_of(items):
    return [SimpleNamespace(text=w["word"], start=w["start"]) for w in items or []]


async def offline(http, url, path):
    began = time.perf_counter()
    async with http.post(
        f"{url}/v1/transcribe",
        data=path.read_bytes(),
        headers={"Content-Type": "application/octet-stream"},
    ) as response:
        response.raise_for_status()
        body = await response.json()
    return body, time.perf_counter() - began


async def stream(http, url, audio, chunk, tail, realtime, timeout):
    """One WebSocket session; returns events as (audio_seconds, message)."""
    data = pcm16(np.concatenate([audio, np.zeros(tail, dtype=np.float32)]))
    step = 2 * chunk
    events = []
    async with http.ws_connect(f"{url}/v1/ws", max_msg_size=0) as ws:
        ready = await ws.receive_json(timeout=30)
        assert ready["type"] == "ready", ready
        await ws.send_json({"type": "configure", "sample_rate": RATE})
        origin = time.perf_counter()

        async def read():
            async for message in ws:
                if message.type.name != "TEXT":
                    continue
                events.append((time.perf_counter() - origin, json.loads(message.data)))

        reader = asyncio.create_task(read())
        for k, at in enumerate(range(0, len(data), step)):
            if realtime:
                due = origin + (k + 1) * chunk / RATE
                await asyncio.sleep(max(0.0, due - time.perf_counter()))
            await ws.send_bytes(data[at : at + step])
        if realtime:
            await asyncio.sleep(max(0.0, origin + len(data) / 2 / RATE - time.perf_counter()))
        await ws.send_json({"type": "stop"})
        sent = time.perf_counter() - origin
        await asyncio.wait_for(reader, timeout)
    return events, sent, time.perf_counter() - origin


def streaming_row(events, wall, audio_s, anchor, offline_words, realtime):
    finals = [(t, m) for t, m in events if m["type"] == "final" and m.get("text", "").strip()]
    partials = [(t, m) for t, m in events if m["type"] == "partial" and m.get("text", "").strip()]
    errors = [m for _, m in events if m["type"] == "error"]
    words = [w for _, m in finals for w in words_of(m.get("words"))]
    shown = []
    for _, m in events:
        if m["type"] == "partial" and m.get("text", "").strip():
            shown.append(m["text"].split())
        elif m["type"] == "final":
            shown.append(None)
    first = partials[0][0] if partials else None
    last = finals[-1][0] if finals else None
    timed = realtime and anchor and "first_start" in anchor
    return {
        "streaming": " ".join(m["text"].strip() for _, m in finals),
        "stream_compute_s": wall,
        "stream_audio_s": audio_s,
        "first_partial_s": first - anchor["first_start"] if timed and first is not None else None,
        "final_lag_s": last - anchor["last_end"] if timed and last is not None else None,
        "partials": len(partials),
        "revisions": revisions(shown),
        "word_offsets_s": word_offsets(words, offline_words),
        "finals": len(finals),
        "final_by_stop": bool(finals) and finals[-1][1].get("endpoint_reason") == "stop",
        "errors": len(errors),
    }


async def run(args):
    import aiohttp
    import soundfile as sf

    items, skipped = clip_list(args.data, args.clips)
    if args.limit:
        items = [
            item
            for name in sorted({item["set"] for item in items})
            for item in [i for i in items if i["set"] == name][: args.limit]
        ]
    anchors = {}
    if args.anchors:
        for line in args.anchors.open(encoding="utf-8"):
            anchor = json.loads(line)
            anchors[(anchor["set"], anchor["id"])] = anchor
    realtime = args.pace == "realtime"
    chunk, tail = args.chunk_ms * RATE // 1000, args.tail_ms * RATE // 1000
    rows = [dict(row) for row in skipped]
    # Clips are appended as they finish, so an interrupted run resumes where it stopped.
    progress = args.out.with_suffix(".clips.jsonl")
    done = []
    if progress.exists():
        done = [json.loads(line) for line in progress.open(encoding="utf-8")]
    rows += [extended_row(row) for row in done]
    finished = {(row["set"], row["id"]) for row in done}
    todo = [item for item in items if (item["set"], item["id"]) not in finished]
    progress.parent.mkdir(parents=True, exist_ok=True)
    log = progress.open("a", encoding="utf-8")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as http:
        async with http.get(f"{args.url}/v1/models") as response:
            server = await response.json()
        # Warm up both paths so the first clip is not charged for it.
        first = args.data / items[0]["audio"]
        await offline(http, args.url, first)
        await stream(http, args.url, np.zeros(RATE * 3, np.float32), chunk, 0, False, 60)
        for number, item in enumerate(todo, len(done) + 1):
            path = args.data / item["audio"]
            audio, rate = sf.read(path, dtype="float32")
            assert rate == RATE
            cpu = cpu_seconds(args.server_pid)
            body, offline_s = await offline(http, args.url, path)
            cpu_mid = cpu_seconds(args.server_pid)
            events, _, wall = await stream(
                http, args.url, audio, chunk, tail, realtime, args.stop_timeout
            )
            cpu_end = cpu_seconds(args.server_pid)
            row = {
                "set": item["set"],
                "id": item["id"],
                "seconds": item["seconds"],
                "reference": item["text"],
                "offline": body.get("text", ""),
                "offline_compute_s": offline_s,
            }
            row |= streaming_row(
                events,
                wall,
                (audio.size + tail) / RATE,
                anchors.get((item["set"], item["id"])),
                words_of(body.get("words")),
                realtime,
            )
            if cpu is not None:
                row |= {"offline_cpu_s": cpu_mid - cpu, "stream_cpu_s": cpu_end - cpu_mid}
            clip = row | {"events": [[round(t, 3), m] for t, m in events]}
            log.write(json.dumps(clip, ensure_ascii=False) + "\n")
            log.flush()
            rows.append(extended_row(row))
            if number % 50 == 0:
                print(f"{number}/{len(items)}", flush=True)
    run = {
        "system": "gigastt",
        "server": {
            key: server.get(key)
            for key in ("id", "version", "variant", "encoder", "execution_provider", "pool_size")
        },
        "settings": {
            "pace": args.pace,
            "chunk_ms": args.chunk_ms,
            "tail_ms": args.tail_ms,
            "clips": args.clips.name,
            "anchors": args.anchors.name if args.anchors else None,
            "serve_flags": args.serve_flags,
        },
        "compute": (
            "stream_compute_s is the session's wall time "
            + ("at real-time pace" if realtime else "with frames sent as fast as accepted")
            + "; offline_compute_s is the wall time of POST /v1/transcribe"
        ),
        "environment": {"cpu": cpu_model(), "python": platform.python_version()},
        "server_peak_rss_mb": peak_rss_mb(args.server_pid) if args.server_pid else None,
        "resumed_after_clips": len(done) or None,
        "note": args.note,
    }
    log.close()
    write_report(args.out, run, rows)
    cpu_report(args.out, rows)
    write_jsonl(args.out.with_suffix(".rows.jsonl"), [{"run": run}, *rows])


def extended_row(row):
    """The public row plus gigastt's endpoint, error and server CPU fields."""
    extra = ("final_by_stop", "errors", "offline_cpu_s", "stream_cpu_s")
    return public_row(row) | {
        key: round(row[key], 4) if isinstance(row[key], float) else row[key]
        for key in extra
        if key in row
    }


def cpu_report(out, rows):
    """Add server CPU seconds per audio second next to the wall-time figures."""
    rows = [row for row in rows if "stream_cpu_s" in row]
    if not rows:
        return
    report = json.loads(out.read_text(encoding="utf-8"))
    for name in [*report["sets"], "all"]:
        part = [row for row in rows if name == "all" or row["set"] == name]
        audio = sum(row["seconds"] for row in part)
        target = report["all"] if name == "all" else report["sets"][name]
        target["server_cpu_s_per_audio_s"] = {
            "offline": round(sum(row["offline_cpu_s"] for row in part) / audio, 4),
            "streaming": round(sum(row["stream_cpu_s"] for row in part) / audio, 4),
        }
        target["final_by_stop_clips"] = sum(row["final_by_stop"] for row in part)
        target["error_messages"] = sum(row["errors"] for row in part)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--clips",
        type=Path,
        default=HERE / "results" / "v3_e2e_rnnt.rows.jsonl",
        help="rows file whose clips (and skipped clips) to use",
    )
    parser.add_argument("--url", default="http://127.0.0.1:9876")
    parser.add_argument("--pace", choices=("fast", "realtime"), default="fast")
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--tail-ms", type=int, default=2000)
    parser.add_argument("--anchors", type=Path, help="offline word times, see --write-anchors")
    parser.add_argument("--server-pid", type=int, help="record the server's CPU time and memory")
    parser.add_argument("--serve-flags", default="", help="the serve command line, for the record")
    parser.add_argument("--stop-timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=0, help="clips per set (0 = all)")
    parser.add_argument("--note", default="", help="run conditions")
    parser.add_argument("--write-anchors", action="store_true", help="write offline word times")
    parser.add_argument("--model", default="v3_e2e_rnnt", help="with --write-anchors")
    parser.add_argument("--device", default="cuda", help="with --write-anchors")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.write_anchors:
        write_anchors(args)
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
