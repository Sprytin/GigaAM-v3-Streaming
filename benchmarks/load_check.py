"""Start-up cost of the CPU mode: time to the first result and peak memory, in fresh processes.

Three ways to bring up the same INT8 encoder, each in ``--repeat`` new
processes (medians reported, every run kept):

* ``prototype``: the whole stock model (``load_model(..., "cpu")``, PyTorch
  encoder included) plus ONNX Runtime optimizing the graph on every start,
  as ``run_cpu.py --encoder`` did before the CPU modes.
* ``lean``: only the frontend, RNN-T head and tokenizer from the
  memory-mapped checkpoint, graph still optimized on every start.
* ``cached``: ``load_engine("cpu")`` as shipped, the lean model plus the
  cached optimized graph.

Each child reports seconds spent importing, loading and on its first 10 s
window (encode and decode), and its peak and final resident memory
(``VmHWM``, ``VmRSS``). The files stay in the page cache between runs (no
root to drop it), so this is a warm-disk start, the usual case for a service
restart; the first build of the encoder is timed separately by ``prepare``.

    python benchmarks/load_check.py --cache-dir /tmp/cache --out benchmarks/results/cpu_load.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

VARIANTS = ("prototype", "lean", "cached")


def memory() -> dict:
    out = {}
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            key, _, value = line.partition(":")
            if key in ("VmRSS", "VmHWM"):
                out[key] = round(int(value.split()[0]) / 1024, 1)
    return out


def child(variant: str, name: str, cache: Path, threads: int) -> dict:
    started = time.perf_counter()
    import numpy as np
    import torch

    from gigaam_v3_streaming import cpu, load_engine
    from gigaam_v3_streaming.onnx_backend import OnnxEncoder

    imported = time.perf_counter()
    torch.set_num_threads(1)
    encoder = cpu.prepare(name, cache_dir=cache)
    if variant == "prototype":
        from gigaam_v3_streaming import load_model

        model = load_model(name, "cpu", fp16_encoder=False)
        backend = OnnxEncoder(model, encoder, threads)
    elif variant == "lean":
        model = cpu.CpuModel(*cpu.checkpoint_files(name, None))
        backend = OnnxEncoder(model, encoder, threads)
    else:
        engine = load_engine("cpu", name, threads=threads, cache_dir=cache)
        model, backend = engine.model, engine.backend
        assert backend.cached_graph is not None and backend.cached_graph.is_file()
    loaded = time.perf_counter()
    window = np.random.default_rng(0).normal(0, 0.1, 160_000).astype(np.float32)
    with torch.inference_mode():
        list(backend.new_decoder(10).decode(backend.encode(window)))
    first = time.perf_counter()
    return {
        "variant": variant,
        "import_s": round(imported - started, 3),
        "load_s": round(loaded - imported, 3),
        "first_window_s": round(first - loaded, 3),
        "ready_s": round(first - started, 3),
        "peak_rss_mb": memory()["VmHWM"],
        "rss_mb": memory()["VmRSS"],
    }


def run_child(variant: str, args) -> dict:
    command = [sys.executable, __file__, "--child", variant, "--model", args.model]
    command += ["--cache-dir", str(args.cache_dir), "--threads", str(args.threads)]
    started = time.perf_counter()
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    row = json.loads(result.stdout.strip().splitlines()[-1])
    row["process_s"] = round(time.perf_counter() - started, 3)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", default="v3_e2e_rnnt")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--child", choices=VARIANTS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.child:
        print(json.dumps(child(args.child, args.model, args.cache_dir, args.threads)))
        return

    import onnxruntime
    import torch

    from gigaam_v3_streaming import cpu
    from gigaam_v3_streaming.onnx_backend import cpu_isa

    started = time.perf_counter()
    encoder = cpu.prepare(args.model, cache_dir=args.cache_dir)
    build_s = time.perf_counter() - started
    variants = args.variants.split(",")
    if "cached" in variants:
        run_child("cached", args)  # makes sure the optimized graph is in the cache
    runs = {variant: [] for variant in variants}
    for _ in range(args.repeat):  # interleaved, so a slow minute hits every variant alike
        for variant in variants:
            runs[variant].append(run_child(variant, args))
    keys = ("process_s", "ready_s", "import_s", "load_s", "first_window_s", "peak_rss_mb", "rss_mb")
    report = {
        "model": args.model,
        "encoder": encoder.name,
        "threads": args.threads,
        "repeat": args.repeat,
        "prepare_s": round(build_s, 1),
        "environment": {
            "cpu_isa": cpu_isa(),
            "machine": platform.machine(),
            "onnxruntime": onnxruntime.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "median": {
            variant: {key: statistics.median(row[key] for row in rows) for key in keys}
            for variant, rows in runs.items()
        },
        "runs": runs,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
