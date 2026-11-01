"""Memory and compute of offline decoding, streaming, and streaming without preview.

``run_benchmark.py`` runs offline and streaming in one process, so its GPU
peak covers both. This script measures each mode on its own: the peak memory
(``torch.cuda.max_memory_allocated``; the CUDA context and allocator cache
come on top) is reset before every mode, and compute per audio second is
taken over the same clips. It also reports the parameter dtypes, since
``gigaam.load_model`` keeps the encoder in fp16 by default. ``--device cpu``
measures compute only.

Usage::

    python benchmarks/memory_check.py --data data --per-set 10 \
        --out benchmarks/results/v3_e2e_rnnt_memory.json
    python benchmarks/memory_check.py --data data --per-set 1 --device cpu \
        --out benchmarks/results/v3_e2e_rnnt_cpu.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from gigaam_v3_streaming import RecognizerConfig, StreamingRecognizer, load_model

RATE = 16_000
MODES = {
    "offline": None,
    "streaming": RecognizerConfig(),
    "streaming_without_preview": RecognizerConfig(preview=None),
}


def stream(model, audio, config, chunk):
    recognizer = StreamingRecognizer(model, config)
    for at in range(0, audio.size, chunk):
        recognizer.feed(audio[at : at + chunk])
    recognizer.finish()


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="v3_e2e_rnnt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--download-root", default=None)
    parser.add_argument("--per-set", type=int, default=10, help="first N clips of each set")
    parser.add_argument("--max-seconds", type=float, default=25.0, help="offline transcribe limit")
    parser.add_argument("--chunk-ms", type=int, default=80)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    items = [json.loads(line) for line in (args.data / "manifest.jsonl").open(encoding="utf-8")]
    clips = []
    for name in sorted({item["set"] for item in items}):
        fit = [i for i in items if i["set"] == name and i["seconds"] <= args.max_seconds]
        clips += fit[: args.per_set]

    model = load_model(args.model, args.device, download_root=args.download_root)
    cuda = args.device.startswith("cuda")
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    weights = torch.cuda.memory_allocated() if cuda else None
    chunk = RATE * args.chunk_ms // 1000
    # Warm up kernels so the first measured clip pays no one-off costs.
    for config in MODES.values():
        if config is not None:
            stream(model, np.zeros(3 * RATE, np.float32), config, chunk)
    model.transcribe(str(args.data / clips[0]["audio"]))

    modes = {}
    for name, config in MODES.items():
        sync()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        compute = seconds = 0.0
        for item in clips:
            path = args.data / item["audio"]
            audio, _ = sf.read(path, dtype="float32")
            began = time.perf_counter()
            if config is None:
                model.transcribe(str(path))
            else:
                stream(model, audio, config, chunk)
            sync()
            compute += time.perf_counter() - began
            seconds += audio.size / RATE
        modes[name] = {
            "peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None,
            "rtf": round(compute / seconds, 4),
        }

    result = {
        "model": args.model,
        "clips": len(clips),
        "audio_seconds": round(sum(i["seconds"] for i in clips), 1),
        "environment": {
            "gpu": torch.cuda.get_device_name() if cuda else None,
            "cpu_threads": None if cuda else torch.get_num_threads(),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "dtypes": {
            "encoder": sorted({str(p.dtype) for p in model.encoder.parameters()}),
            "head": sorted({str(p.dtype) for p in model.head.parameters()}),
        },
        "weights_mb": round(weights / 2**20, 1) if cuda else None,
        "modes": modes,
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
