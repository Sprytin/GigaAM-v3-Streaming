"""Download seeded subsets of public Russian test sets and write a manifest.

Nothing downloaded here is redistributed by this repository; the audio stays in
the directory you pass as ``--out``.

Sets:

* ``golos_crowd`` / ``golos_farfield`` -- SberDevices Golos test splits, read
  from the ``bond005/sberdevices_golos_*`` parquet mirrors on the Hugging Face
  Hub (Golos license: see https://github.com/salute-developers/golos).
* ``fleurs_ru`` -- Google FLEURS ``ru_ru`` test split (CC BY 4.0).

Each output line of ``manifest.jsonl`` is
``{"set", "id", "audio", "text", "seconds"}`` with 16 kHz mono PCM16 WAV audio.

Usage::

    python benchmarks/prepare_data.py --out data --per-set 300 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

HUB = "https://huggingface.co/datasets"
GOLOS = {
    "golos_crowd": (
        "bond005/sberdevices_golos_10h_crowd",
        [
            "data/test-00000-of-00003-cad0cfaddbc8fa71.parquet",
            "data/test-00001-of-00003-9d9aa103f1ceb4c7.parquet",
            "data/test-00002-of-00003-91f7e0c1d8f0af36.parquet",
        ],
    ),
    "golos_farfield": (
        "bond005/sberdevices_golos_100h_farfield",
        ["data/test-00000-of-00001-7fb58d4cda8ff9fe.parquet"],
    ),
}
FLEURS = "google/fleurs"
RATE = 16_000


def fetch(url: str, path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    print(f"downloading {url}", flush=True)
    with urllib.request.urlopen(url) as response, partial.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    partial.rename(path)
    return path


def to_16k_mono(audio: np.ndarray, rate: int) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != RATE:
        # Linear resampling is enough for the 16 kHz sources used here; all
        # three sets are already 16 kHz, so this branch is a guard.
        positions = np.arange(0, len(audio), rate / RATE)
        audio = np.interp(positions, np.arange(len(audio)), audio)
    return audio.astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, RATE, subtype="PCM_16")
    return len(audio) / RATE


def golos(name: str, out: Path, count: int, seed: int) -> list[dict]:
    import pyarrow.parquet as pq

    repo, files = GOLOS[name]
    rows = []
    for file in files:
        local = fetch(f"{HUB}/{repo}/resolve/main/{file}", out / "raw" / name / Path(file).name)
        table = pq.read_table(local)
        rows.extend(table.to_pylist())
    picked = random.Random(seed).sample(range(len(rows)), min(count, len(rows)))
    items = []
    for index in sorted(picked):
        row = rows[index]
        audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        ident = f"{name}-{index:05d}"
        path = out / name / f"{ident}.wav"
        seconds = write_wav(path, to_16k_mono(audio, rate))
        items.append(
            {
                "set": name,
                "id": ident,
                "audio": str(path.relative_to(out)),
                "text": row["transcription"],
                "seconds": round(seconds, 3),
            }
        )
    return items


def fleurs(out: Path, count: int, seed: int) -> list[dict]:
    tsv = fetch(
        f"{HUB}/{FLEURS}/resolve/main/data/ru_ru/test.tsv", out / "raw" / "fleurs_ru" / "test.tsv"
    )
    with tsv.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE))
    # Columns: id, file name, raw transcription, normalized transcription, ...
    by_file = {row[1]: row for row in rows}
    picked = set(random.Random(seed).sample(sorted(by_file), min(count, len(by_file))))
    archive = fetch(
        f"{HUB}/{FLEURS}/resolve/main/data/ru_ru/audio/test.tar.gz",
        out / "raw" / "fleurs_ru" / "test.tar.gz",
    )
    items = []
    with tarfile.open(archive) as tar:
        for member in tar:
            name = Path(member.name).name
            if name not in picked or not member.isfile():
                continue
            audio, rate = sf.read(io.BytesIO(tar.extractfile(member).read()), dtype="float32")
            ident = f"fleurs_ru-{Path(name).stem}"
            path = out / "fleurs_ru" / f"{ident}.wav"
            seconds = write_wav(path, to_16k_mono(audio, rate))
            items.append(
                {
                    "set": "fleurs_ru",
                    "id": ident,
                    "audio": str(path.relative_to(out)),
                    "text": by_file[name][2],
                    "seconds": round(seconds, 3),
                }
            )
    return sorted(items, key=lambda item: item["id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-set", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sets", default="golos_crowd,golos_farfield,fleurs_ru")
    args = parser.parse_args()
    items = []
    for name in args.sets.split(","):
        items.extend(
            fleurs(args.out, args.per_set, args.seed)
            if name == "fleurs_ru"
            else golos(name, args.out, args.per_set, args.seed)
        )
    manifest = args.out / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    hours = sum(item["seconds"] for item in items) / 3600
    print(f"wrote {len(items)} items, {hours:.2f} h, to {manifest}")


if __name__ == "__main__":
    main()
