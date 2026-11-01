"""Stream a WAV file through the recognizer as if it came from a microphone.

    python examples/transcribe_file.py speech.wav [--realtime]

Prints partials on one updating line and each final on its own line with
word times. Any sample rate and channel count is accepted; the audio is mixed
to mono and resampled to 16 kHz.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import soundfile as sf

from gigaam_v3_streaming import Final, StreamingRecognizer

RATE = 16_000
CHUNK = RATE // 10  # 100 ms, a typical microphone buffer


def load(path: str) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != RATE:
        import torch
        import torchaudio

        audio = torchaudio.functional.resample(torch.from_numpy(audio), rate, RATE).numpy()
    return audio


def show(event, width: int) -> None:
    if isinstance(event, Final):
        words = " ".join(f"{w.text}[{w.start:.2f}-{w.end:.2f}]" for w in event.words)
        clear = f"\r{' ' * width}\r"
        sys.stdout.write(f"{clear}{event.start:7.2f}s  {event.text}\n           {words}\n")
    else:
        sys.stdout.write(f"\r{' ' * width}\r      ...  {event.text[-width + 12 :]}")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("wav")
    parser.add_argument("--model", default="v3_e2e_rnnt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--realtime", action="store_true", help="feed at 1x speed")
    args = parser.parse_args()

    audio = load(args.wav)
    recognizer = StreamingRecognizer.from_pretrained(args.model, args.device)
    width = 100
    began = time.monotonic()
    for at in range(0, audio.size, CHUNK):
        if args.realtime:
            time.sleep(max(0.0, began + at / RATE - time.monotonic()))
        for event in recognizer.feed(audio[at : at + CHUNK]):
            show(event, width)
    for event in recognizer.finish():
        show(event, width)
    print()


if __name__ == "__main__":
    main()
