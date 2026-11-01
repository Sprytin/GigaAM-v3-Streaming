"""A deterministic stand-in for a GigaAM RNN-T model.

Audio is a sequence of constant levels: silence is 0, and each "word piece" is
a level from ``LEVELS``. The fake encoder averages each 40 ms encoder frame and
the fake decoder emits a piece on the first frame of each new non-zero level,
so every token has a known emission time.
"""

from __future__ import annotations

import numpy as np

from gigaam_v3_streaming import RecognizerConfig, StreamingRecognizer

RATE = 16_000
HOP = 640  # 160-sample feature hop x4 subsampling, as in stock GigaAM.
LEVELS = {0.1: 1, 0.2: 2, 0.3: 3, 0.4: 4, 0.5: 5}
PIECES = {1: "▁да", 2: "▁нет", 3: "▁три", 4: "ста", 5: "▁,"}


class FakeTokenizer:
    charwise = False

    def id_to_str(self, token: int) -> str:
        return PIECES[token]


class FakeDecoder:
    def __init__(self) -> None:
        self.previous = 0.0

    def decode(self, frames):
        tokens = []
        for index, value in enumerate(np.round(np.asarray(frames, np.float64).ravel(), 3)):
            level = float(value)
            if level and level != self.previous and level in LEVELS:
                tokens.append((LEVELS[level], index))
            self.previous = level
        return tokens


class FakeModel:
    """Frontend geometry and tokenizer of a GigaAM model, and a fake backend for it."""

    sample_rate = RATE
    feature_hop_samples = 160
    subsampling_factor = 4
    tokenizer = FakeTokenizer()

    def __init__(self) -> None:
        self.windows = 0

    def encode(self, window):
        self.windows += 1
        padded = np.zeros(-(-window.size // HOP) * HOP, dtype=np.float32)
        padded[: window.size] = window
        return padded.reshape(1, 1, -1, HOP).mean(axis=-1)

    def new_decoder(self, max_symbols_per_step):
        return FakeDecoder()


def fake_recognizer(config: RecognizerConfig | None = None) -> StreamingRecognizer:
    model = FakeModel()
    return StreamingRecognizer(model, config, backend=model)


def compose(*parts: tuple[float, int]) -> np.ndarray:
    """Build audio from ``(level, milliseconds)`` parts; ms must be 40 ms multiples."""
    chunks = []
    for level, ms in parts:
        assert ms % 40 == 0
        chunks.append(np.full(ms * RATE // 1000, level, dtype=np.float32))
    return np.concatenate(chunks)
