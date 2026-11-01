"""Checks against a real GigaAM checkpoint.

Skipped unless ``gigaam`` is installed and ``GIGAAM_TEST_WAV`` names a 16 kHz
mono WAV with speech (no audio ships with this repository; any public clip,
for example one written by ``benchmarks/prepare_data.py``, will do).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from gigaam_v3_streaming import Final, StreamingRecognizer, load_model
from gigaam_v3_streaming.metrics import char_error_counts, normalize_text

gigaam = pytest.importorskip("gigaam")
sf = pytest.importorskip("soundfile")
WAV = os.environ.get("GIGAAM_TEST_WAV")
pytestmark = pytest.mark.skipif(not WAV, reason="set GIGAAM_TEST_WAV to a speech WAV")


@pytest.fixture(scope="module")
def model():
    return load_model(os.environ.get("GIGAAM_TEST_MODEL", "v3_e2e_rnnt"))


@pytest.fixture(scope="module")
def audio():
    samples, rate = sf.read(WAV, dtype="float32")
    assert rate == 16_000 and samples.ndim == 1
    return samples


def stream(model, audio, chunk):
    recognizer = StreamingRecognizer(model)
    events = []
    for at in range(0, audio.size, chunk):
        events.extend(recognizer.feed(audio[at : at + chunk]))
    events.extend(recognizer.finish())
    return [event for event in events if isinstance(event, Final)]


def test_streaming_matches_offline(model, audio):
    finals = stream(model, audio, 1600)
    offline = model.transcribe(WAV).text
    streamed = " ".join(final.text for final in finals)
    errors = char_error_counts(normalize_text(offline), normalize_text(streamed))
    assert errors.rate <= 0.05, (offline, streamed)
    end = audio.size / 16_000
    for final in finals:
        assert 0 <= final.start <= final.end <= end


def test_finals_do_not_depend_on_chunking(model, audio):
    assert stream(model, audio, 1600) == stream(model, audio, 777)


def test_silence_gives_no_text(model):
    assert stream(model, np.zeros(16_000 * 3, dtype=np.float32), 1600) == []
