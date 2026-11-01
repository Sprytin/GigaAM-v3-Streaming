from __future__ import annotations

import numpy as np
import pytest

from gigaam_v3_streaming import StreamingConfig, StreamingSession


class _FakeDecoder:
    def decode(self, frames):
        # The fixed central values make event equality sensitive to duplicate or
        # missing frame scheduling, independent of packet fragmentation. Each
        # token sits on the window's first central frame.
        return [(1 + (int(abs(value) * 1000) % 2), 0) for value in frames.ravel()[::2]]


class _FakeModel:
    """Frontend geometry plus a backend returning one frame per 40-ms encoder step."""

    sample_rate = 16_000
    feature_hop_samples = 160
    subsampling_factor = 4

    def __init__(self) -> None:
        self.windows: list[np.ndarray] = []

    def encode(self, window):
        self.windows.append(window.copy())
        padded = np.zeros(-(-window.size // 640) * 640, dtype=np.float32)
        padded[: window.size] = window
        return padded.reshape(1, 1, -1, 640).mean(axis=-1)

    def new_decoder(self, max_symbols_per_step):
        return _FakeDecoder()


def _session(config):
    model = _FakeModel()
    return model, StreamingSession(model, config, backend=model)


def _run_plan(pieces: list[int]):
    config = StreamingConfig(emit_ms=80, left_context_ms=160, right_context_ms=80)
    model, session = _session(config)
    pcm = np.arange(16_000, dtype=np.float32) / 16_000
    at = 0
    events = []
    occupancy = []
    for size in pieces:
        events.extend(session.push(pcm[at : at + size], sample_offset=at))
        at += size
        occupancy.append(session.ring_occupancy_samples)
    assert at == pcm.size
    events.extend(session.finish())
    return events, model.windows, session.stats, occupancy


def test_fragment_plans_have_identical_logical_windows_and_events() -> None:
    one = _run_plan([16_000])
    many = _run_plan([17] * 941 + [3])

    one_events, one_windows, one_stats, _ = one
    many_events, many_windows, many_stats, _ = many
    assert one_events == many_events
    assert [window.tobytes() for window in one_windows] == [
        window.tobytes() for window in many_windows
    ]
    assert one_stats.logical_windows == many_stats.logical_windows
    assert one_stats.peak_ring_samples <= one_stats.ring_capacity_samples
    assert many_stats.peak_ring_samples <= many_stats.ring_capacity_samples


def test_scheduler_rejects_gaps_and_overlaps_and_clears_on_finish() -> None:
    config = StreamingConfig(emit_ms=80, left_context_ms=160, right_context_ms=80)
    _, session = _session(config)
    session.push(np.zeros(100, dtype=np.float32))
    with pytest.raises(ValueError, match="gap"):
        session.push(np.zeros(1, dtype=np.float32), sample_offset=101)
    with pytest.raises(ValueError, match="overlap"):
        session.push(np.zeros(1, dtype=np.float32), sample_offset=99)
    events = session.finish()
    assert events[-1].is_final
    assert session.ring_occupancy_samples == 0
    assert session.stats.finished


def test_s16le_input_is_normalized() -> None:
    config = StreamingConfig(emit_ms=80, left_context_ms=160, right_context_ms=80)
    _, session = _session(config)
    session.push(np.array([0, 32767], dtype="<i2").tobytes())
    # Not enough data to run yet, but byte parsing and timeline accounting are.
    assert session.stats.input_samples == 2


def test_config_rejects_non_encoder_aligned_schedule() -> None:
    model = _FakeModel()
    config = StreamingConfig(emit_ms=100, left_context_ms=160, right_context_ms=80)
    with pytest.raises(ValueError, match="encoder hop"):
        StreamingSession(model, config, backend=model)


def test_token_samples_follow_the_emitting_frame() -> None:
    config = StreamingConfig(emit_ms=80, left_context_ms=160, right_context_ms=80)
    events, _, _, _ = _run_plan([16_000])
    for event in events:
        if event.token_ids:
            assert len(event.token_samples) == len(event.token_ids)
            # The fake decoder puts every token on the window's first frame.
            assert all(sample == event.start_sample for sample in event.token_samples)
    assert config.emit_samples == 1280


def test_processed_samples_is_the_settled_clock() -> None:
    config = StreamingConfig(emit_ms=80, left_context_ms=160, right_context_ms=80)
    _, session = _session(config)
    session.push(np.zeros(config.emit_samples + config.right_context_samples - 1, np.float32))
    assert session.processed_samples == 0
    session.push(np.zeros(1, np.float32))
    assert session.processed_samples == config.emit_samples
