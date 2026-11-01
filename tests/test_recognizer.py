from __future__ import annotations

import random

import numpy as np
import pytest
from fake_model import RATE, compose, fake_recognizer

from gigaam_v3_streaming import Final, Partial, RecognizerConfig

# "да"+"ста" (piece without a word boundary), a short pause inside the
# utterance, "нет", a long pause (endpoint), then "три" right at the end.
SPEECH = compose(
    (0.0, 200),
    (0.1, 400),
    (0.4, 200),
    (0.0, 280),
    (0.2, 400),
    (0.0, 1200),
    (0.3, 400),
)


def run(audio, plan, config=None):
    recognizer = fake_recognizer(config)
    events, at = [], 0
    for size in plan:
        events.extend(recognizer.feed(audio[at : at + size], sample_offset=at))
        at += size
    assert at == audio.size
    events.extend(recognizer.finish())
    return events


def finals(events):
    return [event for event in events if isinstance(event, Final)]


def random_plan(total, seed):
    rng, plan = random.Random(seed), []
    while sum(plan) < total:
        plan.append(min(rng.randint(1, 16_000), total - sum(plan)))
    return plan


def test_endpoint_closes_each_utterance_with_word_times():
    result = finals(run(SPEECH, [SPEECH.size]))
    assert [final.text for final in result] == ["даста нет", "три"]
    first = result[0]
    assert [word.text for word in first.words] == ["даста", "нет"]
    # Emission times sit on the 40 ms frame where each word begins.
    assert first.words[0].start == pytest.approx(0.200)
    assert first.words[1].start == pytest.approx(1.080)
    assert first.start == first.words[0].start and first.end == first.words[-1].end
    for final in result:
        for word in final.words:
            assert final.start <= word.start < word.end <= final.end
    assert result[1].words[0].start == pytest.approx(2.680)


def test_finals_do_not_depend_on_chunking():
    reference = finals(run(SPEECH, [SPEECH.size]))
    for plan in (
        [160] * (SPEECH.size // 160),
        [17] * (SPEECH.size // 17) + [SPEECH.size % 17],
        random_plan(SPEECH.size, 1),
        random_plan(SPEECH.size, 2),
    ):
        assert finals(run(SPEECH, plan)) == reference


def test_partials_lead_and_final_supersedes():
    events = run(SPEECH, [640] * (SPEECH.size // 640))
    first_partial = next(i for i, e in enumerate(events) if isinstance(e, Partial))
    first_final = next(i for i, e in enumerate(events) if isinstance(e, Final))
    assert first_partial < first_final
    shown = ""
    for event in events:
        if isinstance(event, Partial):
            assert event.text.startswith(event.stable_text)
            assert event.text != shown
            shown = event.text
        else:
            # The final replaces whatever partial was on screen.
            assert shown == "" or event.text.startswith(shown.split()[0])
            shown = ""


def test_partial_appears_before_stable_lookahead_ends():
    # Preview lane: first text well before the stable lane could settle it.
    audio = compose((0.0, 200), (0.1, 400), (0.0, 2000))
    recognizer = fake_recognizer()
    seen = None
    for at in range(0, audio.size, 640):
        if any(isinstance(e, Partial) for e in recognizer.feed(audio[at : at + 640])):
            seen = (at + 640) / RATE
            break
    assert seen is not None and seen <= 0.2 + 0.6


def test_finish_decodes_the_tail_with_lookahead():
    audio = compose((0.0, 200), (0.2, 120))
    result = finals(run(audio, [audio.size]))
    assert [final.text for final in result] == ["нет"]
    assert result[0].end <= audio.size / RATE


def test_punctuation_piece_joins_the_word_before_it():
    audio = compose((0.0, 200), (0.1, 200), (0.5, 80), (0.2, 200), (0.0, 1200))
    (final,) = finals(run(audio, [audio.size]))
    assert final.text == "да, нет"
    assert [word.text for word in final.words] == ["да,", "нет"]
    assert final.words[0].end == pytest.approx(0.440)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_late_piece_of_a_word_does_not_split_it(seed):
    # The model emits "ста" 1 s after "да": longer than the endpoint, but the
    # preview lane sees it before the stable clock would close the utterance.
    audio = compose((0.0, 200), (0.1, 200), (0.0, 800), (0.4, 200), (0.0, 1600))
    for plan in ([audio.size], random_plan(audio.size, seed)):
        result = finals(run(audio, plan))
        assert [final.text for final in result] == ["даста"]
        assert result[0].words[0].start == pytest.approx(0.200)


def test_late_punctuation_joins_the_utterance_instead_of_a_final_of_its_own():
    audio = compose((0.0, 200), (0.1, 200), (0.0, 800), (0.5, 80), (0.0, 1600))
    assert [final.text for final in finals(run(audio, [audio.size]))] == ["да,"]


def test_punctuation_after_the_utterance_closed_is_dropped():
    audio = compose((0.0, 200), (0.1, 200), (0.0, 2400), (0.5, 80), (0.0, 1600))
    assert [final.text for final in finals(run(audio, [audio.size]))] == ["да"]


def test_endpointing_off_gives_one_final_at_finish():
    result = finals(run(SPEECH, [SPEECH.size], RecognizerConfig(endpoint_ms=None)))
    assert [final.text for final in result] == ["даста нет три"]


def test_preview_off_still_streams_stable_partials():
    events = run(
        SPEECH,
        [1280] * (SPEECH.size // 1280) + [SPEECH.size % 1280],
        RecognizerConfig(preview=None),
    )
    assert [f.text for f in finals(events)] == ["даста нет", "три"]
    assert all(e.text == e.stable_text for e in events if isinstance(e, Partial))


def test_max_utterance_cuts_only_at_a_word_boundary():
    words = [(0.1, 200), (0.4, 200), (0.0, 80), (0.2, 200), (0.0, 80)] * 8
    audio = compose(*words)
    config = RecognizerConfig(endpoint_ms=None, max_utterance_ms=1000)
    result = finals(run(audio, [audio.size], config))
    assert len(result) > 1
    assert all(not final.text.startswith("ста") for final in result)
    assert " ".join(final.text for final in result) == " ".join(["даста нет"] * 8)


def test_silence_only_emits_nothing():
    audio = np.zeros(RATE * 3, dtype=np.float32)
    assert run(audio, [4000] * 12) == []


def test_input_formats_agree():
    pcm16 = (SPEECH * 32768 / 1.0).clip(-32768, 32767).astype(np.int16)
    as_float = finals(run(pcm16.astype(np.float32) / 32768, [SPEECH.size]))
    as_int = finals(run(pcm16, [SPEECH.size]))
    recognizer = fake_recognizer()
    as_bytes = finals(recognizer.feed(pcm16.astype("<i2").tobytes()) + recognizer.finish())
    assert as_float == as_int == as_bytes


def test_lifecycle_errors():
    recognizer = fake_recognizer()
    recognizer.feed(np.zeros(100, np.float32))
    with pytest.raises(ValueError, match="gap"):
        recognizer.feed(np.zeros(1, np.float32), sample_offset=200)
    with pytest.raises(ValueError, match="overlap"):
        recognizer.feed(np.zeros(1, np.float32), sample_offset=50)
    assert recognizer.finish() == []
    assert recognizer.finish() == []
    with pytest.raises(RuntimeError):
        recognizer.feed(np.zeros(1, np.float32))
    cancelled = fake_recognizer()
    cancelled.feed(SPEECH[:RATE])
    cancelled.cancel()
    assert cancelled.finish() == []


def test_tiny_stream_ends_cleanly():
    recognizer = fake_recognizer()
    recognizer.feed(np.zeros(10, np.float32))
    assert recognizer.finish() == []


def test_memory_stays_bounded_over_a_long_stream():
    pattern = compose((0.1, 400), (0.0, 800), (0.2, 400), (0.0, 800))
    recognizer = fake_recognizer()
    count = 0
    for _ in range(100):  # 4 minutes
        count += len(finals(recognizer.feed(pattern)))
        assert len(recognizer._utterance) <= 10 and len(recognizer._tail) <= 10
        stable = recognizer._stable
        assert stable.ring_occupancy_samples <= stable.stats.ring_capacity_samples
    assert count >= 199
