"""Small, dependency-free scoring primitives used by the benchmark.

The benchmark deliberately keeps these functions separate from torch and the
streaming engine.  That makes the definitions of WER, CER, and event latency
easy to unit-test and prevents a model implementation from quietly changing
how it is measured.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


def normalize_text(text: str) -> str:
    """Return the lexical normalization used for the ``normalized`` scores.

    This mirrors GigaAM's public ``normalize_raw_text`` behaviour without
    importing GigaAM: lowercase, ``ё`` -> ``е``, punctuation removal, and
    whitespace collapse.  Dash-like punctuation is a word separator.
    """

    text = str(text).replace("ё", "е").replace("Ё", "Е").lower()
    for quote in ("’", "‘", "ʻ", "ʼ"):
        text = text.replace(quote, "'")
    output: list[str] = []
    for index, char in enumerate(text):
        if char == "'":
            if (
                0 < index < len(text) - 1
                and text[index - 1].isalnum()
                and text[index + 1].isalnum()
            ):
                output.append(char)
        elif char.isalnum() or char.isspace():
            output.append(char)
        elif unicodedata.category(char) == "Pd":
            output.append(" ")
    return " ".join("".join(output).split())


@dataclass(frozen=True)
class ErrorCounts:
    """Levenshtein counts, with ``reference_units`` as the denominator."""

    errors: int = 0
    reference_units: int = 0

    @property
    def rate(self) -> float:
        return self.errors / self.reference_units if self.reference_units else 0.0

    def __add__(self, other: ErrorCounts) -> ErrorCounts:
        if not isinstance(other, ErrorCounts):
            return NotImplemented
        return ErrorCounts(
            errors=self.errors + other.errors,
            reference_units=self.reference_units + other.reference_units,
        )

    def as_dict(self) -> dict[str, int | float]:
        return {
            "errors": self.errors,
            "reference_units": self.reference_units,
            "rate": self.rate,
        }


def levenshtein_distance(reference: Sequence[T], hypothesis: Sequence[T]) -> int:
    """Return a memory-bounded Levenshtein distance.

    Only one dynamic-programming row is retained; this matters for long
    concatenated streaming transcripts.
    """

    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_unit in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_unit in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[hyp_index] + 1,
                    current[hyp_index - 1] + 1,
                    previous[hyp_index - 1] + (ref_unit != hyp_unit),
                )
            )
        previous = current
    return previous[-1]


def word_error_counts(reference: str, hypothesis: str) -> ErrorCounts:
    """Word-level edit counts; callers choose raw or normalized input."""

    reference_words = str(reference).split()
    hypothesis_words = str(hypothesis).split()
    return ErrorCounts(
        levenshtein_distance(reference_words, hypothesis_words), len(reference_words)
    )


def char_error_counts(reference: str, hypothesis: str) -> ErrorCounts:
    """Character-level edit counts, ignoring spaces as customary for CER."""

    reference_chars = list("".join(str(reference).split()))
    hypothesis_chars = list("".join(str(hypothesis).split()))
    return ErrorCounts(
        levenshtein_distance(reference_chars, hypothesis_chars), len(reference_chars)
    )


def transcript_scores(reference: str, hypothesis: str) -> dict[str, dict[str, int | float]]:
    """Return both literal (E2E/raw) and lexical-normalized WER/CER counts."""

    raw_wer = word_error_counts(reference, hypothesis)
    raw_cer = char_error_counts(reference, hypothesis)
    normalized_reference = normalize_text(reference)
    normalized_hypothesis = normalize_text(hypothesis)
    normalized_wer = word_error_counts(normalized_reference, normalized_hypothesis)
    normalized_cer = char_error_counts(normalized_reference, normalized_hypothesis)
    return {
        "raw": {"wer": raw_wer.as_dict(), "cer": raw_cer.as_dict()},
        "normalized": {
            "wer": normalized_wer.as_dict(),
            "cer": normalized_cer.as_dict(),
        },
    }


def aggregate_transcript_scores(
    pairs: Iterable[tuple[str, str]],
) -> dict[str, dict[str, int | float]]:
    """Compute corpus micro scores (sum errors / sum reference units)."""

    totals = {
        "raw": {"wer": ErrorCounts(), "cer": ErrorCounts()},
        "normalized": {"wer": ErrorCounts(), "cer": ErrorCounts()},
    }
    for reference, hypothesis in pairs:
        score = transcript_scores(reference, hypothesis)
        for mode in totals:
            for metric in totals[mode]:
                values = score[mode][metric]
                totals[mode][metric] = totals[mode][metric] + ErrorCounts(
                    int(values["errors"]), int(values["reference_units"])
                )
    return {
        mode: {metric: counts.as_dict() for metric, counts in metrics.items()}
        for mode, metrics in totals.items()
    }


def percentile(values: Sequence[float], percent: float) -> float | None:
    """Linear-interpolated percentile, returning ``None`` for no samples."""

    if not values:
        return None
    if not 0 <= percent <= 100:
        raise ValueError("percent must be in [0, 100]")
    ordered = sorted(float(value) for value in values)
    location = (len(ordered) - 1) * (percent / 100.0)
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)
