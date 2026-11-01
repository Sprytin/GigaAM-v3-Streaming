from gigaam_v3_streaming.metrics import (
    aggregate_transcript_scores,
    levenshtein_distance,
    normalize_text,
    transcript_scores,
)


def test_normalization_matches_expected_russian_lexical_form():
    assert normalize_text("Ёлка — это  'тест'!") == "елка это тест"
    assert normalize_text("Don't  stop") == "don't stop"


def test_levenshtein_and_corpus_micro_counts():
    assert levenshtein_distance(["a", "b", "c"], ["a", "x", "c", "d"]) == 2
    scores = aggregate_transcript_scores([("one two", "one"), ("a", "a b")])
    assert scores["raw"]["wer"]["errors"] == 2
    assert scores["raw"]["wer"]["reference_units"] == 3
    assert scores["raw"]["wer"]["rate"] == 2 / 3


def test_raw_and_normalized_scores_are_distinct():
    scores = transcript_scores("Привет, ёлка!", "привет елка")
    assert scores["raw"]["wer"]["errors"] == 2
    assert scores["normalized"]["wer"]["errors"] == 0
    assert scores["normalized"]["cer"]["rate"] == 0
