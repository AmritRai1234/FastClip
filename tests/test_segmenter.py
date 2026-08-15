"""Tests for natural segmentation quote resolution."""

from fast_clip.pipeline.segmenter import _norm, resolve_quote

SEGS = [
    {"text": "La capital, el suel", "start": 2.9, "end": 19.2},
    {"text": "thank you so much everybody", "start": 19.2, "end": 24.9},
    {"text": "to Mayor for bringing us together", "start": 24.9, "end": 29.1},
]


def test_norm_lowercases_and_strips_punct():
    assert _norm("Hello, World!") == "hello world"


def test_resolve_quote_exact_match():
    assert resolve_quote("thank you so much everybody", SEGS) == 1


def test_resolve_quote_word_reorder_still_matches():
    # Word-overlap scoring is robust to the LLM reordering/dropping words.
    assert resolve_quote("everybody so much", SEGS) == 1


def test_resolve_quote_no_match_returns_none():
    assert resolve_quote("completely unrelated phrase", SEGS) is None
