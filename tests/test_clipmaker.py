"""Tests for clip-plan resolution."""

from fast_clip.pipeline.clipmaker import resolve_shorts

SUB = [
    {"start": 0.0, "end": 1.0, "text": "a"},
    {"start": 1.0, "end": 2.0, "text": "b"},
    {"start": 2.0, "end": 3.0, "text": "c"},
]


def test_resolve_shorts_single_contiguous():
    data = {"shorts": [{"title": "T", "hook": "h", "keep": [0, 1, 2]}]}
    shorts = resolve_shorts(data, SUB)
    assert len(shorts) == 1
    assert shorts[0]["segments"] == [{"start": 0.0, "end": 3.0}]
    assert shorts[0]["duration"] == 3.0


def test_resolve_shorts_merges_gaps_into_ranges():
    data = {"shorts": [{"title": "T", "hook": "h", "keep": [0, 2]}]}
    shorts = resolve_shorts(data, SUB)
    assert shorts[0]["segments"] == [
        {"start": 0.0, "end": 1.0},
        {"start": 2.0, "end": 3.0},
    ]
    assert shorts[0]["duration"] == 2.0


def test_resolve_shorts_clamps_out_of_range_indices():
    data = {"shorts": [{"title": "T", "hook": "h", "keep": [0, 99, -1, 2]}]}
    shorts = resolve_shorts(data, SUB)
    assert shorts[0]["keep"] == [0, 2]


def test_resolve_shorts_ignores_empty_keep():
    data = {"shorts": [{"title": "T", "hook": "h", "keep": []}]}
    assert resolve_shorts(data, SUB) == []
