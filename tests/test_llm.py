"""Tests for tolerant JSON parsing in utils.llm."""

from fast_clip.utils.llm import extract_json_object, extract_json_objects


def test_extract_json_object_plain():
    assert extract_json_object('{"keep": [1, 2, 3]}') == {"keep": [1, 2, 3]}


def test_extract_json_object_fenced():
    raw = '```json\n{"keep": [1, 2, 3]}\n```'
    assert extract_json_object(raw) == {"keep": [1, 2, 3]}


def test_extract_json_object_empty():
    assert extract_json_object("no json here") == {}


def test_extract_json_objects_array():
    raw = '[{"a": 1}, {"b": 2}]'
    assert extract_json_objects(raw) == [{"a": 1}, {"b": 2}]


def test_extract_json_objects_truncated():
    # The model stops mid-array; the complete objects must still be recovered.
    raw = '[{"a": 1}, {"b": 2}, {"c":'
    objs = extract_json_objects(raw)
    assert objs == [{"a": 1}, {"b": 2}]


def test_extract_json_objects_fenced_truncated():
    raw = '```json\n[{"a": 1}, {"b": 2},'
    assert extract_json_objects(raw) == [{"a": 1}, {"b": 2}]
