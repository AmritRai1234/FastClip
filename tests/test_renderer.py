"""Tests for the ASS caption generator."""

from fast_clip.pipeline.renderer import fmt_ass_ts, make_word_ass
from fast_clip.pipeline.transcriber import Segment, Transcript, Word


def _transcript() -> Transcript:
    segs = [
        Segment(
            start=0.0,
            end=2.0,
            text="hello world",
            words=[
                Word(start=0.0, end=1.0, word="hello"),
                Word(start=1.0, end=2.0, word="world"),
            ],
        )
    ]
    return Transcript(language="en", language_probability=1.0, duration=2.0, segments=segs)


def test_fmt_ass_ts():
    assert fmt_ass_ts(0.0) == "0:00:00.00"
    assert fmt_ass_ts(65.5) == "0:01:05.50"
    assert fmt_ass_ts(3661.25) == "1:01:01.25"


def test_make_word_ass_karaoke_tags(tmp_path):
    out = make_word_ass(_transcript(), tmp_path / "t.ass")
    content = out.read_text()
    # Karaoke tag with 100 centiseconds per word.
    assert "{\\k100}hello" in content
    assert "{\\k100}world" in content
    assert content.count("Dialogue:") == 1
    assert "PlayResY: 1920" in content


def test_make_word_ass_empty_transcript(tmp_path):
    tx = Transcript(language="en", language_probability=1.0, duration=0.0, segments=[])
    out = make_word_ass(tx, tmp_path / "empty.ass")
    content = out.read_text()
    assert "Dialogue:" not in content
    assert "[Script Info]" in content
