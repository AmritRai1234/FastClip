"""Natural segmentation — the third pipeline stage.

Divides a transcript into its NATURAL topical sections using the LLM. Unlike a
naive "find N clips" pass, this asks the model for the real boundaries (topic /
speaker / story-beat shifts) with no forced count or duration. Timestamps are
resolved from VERBATIM anchor quotes so the model never invents a time.
"""

from __future__ import annotations

import difflib
import re

from fast_clip.pipeline.transcriber import Transcript
from fast_clip.utils.llm import chat, extract_json_objects

SEGMENT_PROMPT = """You are a thoughtful video editor. Below is a full transcript of a video, one line per spoken segment, each prefixed with a segment ID like [S12] and followed by its timestamp.

Do NOT count clips. Do NOT aim for a target number. Instead, divide this video into its NATURAL sections — the real shifts in topic, speaker, or story beat — the way a human editor watching the footage would break it up. Find exactly as many sections as genuinely exist, no more, no fewer. A section can be 20 seconds or 3 minutes; follow the content, not a clock.

For EACH section, return:
  "title": a short label for what this section is about,
  "summary": one sentence on what's covered and why it's a distinct beat,
  "speaker": who is talking (or "multiple" if it changes),
  "start_quote": copy 6-12 EXACT consecutive words from the transcript that mark where this section begins,
  "end_quote": copy 6-12 EXACT consecutive words that mark where this section ends,
  "clip_potential": "high" | "medium" | "low" — how well this beat would stand alone as a YouTube Short.

CRITICAL RULES:
- start_quote and end_quote MUST be verbatim substrings of the transcript — copy them exactly, don't paraphrase.
- Choose quotes that are UNIQUE in the transcript so they can be located precisely.
- List sections in chronological order and cover the ENTIRE video (first section starts at the beginning, last ends at the end).

Respond ONLY with a JSON array (no markdown fences).

Transcript:

"""


def _norm(s: str) -> str:
    """Normalize text for fuzzy matching: lowercase, strip punctuation, collapse whitespace."""
    return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())


def resolve_quote(quote: str, segs: list[dict]) -> int | None:
    """Find the segment index whose text best matches a verbatim quote.

    Uses word-overlap + difflib ratio so it's robust to the LLM dropping or
    adding a word. Returns None if no good match.
    """
    q = _norm(quote)
    qwords = set(q.split())
    best_idx, best_score = None, 0.0
    for i, s in enumerate(segs):
        t = _norm(s["text"])
        twords = set(t.split())
        if not qwords or not twords:
            continue
        overlap = len(qwords & twords)
        ratio = difflib.SequenceMatcher(None, q, t).ratio()
        score = overlap + ratio * 2
        if score > best_score:
            best_score, best_idx = score, i
    if best_idx is not None and best_score < 2.0:
        return None
    return best_idx


def segment_transcript(
    transcript: Transcript,
    *,
    model: str,
    client=None,
) -> list[dict]:
    """Divide a transcript into natural sections with resolved timestamps.

    Returns a list of section dicts, each with ``resolved_start``/``resolved_end``
    (absolute seconds) and ``resolved`` (bool) plus the LLM's title/summary/etc.
    """
    from fast_clip.utils.llm import get_client

    if client is None:
        client = get_client(model)

    segs = [{"text": s.text, "start": s.start, "end": s.end} for s in transcript.segments]
    lines = [f"[S{i}] ({s['start']:.1f}-{s['end']:.1f}s) {s['text']}" for i, s in enumerate(segs)]
    transcript_text = "\n".join(lines)

    raw = chat(client, SEGMENT_PROMPT + transcript_text, model=model)
    sections = extract_json_objects(raw)
    if not sections:
        raise RuntimeError(f"Segmenter failed to parse response: {raw[:300]}")

    resolved = []
    for sec in sections:
        si = resolve_quote(sec.get("start_quote", ""), segs)
        ei = resolve_quote(sec.get("end_quote", ""), segs)
        if si is None or ei is None:
            sec["resolved_start"] = sec["resolved_end"] = None
            sec["resolved"] = False
        else:
            sec["resolved_start"] = round(segs[si]["start"], 1)
            sec["resolved_end"] = round(segs[ei]["end"], 1)
            sec["resolved"] = True
        resolved.append(sec)

    return resolved
