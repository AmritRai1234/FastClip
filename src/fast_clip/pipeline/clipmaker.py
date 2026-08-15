"""Clip planning — the fourth pipeline stage.

For each natural section, the LLM decides whether it's ONE coherent moment
(a single Short) or MULTIPLE self-contained moments (split into 2-4 Shorts),
based on the content, not a count or clock. The plan is frozen to JSON so
rendering is deterministic and human-reviewable (the plan/render split).
"""

from __future__ import annotations

from fast_clip.pipeline.transcriber import Transcript
from fast_clip.utils.llm import chat, extract_json_object

MAKE_SHORTS_PROMPT = """You are a YouTube Shorts editor. Below is the transcript of ONE section of a video, one line per spoken segment, labeled [S0], [S1], etc., with timestamps.

Turn this section into one or more YouTube Shorts:

- If the section is ONE coherent story or moment, return a SINGLE short. Trim filler (false starts, repeated thanks, "um"/"uh", applause, audience noise, rambling, off-topic tangents) and keep the tightest version of that one story.

- If the section contains MULTIPLE distinct, self-contained moments (different topics, quotes, story beats, or emotional lines), SPLIT it into 2-4 shorts. Each short must be a complete, standalone moment that makes sense on its own — strong opening, clear point, clean ending.

Decide from the CONTENT and STORY, not a clock. A short is usually 15-60s, but longer is fine if the story genuinely needs it. Don't force a count: if there's one moment, return one; if there are four, return four.

CRITICAL: cut filler and noise aggressively — this is raw live speech, it always has filler. "Keep everything" is never correct. Never cut the single best hook or emotional payoff of a moment.

EXAMPLE of correct output for a section that is ONE story:
{"shorts": [{"title": "My Father's Journey to NYC", "hook": "He arrived with nothing but a promise.", "keep": [2, 3, 4, 5, 6, 7, 8]}]}

EXAMPLE for a section with THREE distinct moments (split it):
{"shorts": [
  {"title": "First moment", "hook": "...", "keep": [0, 1, 2, 3]},
  {"title": "Second moment", "hook": "...", "keep": [6, 7, 8, 9]},
  {"title": "Third moment", "hook": "...", "keep": [12, 13, 14, 15]}
]}
(Note the gaps — segments 4, 5, 10, 11 were filler and were dropped.)

Return a JSON object (no markdown):
{
  "shorts": [
    {"title": "<catchy title, max 60 chars>", "hook": "<opening caption, max 120 chars>", "keep": [<segment indices, ascending>]},
    ...
  ]
}

Each "keep" list is the segments that make up that short, in order. A segment belongs to at most one short. Segment indices must reference real segments (S0 -> 0, etc.). Cover the important content; you may drop filler-only segments entirely.

Transcript:

"""

# Reject a "lazy" answer that keeps >80% of a long section as one clip.
LAZY_KEEP_FRACTION = 0.8
LAZY_MIN_SECTION_SECONDS = 60.0
LAZY_NUDGE = (
    "\n\nYou previously kept nearly everything as one clip. That is wrong for this "
    "raw speech. Identify the DISTINCT moments and split into 2-4 shorts, cutting "
    "the filler between them."
)


def resolve_shorts(data: dict, sub: list[dict]) -> list[dict]:
    """Convert the LLM's raw shorts into timestamp-resolved shorts.

    ``sub`` is the section's segment list (dicts with start/end/text). Returns
    shorts with ``keep`` (segment indices) and ``segments`` (start/end ranges).
    """
    shorts = []
    for s in data.get("shorts", []):
        keep = sorted({i for i in s.get("keep", []) if isinstance(i, int) and 0 <= i < len(sub)})
        if not keep:
            continue
        ranges = []
        for i in keep:
            if ranges and i == ranges[-1][1]:
                ranges[-1] = (ranges[-1][0], i + 1)
            else:
                ranges.append((i, i + 1))
        seg_ranges = [{"start": sub[a]["start"], "end": sub[b - 1]["end"]} for a, b in ranges]
        dur = sum(r["end"] - r["start"] for r in seg_ranges)
        shorts.append(
            {
                "title": s.get("title", "Untitled"),
                "hook": s.get("hook", ""),
                "keep": keep,
                "segments": seg_ranges,
                "duration": round(dur, 1),
            }
        )
    return shorts


def plan_section(
    section: dict,
    transcript: Transcript,
    *,
    model: str,
    client=None,
) -> list[dict]:
    """Plan the shorts for one natural section, with lazy-answer rejection.

    Returns a list of short dicts (each with title/hook/keep/segments/duration).
    """
    from fast_clip.utils.llm import get_client

    if client is None:
        client = get_client(model)

    s_start, s_end = section["resolved_start"], section["resolved_end"]
    sub = [
        {"text": s.text, "start": s.start, "end": s.end}
        for s in transcript.segments
        if s.start >= s_start - 0.5 and s.end <= s_end + 0.5
    ]
    lines = [f"[S{i}] ({s['start']:.1f}-{s['end']:.1f}s) {s['text']}" for i, s in enumerate(sub)]
    transcript_text = "\n".join(lines)

    data: dict = {}
    nudge = ""
    for attempt in range(4):
        raw = chat(client, MAKE_SHORTS_PROMPT + transcript_text + nudge, model=model)
        data = extract_json_object(raw)
        if not data.get("shorts"):
            continue
        shorts_tmp = resolve_shorts(data, sub)
        kept_frac = sum(len(s["keep"]) for s in shorts_tmp) / max(len(sub), 1)
        if len(shorts_tmp) == 1 and kept_frac > LAZY_KEEP_FRACTION and (s_end - s_start) > LAZY_MIN_SECTION_SECONDS:
            nudge = LAZY_NUDGE
            data = {}
            continue
        break

    shorts = resolve_shorts(data, sub)
    if not shorts and sub:
        # Graceful fallback: if the LLM produced no valid plan after retries,
        # treat the whole section as a single short instead of failing the job.
        seg_ranges = [{"start": sub[0]["start"], "end": sub[-1]["end"]}]
        dur = round(sum(r["end"] - r["start"] for r in seg_ranges), 1)
        shorts = [{
            "title": section.get("title", "Untitled"),
            "hook": "",
            "keep": list(range(len(sub))),
            "segments": seg_ranges,
            "duration": dur,
        }]
    return shorts
