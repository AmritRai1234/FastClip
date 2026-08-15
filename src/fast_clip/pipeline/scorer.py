"""Moment scorer — the third pipeline stage.

Sends the transcript to DeepSeek via OpenRouter and returns clip
recommendations — which segments are worth extracting for YouTube Shorts,
what to merge together, and suggested titles/captions/hooks.

This is the "intelligence" layer that separates FastClip from a dumb
video-grabbing script. The LLM reads the full transcript in one shot
(DeepSeek has a 1M-token context window — no chunking needed) and outputs
structured JSON with clip specifications.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

from fast_clip.pipeline.transcriber import Segment, Transcript

# ── Data models ─────────────────────────────────────────────────────────────────


@dataclass
class ClipIdea:
    """A single clip recommendation from the LLM."""

    start: float  # seconds into the video
    end: float  # seconds
    title: str  # suggested YouTube Short title
    hook: str  # opening line / hook for the Short
    caption: str  # suggested description / hashtags
    reasoning: str  # why the LLM thinks this clip works


@dataclass
class ScoreResult:
    """Full scoring result from the LLM."""

    clips: list[ClipIdea] = field(default_factory=list)
    raw_response: str = ""  # for debugging
    tokens_used: dict[str, int] = field(default_factory=dict)

    @property
    def clip_count(self) -> int:
        return len(self.clips)

    def to_json(self) -> str:
        """Serialize clips to structured JSON for the renderer stage."""
        return json.dumps(
            {
                "clips": [
                    {
                        "start": round(c.start, 3),
                        "end": round(c.end, 3),
                        "duration": round(c.end - c.start, 1),
                        "title": c.title,
                        "hook": c.hook,
                        "caption": c.caption,
                        "reasoning": c.reasoning,
                    }
                    for c in self.clips
                ],
                "tokens": self.tokens_used,
            },
            ensure_ascii=False,
            indent=2,
        )

    def save(self, path: Path) -> Path:
        """Write scoring results to a JSON file. Returns the path."""
        path = Path(path)
        path.write_text(self.to_json(), encoding="utf-8")
        return path


# ── Scorer ──────────────────────────────────────────────────────────────────────


# System prompt that teaches the model what to look for and how to respond.
SYSTEM_PROMPT = """You are a world-class YouTube Shorts editor. Your job: read a video transcript with timestamps and identify the best 3–5 moments that would make engaging, viral-worthy Shorts.

For each clip, think about:
- **Hook potential** — does it start with something surprising, emotional, or curiosity-sparking?
- **Completeness** — is it a self-contained story beat? Does it have a satisfying end?
- **Pacing** — natural breaks make good start/end boundaries.
- **Shareability** — would someone send this to a friend?

Consider merging adjacent segments if they form a better, more coherent narrative together. Shorts work best at 30–60 seconds, but exceptional clips can be 15–25s or 60–90s.

Respond ONLY with a JSON array. Each object:

{
  "start": <float seconds>,
  "end": <float seconds>,
  "title": "<catchy Short title, max 60 chars>",
  "hook": "<opening caption/hook line>",
  "caption": "<description with 2-4 hashtags>",
  "reasoning": "<1 sentence why this clip works>"
}

CRITICAL: Use the EXACT timestamps from the transcript segments. Do not invent or offset them. start and end must match real segment boundaries from the transcript below."""


# Approximate token limit for the prompt (not the model's full 1M context).
# We cap the transcript at ~60K chars (~15K tokens) to leave room for the
# response and avoid paying for tokens we won't use. The actual token count
# varies — tiktoken would give a precise count, but a 4-char-per-token
# heuristic is good enough for a safety cutoff.
MAX_TRANSCRIPT_CHARS = 240_000  # ~60K tokens, well within 1M context


def _load_api_key() -> str:
    """Return the OpenRouter API key from environment or config files."""
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if key:
        return key

    # Try common env-file locations (used during dev, not by end users).
    for candidate in (
        Path.home() / ".fastclip" / ".env",
        Path.home() / "fast-clip" / ".env",
        Path.home() / ".hermes" / ".env",
    ):
        if candidate.exists():
            text = candidate.read_text()
            m = re.search(r"OPENROUTER_API_KEY[= ]*([^\n\r]+)", text)
            if m:
                return m.group(1).strip().strip("\"'")

    raise RuntimeError(
        "OPENROUTER_API_KEY not set. Set it in your environment or "
        "~/.hermes/.env:  export OPENROUTER_API_KEY=sk-or-v1-..."
    )


def _build_prompt(transcript: Transcript, language: str | None = None) -> str:
    """Build the user prompt from a transcript.

    Each segment is formatted as '[start → end] text' so the LLM sees exact
    timestamps it can reference in its recommendations.
    """
    segments_text: list[str] = []
    for seg in transcript.segments:
        ts = f"[{seg.start:.1f} → {seg.end:.1f}]"
        segments_text.append(f"{ts} {seg.text}")

    body = "\n".join(segments_text)

    # Truncate if unusually long (rare — DeepSeek has 1M context).
    if len(body) > MAX_TRANSCRIPT_CHARS:
        body = body[:MAX_TRANSCRIPT_CHARS]
        body += "\n\n[TRANSCRIPT TRUNCATED — this video is extremely long]"

    meta = f"Language: {transcript.language} ({transcript.language_probability:.0%} confidence)"
    header = (
        "Below is the transcript of a video. Find the 3-5 best moments "
        "for YouTube Shorts. Return a JSON array of clip objects.\n"
    )
    return f"{header}\n{meta}\n\n{body}"


def _parse_response(raw: str) -> list[dict]:
    """Parse the LLM's JSON response, handling markdown fences and edge cases."""
    raw = raw.strip()

    # Strip markdown fences — OpenRouter models often wrap JSON in ```json.
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
    raw = re.sub(r"\n?\s*```$", "", raw)

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        # Sometimes the model returns {"clips": [...]} instead of a bare array.
        if isinstance(data, dict):
            for key in ("clips", "moments", "highlights", "results"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            # If it's a single clip, wrap it.
            if "start" in data:
                return [data]
        raise ValueError(f"Expected a JSON array of clips, got {type(data).__name__}")
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: regex-extract a JSON array from the text.
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Failed to parse LLM response as JSON. "
        f"Raw (first 500 chars): {raw[:500]}"
    )


def _build_clip_ideas(data: list[dict]) -> list[ClipIdea]:
    """Validate and convert parsed JSON objects to ClipIdea dataclasses."""
    clips: list[ClipIdea] = []
    for i, obj in enumerate(data):
        try:
            clips.append(
                ClipIdea(
                    start=float(obj["start"]),
                    end=float(obj["end"]),
                    title=str(obj.get("title", f"Clip {i+1}")),
                    hook=str(obj.get("hook", "")),
                    caption=str(obj.get("caption", "")),
                    reasoning=str(obj.get("reasoning", "")),
                )
            )
        except (KeyError, ValueError, TypeError) as e:
            print(f"[scorer] Warning: skipping invalid clip {i}: {e} — {obj}")
    return clips


class Scorer:
    """LLM-powered moment scorer.

    Sends the transcript to DeepSeek via OpenRouter and returns a ranked list
    of clip recommendations with titles, hooks, and captions.

    Usage:
        scorer = Scorer()
        result = scorer.score(transcript)
        for clip in result.clips:
            print(f"{clip.start:.1f}-{clip.end:.1f}: {clip.title}")
    """

    def __init__(
        self,
        *,
        model: str = "deepseek/deepseek-v4-flash",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_retries: int = 3,
        http_referer: str = "http://localhost:8900",
        app_title: str = "FastClip",
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.http_referer = http_referer
        self.app_title = app_title

        api_key = _load_api_key()
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def score(self, transcript: Transcript, *, language: str | None = None) -> ScoreResult:
        """Score a transcript and return clip recommendations.

        Args:
            transcript: The transcribed segments with timestamps.
            language: Optional language override (auto-detected if None).

        Returns:
            ScoreResult with ranked ClipIdea objects.

        Raises:
            RuntimeError: If the LLM fails after max_retries or returns no clips.
        """
        prompt = _build_prompt(transcript, language=language)

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    extra_headers={
                        "HTTP-Referer": self.http_referer,
                        "X-Title": self.app_title,
                    },
                )
            except Exception as exc:
                last_error = exc
                print(f"[scorer] Attempt {attempt + 1} failed: {exc}")
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                continue

            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            if content is None:
                last_error = RuntimeError(
                    f"LLM returned empty response "
                    f"(finish_reason={finish_reason})"
                )
                print(f"[scorer] Attempt {attempt + 1}: {last_error}")
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                continue

            # Parse the JSON response.
            try:
                raw_data = _parse_response(content)
                clips = _build_clip_ideas(raw_data)
            except ValueError as exc:
                last_error = exc
                print(f"[scorer] Attempt {attempt + 1} parse error: {exc}")
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                continue

            usage = {}
            if response.usage:
                usage = {
                    "input": response.usage.prompt_tokens or 0,
                    "output": response.usage.completion_tokens or 0,
                    "total": response.usage.total_tokens or 0,
                }

            return ScoreResult(clips=clips, raw_response=content, tokens_used=usage)

        raise RuntimeError(
            f"Scorer failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )