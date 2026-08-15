"""Shared LLM helpers — OpenRouter client + tolerant JSON parsing.

Centralizes the API-key lookup and the tolerant JSON extraction that every
pipeline stage needs. See the scorer's original logic and the hard-won lessons
in the fastclip-shorts-pipeline skill.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from openai import OpenAI

# Preferred model for deterministic structured editing. deepseek-v4-flash is a
# reasoning model that returns empty at max_tokens<8192 and is non-deterministic
# even at temp=0.
DEFAULT_MODEL = "deepseek/deepseek-chat"

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def load_api_key() -> str:
    """Return the OpenRouter API key from env or the project .env file.

    Priority: process env var, then the project-local .env (cwd or package
    root), then home-dir fallbacks (~/fast-clip/.env, ~/.hermes/.env).
    """
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    _root = Path(__file__).resolve().parents[3]  # .../fast-clip (project root)
    for candidate in (
        Path.cwd() / ".env",
        _root / ".env",
        Path.home() / "fast-clip" / ".env",
        Path.home() / ".hermes" / ".env",
    ):
        if candidate.exists():
            m = re.search(r"OPENROUTER_API_KEY[= ]*([^\n\r]+)", candidate.read_text())
            if m:
                return m.group(1).strip().strip("\"'")
    raise RuntimeError(
        "OPENROUTER_API_KEY not set. Add it to fast-clip/.env or set the env var."
    )


def get_client(model: str | None = None) -> OpenAI:
    """Return an OpenAI client pointed at OpenRouter."""
    return OpenAI(base_url=OPENROUTER_BASE, api_key=load_api_key())


def extract_json_object(text: str) -> dict:
    """Extract the first { ... } JSON object from a possibly-fenced response."""
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return {}


def extract_json_objects(text: str) -> list[dict]:
    """Tolerantly extract JSON objects from a (possibly truncated) array.

    The model sometimes stops mid-array. This scans for balanced { ... } objects
    and parses each individually, so a partial response still yields every
    complete section.
    """
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?\s*```$", "", text)
    objs: list[dict] = []
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    objs.append(json.loads(text[start : i + 1]))
                except json.JSONDecodeError:
                    pass
                start = -1
    return objs


def chat(client: OpenAI, prompt: str, *, model: str, temperature: float = 0.0, max_tokens: int = 8192, retries: int = 3) -> str:
    """One chat completion against OpenRouter, returning the text content.

    Retries transient failures (network, rate-limit, 5xx) with a short backoff
    so a single blip doesn't fail the whole pipeline stage.
    """
    import time

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers={"HTTP-Referer": "http://localhost:8900", "X-Title": "FastClip"},
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 — retry any transient failure
            last_exc = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise last_exc  # type: ignore[misc]
