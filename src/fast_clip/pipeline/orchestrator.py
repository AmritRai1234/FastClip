"""Pipeline orchestrator — runs the full clip pipeline with progress reporting.

Shared by the CLI and the FastAPI backend so both drive the same logic. The
progress callback is called at each stage with a small dict describing where
we are, so a console can print it and a web UI can push it over a socket.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from fast_clip.pipeline.clipmaker import plan_section
from fast_clip.pipeline.downloader import Downloader
from fast_clip.pipeline.renderer import render_short
from fast_clip.pipeline.segmenter import segment_transcript
from fast_clip.pipeline.transcriber import Transcriber
from fast_clip.utils.llm import get_client

# progress_cb receives {"stage": str, "current": int, "total": int, "message": str}
ProgressCallback = Callable[[dict[str, Any]], None]


def _noop(_: dict[str, Any]) -> None:
    pass


def run_pipeline(
    media: str | Path,
    *,
    output_dir: Path,
    whisper_model: str = "base",
    llm_model: str | None = None,
    max_height: int | None = None,
    cookies_file: str | Path | None = None,
    cookies_from_browser: str | None = None,
    pan: bool = True,
    language: str | None = None,
    progress: ProgressCallback = _noop,
) -> list[dict]:
    """Run download -> transcribe -> segment -> plan -> render.

    ``media`` is a URL or local file path. Returns a list of short dicts, each
    with ``path`` (rendered file), ``title``, ``hook``, ``duration``, ``section``
    and ``section_title``.
    """
    from fast_clip.pipeline.downloader import VideoMetadata

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Download ──
    progress({"stage": "download", "current": 0, "total": 1, "message": str(media)})
    media_str = str(media)
    if media_str.startswith(("http://", "https://", "www.")):
        dl = Downloader(output_dir=output_dir, max_height=max_height, cookies_file=cookies_file, cookies_from_browser=cookies_from_browser, quiet=True)
        meta: VideoMetadata = dl.download(media_str)
        media_path = meta.filepath
    else:
        media_path = Path(media_str)
        if not media_path.exists():
            raise FileNotFoundError(f"Media file not found: {media_path}")
    progress({"stage": "download", "current": 1, "total": 1, "message": str(media_path)})

    # ── 2. Transcribe ──
    progress({"stage": "transcribe", "current": 0, "total": 1, "message": "transcribing"})
    tr = Transcriber(model_size=whisper_model, vad_filter=True, word_timestamps=True)
    transcript = tr.transcribe(media_path, language=language)
    progress(
        {
            "stage": "transcribe",
            "current": 1,
            "total": 1,
            "message": f"{len(transcript.segments)} segments ({transcript.language})",
        }
    )

    # ── 3. Segment ──
    progress({"stage": "segment", "current": 0, "total": 1, "message": "finding sections"})
    client = get_client(llm_model)
    sections = [s for s in segment_transcript(transcript, model=llm_model, client=client) if s["resolved"]]
    progress({"stage": "segment", "current": 1, "total": 1, "message": f"{len(sections)} sections"})

    # ── 4. Plan shorts ──
    all_shorts: list[dict] = []
    for i, sec in enumerate(sections, 1):
        progress({"stage": "plan", "current": i - 1, "total": len(sections), "message": sec["title"]})
        try:
            shorts = plan_section(sec, transcript, model=llm_model, client=client)
        except Exception as exc:  # noqa: BLE001 — skip a failed section, keep going
            progress({"stage": "plan", "current": i, "total": len(sections), "message": f"skipped '{sec['title']}' ({exc})"})
            continue
        for s in shorts:
            s["section"] = i
            s["section_title"] = sec["title"]
        all_shorts.extend(shorts)
    progress({"stage": "plan", "current": len(sections), "total": len(sections), "message": f"{len(all_shorts)} shorts"})

    # Save the plan (reviewable) and write it to disk.
    plan_path = output_dir / f"{media_path.stem}_plans.json"
    plan_path.write_text(json.dumps(all_shorts, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 5. Render ──
    out_dir = output_dir / "shorts"
    results: list[dict] = []
    for i, short in enumerate(all_shorts, 1):
        progress({"stage": "render", "current": i - 1, "total": len(all_shorts), "message": short["title"]})
        out = render_short(
            media_path,
            short,
            out_dir=out_dir,
            transcriber=tr,
            ffmpeg="ffmpeg",
            pan=pan,
            prefix=f"{i:02d}_",
        )
        results.append({
            "path": out,
            "title": short.get("title") or out.stem,
            "hook": short.get("hook", ""),
            "duration": short.get("duration", 0.0),
            "section": short.get("section", i),
            "section_title": short.get("section_title", ""),
        })
    progress({"stage": "done", "current": len(all_shorts), "total": len(all_shorts), "message": "complete"})

    return results
