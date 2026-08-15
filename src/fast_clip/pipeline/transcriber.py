"""Transcription — the second pipeline stage.

Converts downloaded video audio into timestamped text using faster-whisper
(CTranslate2, int8-quantized — 3-4x faster than openai-whisper on CPU).

The transcript (segments with start/end times) is what the downstream
"moment scorer" stage consumes to pick the best clip candidates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel


# ── Data models ─────────────────────────────────────────────────────────────────


@dataclass
class Word:
    """A single word with its timing (for precise captioning)."""

    start: float
    end: float
    word: str


@dataclass
class Segment:
    """A continuous chunk of speech."""

    start: float  # seconds
    end: float  # seconds
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    """Full transcription result."""

    language: str
    language_probability: float
    duration: float  # audio duration in seconds
    segments: list[Segment] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Full plain-text transcript (no timestamps)."""
        return " ".join(s.text.strip() for s in self.segments)

    def to_txt(self) -> str:
        """Human-readable transcript with per-segment timestamps."""
        lines = []
        for seg in self.segments:
            ts = f"[{_fmt_ts(seg.start)} -> {_fmt_ts(seg.end)}]"
            lines.append(f"{ts}  {seg.text.strip()}")
        return "\n".join(lines)

    def to_srt(self) -> str:
        """Standard SubRip subtitle format (for captions/editing tools)."""
        blocks = []
        for i, seg in enumerate(self.segments, start=1):
            blocks.append(
                f"{i}\n"
                f"{_fmt_srt_ts(seg.start)} --> {_fmt_srt_ts(seg.end)}\n"
                f"{seg.text.strip()}\n"
            )
        return "\n".join(blocks)

    def to_json(self) -> str:
        """Structured JSON for the pipeline / API."""
        return json.dumps(
            {
                "language": self.language,
                "language_probability": round(self.language_probability, 4),
                "duration": round(self.duration, 3),
                "segments": [
                    {
                        "start": round(s.start, 3),
                        "end": round(s.end, 3),
                        "text": s.text.strip(),
                        "words": [
                            {"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word}
                            for w in s.words
                        ],
                    }
                    for s in self.segments
                ],
            },
            ensure_ascii=False,
            indent=2,
        )

    def save(self, path: Path, fmt: str = "txt") -> Path:
        """Write the transcript to disk (.txt, .srt, or .json)."""
        path = Path(path)
        if fmt == "txt":
            content = self.to_txt()
        elif fmt == "srt":
            content = self.to_srt()
        elif fmt == "json":
            content = self.to_json()
        else:
            raise ValueError(f"Unknown format {fmt!r} — use txt, srt, or json")
        path.write_text(content, encoding="utf-8")
        return path


# ── Transcriber ─────────────────────────────────────────────────────────────────


# Model sizes available in faster-whisper (ordered by size/accuracy).
MODEL_SIZES = ("tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3")

# Rough model download sizes (for user awareness)
MODEL_SIZE_MB = {
    "tiny": 75,
    "base": 145,
    "small": 484,
    "medium": 1536,
    "large-v1": 3092,
    "large-v2": 3092,
    "large-v3": 3092,
}


class Transcriber:
    """Speech-to-text using faster-whisper, optimized for CPU.

    Usage:
        tr = Transcriber(model_size="base")
        transcript = tr.transcribe(Path("video.mp4"))
        transcript.save(Path("video.txt"))
    """

    def __init__(
        self,
        model_size: str = "base",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 0,  # 0 = use all cores
        word_timestamps: bool = False,
        vad_filter: bool = True,
    ) -> None:
        """
        Args:
            model_size: tiny | base | small | medium | large-v3.
                        tiny/base are fastest on CPU; small is the accuracy
                        sweet spot; large-v3 is slow but most accurate.
            device: "cpu" (this box has no GPU) or "cuda".
            compute_type: "int8" (fast, recommended) or "float16"/"float32".
            cpu_threads: Number of CPU threads. 0 = auto (all cores).
            word_timestamps: Emit word-level timestamps (for precise captions).
                             Slower; segment-level is usually enough.
            vad_filter: Skip silence/non-speech regions via Silero VAD.
                        Recommended for talking videos, but DISABLE for
                        music-heavy content (it can filter out everything).
        """
        if model_size not in MODEL_SIZES:
            raise ValueError(f"model_size must be one of {MODEL_SIZES}")
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.word_timestamps = word_timestamps
        self.vad_filter = vad_filter

        # Lazily load the model on first transcribe() call.
        self._model: WhisperModel | None = None

    @property
    def model(self) -> WhisperModel:
        """Load (and cache) the WhisperModel on first use."""
        if self._model is None:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
            )
        return self._model

    def transcribe(self, media_path: Path, *, language: str | None = None) -> Transcript:
        """Transcribe a video or audio file.

        Args:
            media_path: Path to the video (or audio) file. Audio is decoded
                        from the video container via PyAV automatically.
            language: ISO code (e.g. "en") or None to auto-detect.

        Returns:
            Transcript with timestamped segments.
        """
        media_path = Path(media_path)
        if not media_path.exists():
            raise FileNotFoundError(f"Media file not found: {media_path}")

        segments_iter, info = self.model.transcribe(
            str(media_path),
            language=language,
            beam_size=5,
            word_timestamps=self.word_timestamps,
            vad_filter=self.vad_filter,
        )

        segments: list[Segment] = []
        for seg in segments_iter:
            words = [
                Word(start=w.start, end=w.end, word=w.word)
                for w in (seg.words or [])
            ] if self.word_timestamps else []
            segments.append(
                Segment(start=seg.start, end=seg.end, text=seg.text.strip(), words=words)
            )

        return Transcript(
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            segments=segments,
        )


# ── Formatting helpers ──────────────────────────────────────────────────────────


def _fmt_ts(seconds: float) -> str:
    """Format seconds as [mm:ss.cc] for the txt transcript."""
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes):02d}:{secs:05.2f}"


def _fmt_srt_ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS,mmm for SRT."""
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    millis = int((secs - int(secs)) * 1000)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d},{millis:03d}"