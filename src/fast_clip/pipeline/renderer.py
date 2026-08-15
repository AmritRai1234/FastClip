"""Rendering — the fifth pipeline stage.

Cuts each planned short from the source, re-transcribes it (so captions match
its exact timeline), then renders a 9:16 vertical with face-following pan and
word-by-word captions. Heavy lifting via ffmpeg + OpenCV (YuNet).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from fast_clip.pipeline.transcriber import Transcriber, Transcript

# 9:16 target resolution.
OUT_W, OUT_H = 1080, 1920

# Word-by-word karaoke captions. PlayRes matches the output so Fontsize/MarginV
# are in real pixels (no 288-unit confusion). PrimaryColour = yellow highlight,
# SecondaryColour = fully transparent (unspoken words hidden, so words appear
# one at a time). Alignment 5 = middle-center. WrapStyle 0 = smart word wrap.
ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Liberation Sans,100,&H0000FFFF,&HFFFFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,5,60,60,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def slugify(title: str) -> str:
    keep = "".join(c if c.isalnum() or c in " -_" else "" for c in title)
    return "_".join(keep.split()).lower()


def fmt_ass_ts(sec: float) -> str:
    """ASS timestamp: H:MM:SS.cc (centiseconds)."""
    cs = int(round(sec * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def make_word_ass(transcript: Transcript, out: Path, max_words: int = 4) -> Path:
    """Generate a karaoke ASS (word-by-word highlight) from a transcript.

    The transcript's words are already 0-based relative to the short. Groups
    words into lines of ~max_words; each word carries a \\k tag with its spoken
    duration so libass sweeps the highlight across the line in sync.
    """
    words: list[dict] = []
    for seg in transcript.segments:
        for w in seg.words:
            words.append({"word": w.word, "start": w.start, "end": w.end})
    if not words:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(ASS_HEADER, encoding="utf-8")
        return out

    lines = [words[i : i + max_words] for i in range(0, len(words), max_words)]
    events = []
    for line in lines:
        ls, le = line[0]["start"], line[-1]["end"]
        parts = []
        for w in line:
            dur = max(1, int(round((w["end"] - w["start"]) * 100)))
            parts.append(f"{{\\k{dur}}}{w['word']}")
        events.append(f"Dialogue: 0,{fmt_ass_ts(ls)},{fmt_ass_ts(le)},Caption,,0,0,0,,{' '.join(parts)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ASS_HEADER + "\n".join(events) + "\n", encoding="utf-8")
    return out


# ── Panning (OpenCV YuNet face tracking) ───────────────────────────────────────


def _find_yunet_model() -> Path:
    for candidate in (
        Path.home() / ".clipme_yunet.onnx",
        Path.home() / ".yunet.onnx",
        Path("yunet.onnx"),
    ):
        if candidate.exists():
            return candidate
    url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    dst = Path.home() / ".yunet.onnx"
    import urllib.request

    urllib.request.urlretrieve(url, dst)
    return dst


def compute_pan(video_path: Path, crop_w: float, sample_interval: float = 0.25) -> list[tuple[float, float]]:
    """Return a (time, crop_x) pan path that keeps the face centered."""
    import cv2

    model_path = str(_find_yunet_model())
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    det_w, det_h = 960, 540
    detector = cv2.FaceDetectorYN_create(model_path, "", (det_w, det_h), 0.85, 0.3, 5000)

    stride = max(1, int(fps * sample_interval))
    pan: list[tuple[float, float]] = []
    last_cx = width / 2.0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % stride != 0 and frame_idx != 1:
            continue
        t = frame_idx / fps
        small = cv2.resize(frame, (det_w, det_h))
        _, faces = detector.detect(small)
        if faces is not None and len(faces):
            best = max(faces, key=lambda f: f[14])
            fx, fw = best[0] * width / det_w, best[2] * width / det_w
            last_cx = fx + fw / 2.0
        x_offset = max(0.0, min(width - crop_w, last_cx - crop_w / 2.0))
        pan.append((t, x_offset))
    cap.release()

    # Smooth + downsample (ffmpeg's expression parser chokes on deep if() chains).
    if len(pan) > 2:
        w = 5
        xs = [x for _, x in pan]
        smooth = [sum(xs[max(0, i - w // 2) : min(len(xs), i + w // 2 + 1)]) / (min(len(xs), i + w // 2 + 1) - max(0, i - w // 2)) for i in range(len(xs))]
        pan = [(t, x) for (t, _), x in zip(pan, smooth)]
    max_keyframes = 60
    if len(pan) > max_keyframes:
        idxs = [round(i * (len(pan) - 1) / (max_keyframes - 1)) for i in range(max_keyframes)]
        pan = [pan[i] for i in idxs]
    return pan


# ── ffmpeg render ──────────────────────────────────────────────────────────────


def _probe(ffmpeg: str, path: Path) -> tuple[int, int]:
    p = subprocess.run([ffmpeg, "-i", str(path)], capture_output=True, text=True)
    m = re.search(r"(\d{2,5})x(\d{2,5})", p.stderr)
    return (int(m.group(1)), int(m.group(2))) if m else (1920, 1080)


def cut_segments(ffmpeg: str, src: Path, segments: list[dict], out: Path) -> Path:
    """Cut one or more segment ranges and concatenate into a single clip."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        parts = []
        for i, r in enumerate(segments):
            part = tmp / f"p{i}.mp4"
            cmd = [
                ffmpeg, "-y", "-ss", f"{r['start']:.3f}", "-to", f"{r['end']:.3f}",
                "-i", str(src),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
                "-c:a", "aac", "-b:a", "192k",
                "-avoid_negative_ts", "make_zero", str(part),
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            parts.append(part)
        if len(parts) == 1:
            shutil.copy(parts[0], out)
        else:
            lst = tmp / "concat.txt"
            lst.write_text("".join(f"file '{p}'\n" for p in parts))
            cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)]
            subprocess.run(cmd, check=True, capture_output=True)
    return out


def to_vertical(ffmpeg: str, src: Path, ass: Path, out: Path, pan_x: list[tuple[float, float]] | None, crop_w: float) -> Path:
    """Crop 16:9 -> 9:16 (center or pan), scale, burn captions, re-encode."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if pan_x:
        # Piecewise-LINEAR interpolation between keyframes (smooth glide,
        # not the old step/snap behavior). x(t) = lerp(x_i, x_{i+1}, frac)
        # within each segment, held at the final value past the last keyframe.
        inner = f"{pan_x[-1][1]:.1f}"
        for i in range(len(pan_x) - 2, -1, -1):
            t0, x0 = pan_x[i]
            t1, x1 = pan_x[i + 1]
            dur = max(t1 - t0, 1e-3)
            seg = f"lerp({x0:.1f},{x1:.1f},clip((t-{t0:.3f})/{dur:.3f},0,1))"
            inner = f"if(lt(t,{t1:.3f}),{seg},{inner})"
        x_expr = inner
    else:
        x_expr = f"(iw-{crop_w:.1f})/2"

    vf = f"crop=w={crop_w:.1f}:h=ih:x='{x_expr}':y=0,scale={OUT_W}:{OUT_H},ass='{ass}'"
    cmd = [
        ffmpeg, "-y", "-i", str(src), "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


# ── Orchestration ──────────────────────────────────────────────────────────────


def render_short(
    source: Path,
    short: dict,
    *,
    out_dir: Path,
    transcriber: Transcriber,
    ffmpeg: str,
    pan: bool = True,
    prefix: str = "",
) -> Path:
    """Cut a short from source, re-transcribe, and render 9:16 + captions."""
    ffmpeg = ffmpeg or shutil.which("ffmpeg") or "ffmpeg"
    out_dir = Path(out_dir)
    name = f"{prefix}{slugify(short['title'])}.mp4"

    # 1. Cut the horizontal short from source.
    horizontal = out_dir / "_h" / name
    cut_segments(ffmpeg, source, short["segments"], horizontal)

    # 2. Re-transcribe the short so captions match its exact timeline.
    transcript = transcriber.transcribe(horizontal)

    # 3. Word-by-word ASS.
    ass = out_dir / "_h" / f"{horizontal.stem}.ass"
    make_word_ass(transcript, ass)

    # 4. Pan + vertical + captions.
    src_w, src_h = _probe(ffmpeg, horizontal)
    crop_w = src_h * OUT_W / OUT_H
    pan_x = compute_pan(horizontal, crop_w) if pan else None
    out = out_dir / name
    to_vertical(ffmpeg, horizontal, ass, out, pan_x, crop_w)
    return out
