"""Video downloader — the first pipeline stage.

Built on yt-dlp with quality-first defaults for YouTube Shorts clipping.
Handles regular videos, Shorts, and playlists. Returns structured metadata
for downstream pipeline stages (transcriber, scorer, renderer).

Quality model
-------------
FastClip downloads the HIGHEST quality by default. Rationale: we crop a
16:9 landscape source into a 9:16 vertical Short, keeping only the vertical
slice. A 1080p source yields only 1080 vertical pixels; a 4K source yields
2160 — far sharper after the crop. So maximum source resolution directly
determines output quality.

Speed toggle: pass ``max_height`` (e.g. 720) to cap resolution and trade
quality for download speed during rapid iteration.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yt_dlp

# JS runtimes yt-dlp can use to generate YouTube "PO tokens" for 1080p+
# formats (ordered by startup speed / common availability).
_JS_RUNTIMES = ("node", "bun", "deno", "quickjs")


def _detect_js_runtimes() -> dict[str, dict[str, str]]:
    """Return a yt-dlp ``js_runtimes`` config for the first runtime on PATH.

    YouTube now requires a JS-runtime-generated "PO token" to serve 1080p+
    formats; without one those streams return HTTP 403. This auto-detects
    node/bun/deno/quickjs so the downloader "just works" at max quality.
    """
    for name in _JS_RUNTIMES:
        path = shutil.which(name)
        if path:
            return {name: {"path": path}}
    return {}


# ── Data model ──────────────────────────────────────────────────────────────────


@dataclass
class VideoMetadata:
    """Structured metadata for a downloaded video."""

    video_id: str
    title: str
    description: str
    duration: float  # seconds
    width: int
    height: int
    fps: float
    filepath: Path
    upload_date: str | None = None  # YYYYMMDD
    channel: str | None = None
    channel_url: str | None = None
    view_count: int | None = None
    thumbnail_url: str | None = None
    tags: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    is_short: bool = False  # detected from URL or aspect ratio
    video_codec: str | None = None  # e.g. avc1, vp9, av01
    audio_codec: str | None = None  # e.g. mp4a, opus
    is_hdr: bool = False  # HDR10/HLG source (needs tone-mapping downstream)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0.0

    @property
    def is_vertical(self) -> bool:
        return self.height > self.width

    @property
    def duration_str(self) -> str:
        mins, secs = divmod(int(self.duration), 60)
        return f"{mins}:{secs:02d}"


# ── Callback protocol ───────────────────────────────────────────────────────────

ProgressCallback = Callable[[dict[str, Any]], None]


# ── Downloader ──────────────────────────────────────────────────────────────────


class Downloader:
    """Quality-first YouTube video downloader.

    Downloads best video + best audio (merged via ffmpeg) at the highest
    resolution available, preferring H.264/AAC (mp4/m4a) containers so the
    result is readable by MoviePy/OpenCV. Uses yt-dlp's concurrent fragment
    downloads for parallelized chunk retrieval.

    Usage:
        dl = Downloader(output_dir=Path("./videos"))       # highest quality
        meta = dl.download("https://youtube.com/watch?v=...")
        print(meta.title, meta.duration_str, meta.filepath)

        fast = Downloader(output_dir=Path("./videos"), max_height=720)  # speed
    """

    # Allowed explicit height caps (None = no cap = highest quality)
    MAX_HEIGHT_OPTIONS = {360, 480, 720, 1080, 1440, 2160, 4320}

    def __init__(
        self,
        output_dir: Path,
        *,
        max_height: int | None = None,
        prefer_compatible: bool = True,
        concurrent_fragments: int = 8,
        throttle_rate: int | None = None,  # bytes/sec limit, None = no limit
        progress_callback: ProgressCallback | None = None,
        use_js_runtime: bool = True,
        cookies_file: str | Path | None = None,
        cookies_from_browser: str | None = None,
        quiet: bool = False,
    ) -> None:
        """
        Args:
            output_dir: Where to save downloaded videos.
            max_height: Maximum video height. None (default) = highest
                        quality available. Set e.g. 720 to download faster.
            prefer_compatible: Prefer H.264/AAC (mp4/m4a) over VP9/AV1 so
                        downstream tools (MoviePy, OpenCV) can always read
                        the file. Set False for the literal best codec.
            concurrent_fragments: Parallel fragment downloads for HLS/DASH.
            throttle_rate: Optional download speed limit in bytes/sec.
            progress_callback: Called with yt-dlp progress dict on every
                               progress update. Use for Rich progress bars.
            use_js_runtime: Auto-detect a JS runtime (node/bun/deno) so
                        yt-dlp can generate PO tokens for 1080p+ formats.
                        Disable only if no JS runtime is available.
            cookies_file: Path to a Netscape-format cookies.txt exported from
                        a browser logged into YouTube. Required (alongside the
                        bgutil PO token) for 1080p+ formats since YouTube's
                        SABR rollout. Without it, downloads fall back to 360p.
            cookies_from_browser: Browser name to read YouTube login cookies
                        from directly (e.g. "chrome", "firefox", "edge").
                        Alternative to cookies_file — no manual export needed.
            quiet: Suppress yt-dlp output.
        """
        if max_height is not None and max_height not in self.MAX_HEIGHT_OPTIONS:
            raise ValueError(f"max_height must be one of {sorted(self.MAX_HEIGHT_OPTIONS)} or None")
        self.output_dir = Path(output_dir)
        self.max_height = max_height
        self.prefer_compatible = prefer_compatible
        self.concurrent_fragments = concurrent_fragments
        self.throttle_rate = throttle_rate
        self.progress_callback = progress_callback
        self.use_js_runtime = use_js_runtime
        self.cookies_file = Path(cookies_file) if cookies_file else None
        self.cookies_from_browser = cookies_from_browser
        self.quiet = quiet
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ──────────────────────────────────────────────────────────

    def download(self, url: str) -> VideoMetadata:
        """Download a video at highest quality and return its metadata.

        Args:
            url: YouTube video or Short URL.

        Returns:
            VideoMetadata with the downloaded filepath and all extracted info.
        """
        ydl_opts = self._build_options()
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return self._parse_info(info)

    def probe(self, url: str) -> VideoMetadata:
        """Extract metadata WITHOUT downloading.

        Uses yt-dlp's extract_info with download=False. Fast check before
        committing to a full download — validate the video exists, check
        duration, resolution, codec, HDR, etc.

        Args:
            url: YouTube video or Short URL.

        Returns:
            VideoMetadata (filepath empty — not downloaded).
        """
        ydl_opts = self._build_options()
        ydl_opts["quiet"] = True
        ydl_opts["no_warnings"] = True
        ydl_opts["noplaylist"] = True
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            result = self._parse_info(info)
            result.filepath = Path()  # not downloaded
            return result

    # ── Internals ───────────────────────────────────────────────────────────

    def _build_format(self) -> str:
        """Build the yt-dlp format selector string.

        Strategy (highest quality, downstream-compatible):
            1. Best H.264 video + best AAC audio (mp4/m4a) — universally
               readable by MoviePy/OpenCV.
            2. Fall back to best any-codec video + best any audio.
            3. Fall back to a single best combined format.

        When max_height is set, every tier is capped at that height.
        """
        height = f"[height<={self.max_height}]" if self.max_height is not None else ""

        if self.prefer_compatible:
            return (
                f"bestvideo[ext=mp4]{height}+bestaudio[ext=m4a]/"
                f"bestvideo{height}+bestaudio/best{height}"
            )
        return f"bestvideo{height}+bestaudio/best{height}"

    def _build_options(self) -> dict[str, Any]:
        """Build yt-dlp options dict optimized for highest-quality download."""
        outtmpl = str(self.output_dir / "%(id)s.%(ext)s")

        options: dict[str, Any] = {
            "format": self._build_format(),
            "outtmpl": outtmpl,
            "merge_output_format": "mp4",
            "concurrent_fragment_downloads": self.concurrent_fragments,
            # Metadata extraction
            "writesubtitles": False,
            "writeautomaticsub": False,
            "writeinfojson": False,
            "writethumbnail": False,
            # Don't download playlists by default
            "noplaylist": True,
            # Speed
            "noprogress": True,  # we handle progress via callback
            "quiet": self.quiet,
            "no_warnings": self.quiet,
            "extract_flat": False,
            # Post-processing: yt-dlp handles the merge, nothing extra needed
            "postprocessors": [],
        }

        if self.throttle_rate is not None:
            options["throttledratelimit"] = self.throttle_rate

        # YouTube's SABR / GVS-PO-token rollout (yt-dlp issue #12482) broke 1080p+
        # downloads. Two auth signals are now needed for full HD: a bgutil PO token
        # AND a logged-in session (cookies). We support three tiers:
        #   - cookies + PO token  -> default web client, 1080p+ (best).
        #   - no cookies          -> `android` client, 360p fallback (no 403).
        if self.cookies_file:
            if self.cookies_file.exists():
                options["cookiefile"] = str(self.cookies_file)
                options["extractor_args"] = {"youtube": {"player_client": ["default"]}}
            else:
                raise FileNotFoundError(f"Cookies file not found: {self.cookies_file}")
        elif self.cookies_from_browser:
            options["cookiesfrombrowser"] = (self.cookies_from_browser,)
            options["extractor_args"] = {"youtube": {"player_client": ["default"]}}
        else:
            options["extractor_args"] = {"youtube": {"player_client": ["android"]}}

        # Enable PO token / anti-bot challenge solving for 1080p+ formats.
        # Needs BOTH a JS runtime (node/bun/deno) AND the EJS challenge-solver
        # script (a remote component fetched from GitHub). Without the remote
        # component, the JS runtime is skipped and YouTube returns HTTP 403.
        if self.use_js_runtime:
            js_runtimes = _detect_js_runtimes()
            if js_runtimes:
                options["js_runtimes"] = js_runtimes
                options["remote_components"] = ["ejs:github"]

        if self.progress_callback is not None:
            options["progress_hooks"] = [self._progress_hook]

        return options

    def _progress_hook(self, d: dict[str, Any]) -> None:
        """Internal yt-dlp progress hook → user callback.

        Re-raises DownloadCancelled so a callback can abort the download
        cleanly (used by the GUI's cancel button).
        """
        if self.progress_callback is not None:
            self.progress_callback(d)

    @staticmethod
    def _parse_info(info: dict[str, Any]) -> VideoMetadata:
        """Parse yt-dlp info dict into a clean VideoMetadata."""
        is_short = _detect_short(info)

        # Resolve the final filepath (merged mp4 after post-processing).
        filepath = info.get("_filename") or ""
        if not filepath:
            requested = info.get("requested_downloads") or []
            if requested:
                filepath = requested[0].get("filepath", "")

        video_codec = info.get("vcodec")
        audio_codec = info.get("acodec")

        # HDR detection: dynamic_range field (HDR10, HLG) or HDR in format.
        dyn_range = (info.get("dynamic_range") or "").upper()
        is_hdr = dyn_range in {"HDR10", "HDR10+", "HLG", "DOLBY VISION"}

        return VideoMetadata(
            video_id=info.get("id", ""),
            title=info.get("title", ""),
            description=info.get("description", ""),
            duration=info.get("duration", 0.0),
            width=info.get("width", 0),
            height=info.get("height", 0),
            fps=info.get("fps", 0.0),
            filepath=Path(filepath),
            upload_date=info.get("upload_date"),
            channel=info.get("channel") or info.get("uploader"),
            channel_url=info.get("channel_url") or info.get("uploader_url"),
            view_count=info.get("view_count"),
            thumbnail_url=info.get("thumbnail"),
            tags=info.get("tags", []) or [],
            categories=info.get("categories", []) or [],
            is_short=is_short,
            video_codec=video_codec,
            audio_codec=audio_codec,
            is_hdr=is_hdr,
        )


def _detect_short(info: dict[str, Any]) -> bool:
    """Detect whether a video is a YouTube Short."""
    # Method 1: Original URL contains /shorts/
    url = info.get("original_url", "")
    webpage_url = info.get("webpage_url", "")
    if "/shorts/" in f"{url}{webpage_url}":
        return True

    # Method 2: Vertical aspect ratio + short duration
    w = info.get("width", 0)
    h = info.get("height", 0)
    dur = info.get("duration", 0)
    if h > w and 0 < dur <= 61:
        return True

    return False