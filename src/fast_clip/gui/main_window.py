"""FastClip main window — URL input, settings, download + transcribe with progress."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QClipboard, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from fast_clip.gui.worker import DownloadThread, TranscribeThread, ScorerThread
from fast_clip.pipeline.downloader import VideoMetadata
from fast_clip.pipeline.scorer import ScoreResult
from fast_clip.pipeline.transcriber import Transcript


# ── Constants ───────────────────────────────────────────────────────────────────

QUALITY_OPTIONS = {
    "Max (highest)": None,
    "1080p (Full HD)": 1080,
    "720p (HD)": 720,
    "480p (SD)": 480,
    "360p (low)": 360,
}

# Practical whisper model sizes for the GUI (skip legacy large-v1/v2).
MODEL_OPTIONS = ("tiny", "base", "small", "medium", "large-v3")

WINDOW_TITLE = "FastClip — YouTube Shorts Engine"
WINDOW_SIZE = (640, 960)


# ── Main Window ─────────────────────────────────────────────────────────────────


class MainWindow(QMainWindow):
    """Main FastClip GUI window."""

    def __init__(self) -> None:
        super().__init__()
        self._dl_thread: DownloadThread | None = None
        self._tx_thread: TranscribeThread | None = None
        self._sc_thread: ScorerThread | None = None
        self._output_dir = Path.home() / "Videos" / "FastClip"
        self._last_video_path: Path | None = None
        self._last_transcript: Transcript | None = None

        self.setWindowTitle(WINDOW_TITLE)
        self.resize(*WINDOW_SIZE)
        self.setMinimumSize(520, 620)

        # Central widget + layout
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)

        # ── Title bar ──
        title = QLabel("FastClip")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title.setFont(title_font)
        root.addWidget(title)

        subtitle = QLabel("AI-powered YouTube Shorts engine")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #888; margin-bottom: 8px;")
        root.addWidget(subtitle)

        # ═══════════════════════════════════════════════════════════════════
        # DOWNLOAD section
        # ═══════════════════════════════════════════════════════════════════

        dl_group = QGroupBox("Download")
        dl_layout = QVBoxLayout(dl_group)
        dl_layout.setSpacing(8)

        # ── URL row ──
        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel("URL:"))
        self._url_input = QLineEdit()
        self._url_input.setPlaceholderText("https://youtube.com/watch?v=... or /shorts/...")
        self._url_input.returnPressed.connect(self._on_download)
        url_layout.addWidget(self._url_input, stretch=1)
        paste_btn = QPushButton("Paste")
        paste_btn.clicked.connect(self._on_paste)
        url_layout.addWidget(paste_btn)
        dl_layout.addLayout(url_layout)

        # ── Output + Quality row ──
        settings = QHBoxLayout()
        settings.addWidget(QLabel("Output:"))
        self._output_label = QLabel(str(self._output_dir))
        self._output_label.setStyleSheet("color: #aaa;")
        self._output_label.setToolTip(str(self._output_dir))
        settings.addWidget(self._output_label, stretch=1)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._on_browse)
        settings.addWidget(browse_btn)
        dl_layout.addLayout(settings)

        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Quality:"))
        self._quality_combo = QComboBox()
        self._quality_combo.addItems(QUALITY_OPTIONS.keys())
        self._quality_combo.setCurrentText("Max (highest)")
        quality_layout.addWidget(self._quality_combo)
        quality_layout.addStretch()
        dl_layout.addLayout(quality_layout)

        # ── Download button ──
        self._dl_btn = QPushButton("Download")
        self._dl_btn.setMinimumHeight(40)
        self._dl_btn.setStyleSheet("font-size: 14px; font-weight: bold;")
        self._dl_btn.clicked.connect(self._on_download)
        dl_layout.addWidget(self._dl_btn)

        # ── Progress bar ──
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("Ready")
        dl_layout.addWidget(self._progress_bar)

        # Status label
        self._status = QLabel("Ready — paste a YouTube URL to start.")
        self._status.setStyleSheet("color: #666;")
        dl_layout.addWidget(self._status)

        # ── Video info ──
        info_group = QGroupBox("Video Info")
        info_layout = QVBoxLayout(info_group)
        self._info_text = QTextEdit()
        self._info_text.setReadOnly(True)
        self._info_text.setMaximumHeight(130)
        self._info_text.setStyleSheet("background: #1a1a1a; border: 1px solid #333;")
        info_layout.addWidget(self._info_text)
        dl_layout.addWidget(info_group)

        root.addWidget(dl_group)

        # ═══════════════════════════════════════════════════════════════════
        # TRANSCRIBE section
        # ═══════════════════════════════════════════════════════════════════

        tx_group = QGroupBox("Transcribe")
        tx_layout = QVBoxLayout(tx_group)
        tx_layout.setSpacing(8)

        # ── Model + VAD row ──
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self._model_combo = QComboBox()
        self._model_combo.addItems(MODEL_OPTIONS)
        self._model_combo.setCurrentText("base")
        self._model_combo.setToolTip(
            "tiny=fastest (75MB) | base=balanced (145MB) | "
            "small=accurate (484MB) | large-v3=best (3GB)"
        )
        model_row.addWidget(self._model_combo)
        self._vad_check = QCheckBox("Skip silence (VAD)")
        self._vad_check.setChecked(True)
        self._vad_check.setToolTip(
            "Enable to skip non-speech regions. "
            "Disable for music-heavy videos (VAD can filter out everything)."
        )
        model_row.addWidget(self._vad_check)
        model_row.addStretch()
        tx_layout.addLayout(model_row)

        # ── Source file + button row ──
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("Source:"))
        self._tx_source_input = QLineEdit()
        self._tx_source_input.setReadOnly(True)
        self._tx_source_input.setPlaceholderText("Auto-filled after download")
        file_row.addWidget(self._tx_source_input, stretch=1)
        self._tx_btn = QPushButton("Transcribe")
        self._tx_btn.setMinimumHeight(36)
        self._tx_btn.setStyleSheet("font-weight: bold;")
        self._tx_btn.clicked.connect(self._on_transcribe)
        file_row.addWidget(self._tx_btn)
        tx_layout.addLayout(file_row)

        # ── Transcript preview ──
        self._transcript_text = QTextEdit()
        self._transcript_text.setReadOnly(True)
        self._transcript_text.setMinimumHeight(140)
        self._transcript_text.setStyleSheet("background: #1a1a1a; border: 1px solid #333;")
        self._transcript_text.setPlaceholderText("Transcript will appear here...")
        tx_layout.addWidget(self._transcript_text)

        root.addWidget(tx_group)

        # ═══════════════════════════════════════════════════════════════════
        # SCORE section
        # ═══════════════════════════════════════════════════════════════════

        sc_group = QGroupBox("AI Clip Scoring")
        sc_layout = QVBoxLayout(sc_group)
        sc_layout.setSpacing(8)

        score_btn_row = QHBoxLayout()
        self._score_btn = QPushButton("Score Clips with AI")
        self._score_btn.setMinimumHeight(36)
        self._score_btn.setStyleSheet("font-weight: bold; color: #0ff;")
        self._score_btn.setToolTip("Send the transcript to DeepSeek and get clip recommendations with titles, hooks, and timestamps.")
        self._score_btn.clicked.connect(self._on_score)
        score_btn_row.addWidget(self._score_btn)
        score_btn_row.addStretch()
        sc_layout.addLayout(score_btn_row)

        self._clips_text = QTextEdit()
        self._clips_text.setReadOnly(True)
        self._clips_text.setMinimumHeight(160)
        self._clips_text.setStyleSheet("background: #1a1a1a; border: 1px solid #333;")
        self._clips_text.setPlaceholderText("Clip recommendations will appear here...")
        sc_layout.addWidget(self._clips_text)

        root.addWidget(sc_group)

    # ── Slots: Download ────────────────────────────────────────────────────

    def _on_paste(self) -> None:
        """Paste from clipboard into URL field."""
        clipboard: QClipboard = QApplication.clipboard()
        text = clipboard.text()
        if text:
            self._url_input.setText(text.strip())

    def _on_browse(self) -> None:
        """Open directory picker for output location."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", str(self._output_dir)
        )
        if path:
            self._output_dir = Path(path)
            self._output_label.setText(path)
            self._output_label.setToolTip(path)

    def _on_download(self) -> None:
        """Start download or cancel current."""
        if self._dl_thread is not None and self._dl_thread.isRunning():
            self._dl_thread.cancel()
            return

        url = self._url_input.text().strip()
        if not url:
            self._set_status("Enter a YouTube URL first.", error=True)
            return

        quality_label = self._quality_combo.currentText()
        max_height = QUALITY_OPTIONS[quality_label]

        self._dl_thread = DownloadThread(url, self._output_dir, max_height)
        self._dl_thread.progress.connect(self._on_progress)
        self._dl_thread.finished_ok.connect(self._on_dl_finished)
        self._dl_thread.error.connect(self._on_dl_error)
        self._dl_thread.cancelled.connect(self._on_dl_cancelled)
        self._dl_thread.finished.connect(self._on_dl_cleanup)

        self._set_downloading(True)
        self._dl_thread.start()

    def _on_dl_cancelled(self) -> None:
        """User cancelled the download."""
        self._reset_dl_ui()
        self._set_status("Download cancelled.")

    def _on_progress(self, d: dict) -> None:
        """Update progress bar from yt-dlp progress dict."""
        status = d.get("status", "")
        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            speed = d.get("speed") or 0
            eta = d.get("eta") or 0

            if total > 0:
                pct = int(downloaded / total * 100)
                self._progress_bar.setValue(pct)
                self._progress_bar.setFormat(f"{pct}%")

            speed_str = _fmt_speed(speed)
            eta_str = _fmt_eta(eta)
            self._set_status(f"Downloading... {speed_str}  |  ETA {eta_str}")
        elif status == "finished":
            self._set_status("Download complete — merging...")
            self._progress_bar.setFormat("Merging")
        elif status == "error":
            self._set_status("Download error", error=True)

    def _on_dl_finished(self, result: VideoMetadata) -> None:
        """Download complete — display metadata, stash path for transcribe."""
        self._reset_dl_ui()
        self._progress_bar.setValue(100)
        self._progress_bar.setFormat("Done")
        self._set_status(
            f"Saved: {result.filepath.name}  ({result.duration_str}, "
            f"{result.width}x{result.height})"
        )
        self._display_info(result)

        # Auto-fill the transcriber source field.
        self._last_video_path = result.filepath
        self._tx_source_input.setText(str(self._last_video_path))
        self._tx_source_input.setToolTip(str(self._last_video_path))

    def _on_dl_error(self, msg: str) -> None:
        self._reset_dl_ui()
        self._set_status(f"Error: {msg}", error=True)

    def _on_dl_cleanup(self) -> None:
        """Reap the finished download thread: join it, then drop our reference.

        Runs on the QThread ``finished`` signal (emitted after ``run()`` has
        returned). ``wait()`` here is instant — it just guarantees the native
        thread has fully joined before we release the Python reference, so the
        GC can't destroy a still-running QThread.
        """
        if self._dl_thread is not None:
            self._dl_thread.wait()
            self._dl_thread = None

    # ── Slots: Transcribe ──────────────────────────────────────────────────

    def _on_transcribe(self) -> None:
        """Start transcription or cancel current."""
        if self._tx_thread is not None and self._tx_thread.isRunning():
            return  # already transcribing; could cancel but whisper doesn't support it

        if self._last_video_path is None:
            self._set_status("Download a video first, then transcribe.", error=True)
            return

        if not self._last_video_path.exists():
            self._set_status(
                f"File not found: {self._last_video_path.name}", error=True
            )
            return

        model = self._model_combo.currentText()
        vad = self._vad_check.isChecked()

        self._tx_thread = TranscribeThread(
            self._last_video_path,
            model_size=model,
            vad_filter=vad,
        )
        self._tx_thread.finished_ok.connect(self._on_tx_finished)
        self._tx_thread.error.connect(self._on_tx_error)
        self._tx_thread.finished.connect(self._on_tx_cleanup)

        self._tx_btn.setEnabled(False)
        self._tx_btn.setText("Transcribing...")
        self._set_status("Transcribing — loading model + processing audio...")
        self._tx_thread.start()

    def _on_tx_finished(self, transcript: Transcript) -> None:
        """Transcription done — display preview and auto-save."""
        self._tx_btn.setEnabled(True)
        self._tx_btn.setText("Transcribe")

        if not transcript.segments:
            self._transcript_text.setPlainText(
                "[No speech detected. Try disabling VAD for music-heavy video.]"
            )
            self._set_status("Transcription returned 0 segments. Try disabling VAD.", error=True)
            return

        # Show in text area
        self._transcript_text.setPlainText(transcript.to_txt())

        # Auto-save .txt + .srt next to the source file
        src = self._last_video_path
        stem = src.stem if src else "transcript"
        out_dir = src.parent if src else self._output_dir
        transcript.save(out_dir / f"{stem}.txt", "txt")
        transcript.save(out_dir / f"{stem}.srt", "srt")

        self._set_status(
            f"Transcribed: {transcript.language} "
            f"({transcript.language_probability:.0%}), "
            f"{len(transcript.segments)} segments — "
            f"saved {stem}.txt + {stem}.srt"
        )

        # Enable the score button — we have a transcript to work with.
        self._score_btn.setEnabled(True)
        self._last_transcript = transcript

    def _on_tx_error(self, msg: str) -> None:
        self._tx_btn.setEnabled(True)
        self._tx_btn.setText("Transcribe")
        self._set_status(f"Transcription error: {msg}", error=True)

    def _on_tx_cleanup(self) -> None:
        """Reap the finished transcription thread: join it, then drop our reference.

        See ``_on_dl_cleanup`` — same rationale. Runs on the QThread ``finished``
        signal so we never release the reference while the native thread lives.
        """
        if self._tx_thread is not None:
            self._tx_thread.wait()
            self._tx_thread = None

    # ── Slots: Score ──────────────────────────────────────────────────────

    def _on_score(self) -> None:
        """Start LLM clip scoring."""
        if self._sc_thread is not None and self._sc_thread.isRunning():
            return  # already scoring

        if self._last_transcript is None:
            self._set_status("Transcribe a video first, then score.", error=True)
            return

        self._sc_thread = ScorerThread(self._last_transcript)
        self._sc_thread.finished_ok.connect(self._on_sc_finished)
        self._sc_thread.error.connect(self._on_sc_error)
        self._sc_thread.finished.connect(self._on_sc_cleanup)

        self._score_btn.setEnabled(False)
        self._score_btn.setText("Scoring with AI...")
        self._set_status("Scoring transcript with DeepSeek — finding the best clips...")
        self._sc_thread.start()

    def _on_sc_finished(self, result: ScoreResult) -> None:
        """Scoring done — display clip recommendations."""
        self._score_btn.setEnabled(True)
        self._score_btn.setText("Score Clips with AI")

        if not result.clips:
            self._clips_text.setPlainText("[No clips identified — try a different video.]")
            self._set_status("Scoring returned no clips.", error=True)
            return

        lines = []
        for i, clip in enumerate(result.clips, 1):
            dur = clip.end - clip.start
            lines.append(f"#{i}  [{clip.start:.1f}s → {clip.end:.1f}s, {dur:.0f}s]  {clip.title}")
            lines.append(f"    Hook: {clip.hook}")
            lines.append(f"    Caption: {clip.caption}")
            lines.append(f"    Why: {clip.reasoning}")
            lines.append("")

        tokens = result.tokens_used
        if tokens:
            lines.append(
                f"Tokens: {tokens['input']:,} in / {tokens['output']:,} out "
                f"({tokens['total']:,} total)"
            )

        self._clips_text.setPlainText("\n".join(lines))
        self._set_status(
            f"Scored: {len(result.clips)} clips found "
            f"({tokens.get('total', 0):,} LLM tokens)"
        )

    def _on_sc_error(self, msg: str) -> None:
        self._score_btn.setEnabled(True)
        self._score_btn.setText("Score Clips with AI")
        self._set_status(f"Scoring error: {msg}", error=True)

    def _on_sc_cleanup(self) -> None:
        """Reap the finished scorer thread."""
        if self._sc_thread is not None:
            self._sc_thread.wait()
            self._sc_thread = None

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _set_downloading(self, active: bool) -> None:
        """Toggle download UI between idle and active states."""
        self._url_input.setEnabled(not active)
        self._quality_combo.setEnabled(not active)
        self._tx_btn.setEnabled(not active)
        if active:
            self._dl_btn.setText("Cancel")
        else:
            self._dl_btn.setText("Download")

    def _set_status(self, msg: str, *, error: bool = False) -> None:
        self._status.setText(msg)
        self._status.setStyleSheet(
            "color: #e44;" if error else "color: #666;"
        )

    def _reset_dl_ui(self) -> None:
        """Return download UI to idle state."""
        self._set_downloading(False)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("Ready")

    def _display_info(self, meta: VideoMetadata) -> None:
        """Populate the video info box."""
        lines = [
            f"Title:      {meta.title}",
            f"Channel:    {meta.channel or '—'}",
            f"Duration:   {meta.duration_str}",
            f"Resolution: {meta.width}x{meta.height} @ {meta.fps:.0f}fps",
            f"Codec:      {meta.video_codec or '—'} / {meta.audio_codec or '—'}",
            f"File:       {meta.filepath}",
        ]
        if meta.is_hdr:
            lines.append("HDR:        Yes (needs tone-mapping for Shorts)")
        if meta.is_short:
            lines.append("Type:       YouTube Short")
        self._info_text.setPlainText("\n".join(lines))


# ── Formatting helpers ──────────────────────────────────────────────────────────


def _fmt_speed(speed: float) -> str:
    """Format bytes/sec into human-readable string."""
    if speed is None or speed == 0:
        return "—"
    if speed >= 1_000_000:
        return f"{speed / 1_000_000:.1f} MB/s"
    if speed >= 1_000:
        return f"{speed / 1_000:.0f} KB/s"
    return f"{speed:.0f} B/s"


def _fmt_eta(eta: int) -> str:
    """Format ETA in seconds to mm:ss or h:mm:ss."""
    if eta is None or eta <= 0:
        return "—"
    if eta >= 3600:
        return f"{eta // 3600}:{(eta % 3600) // 60:02d}:{eta % 60:02d}"
    return f"{eta // 60}:{eta % 60:02d}"