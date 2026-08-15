"""Background worker threads — run long pipeline stages without blocking the UI."""

from __future__ import annotations

import threading
from pathlib import Path

import yt_dlp
from PySide6.QtCore import QThread, Signal

from fast_clip.pipeline.downloader import Downloader, VideoMetadata
from fast_clip.pipeline.scorer import Scorer, ScoreResult
from fast_clip.pipeline.transcriber import Transcriber, Transcript


class DownloadThread(QThread):
    """Downloads a video in a background thread.

    Subclasses QThread directly (the simplest, most robust pattern for a
    one-shot worker). Override ``run()`` and emit signals — they are
    auto-delivered to the UI thread via queued connections.

    Usage (from the UI thread):

        thread = DownloadThread(url, output_dir, max_height)
        thread.finished_ok.connect(on_done)     # VideoMetadata
        thread.error.connect(on_error)          # str
        thread.progress.connect(on_progress)    # dict
        thread.start()
    """

    progress = Signal(dict)  # raw yt-dlp progress dict
    finished_ok = Signal(object)  # VideoMetadata
    error = Signal(str)  # error message
    cancelled = Signal()  # emitted on user cancellation

    def __init__(
        self,
        url: str,
        output_dir: Path,
        max_height: int | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._url = url
        self._output_dir = Path(output_dir)
        self._max_height = max_height
        self._cancel_event = threading.Event()

    def run(self) -> None:
        """Execute the download (runs in the worker thread)."""
        try:
            dl = Downloader(
                output_dir=self._output_dir,
                max_height=self._max_height,
                quiet=True,
                progress_callback=self._on_progress,
            )
            result = dl.download(self._url)
            self.finished_ok.emit(result)
        except yt_dlp.utils.DownloadCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.error.emit(str(exc))

    def _on_progress(self, d: dict) -> None:
        """Forward yt-dlp progress and honor cancellation requests."""
        if self._cancel_event.is_set():
            raise yt_dlp.utils.DownloadCancelled("Cancelled by user")
        self.progress.emit(d)

    def cancel(self) -> None:
        """Request cancellation — aborts on the next progress tick."""
        self._cancel_event.set()


class TranscribeThread(QThread):
    """Transcribes a local media file in a background thread.

    Same one-shot QThread-subclass pattern as ``DownloadThread``: override
    ``run()``, emit ``finished_ok``/``error`` signals, and they are delivered
    to the UI thread via queued connections.

    Usage (from the UI thread):

        thread = TranscribeThread(media_path, model_size="base", vad_filter=True)
        thread.finished_ok.connect(on_done)   # Transcript
        thread.error.connect(on_error)        # str
        thread.start()
    """

    finished_ok = Signal(object)  # Transcript
    error = Signal(str)  # error message

    def __init__(
        self,
        media_path: Path,
        *,
        model_size: str = "base",
        vad_filter: bool = True,
        language: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._media_path = Path(media_path)
        self._model_size = model_size
        self._vad_filter = vad_filter
        self._language = language

    def run(self) -> None:
        """Transcribe the file (runs in the worker thread)."""
        try:
            tr = Transcriber(model_size=self._model_size, vad_filter=self._vad_filter)
            result = tr.transcribe(self._media_path, language=self._language)
            self.finished_ok.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class ScorerThread(QThread):
    """Scores a transcript with the LLM in a background thread.

    Same one-shot QThread-subclass pattern. The ``Scorer`` class auto-loads
    the OpenRouter API key from the environment on first use.

    Usage (from the UI thread):

        thread = ScorerThread(transcript, llm_model="deepseek/deepseek-v4-flash")
        thread.finished_ok.connect(on_done)   # ScoreResult
        thread.error.connect(on_error)        # str
        thread.start()
    """

    finished_ok = Signal(object)  # ScoreResult
    error = Signal(str)  # error message

    def __init__(
        self,
        transcript: Transcript,
        *,
        llm_model: str = "deepseek/deepseek-v4-flash",
        temperature: float = 0.7,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._transcript = transcript
        self._llm_model = llm_model
        self._temperature = temperature

    def run(self) -> None:
        """Score the transcript via LLM (runs in the worker thread)."""
        try:
            scorer = Scorer(model=self._llm_model, temperature=self._temperature)
            result = scorer.score(self._transcript)
            self.finished_ok.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))