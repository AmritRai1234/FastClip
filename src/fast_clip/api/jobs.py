"""Job manager — runs clip pipelines in background threads.

Each job runs the orchestrator's ``run_pipeline`` in a daemon thread and records
its progress in a dict the API can poll. Simple and local-first (no queue broker).
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fast_clip.pipeline.orchestrator import run_pipeline


@dataclass
class Job:
    id: str
    url: str
    status: str = "queued"  # queued | running | done | error
    stage: str = ""
    current: int = 0
    total: int = 0
    message: str = ""
    shorts: list[dict] = field(default_factory=list)
    error: str | None = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "url": self.url,
            "status": self.status,
            "stage": self.stage,
            "current": self.current,
            "total": self.total,
            "message": self.message,
            "shorts": self.shorts,
            "error": self.error,
            "created_at": self.created_at,
        }


class JobManager:
    """In-memory job registry with a lock for cross-thread access."""

    def __init__(self, jobs_dir: Path) -> None:
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def _set(self, job: Job, **fields: Any) -> None:
        with self._lock:
            for k, v in fields.items():
                setattr(job, k, v)

    def create(self, url: str, *, whisper_model: str, llm_model: str | None, pan: bool, language: str | None, cookies_file: str | None = None, cookies_from_browser: str | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], url=url)
        with self._lock:
            self._jobs[job.id] = job
        thread = threading.Thread(
            target=self._run,
            args=(job,),
            kwargs={
                "whisper_model": whisper_model,
                "llm_model": llm_model,
                "pan": pan,
                "language": language,
                "cookies_file": cookies_file,
                "cookies_from_browser": cookies_from_browser,
            },
            daemon=True,
        )
        thread.start()
        return job

    def _run(self, job: Job, *, whisper_model: str, llm_model: str | None, pan: bool, language: str | None, cookies_file: str | None = None, cookies_from_browser: str | None = None) -> None:
        self._set(job, status="running", stage="queued", message="starting")

        def _progress(d: dict[str, Any]) -> None:
            self._set(job, stage=d["stage"], current=d["current"], total=d["total"], message=d["message"])

        out_dir = self.jobs_dir / job.id
        try:
            results = run_pipeline(
                job.url,
                output_dir=out_dir,
                whisper_model=whisper_model,
                llm_model=llm_model,
                cookies_file=cookies_file,
                cookies_from_browser=cookies_from_browser,
                pan=pan,
                language=language,
                progress=_progress,
            )
            shorts = [
                {
                    "title": r["title"],
                    "hook": r["hook"],
                    "duration": r["duration"],
                    "section": r["section"],
                    "section_title": r["section_title"],
                    "filename": r["path"].name,
                    "url": f"/api/jobs/{job.id}/shorts/{r['path'].name}",
                }
                for r in results
            ]
            self._set(job, status="done", stage="done", shorts=shorts, message=f"{len(shorts)} shorts")
        except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
            self._set(job, status="error", error=str(exc), message="failed")

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def short_path(self, job_id: str, filename: str) -> Path | None:
        job = self.get(job_id)
        if job is None:
            return None
        path = self.jobs_dir / job_id / "shorts" / filename
        # Guard against path traversal via filename.
        if path.resolve().parent != (self.jobs_dir / job_id / "shorts").resolve():
            return None
        return path if path.exists() else None
