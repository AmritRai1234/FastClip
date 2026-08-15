"""FastAPI routes for FastClip — REST API over the clip pipeline."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from fast_clip.api.jobs import JobManager
from fast_clip.utils.llm import DEFAULT_MODEL


class ClipRequest(BaseModel):
    url: str
    whisper_model: str = "base"
    llm_model: str | None = DEFAULT_MODEL
    pan: bool = True
    language: str | None = None
    cookies_file: str | None = None
    cookies_from_browser: str | None = None


class JobOut(BaseModel):
    id: str
    url: str
    status: str
    stage: str
    current: int
    total: int
    message: str
    shorts: list[dict]
    error: str | None
    created_at: float


def create_app(jobs_dir: Path | None = None) -> FastAPI:
    manager = JobManager(jobs_dir or Path("output/jobs"))

    app = FastAPI(title="FastClip", version="0.1.0")

    # Allow the Next.js dev server (and any local frontend) to call us.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/api/jobs", response_model=JobOut)
    async def create_job(req: ClipRequest) -> JobOut:
        job = manager.create(
            req.url,
            whisper_model=req.whisper_model,
            llm_model=req.llm_model,
            pan=req.pan,
            language=req.language,
            cookies_file=req.cookies_file,
            cookies_from_browser=req.cookies_from_browser,
        )
        return JobOut(**job.to_dict())

    @app.post("/api/upload", response_model=JobOut)
    async def upload_video(
        file: UploadFile = File(...),
        whisper_model: str = Form("base"),
        llm_model: str | None = Form(DEFAULT_MODEL),
        pan: bool = Form(True),
    ) -> JobOut:
        """Accept an uploaded video file and run the pipeline on it directly."""
        uploads = manager.jobs_dir.parent / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        safe_name = Path(file.filename or "upload.mp4").name  # strip any path
        dest = uploads / safe_name
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)

        job = manager.create(
            str(dest),
            whisper_model=whisper_model,
            llm_model=llm_model,
            pan=pan,
            language=None,
        )
        return JobOut(**job.to_dict())

    @app.get("/api/jobs", response_model=list[JobOut])
    async def list_jobs() -> list[JobOut]:
        return [JobOut(**j.to_dict()) for j in manager.list()]

    @app.get("/api/jobs/{job_id}", response_model=JobOut)
    async def get_job(job_id: str) -> JobOut:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return JobOut(**job.to_dict())

    @app.get("/api/jobs/{job_id}/shorts/{filename}")
    async def get_short(job_id: str, filename: str) -> FileResponse:
        path = manager.short_path(job_id, filename)
        if path is None:
            raise HTTPException(status_code=404, detail="short not found")
        return FileResponse(path, media_type="video/mp4")

    app.state.manager = manager
    return app
