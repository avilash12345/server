"""FastAPI application for the TubeFlow YouTube downloader."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import quote, urlparse

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .config import FFMPEG_PATH
from .jobs import TooManyJobsError, job_store
from .youtube import (
    InvalidUrlError,
    VideoUnavailableError,
    extract_video_info,
    normalize_url,
    youtube_tools_version,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("tubeflow")

APP_VERSION = "1.0.0"

ALLOWED_THUMBNAIL_HOSTS = (
    "ytimg.com",
    "ggpht.com",
    "googleusercontent.com",
    "youtube.com",
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    job_store.start()
    logger.info("TubeFlow API ready (yt-dlp %s)", youtube_tools_version())
    try:
        yield
    finally:
        job_store.stop()


app = FastAPI(
    title="TubeFlow API",
    description=(
        "A small, self-hosted YouTube downloader API. Inspect available video "
        "resolutions and audio formats, then download a merged file."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Length"],
)


class VideoFormatOut(BaseModel):
    height: int
    fps: int | None = None
    ext: str
    format_id: str | None = None
    vcodec: str | None = None
    acodec: str | None = None
    filesize: int | None = None
    needs_merge: bool
    container: str
    label: str
    quality: str


class AudioFormatOut(BaseModel):
    id: str
    label: str
    description: str
    ext: str
    abr: int | None = None
    filesize: int | None = None
    converted: bool
    recommended: bool


class VideoInfoOut(BaseModel):
    id: str | None = None
    title: str
    description: str
    author: str
    channel_url: str | None = None
    thumbnail: str | None = None
    duration: int | None = None
    view_count: int | None = None
    like_count: int | None = None
    upload_date: str | None = None
    webpage_url: str
    is_live: bool
    video_formats: list[VideoFormatOut]
    audio_formats: list[AudioFormatOut]


class DownloadRequest(BaseModel):
    url: str = Field(min_length=3, description="A YouTube URL or video id")
    kind: str = Field(default="video", pattern="^(video|audio)$")
    height: int | None = Field(default=1080, ge=144, le=4320)
    container: str | None = Field(default="mp4", pattern="^(mp4|webm)$")
    audio_id: str | None = Field(default="mp3", pattern="^(mp3|m4a|opus)$")


class JobOut(BaseModel):
    id: str
    status: str
    progress: float
    message: str
    kind: str
    quality: str
    height: int | None = None
    container: str | None = None
    audio_id: str | None = None
    title: str | None = None
    thumbnail: str | None = None
    duration: int | None = None
    filename: str | None = None
    filesize: int | None = None
    speed: str | None = None
    eta: int | None = None
    downloaded_bytes: int | None = None
    total_bytes: int | None = None
    error: str | None = None
    download_url: str | None = None
    created_at: float
    updated_at: float


def _base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/")


def _public_job(job: Any, request: Request) -> dict[str, Any]:
    return job.public(_base_url(request))


@app.get("/", tags=["meta"])
async def root() -> dict[str, Any]:
    return {
        "name": "TubeFlow API",
        "version": APP_VERSION,
        "docs": "/docs",
        "endpoints": {
            "info": "GET /api/info?url=<youtube-url>",
            "create_download": "POST /api/downloads",
            "job_status": "GET /api/downloads/{id}",
            "download_file": "GET /api/downloads/{id}/file",
        },
    }


@app.get("/api/health", tags=["meta"])
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": APP_VERSION,
        "yt_dlp": youtube_tools_version(),
        "ffmpeg": FFMPEG_PATH,
        "ffmpeg_available": Path(FFMPEG_PATH).is_file(),
    }


@app.get("/api/info", response_model=VideoInfoOut, tags=["youtube"])
async def video_info(url: str = Query(..., description="YouTube URL or video id")):
    try:
        info = await run_in_threadpool(extract_video_info, url)
    except InvalidUrlError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except VideoUnavailableError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return info


@app.post("/api/downloads", response_model=JobOut, status_code=202, tags=["downloads"])
async def create_download(payload: DownloadRequest, request: Request):
    try:
        job = job_store.create(
            url=payload.url,
            kind=payload.kind,
            height=payload.height,
            container=payload.container,
            audio_id=payload.audio_id,
        )
    except InvalidUrlError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TooManyJobsError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _public_job(job, request)


@app.get("/api/downloads/{job_id}", response_model=JobOut, tags=["downloads"])
async def download_status(job_id: str, request: Request):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Download job not found.")
    return _public_job(job, request)


@app.get("/api/downloads", response_model=list[JobOut], tags=["downloads"])
async def list_downloads(request: Request, limit: int = Query(default=20, ge=1, le=100)):
    jobs = job_store.recent(limit)
    return [_public_job(job, request) for job in jobs]


@app.get("/api/downloads/{job_id}/file", tags=["downloads"])
async def download_file(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Download job not found.")
    if job.status != "ready" or job.filepath is None or not job.filepath.is_file():
        raise HTTPException(status_code=409, detail="This file is not ready yet.")
    if job.cancel_requested:
        raise HTTPException(status_code=410, detail="This download was cancelled.")

    suffix = job.filepath.suffix.lstrip(".").lower()
    media_types = {
        "mp4": "video/mp4",
        "webm": "video/webm",
        "m4a": "audio/mp4",
        "mp3": "audio/mpeg",
        "opus": "audio/ogg",
    }
    filename = job.filename or f"tubeflow-{job.quality_label.lower()}.{suffix}"
    headers = {
        "Content-Disposition": (
            f'attachment; filename="tubeflow-{job.quality_label.lower()}.{suffix}"; '
            f"filename*=UTF-8''{quote(filename)}"
        ),
        "Cache-Control": "no-store",
        "Access-Control-Expose-Headers": "Content-Disposition, Content-Length",
    }
    return FileResponse(
        path=job.filepath,
        media_type=media_types.get(suffix, "application/octet-stream"),
        headers=headers,
    )


@app.delete("/api/downloads/{job_id}", tags=["downloads"])
async def delete_download(job_id: str):
    removed = job_store.remove(job_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Download job not found.")
    return {"ok": True}


@app.get("/api/thumbnail", tags=["youtube"])
async def proxy_thumbnail(url: str = Query(..., min_length=8)):
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or not any(
        host == allowed or host.endswith(f".{allowed}") for allowed in ALLOWED_THUMBNAIL_HOSTS
    ):
        raise HTTPException(status_code=400, detail="Only YouTube thumbnail URLs are allowed.")

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
            response = await client.get(url, headers={"User-Agent": "TubeFlow/1.0"})
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Could not load the thumbnail.") from exc

    content_type = response.headers.get("content-type", "image/jpeg")
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=502, detail="Thumbnail response was not an image.")
    if len(response.content) > 8 * 1024 * 1024:
        raise HTTPException(status_code=502, detail="Thumbnail is too large.")

    return Response(
        content=response.content,
        media_type=content_type,
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/api/validate", tags=["youtube"])
async def validate_url(url: str = Query(...)):
    try:
        return {"valid": True, "normalized_url": normalize_url(url)}
    except InvalidUrlError as exc:
        return {"valid": False, "error": str(exc)}
