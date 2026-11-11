"""Background download jobs with progress, cancellation, and cleanup."""

from __future__ import annotations

import logging
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from .config import (
    DOWNLOAD_DIR,
    JOB_TTL_SECONDS,
    MAX_ACTIVE_JOBS,
    MAX_CONCURRENT_DOWNLOADS,
)
from .youtube import (
    VideoUnavailableError,
    _human_error,
    build_download_options,
    normalize_url,
    sanitize_filename,
)

logger = logging.getLogger("tubeflow.jobs")

ACTIVE_STATUSES = {"queued", "preparing", "downloading", "processing"}
FINAL_STATUSES = {"ready", "error", "cancelled"}


class TooManyJobsError(RuntimeError):
    pass


class DownloadCancelled(RuntimeError):
    pass


@dataclass
class DownloadJob:
    id: str
    url: str
    kind: str
    height: int | None = None
    container: str | None = None
    audio_id: str | None = None
    status: str = "queued"
    progress: float = 0.0
    message: str = "Waiting in queue"
    title: str | None = None
    thumbnail: str | None = None
    duration: int | None = None
    filename: str | None = None
    filesize: int | None = None
    error: str | None = None
    speed: str | None = None
    eta: int | None = None
    downloaded_bytes: int | None = None
    total_bytes: int | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    filepath: Path | None = None
    workdir: Path | None = None
    cancel_requested: bool = False

    @property
    def quality_label(self) -> str:
        if self.kind == "audio":
            return (self.audio_id or "mp3").upper()
        return f"{self.height or 1080}p"

    def public(self, base_url: str = "") -> dict[str, Any]:
        download_url = None
        if self.status == "ready":
            download_url = f"{base_url}/api/downloads/{self.id}/file"
        return {
            "id": self.id,
            "status": self.status,
            "progress": round(self.progress, 4),
            "message": self.message,
            "kind": self.kind,
            "quality": self.quality_label,
            "height": self.height,
            "container": self.container,
            "audio_id": self.audio_id,
            "title": self.title,
            "thumbnail": self.thumbnail,
            "duration": self.duration,
            "filename": self.filename,
            "filesize": self.filesize,
            "speed": self.speed,
            "eta": self.eta,
            "downloaded_bytes": self.downloaded_bytes,
            "total_bytes": self.total_bytes,
            "error": self.error,
            "download_url": download_url,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def _fmt_speed(value: float | None) -> str | None:
    if not value:
        return None
    units = ["B/s", "KB/s", "MB/s", "GB/s"]
    size = float(value)
    index = 0
    while size >= 1024 and index < len(units) - 1:
        size /= 1024
        index += 1
    return f"{size:.1f} {units[index]}"


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, DownloadJob] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=MAX_CONCURRENT_DOWNLOADS,
            thread_name_prefix="tubeflow-download",
        )
        self._stop_event = threading.Event()
        self._cleaner = threading.Thread(
            target=self._cleanup_loop,
            name="tubeflow-cleaner",
            daemon=True,
        )
        self._started = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        self._cleaner.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _cleanup_loop(self) -> None:
        while not self._stop_event.wait(60):
            cutoff = time.time() - JOB_TTL_SECONDS
            with self._lock:
                expired = [
                    job
                    for job in self._jobs.values()
                    if job.status in FINAL_STATUSES and job.updated_at < cutoff
                ]
            for job in expired:
                self._remove_job(job.id)

    def _active_count(self) -> int:
        return sum(1 for job in self._jobs.values() if job.status in ACTIVE_STATUSES)

    def create(
        self,
        *,
        url: str,
        kind: str,
        height: int | None,
        container: str | None,
        audio_id: str | None,
    ) -> DownloadJob:
        normalized = normalize_url(url)
        if kind not in {"video", "audio"}:
            raise ValueError("kind must be 'video' or 'audio'")

        with self._lock:
            if self._active_count() >= MAX_ACTIVE_JOBS:
                raise TooManyJobsError(
                    "The server is busy with other downloads. Please try again in a moment."
                )
            job = DownloadJob(
                id=uuid.uuid4().hex,
                url=normalized,
                kind=kind,
                height=height,
                container=container,
                audio_id=(audio_id or "mp3").lower() if kind == "audio" else None,
            )
            self._jobs[job.id] = job

        self._executor.submit(self._run, job)
        return job

    def get(self, job_id: str) -> DownloadJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, limit: int = 20) -> list[DownloadJob]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda job: job.created_at, reverse=True)
        return jobs[:limit]

    def cancel(self, job_id: str) -> DownloadJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status in ACTIVE_STATUSES:
                job.cancel_requested = True
                job.status = "cancelled"
                job.message = "Cancelled"
                job.updated_at = time.time()
            return job

    def remove(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None:
            return False
        if job.status in ACTIVE_STATUSES:
            self.cancel(job_id)
            return True
        self._remove_job(job_id)
        return True

    def _remove_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is not None:
            self._cleanup_files(job)

    @staticmethod
    def _cleanup_files(job: DownloadJob) -> None:
        if job.workdir and job.workdir.exists():
            shutil.rmtree(job.workdir, ignore_errors=True)
            job.workdir = None
            job.filepath = None

    def _update(self, job: DownloadJob, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = time.time()

    def _run(self, job: DownloadJob) -> None:
        job.workdir = DOWNLOAD_DIR / job.id
        job.workdir.mkdir(parents=True, exist_ok=True)
        output_template = str(job.workdir / "media.%(ext)s")

        def progress_hook(data: dict[str, Any]) -> None:
            if job.cancel_requested:
                raise DownloadCancelled("Download cancelled")

            status = data.get("status")
            if status == "downloading":
                total = data.get("total_bytes") or data.get("total_bytes_estimate")
                downloaded = data.get("downloaded_bytes") or 0
                ratio = 0.0
                if total:
                    ratio = min(0.94, max(0.0, downloaded / total * 0.94))
                self._update(
                    job,
                    status="downloading",
                    progress=ratio,
                    message=f"Downloading {job.quality_label}",
                    speed=_fmt_speed(data.get("speed")),
                    eta=data.get("eta"),
                    downloaded_bytes=downloaded,
                    total_bytes=int(total) if total else None,
                )
            elif status == "finished":
                self._update(
                    job,
                    progress=0.95,
                    message="Merging video and audio",
                    speed=None,
                    eta=None,
                )

        def postprocessor_hook(data: dict[str, Any]) -> None:
            if job.cancel_requested:
                raise DownloadCancelled("Download cancelled")
            if data.get("status") == "started":
                self._update(job, status="processing", progress=0.96, message="Processing media")
            elif data.get("status") == "finished":
                self._update(job, progress=0.99, message="Finalizing file")

        try:
            self._update(job, status="preparing", message="Reading video details")
            options = build_download_options(
                url=job.url,
                kind=job.kind,
                height=job.height,
                container=job.container,
                audio_id=job.audio_id,
                output_template=output_template,
                progress_hook=progress_hook,
                postprocessor_hook=postprocessor_hook,
            )

            with YoutubeDL(options) as ydl:
                info = ydl.extract_info(job.url, download=True)

            if job.cancel_requested:
                raise DownloadCancelled("Download cancelled")

            if info:
                with self._lock:
                    job.title = job.title or info.get("title")
                    job.duration = job.duration or (
                        int(info["duration"]) if info.get("duration") else None
                    )

            files = [
                path
                for path in job.workdir.iterdir()
                if path.is_file() and not path.name.endswith((".part", ".ytdl", ".temp"))
            ]
            if not files:
                raise VideoUnavailableError("The download finished but no file was produced.")

            media = max(files, key=lambda path: path.stat().st_size)
            suffix = media.suffix.lstrip(".").lower() or ("mp3" if job.kind == "audio" else "mp4")
            if job.kind == "video":
                quality_suffix = f"{job.height or 1080}p"
            else:
                quality_suffix = (job.audio_id or "mp3").upper()

            title = sanitize_filename(job.title or "youtube-download")
            filename = f"{title} [{quality_suffix}].{suffix}"

            self._update(
                job,
                status="ready",
                progress=1.0,
                message="Ready to save",
                filepath=media,
                filename=filename,
                filesize=media.stat().st_size,
                speed=None,
                eta=None,
            )
        except DownloadCancelled:
            self._cleanup_files(job)
            self._update(job, status="cancelled", message="Download cancelled", progress=0.0)
        except DownloadError as exc:
            logger.warning("Download failed for %s: %s", job.url, exc)
            self._cleanup_files(job)
            self._update(
                job,
                status="error",
                message="Download failed",
                error=_human_error(exc),
            )
        except Exception as exc:  # noqa: BLE001 - job errors are reported to the client
            logger.exception("Unexpected download error for %s", job.url)
            self._cleanup_files(job)
            self._update(
                job,
                status="error",
                message="Download failed",
                error=_human_error(exc),
            )


job_store = JobStore()
