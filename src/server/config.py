"""Runtime configuration for the TubeFlow download API."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from imageio_ffmpeg import get_ffmpeg_exe


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser().resolve()


# yt-dlp needs a real ffmpeg binary to merge adaptive video + audio streams.
FFMPEG_PATH: str = os.getenv("FFMPEG_PATH") or get_ffmpeg_exe()

# Files are written to a dedicated temp directory and cleaned up automatically.
DOWNLOAD_DIR: Path = _env_path(
    "TUBEFLOW_DOWNLOAD_DIR",
    str(Path(tempfile.gettempdir()) / "tubeflow-downloads"),
)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Optional Netscape-format cookies exported from a browser. This helps when
# YouTube asks the server to "confirm you're not a bot".
COOKIES_FILE: Path | None = None
if cookies := os.getenv("TUBEFLOW_COOKIES_FILE"):
    candidate = Path(cookies).expanduser().resolve()
    if candidate.is_file():
        COOKIES_FILE = candidate

# How long a finished file is kept before the cleanup worker removes it.
JOB_TTL_SECONDS: int = int(os.getenv("TUBEFLOW_JOB_TTL_SECONDS", "1800"))
MAX_CONCURRENT_DOWNLOADS: int = int(os.getenv("TUBEFLOW_MAX_CONCURRENT", "3"))
MAX_ACTIVE_JOBS: int = int(os.getenv("TUBEFLOW_MAX_ACTIVE_JOBS", "12"))

# Some videos are restricted unless a proxy is supplied.
PROXY_URL: str | None = os.getenv("TUBEFLOW_PROXY") or None
