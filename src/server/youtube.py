"""YouTube metadata extraction and yt-dlp option builders."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qs, urlparse

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from .config import COOKIES_FILE, FFMPEG_PATH, PROXY_URL

logger = logging.getLogger("tubeflow.youtube")

YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,20}$")


class InvalidUrlError(ValueError):
    """Raised when a URL is not a supported YouTube link."""


class VideoUnavailableError(RuntimeError):
    """Raised when metadata extraction fails."""


def normalize_url(raw: str) -> str:
    """Validate and normalize a YouTube URL.

    Accepts watch links, youtu.be links, shorts, live links, and bare IDs.
    """

    value = (raw or "").strip()
    if not value:
        raise InvalidUrlError("Please provide a YouTube video URL.")

    # Allow pasting a bare video id.
    if _VIDEO_ID_RE.match(value) and "://" not in value:
        return f"https://www.youtube.com/watch?v={value}"

    if "://" not in value:
        value = f"https://{value}"

    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    if host not in YOUTUBE_HOSTS:
        raise InvalidUrlError("Only YouTube links are supported.")

    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/")[0]
        if not video_id:
            raise InvalidUrlError("That YouTube link is missing a video id.")
        return f"https://www.youtube.com/watch?v={video_id}"

    if parsed.path.startswith("/shorts/") or parsed.path.startswith("/live/"):
        video_id = parsed.path.strip("/").split("/")[1]
        return f"https://www.youtube.com/watch?v={video_id}"

    if parsed.path.startswith("/embed/"):
        video_id = parsed.path.strip("/").split("/")[1]
        return f"https://www.youtube.com/watch?v={video_id}"

    query = parse_qs(parsed.query)
    if "v" in query and query["v"]:
        return f"https://www.youtube.com/watch?v={query['v'][0]}"

    raise InvalidUrlError("Could not find a video in that YouTube link.")


def ydl_base_options() -> dict[str, Any]:
    options: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "cachedir": False,
        "socket_timeout": 30,
        "retries": 3,
        "fragment_retries": 3,
        "extractor_retries": 2,
        "ffmpeg_location": FFMPEG_PATH,
        "http_headers": {"User-Agent": _USER_AGENT},
    }
    if COOKIES_FILE is not None:
        options["cookiefile"] = str(COOKIES_FILE)
    if PROXY_URL:
        options["proxy"] = PROXY_URL
    return options


def _human_error(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    if "sign in to confirm" in lowered or "not a bot" in lowered:
        return (
            "YouTube is asking for verification. Try again later or configure "
            "TUBEFLOW_COOKIES_FILE with exported browser cookies."
        )
    if "private video" in lowered:
        return "This video is private."
    if "video unavailable" in lowered or "has been removed" in lowered:
        return "This video is unavailable or has been removed."
    if "age" in lowered and "restrict" in lowered:
        return "This video is age-restricted and cannot be downloaded without cookies."
    if "unsupported url" in lowered:
        return "That link is not a supported YouTube video."
    # Keep the message useful but avoid dumping a huge traceback to the client.
    return message.split("\n")[0][:300] or "Could not read this YouTube video."


def _best_audio_size(formats: list[dict[str, Any]]) -> tuple[int | None, dict[str, Any] | None]:
    audio_formats = [
        f
        for f in formats
        if (f.get("vcodec") in (None, "none"))
        and (f.get("acodec") not in (None, "none"))
        and f.get("ext") in {"m4a", "webm", "mp3", "opus"}
    ]
    if not audio_formats:
        return None, None

    def score(fmt: dict[str, Any]) -> tuple[int, float, int]:
        return (
            1 if fmt.get("ext") == "m4a" else 0,
            float(fmt.get("abr") or fmt.get("tbr") or 0),
            int(fmt.get("filesize") or fmt.get("filesize_approx") or 0),
        )

    best = max(audio_formats, key=score)
    size = best.get("filesize") or best.get("filesize_approx")
    return (int(size) if size else None), best


def _video_formats(
    formats: list[dict[str, Any]], duration: float | None
) -> list[dict[str, Any]]:
    audio_size, _ = _best_audio_size(formats)
    by_height: dict[int, dict[str, Any]] = {}

    for fmt in formats:
        vcodec = fmt.get("vcodec")
        if not vcodec or vcodec == "none":
            continue
        ext = fmt.get("ext")
        if ext not in {"mp4", "webm"}:
            continue
        height = fmt.get("height")
        if not height:
            continue
        height = int(height)

        has_audio = fmt.get("acodec") not in (None, "none")
        tbr = float(fmt.get("tbr") or 0)
        fps = float(fmt.get("fps") or 0)
        size = fmt.get("filesize") or fmt.get("filesize_approx")

        # Prefer MP4 for maximum device compatibility, then bitrate/fps.
        rank = (1 if ext == "mp4" else 0, tbr, fps)
        previous = by_height.get(height)
        if previous is not None:
            previous_rank = (
                1 if previous["ext"] == "mp4" else 0,
                float(previous.get("_tbr") or 0),
                float(previous.get("fps") or 0),
            )
            if rank <= previous_rank:
                continue

        needs_merge = not has_audio
        estimated_size = int(size) if size else None
        if estimated_size is None and duration and tbr:
            estimated_size = int((tbr * 1000 / 8) * duration)
        if needs_merge and audio_size:
            estimated_size = (estimated_size or 0) + audio_size

        label = f"{height}p"
        if fps >= 50:
            label += f"{int(fps)}"
        if height >= 2160:
            quality = "4K"
        elif height >= 1440:
            quality = "2K"
        elif height >= 1080:
            quality = "Full HD"
        elif height >= 720:
            quality = "HD"
        else:
            quality = "SD"

        by_height[height] = {
            "height": height,
            "fps": int(fps) if fps else None,
            "ext": ext,
            "format_id": fmt.get("format_id"),
            "vcodec": vcodec,
            "acodec": fmt.get("acodec"),
            "filesize": estimated_size,
            "needs_merge": needs_merge,
            "container": "mp4" if ext == "mp4" else "webm",
            "label": label,
            "quality": quality,
            "_tbr": tbr,
        }

    ordered = sorted(
        by_height.values(),
        key=lambda item: (item["height"], item["fps"] or 0),
        reverse=True,
    )
    for item in ordered:
        item.pop("_tbr", None)
    return ordered


def _audio_formats(
    formats: list[dict[str, Any]], duration: float | None
) -> list[dict[str, Any]]:
    def best_for(ext: str) -> dict[str, Any] | None:
        candidates = [
            f
            for f in formats
            if (f.get("vcodec") in (None, "none"))
            and (f.get("acodec") not in (None, "none"))
            and f.get("ext") == ext
        ]
        if not candidates:
            return None

        def score(fmt: dict[str, Any]) -> tuple[float, int]:
            return (
                float(fmt.get("abr") or fmt.get("tbr") or 0),
                int(fmt.get("filesize") or fmt.get("filesize_approx") or 0),
            )

        return max(candidates, key=score)

    options: list[dict[str, Any]] = []

    m4a = best_for("m4a")
    if m4a is not None:
        size = m4a.get("filesize") or m4a.get("filesize_approx")
        abr = m4a.get("abr")
        options.append(
            {
                "id": "m4a",
                "label": "M4A",
                "description": "Original audio • best compatibility",
                "ext": "m4a",
                "abr": int(abr) if abr else None,
                "filesize": int(size) if size else None,
                "converted": False,
                "recommended": True,
            }
        )

    opus = best_for("webm")
    if opus is not None:
        size = opus.get("filesize") or opus.get("filesize_approx")
        abr = opus.get("abr")
        options.append(
            {
                "id": "opus",
                "label": "WEBM",
                "description": "Opus audio • smaller file, high quality",
                "ext": "webm",
                "abr": int(abr) if abr else None,
                "filesize": int(size) if size else None,
                "converted": False,
                "recommended": False,
            }
        )

    mp3_size = int((192_000 / 8) * duration) if duration else None
    options.append(
        {
            "id": "mp3",
            "label": "MP3",
            "description": "Converted to 192 kbps • plays everywhere",
            "ext": "mp3",
            "abr": 192,
            "filesize": mp3_size,
            "converted": True,
            "recommended": False,
        }
    )
    return options


def _format_upload_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = datetime.strptime(value, "%Y%m%d").replace(tzinfo=timezone.utc)
        return parsed.date().isoformat()
    except ValueError:
        return value


def _pick_thumbnail(info: dict[str, Any]) -> str | None:
    if info.get("thumbnail"):
        return info["thumbnail"]
    thumbnails = info.get("thumbnails") or []
    if thumbnails:
        usable = [t for t in thumbnails if t.get("url")]
        if usable:
            return usable[-1]["url"]
    if info.get("id"):
        return f"https://i.ytimg.com/vi/{info['id']}/hqdefault.jpg"
    return None


def extract_video_info(url: str) -> dict[str, Any]:
    """Extract metadata and available formats for a YouTube video."""

    normalized = normalize_url(url)
    try:
        with YoutubeDL({**ydl_base_options(), "skip_download": True}) as ydl:
            info = ydl.extract_info(normalized, download=False)
    except DownloadError as exc:
        raise VideoUnavailableError(_human_error(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surfaced as a clean API error
        logger.exception("Unexpected metadata error for %s", normalized)
        raise VideoUnavailableError(_human_error(exc)) from exc

    if not info:
        raise VideoUnavailableError("Could not read this YouTube video.")

    if info.get("_type") == "playlist":
        entries = [entry for entry in (info.get("entries") or []) if entry]
        if not entries:
            raise VideoUnavailableError("That playlist does not contain a video.")
        info = entries[0]

    if info.get("_type") == "url":
        # A flat entry has no formats; resolve it properly.
        return extract_video_info(info.get("url") or normalized)

    formats = [f for f in (info.get("formats") or []) if f]
    if not formats:
        raise VideoUnavailableError("No downloadable formats were found for this video.")

    duration = info.get("duration")

    return {
        "id": info.get("id"),
        "title": info.get("title") or "YouTube video",
        "description": (info.get("description") or "")[:2000],
        "author": info.get("uploader") or info.get("channel") or "Unknown creator",
        "channel_url": info.get("channel_url"),
        "thumbnail": _pick_thumbnail(info),
        "duration": int(duration) if duration else None,
        "view_count": info.get("view_count"),
        "like_count": info.get("like_count"),
        "upload_date": _format_upload_date(info.get("upload_date")),
        "webpage_url": info.get("webpage_url") or normalized,
        "is_live": bool(info.get("is_live")),
        "video_formats": _video_formats(formats, duration),
        "audio_formats": _audio_formats(formats, duration),
    }


def build_download_options(
    *,
    url: str,
    kind: str,
    height: int | None,
    container: str | None,
    audio_id: str | None,
    output_template: str,
    progress_hook: Any,
    postprocessor_hook: Any,
) -> dict[str, Any]:
    """Build yt-dlp options for the requested video/audio download."""

    options = {
        **ydl_base_options(),
        "skip_download": False,
        "noplaylist": True,
        "outtmpl": output_template,
        "progress_hooks": [progress_hook],
        "postprocessor_hooks": [postprocessor_hook],
        "overwrites": True,
        "continuedl": True,
        "concurrent_fragment_downloads": 4,
        "retries": 5,
        "fragment_retries": 5,
    }

    if kind == "audio":
        audio_id = (audio_id or "mp3").lower()
        if audio_id == "m4a":
            options["format"] = "bestaudio[ext=m4a]/bestaudio/best"
        elif audio_id == "opus":
            options["format"] = "bestaudio[ext=webm]/bestaudio/best"
        else:
            options["format"] = "bestaudio/best"
            options["postprocessors"] = [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ]
            options["postprocessor_args"] = {"extractaudio": ["-vn"]}
        options["writethumbnail"] = False
        return options

    height = max(144, min(int(height or 1080), 4320))
    container = "webm" if container == "webm" else "mp4"

    if container == "webm":
        video_selector = f"bestvideo[height<={height}][ext=webm]+bestaudio[ext=webm]"
    else:
        video_selector = f"bestvideo[height<={height}][ext=mp4]+bestaudio[ext=m4a]"

    fallbacks = [
        video_selector,
        f"bestvideo[height<={height}]+bestaudio",
        f"best[height<={height}][ext=mp4]",
        f"best[height<={height}]",
        f"bestvideo[height<={height}]+bestaudio",
        "best",
    ]
    options["format"] = "/".join(fallbacks)
    options["merge_output_format"] = container
    options["writethumbnail"] = False
    return options


def sanitize_filename(name: str, fallback: str = "youtube-download") -> str:
    """Create a safe, readable filename for Content-Disposition."""

    cleaned = re.sub(r"[\x00-\x1f\x7f]+", "", name or "").strip()
    cleaned = re.sub(r"[\\/:*?\"<>|]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if not cleaned:
        cleaned = fallback
    return cleaned[:150]


def youtube_tools_version() -> str:
    from importlib.metadata import version

    try:
        return version("yt-dlp")
    except Exception:  # noqa: BLE001
        return "unknown"
