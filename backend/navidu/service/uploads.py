import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

from fastapi import UploadFile

from navidu.contracts.domain import VideoMeta
from navidu.contracts.units import new_check_id

from .config import Settings
from .exceptions import ServiceError


def safe_filename(value: str | None) -> str:
    return (re.split(r"[/\\]", value or "video.mov")[-1].replace("\x00", "") or "video.mov")[:240]


def probe_video(path: Path, settings: Settings) -> VideoMeta:
    try:
        result = subprocess.run([
            settings.ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe", "-show_format",
            "-show_streams", "-print_format", "json", str(path),
        ], capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise ServiceError(503, "NOT_READY", "На сервере не установлен ffprobe для проверки записи.") from exc
    except subprocess.TimeoutExpired as exc:
        raise ServiceError(415, "UNSUPPORTED_MEDIA", "Не удалось прочитать метаданные видео.") from exc
    if result.returncode:
        raise ServiceError(415, "UNSUPPORTED_MEDIA", "Файл не читается как видеозапись.")
    try:
        value = json.loads(result.stdout)
        format_name = value.get("format", {}).get("format_name", "")
        # Playlists can reference other files or URLs and are not submitted video containers.
        if any(name in format_name.split(",") for name in ["hls", "concat", "dash"]):
            raise ValueError("playlist")
        stream = next(stream for stream in value["streams"] if stream.get("codec_type") == "video")
        duration = float(stream.get("duration") or value["format"]["duration"])
        rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
        numerator, denominator = rate.split("/")
        fps = float(numerator) / float(denominator)
        if not math.isfinite(duration) or duration <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError("invalid timing")
        rotation = int(float(stream.get("tags", {}).get("rotate", 0)))
        for side in stream.get("side_data_list", []):
            if "rotation" in side:
                rotation = -int(float(side["rotation"]))
        rotation %= 360
        if rotation not in {0, 90, 180, 270}:
            raise ValueError("invalid rotation")
        width, height = int(stream["width"]), int(stream["height"])
        if rotation in {90, 270}:
            width, height = height, width
        meta = VideoMeta(duration_sec=duration, fps_nominal=fps, width=width, height=height,
                         rotation=rotation)
    except (KeyError, ValueError, TypeError, StopIteration, ZeroDivisionError) as exc:
        raise ServiceError(415, "UNSUPPORTED_MEDIA", "В файле нет пригодной видеодорожки.") from exc
    if meta.duration_sec > settings.max_duration_sec:
        raise ServiceError(422, "VIDEO_TOO_LONG", f"Запись длиннее {settings.max_duration_sec:g} секунд.")
    return meta


async def persist_upload(file: UploadFile | None, settings: Settings, probe=probe_video):
    if file is None:
        raise ServiceError(400, "MISSING_FILE", "Выберите видеофайл.")
    check_id = new_check_id()
    directory = settings.checks_dir / check_id
    directory.mkdir(parents=True, exist_ok=False)
    name = safe_filename(file.filename)
    extension = Path(name).suffix.lower()
    if extension not in {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm"}:
        extension = ".mp4"
    path = directory / f"video{extension}"
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("xb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise ServiceError(413, "FILE_TOO_LARGE", f"Файл больше {settings.max_upload_mb:g} МБ.")
                output.write(chunk)
                digest.update(chunk)
        if not size:
            raise ServiceError(400, "MISSING_FILE", "Файл пуст. Выберите видеозапись.")
        # Probe is CPU/IO work; run outside the event loop.
        from starlette.concurrency import run_in_threadpool
        meta = await run_in_threadpool(probe, path, settings)
        return check_id, name, path, digest.hexdigest(), meta
    except BaseException:
        path.unlink(missing_ok=True)
        directory.rmdir()
        raise
    finally:
        await file.close()
