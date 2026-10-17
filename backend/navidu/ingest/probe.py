"""Read source metadata without decoding the whole recording."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from navidu.contracts.domain import VideoMeta


class MediaProbeError(ValueError):
    """A stable, recoverable media validation error."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ProbeResult:
    video: VideoMeta
    start_time_sec: float
    stream_index: int


def _number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _rotation(stream: dict) -> int:
    raw = stream.get("tags", {}).get("rotate")
    for item in stream.get("side_data_list", []):
        if "rotation" in item:
            raw = item["rotation"]
    value = int(round(_number(raw))) % 360
    return value if value in {0, 90, 180, 270} else 0


def probe_video(path: Path, *, max_duration_sec: float | None = None) -> ProbeResult:
    if not path.is_file():
        raise MediaProbeError("UNSUPPORTED_MEDIA", "Файл записи не найден.")
    command = [
        "ffprobe", "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(path),
    ]
    try:
        # Camera files may contain non-UTF-8 tags or diagnostics. They must not
        # crash otherwise valid numeric metadata parsing (including stderr).
        result = subprocess.run(command, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MediaProbeError("UNSUPPORTED_MEDIA", "Не удалось прочитать метаданные записи.") from exc
    if result.returncode:
        raise MediaProbeError("UNSUPPORTED_MEDIA", "Формат записи не поддерживается или файл повреждён.")
    try:
        payload = json.loads(result.stdout)
        stream = next(item for item in payload["streams"] if item.get("codec_type") == "video")
    except (json.JSONDecodeError, KeyError, StopIteration, TypeError) as exc:
        raise MediaProbeError("UNSUPPORTED_MEDIA", "В файле нет читаемого видеопотока.") from exc
    duration = _number(stream.get("duration"), _number(payload.get("format", {}).get("duration")))
    if duration <= 0:
        raise MediaProbeError("UNSUPPORTED_MEDIA", "Не удалось определить длительность записи.")
    if max_duration_sec is not None and duration > max_duration_sec:
        raise MediaProbeError("VIDEO_TOO_LONG", "Запись превышает допустимую длительность.")
    rate = str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1")
    numerator, _, denominator = rate.partition("/")
    fps = _number(numerator) / max(_number(denominator, 1), 1e-12)
    if fps <= 0:
        fps = 1.0
    rotation = _rotation(stream)
    width, height = int(stream["width"]), int(stream["height"])
    if rotation in {90, 270}:
        width, height = height, width
    start = _number(stream.get("start_time"), _number(payload.get("format", {}).get("start_time")))
    return ProbeResult(
        video=VideoMeta(duration_sec=duration, fps_nominal=fps, width=width, height=height,
                        rotation=rotation),
        start_time_sec=start,
        stream_index=int(stream.get("index", 0)),
    )
