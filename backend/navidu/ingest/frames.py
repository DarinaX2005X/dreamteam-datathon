"""Decode RGB frames on the source PTS clock, including VFR recordings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from .probe import MediaProbeError, ProbeResult, probe_video


@dataclass(frozen=True)
class DecodedFrame:
    timestamp_sec: float
    rgb: Any


def _autorotate(rgb: Any, rotation: int) -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - deployment check
        raise MediaProbeError("VIDEO_DECODE_FAILED", "Для кадров требуется пакет NumPy.") from exc
    # ffprobe reports clockwise display rotation; np.rot90 uses counter-clockwise turns.
    turns = {0: 0, 90: 3, 180: 2, 270: 1}[rotation]
    return np.ascontiguousarray(np.rot90(rgb, turns)) if turns else np.ascontiguousarray(rgb)


def iter_frames(
    path: Path,
    *,
    analysis_fps: float,
    probe: ProbeResult | None = None,
    should_stop: Callable[[], bool] = lambda: False,
) -> Iterator[DecodedFrame]:
    """Yield the first decoded frame at or after each sampling instant.

    Timestamps are actual decoded PTS values relative to the stream start, never
    frame indices divided by nominal FPS.
    """
    try:
        import av
    except ImportError as exc:  # pragma: no cover - deployment check
        raise MediaProbeError("VIDEO_DECODE_FAILED", "Для декодирования требуется пакет PyAV.") from exc
    info = probe or probe_video(path)
    period = 1.0 / analysis_fps
    next_sample = 0.0
    try:
        with av.open(str(path), metadata_errors="ignore") as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                if should_stop():
                    return
                if frame.pts is None or frame.time_base is None:
                    continue
                timestamp = max(0.0, float(frame.pts * frame.time_base) - info.start_time_sec)
                if timestamp + 1e-6 < next_sample:
                    continue
                rgb = _autorotate(frame.to_ndarray(format="rgb24"), info.video.rotation)
                yield DecodedFrame(round(timestamp, 6), rgb)
                while next_sample <= timestamp + 1e-6:
                    next_sample += period
    except MediaProbeError:
        raise
    except Exception as exc:
        raise MediaProbeError("VIDEO_DECODE_FAILED", "Не удалось декодировать кадр записи.") from exc


def frame_at(path: Path, timestamp_sec: float, *, probe: ProbeResult | None = None) -> DecodedFrame:
    """Return the first full-resolution frame at/after a requested source time."""
    try:
        import av
    except ImportError as exc:  # pragma: no cover - deployment check
        raise MediaProbeError("VIDEO_DECODE_FAILED", "Для декодирования требуется пакет PyAV.") from exc
    info = probe or probe_video(path)
    target_absolute = max(0.0, timestamp_sec + info.start_time_sec)
    try:
        with av.open(str(path), metadata_errors="ignore") as container:
            stream = container.streams.video[0]
            seek_pts = int(target_absolute / float(stream.time_base))
            container.seek(max(0, seek_pts), stream=stream, backward=True, any_frame=False)
            prior = None
            for frame in container.decode(stream):
                if frame.pts is None or frame.time_base is None:
                    continue
                actual = max(0.0, float(frame.pts * frame.time_base) - info.start_time_sec)
                decoded = DecodedFrame(round(actual, 6), _autorotate(
                    frame.to_ndarray(format="rgb24"), info.video.rotation
                ))
                prior = decoded
                if actual + 1e-6 >= timestamp_sec:
                    return decoded
            if prior is not None:
                return prior
    except Exception as exc:
        raise MediaProbeError("VIDEO_DECODE_FAILED", "Не удалось извлечь кадр-доказательство.") from exc
    raise MediaProbeError("VIDEO_DECODE_FAILED", "В записи нет декодируемых кадров.")
