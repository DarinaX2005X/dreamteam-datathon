"""Supervised crop selection is separate from label-blind inference sampling."""

from __future__ import annotations

import bisect
import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any

from eval.training import TrainingSample


@dataclass(frozen=True)
class WindowSpec:
    sample: TrainingSample
    start_sec: float
    end_sec: float
    ordinal: int


def _span(frame_count: int, fps: float) -> float:
    if type(frame_count) is not int or not 2 <= frame_count <= 64 \
            or not math.isfinite(fps) or not 0 < fps <= 16:
        raise ValueError("Invalid frame count or temporal sample rate")
    return (frame_count - 1) / fps


def _check_samples(samples: list[TrainingSample]) -> None:
    if len({sample.source_sha256 for sample in samples}) != len(samples):
        raise ValueError("Repeated source hashes must not become independent training examples")
    if any(not math.isfinite(sample.duration_sec) or sample.duration_sec <= 0 for sample in samples):
        raise ValueError("Training sources must have a finite positive duration")


def make_training_windows(
    samples: list[TrainingSample], *, frame_count: int = 16, fps: float = 2.0,
    windows_per_video: int = 6, seed: int = 1729,
) -> list[WindowSpec]:
    """Use the known positive point only for supervised training crop selection.

    Other windows of a positive video have unknown targets. They are never
    converted into negative training examples. Repeated crops from one source
    remain one source for every quality metric.
    """
    span = _span(frame_count, fps)
    _check_samples(samples)
    if type(windows_per_video) is not int or not 1 <= windows_per_video <= 64:
        raise ValueError("Invalid number of supervised windows per source")
    windows = []
    for sample in sorted(samples, key=lambda item: item.source_sha256):
        maximum_start = max(0.0, sample.duration_sec - span)
        lower, upper = 0.0, maximum_start
        if sample.has_violation:
            point = sample.time_sec
            if not sample.time_supervision or point is None or not math.isfinite(point) \
                    or not 0 <= point <= sample.duration_sec:
                raise ValueError("Positive crop training requires one reliable organizer timestamp")
            lower, upper = max(0.0, point - span), min(point, maximum_start)
        local_seed = int.from_bytes(hashlib.sha256(
            f"{seed}:{sample.source_sha256}".encode("ascii")).digest()[:8], "big")
        rng = random.Random(local_seed)
        for ordinal in range(windows_per_video):
            fraction = (ordinal + rng.random()) / windows_per_video
            start = lower + (upper - lower) * fraction
            windows.append(WindowSpec(sample=sample, start_sec=start,
                end_sec=min(sample.duration_sec, start + span), ordinal=ordinal))
    return windows


def make_inference_windows(
    samples: list[TrainingSample], *, frame_count: int = 16, fps: float = 2.0,
    stride_frames: int = 8,
) -> list[WindowSpec]:
    """Cover every source without consulting its class or timestamp annotation."""
    span = _span(frame_count, fps)
    _check_samples(samples)
    if type(stride_frames) is not int or not 1 <= stride_frames < frame_count:
        raise ValueError("Inference windows must overlap")
    stride = stride_frames / fps
    windows = []
    for sample in sorted(samples, key=lambda item: item.source_sha256):
        maximum_start = max(0.0, sample.duration_sec - span)
        starts = [0.0]
        while starts[-1] + stride < maximum_start - 1e-8:
            starts.append(starts[-1] + stride)
        if maximum_start > starts[-1] + 1e-8:
            starts.append(maximum_start)
        windows.extend(WindowSpec(sample, start, min(sample.duration_sec, start + span), ordinal)
                       for ordinal, start in enumerate(starts))
    return windows


def decode_window(
    spec: WindowSpec, *, frame_count: int = 16, fps: float = 2.0,
    cache: dict[tuple[str, float], list[tuple[float, Any]]] | None = None,
) -> list[tuple[float, Any]]:
    """Decode actual source PTS locally; no image or hypothesis is printed."""
    from navidu.ingest import iter_frames

    _span(frame_count, fps)
    key = (spec.sample.source_sha256, fps)
    stored = None if cache is None else cache.get(key)
    if stored is None:
        stored = [(round(frame.timestamp_sec, 3), frame.rgb)
                  for frame in iter_frames(spec.sample.video_path, analysis_fps=fps)]
        if cache is not None:
            cache[key] = stored
    times = [timestamp for timestamp, _ in stored]
    if len(times) < 2:
        raise RuntimeError("TRAINING_INPUT_INVALID: source has fewer than two decodable frames")
    # Pick the nearest sampled frame to the requested start, retaining the last
    # complete window at the source tail. Supervision is aligned to actual PTS.
    index = bisect.bisect_left(times, spec.start_sec)
    if index and (index == len(times) or spec.start_sec - times[index - 1] <= times[index] - spec.start_sec):
        index -= 1
    index = min(index, max(0, len(stored) - frame_count))
    return stored[index:index + frame_count]


def supervised_target(spec: WindowSpec, frames: list[tuple[float, Any]]) -> dict | None:
    """An unannotated positive window is unknown, never a clean example."""
    if len(frames) < 2:
        raise ValueError("A temporal target needs at least two source frames")
    if not spec.sample.has_violation:
        return {"candidate": 0, "frame": None}
    point = spec.sample.time_sec
    if not spec.sample.time_supervision or point is None:
        return None
    times = [float(timestamp) for timestamp, _ in frames]
    # Half an actual sample interval covers quantization around a point label,
    # while windows that do not contain that point remain unsupervised.
    margin = max(times[1] - times[0], times[-1] - times[-2]) / 2
    if not (spec.start_sec - 1e-8 <= point <= spec.end_sec + 1e-8) \
            or not times[0] - margin <= point <= times[-1] + margin:
        return None
    return {"candidate": 1, "frame": min(range(len(times)), key=lambda index: abs(times[index] - point))}
