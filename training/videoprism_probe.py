"""Pure sampling, calibration, and linear-head helpers for the VideoPrism pilot."""

from __future__ import annotations

import math
from typing import Any, Sequence

FRAME_COUNT = 16
FPS = 2.0
CLIP_SPAN_SEC = (FRAME_COUNT - 1) / FPS
INFERENCE_STRIDE_SEC = 4.0
TEMPERATURE = 0.25


def pool_video_tokens(tokens: Any) -> Any:
    """Pool official B x (16 x 16 x 16) x 768 tokens in NumPy or JAX."""
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (4096, 768):
        raise ValueError("unexpected VideoPrism token shape")
    return tokens.reshape(-1, 16, 16, 16, 768).mean(axis=(2, 3)).mean(axis=1)


def letterbox_rgb(frame: Any, size: int = 288) -> Any:
    """Aspect-preserving, label-blind RGB adaptation with zero padding."""
    import numpy as np
    from PIL import Image

    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    scale = min(size / image.width, size / image.height)
    resized = image.resize((max(1, round(image.width * scale)),
                            max(1, round(image.height * scale))), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def coverage_starts(duration_sec: float) -> list[float]:
    if not math.isfinite(duration_sec) or duration_sec <= 0:
        raise ValueError("invalid duration")
    maximum = max(0.0, duration_sec - CLIP_SPAN_SEC)
    starts = [0.0]
    while starts[-1] + INFERENCE_STRIDE_SEC < maximum - 1e-8:
        starts.append(starts[-1] + INFERENCE_STRIDE_SEC)
    if maximum > starts[-1] + 1e-8:
        starts.append(maximum)
    return starts


def onset_start(duration_sec: float, onset_sec: float) -> float:
    if not math.isfinite(onset_sec) or not 0 <= onset_sec <= duration_sec:
        raise ValueError("invalid onset")
    return min(max(0.0, onset_sec - CLIP_SPAN_SEC / 2),
               max(0.0, duration_sec - CLIP_SPAN_SEC))


def clean_training_starts(duration_sec: float, count: int = 6) -> list[float]:
    maximum = max(0.0, duration_sec - CLIP_SPAN_SEC)
    if count < 1:
        raise ValueError("count must be positive")
    return [maximum * (index + 0.5) / count for index in range(count)]


def select_clip(frames: Sequence[tuple[float, Any]], start_sec: float) -> list[tuple[float, Any]]:
    if len(frames) < FRAME_COUNT or any(
            not math.isfinite(float(t)) or float(t) < 0 for t, _ in frames):
        raise ValueError("insufficient or invalid frame PTS")
    if any(float(right[0]) <= float(left[0]) for left, right in zip(frames, frames[1:])):
        raise ValueError("frame PTS must increase")
    index = min(range(len(frames)), key=lambda i: abs(float(frames[i][0]) - start_sec))
    index = min(index, len(frames) - FRAME_COUNT)
    clip = list(frames[index:index + FRAME_COUNT])
    if len(clip) != FRAME_COUNT:
        raise ValueError("incomplete clip")
    gaps = [float(right[0]) - float(left[0]) for left, right in zip(clip, clip[1:])]
    if any(not 0.25 <= gap <= 0.75 for gap in gaps):
        raise ValueError("sampled frame PTS deviates from 2 fps")
    return clip


def normalized_lme_numpy(logits: Any, temperature: float = TEMPERATURE) -> float:
    import numpy as np

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or not values.size:
        raise ValueError("invalid bag logits")
    scaled = values / temperature
    maximum = float(scaled.max())
    return temperature * (maximum + math.log(float(np.exp(scaled - maximum).mean())))


def binary_metrics(truth: Sequence[bool], predicted: Sequence[bool]) -> dict[str, float | int]:
    tp = sum(t and p for t, p in zip(truth, predicted))
    tn = sum(not t and not p for t, p in zip(truth, predicted))
    fp = sum(not t and p for t, p in zip(truth, predicted))
    fn = sum(t and not p for t, p in zip(truth, predicted))
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "recall": recall,
            "specificity": specificity, "balanced_accuracy": (recall + specificity) / 2}


def calibrate(rows: list[dict[str, Any]]) -> tuple[float, dict[str, float | int]]:
    if len(rows) != 24 or len({row["sample_id"] for row in rows}) != 24:
        raise ValueError("expected 24 unique OOF sources")
    scores = [float(row["score"]) for row in rows]
    truth = [row["truth"] for row in rows]
    unique = sorted(set(scores))
    thresholds = [math.nextafter(unique[0], -math.inf),
                  *(a + (b - a) / 2 for a, b in zip(unique, unique[1:])),
                  math.nextafter(unique[-1], math.inf)]
    feasible = []
    for threshold in thresholds:
        metrics = binary_metrics(truth, [score >= threshold for score in scores])
        if metrics["specificity"] >= 0.75 and metrics["recall"] >= 0.5:
            feasible.append((metrics["balanced_accuracy"], metrics["specificity"],
                             metrics["recall"], -threshold, threshold, metrics))
    if not feasible:
        raise ValueError("no OOF threshold satisfies specificity/recall guard")
    *_, threshold, metrics = max(feasible)
    return threshold, metrics
