"""Pure helpers and the small frozen-feature onset-transition probe."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

SEED = 1729
LAG_SEC = 2.0
TEMPERATURE = 0.25
ONSET_SIGMA_SEC = 1.0
ONSET_SUPPORT_RADIUS_SEC = 1.5
OOF_MIN_SPECIFICITY = 0.75
OOF_MIN_RECALL = 0.50


def normalized_log_mean_exp(logits: Any, temperature: float = TEMPERATURE) -> Any:
    """Smooth bag maximum without a bag-length offset."""
    import torch

    if logits.ndim != 1 or logits.numel() == 0 or not math.isfinite(temperature) \
            or temperature <= 0:
        raise ValueError("expected nonempty vector and positive temperature")
    return temperature * (torch.logsumexp(logits / temperature, dim=0)
                          - math.log(logits.numel()))


def transition_pairs(features: Any, times: Any, lag_sec: float = LAG_SEC) -> tuple[Any, Any]:
    """Create causal pre/post transition vectors and endpoint times."""
    import torch

    if features.ndim != 2 or times.ndim != 1 or features.shape[0] != times.numel() \
            or times.numel() < 2 or lag_sec <= 0:
        raise ValueError("invalid frame feature sequence")
    if not bool(torch.all(times[1:] > times[:-1])):
        raise ValueError("frame times must increase")
    pre_indices, post_indices = [], []
    for post in range(1, times.numel()):
        target = times[post] - lag_sec
        if target < times[0]:
            continue
        pre = int(torch.argmin(torch.abs(times[:post] - target)))
        pre_indices.append(pre)
        post_indices.append(post)
    if not post_indices:
        raise ValueError("sequence is shorter than the requested lag")
    pre = features[torch.tensor(pre_indices, device=features.device)]
    post = features[torch.tensor(post_indices, device=features.device)]
    vectors = torch.cat((pre, post, post - pre, torch.abs(post - pre)), dim=-1)
    return vectors, times[torch.tensor(post_indices, device=times.device)]


def build_probe(input_dim: int, dropout: float = 0.15) -> Any:
    import torch.nn as nn

    if type(input_dim) is not int or input_dim < 1:
        raise ValueError("input_dim must be positive")
    return nn.Sequential(
        nn.LayerNorm(input_dim * 4),
        nn.Linear(input_dim * 4, 128),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(128, 1),
    )


def pool_qwen_image_features(output: Any) -> Any:
    """Mean-pool only final Qwen image embeddings, excluding DeepStack features."""
    import torch

    if not isinstance(output, tuple) or len(output) != 2:
        raise ValueError("unexpected Qwen get_image_features return")
    image_embeds, _deepstack_image_embeds = output
    if not isinstance(image_embeds, tuple) or not image_embeds \
            or any(not isinstance(item, torch.Tensor) or item.ndim != 2 for item in image_embeds):
        raise ValueError("unexpected Qwen final image embedding structure")
    return torch.cat(image_embeds, dim=0).float().mean(0)


def source_loss(logits: Any, times: Any, has_violation: bool, onset_sec: float | None,
                localization_weight: float = 1.0) -> tuple[Any, dict[str, float]]:
    """One source contributes one binary term; positive off-onset pairs are unlabeled."""
    import torch
    import torch.nn.functional as functional

    if logits.ndim != 1 or times.shape != logits.shape or logits.numel() == 0:
        raise ValueError("invalid source logits")
    bag = normalized_log_mean_exp(logits)
    binary = functional.binary_cross_entropy_with_logits(
        bag, logits.new_tensor(float(has_violation)))
    localization = logits.new_zeros(())
    if has_violation:
        if onset_sec is None or not math.isfinite(onset_sec):
            raise ValueError("positive source needs a finite onset")
        distance = torch.abs(times - onset_sec)
        support = distance <= ONSET_SUPPORT_RADIUS_SEC
        if not bool(support.any()):
            raise ValueError("no transition endpoint supports the annotated onset")
        weights = torch.exp(-0.5 * (distance[support] / ONSET_SIGMA_SEC) ** 2)
        weights = weights / weights.sum().clamp_min(1e-12)
        # Point supervision establishes positive support near the onset. Pairs
        # elsewhere in the same positive source remain unknown and receive no loss.
        localization = (weights * functional.softplus(-logits[support])).sum()
    total = binary + localization_weight * localization
    return total, {"binary": float(binary.detach()),
                   "localization": float(localization.detach())}


def balanced_source_loss(losses: Iterable[tuple[bool, Any]]) -> Any:
    """Average sources within each class, then give both classes equal weight."""
    positive, negative = [], []
    for label, loss in losses:
        (positive if label else negative).append(loss)
    if not positive or not negative:
        raise ValueError("both source classes are required")
    return 0.5 * sum(positive) / len(positive) + 0.5 * sum(negative) / len(negative)


def binary_metrics(truth: Sequence[bool], predicted: Sequence[bool]) -> dict[str, float | int]:
    if len(truth) != len(predicted) or not truth:
        raise ValueError("truth and predictions must be nonempty and aligned")
    tp = sum(t and p for t, p in zip(truth, predicted))
    tn = sum(not t and not p for t, p in zip(truth, predicted))
    fp = sum(not t and p for t, p in zip(truth, predicted))
    fn = sum(t and not p for t, p in zip(truth, predicted))
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "recall": recall,
            "specificity": specificity, "balanced_accuracy": (recall + specificity) / 2}


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    metrics: dict[str, float | int]


def calibrate_threshold(rows: Sequence[dict[str, Any]]) -> ThresholdResult:
    """Choose one threshold from train-only OOF scores under specificity/recall guards."""
    if len(rows) != 24 or len({row.get("sample_id") for row in rows}) != 24:
        raise ValueError("calibration requires 24 unique OOF train sources")
    scores = [row.get("score") for row in rows]
    truth = [row.get("truth") for row in rows]
    if any(type(value) is not bool for value in truth) or any(
            not isinstance(value, (int, float)) or not math.isfinite(value) for value in scores):
        raise ValueError("invalid OOF rows")
    unique = sorted(set(float(value) for value in scores))
    candidates = [math.nextafter(unique[0], -math.inf),
                  *(left + (right - left) / 2 for left, right in zip(unique, unique[1:])),
                  math.nextafter(unique[-1], math.inf)]
    feasible = []
    for threshold in candidates:
        metrics = binary_metrics(truth, [float(score) >= threshold for score in scores])
        if metrics["specificity"] >= OOF_MIN_SPECIFICITY and metrics["recall"] >= OOF_MIN_RECALL:
            feasible.append((metrics["balanced_accuracy"], metrics["specificity"],
                             metrics["recall"], threshold, metrics))
    if not feasible:
        raise ValueError("no OOF threshold satisfies anti-all-positive guards")
    _, _, _, threshold, metrics = max(feasible, key=lambda item: item[:3] + (-item[3],))
    return ThresholdResult(threshold=threshold, metrics=metrics)


def strict_validation_gate(rows: Sequence[dict[str, Any]], required_positive_ids: set[str],
                           required_negative_ids: set[str], additional_positive_ids: set[str]) -> bool:
    """Preserve both baseline TPs and all TNs, and recover one prior positive miss."""
    if len(rows) != 8 or len({row.get("sample_id") for row in rows}) != 8:
        return False
    by_id = {row["sample_id"]: row for row in rows}
    expected = required_positive_ids | required_negative_ids | additional_positive_ids
    if set(by_id) != expected:
        return False
    if any(type(row.get("candidate")) is not bool
           or not isinstance(row.get("score"), (int, float))
           or not math.isfinite(row["score"]) for row in rows):
        return False
    return (all(by_id[key]["candidate"] for key in required_positive_ids)
            and all(not by_id[key]["candidate"] for key in required_negative_ids)
            and sum(by_id[key]["candidate"] for key in additional_positive_ids) >= 1)
