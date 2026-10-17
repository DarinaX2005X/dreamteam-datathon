"""Small temporal MIL head and pure selection helpers.

Torch is imported only when a model or tensor operation is requested so the
metadata/selection tests remain usable in lightweight environments.
"""

from __future__ import annotations

import math
from typing import Any


def build_temporal_head(input_dim: int = 256, dropout: float = 0.2) -> Any:
    import torch.nn as nn

    class TemporalMILHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv1d(input_dim * 2, 64, 3, padding=1, dilation=1),
                nn.GELU(), nn.Dropout(dropout),
                nn.Conv1d(64, 128, 3, padding=2, dilation=2),
                nn.GELU(), nn.Dropout(dropout),
                nn.Conv1d(128, 64, 3, padding=4, dilation=4),
                nn.GELU(), nn.Dropout(dropout),
                nn.Conv1d(64, 1, 1),
            )

        def forward(self, features: Any) -> Any:
            if features.ndim != 2 or features.shape[-1] != input_dim:
                raise ValueError("expected [time,input_dim] features")
            delta = features.new_zeros(features.shape)
            delta[1:] = features[1:] - features[:-1]
            inputs = __import__("torch").cat((features, delta), dim=-1)
            return self.network(inputs.transpose(0, 1).unsqueeze(0)).flatten()

    return TemporalMILHead()


def paired_improves(candidate: dict, reference: dict, min_localization_gain_sec: float = 0.5) -> bool:
    """Require a paired correction with no regression, or meaningful timing gain."""
    if candidate.get("invalid_predictions", 0):
        return False
    candidate_rows = candidate.get("per_source", [])
    reference_rows = reference.get("per_source", [])
    current = {row["sample_id"]: row for row in candidate_rows}
    previous = {row["sample_id"]: row for row in reference_rows}
    if (not current or current.keys() != previous.keys()
            or len(current) != len(candidate_rows) or len(previous) != len(reference_rows)):
        return False
    if any(row.get("truth") is not previous[key].get("truth") for key, row in current.items()):
        return False
    for row in candidate_rows + reference_rows:
        score, error = row.get("score"), row.get("localization_error_sec")
        if score is not None and (not isinstance(score, (int, float)) or not math.isfinite(score)):
            return False
        if error is not None and (not isinstance(error, (int, float)) or not math.isfinite(error)):
            return False
    if any(type(row.get("truth")) is not bool or type(row.get("correct")) is not bool
           for row in candidate_rows + reference_rows):
        return False
    if any(previous[key]["correct"] and not current[key]["correct"] for key in current):
        return False
    if any(current[key]["correct"] and not previous[key]["correct"] for key in current):
        return True
    current_errors = {key: row.get("localization_error_sec") for key, row in current.items()
                      if row.get("localization_error_sec") is not None}
    previous_errors = {key: row.get("localization_error_sec") for key, row in previous.items()
                       if row.get("localization_error_sec") is not None}
    if not current_errors or current_errors.keys() != previous_errors.keys():
        return False
    previous_mean = sum(previous_errors.values()) / len(previous_errors)
    current_mean = sum(current_errors.values()) / len(current_errors)
    return previous_mean - current_mean >= min_localization_gain_sec


def gaussian_target(times: Any, point_sec: float, sigma_sec: float = 1.0) -> Any:
    import torch

    if times.ndim != 1 or times.numel() == 0 or sigma_sec <= 0:
        raise ValueError("invalid localization target")
    weights = torch.exp(-0.5 * ((times - point_sec) / sigma_sec) ** 2)
    return weights / weights.sum().clamp_min(1e-12)


def mil_loss(logits: Any, has_violation: bool, times: Any, point_sec: float | None,
             localization_weight: float = 1.0) -> tuple[Any, dict[str, float]]:
    import torch.nn.functional as functional

    if logits.ndim != 1 or logits.numel() == 0 or times.shape != logits.shape:
        raise ValueError("logits and times must be matching nonempty vectors")
    bag_logit = logits.max()
    target = logits.new_tensor(float(has_violation))
    binary = functional.binary_cross_entropy_with_logits(bag_logit, target)
    localization = logits.new_zeros(())
    if has_violation and point_sec is not None:
        soft_target = gaussian_target(times, point_sec)
        localization = -(soft_target * functional.log_softmax(logits, dim=0)).sum()
    total = binary + localization_weight * localization
    return total, {"binary": float(binary.detach()), "localization": float(localization.detach())}

