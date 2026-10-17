"""Visibility coverage gaps; missing data never proves absence."""

from __future__ import annotations

from collections.abc import Iterable

from navidu.contracts.domain import Coverage


def derive_coverage(
    timestamps: Iterable[float],
    *,
    expected_period_sec: float,
    duration_sec: float,
) -> list[Coverage]:
    """Mark decoder gaps as whole-frame coverage loss.

    Object/region occlusion must be supplied by an actual perception capability;
    this function does not infer visibility from a missing detection.
    """
    values = sorted(set(float(value) for value in timestamps))
    if not values:
        return [Coverage(start_sec=0, end_sec=duration_sec, scope="whole", reason="out_of_frame")]
    result: list[Coverage] = []
    gap_limit = max(expected_period_sec * 1.5, 0.5)
    if values[0] > gap_limit:
        result.append(Coverage(start_sec=0, end_sec=values[0], scope="whole",
                               reason="out_of_frame"))
    for left, right in zip(values, values[1:]):
        if right - left > gap_limit:
            result.append(Coverage(start_sec=left, end_sec=right, scope="whole",
                                   reason="out_of_frame"))
    if duration_sec - values[-1] > gap_limit:
        result.append(Coverage(start_sec=values[-1], end_sec=duration_sec, scope="whole",
                               reason="out_of_frame"))
    return result
