"""Deterministic conversion of retained geometry into temporal observations."""

from __future__ import annotations

from collections.abc import Iterable

from navidu.contracts.domain import Observation, Role, SceneLayout, Track
from navidu.contracts.units import frame_id
from navidu.scene.roles import _inside


def _runs(samples, predicate, max_gap_sec: float):
    run = []
    for sample in samples:
        if predicate(sample) and (not run or sample.t - run[-1].t <= max_gap_sec):
            run.append(sample)
        else:
            if run:
                yield run
            run = [sample] if predicate(sample) else []
    if run:
        yield run


def derive_observations(
    tracks: Iterable[Track],
    layout: SceneLayout,
    roles: dict[str, Role],
    *,
    max_gap_sec: float = 0.75,
    min_duration_sec: float = 0.25,
    start_index: int = 1,
) -> list[Observation]:
    """Derive only facts directly supported by track geometry.

    Semantic actions (handover, counting, bag damage/opening) must arrive from the
    perception runtime with provenance and are never guessed here.
    """
    observations: list[Observation] = []
    index = start_index
    for track in sorted(tracks, key=lambda item: item.track_id):
        if track.kind != "person":
            continue
        for zone in layout.zones:
            runs = _runs(
                track.samples,
                lambda sample, polygon=zone.polygon: _inside(
                    (sample.bbox[0] + sample.bbox[2]) / 2, sample.bbox[3], polygon
                ),
                max_gap_sec,
            )
            for run in runs:
                if run[-1].t - run[0].t < min_duration_sec:
                    continue
                observations.append(Observation(
                    obs_id=f"ob-{index}", kind="person_in_zone",
                    start_sec=run[0].t, end_sec=run[-1].t,
                    subjects=[track.track_id], zones=[zone.zone_id],
                    score=min(sample.score for sample in run),
                    frame_ids=[frame_id(run[0].t), frame_id(run[-1].t)],
                    provenance="geometry", model_id=None,
                ))
                index += 1
            for previous, current in zip(track.samples, track.samples[1:]):
                was_inside = _inside(
                    (previous.bbox[0] + previous.bbox[2]) / 2,
                    previous.bbox[3], zone.polygon,
                )
                is_inside = _inside(
                    (current.bbox[0] + current.bbox[2]) / 2,
                    current.bbox[3], zone.polygon,
                )
                if was_inside and not is_inside and current.t - previous.t <= max_gap_sec:
                    observations.append(Observation(
                        obs_id=f"ob-{index}", kind="person_left_zone",
                        start_sec=previous.t, end_sec=current.t,
                        subjects=[track.track_id], zones=[zone.zone_id],
                        score=min(previous.score, current.score),
                        frame_ids=[frame_id(previous.t), frame_id(current.t)],
                        provenance="geometry", model_id=None,
                    ))
                    index += 1
    return observations
