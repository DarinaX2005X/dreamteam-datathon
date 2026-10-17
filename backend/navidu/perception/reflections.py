"""Conservative validation and interval filtering for explicit mirror claims."""

from __future__ import annotations

from dataclasses import replace

from navidu.pipeline.interfaces import PerceptionArtifacts


def _overlap(a, b) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if not intersection:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def validate_reflections(payload: dict, frame_tracks: list[dict], timestamps: list[float]) -> list[dict]:
    """Accept only explicit, simultaneous, multi-frame source/reflection pairs."""
    accepted = []
    for value in payload.get("reflections", [])[:12]:
        if not isinstance(value, dict):
            continue
        reflected, source = value.get("reflected_track"), value.get("source_track")
        support = value.get("frames")
        if not isinstance(reflected, str) or not isinstance(source, str) or reflected == source \
                or not isinstance(support, list) or any(type(index) is not int for index in support):
            continue
        indices = sorted(set(support))
        if len(indices) < 3 or any(not 0 <= index < len(timestamps) for index in indices) \
                or any(right != left + 1 for left, right in zip(indices, indices[1:])):
            continue
        if any(reflected not in frame_tracks[index] or source not in frame_tracks[index]
               or _overlap(frame_tracks[index][reflected], frame_tracks[index][source]) > 0.2
               for index in indices):
            continue
        accepted.append({"reflected_track": reflected, "source_track": source,
                         "start_sec": timestamps[indices[0]], "end_sec": timestamps[indices[-1]]})
    return accepted


def apply_reflection_filters(artifacts: PerceptionArtifacts) -> dict[str, float | int]:
    """Remove reflected data in supported intervals and report numeric impact.

    Facts fully inside an interval are rebound to the physical source without
    changing their time or provenance. Partial overlaps are removed whole because
    clipping a model-produced fact would fabricate new endpoints. Track samples
    and point detections remain intact outside the interval.
    """
    claims = [signal for signal in artifacts.semantic_signals
              if signal.kind == "reflection" and len(signal.subjects) == 2]
    valid = []
    for claim in claims:
        reflected, source = claim.subjects
        conflicting = any(other is not claim and not (
            other.end_sec < claim.start_sec or claim.end_sec < other.start_sec
        ) and ((reflected == other.subjects[0] and source != other.subjects[1])
               or source == other.subjects[0] or reflected == other.subjects[1])
                          for other in claims)
        if not conflicting:
            valid.append((reflected, source, claim.start_sec, claim.end_sec))

    # Overlapping windows can confirm one continuous pair. Merge only touching
    # intervals for that same pair; never fill an unsupported temporal gap.
    merged = []
    for reflected, source in sorted({(item[0], item[1]) for item in valid}):
        intervals = sorted((left, right) for target, original, left, right in valid
                           if (target, original) == (reflected, source))
        for left, right in intervals:
            if merged and merged[-1][:2] == (reflected, source) and left <= merged[-1][3]:
                previous = merged.pop()
                merged.append((reflected, source, previous[2], max(previous[3], right)))
            else:
                merged.append((reflected, source, left, right))
    valid = merged

    def suppressed(track_id: str | None, start: float, end: float | None = None) -> bool:
        end = start if end is None else end
        return bool(track_id) and any(track_id == reflected and left <= end and start <= right
                                      for reflected, _, left, right in valid)

    def mapping(subjects, start: float, end: float):
        result = {}
        for subject in subjects:
            matches = {source for reflected, source, left, right in valid
                       if subject == reflected and left <= start and end <= right}
            if len(matches) == 1:
                result[subject] = matches.pop()
        return result

    canonicalized_observations = 0
    rejected_self_handovers = 0
    retained_observations = []
    for item in artifacts.observations:
        remap = mapping(item.subjects, item.start_sec, item.end_sec)
        if any(suppressed(subject, item.start_sec, item.end_sec) and subject not in remap
               for subject in item.subjects):
            continue
        if remap:
            subjects = list(dict.fromkeys(remap.get(subject, subject) for subject in item.subjects))
            actor = remap.get(item.actor_id, item.actor_id)
            recipient = remap.get(item.recipient_id, item.recipient_id)
            if item.kind == "cash_handover" and actor == recipient:
                rejected_self_handovers += 1
                continue
            retained_observations.append(item.model_copy(update={
                "subjects": subjects, "actor_id": actor, "recipient_id": recipient,
            }))
            canonicalized_observations += 1
        elif any(suppressed(subject, item.start_sec, item.end_sec) for subject in item.subjects):
            continue
        else:
            retained_observations.append(item)

    canonicalized_signals = 0
    retained_signals = []
    for item in artifacts.semantic_signals:
        if item.kind == "reflection":
            retained_signals.append(item)
            continue
        remap = mapping(item.subjects, item.start_sec, item.end_sec)
        if any(suppressed(subject, item.start_sec, item.end_sec) and subject not in remap
               for subject in item.subjects):
            continue
        if remap:
            subjects = tuple(dict.fromkeys(remap.get(subject, subject) for subject in item.subjects))
            if item.kind == "cash_handover" and len(subjects) < 2:
                rejected_self_handovers += 1
                continue
            retained_signals.append(replace(item, subjects=subjects))
            canonicalized_signals += 1
        elif any(suppressed(subject, item.start_sec, item.end_sec) for subject in item.subjects):
            continue
        else:
            retained_signals.append(item)

    before_samples = sum(len(track.samples) for track in artifacts.tracks)
    before_detections = len(artifacts.raw_detections)
    before_keypoints = len(artifacts.keypoints)
    before_observations = len(artifacts.observations)
    before_signals = len([item for item in artifacts.semantic_signals if item.kind != "reflection"])
    retained_tracks = []
    for track in artifacts.tracks:
        samples = [sample for sample in track.samples if not suppressed(track.track_id, sample.t)]
        if samples:
            track.samples = samples
            track.first_sec, track.last_sec = samples[0].t, samples[-1].t
            retained_tracks.append(track)
    artifacts.tracks = retained_tracks
    artifacts.raw_detections = [item for item in artifacts.raw_detections
                                if not suppressed(item.track_id, item.timestamp_sec)]
    artifacts.keypoints = [item for item in artifacts.keypoints
                           if not suppressed(item.track_id, item.timestamp_sec)]
    artifacts.observations = retained_observations
    artifacts.semantic_signals = retained_signals
    intervals = {(reflected, source, left, right) for reflected, source, left, right in valid}
    return {
        "accepted_interval_count": len(intervals),
        "affected_duration_sec": round(sum(right - left for _, _, left, right in intervals), 3),
        "removed_track_samples": before_samples - sum(len(track.samples) for track in artifacts.tracks),
        "removed_raw_detections": before_detections - len(artifacts.raw_detections),
        "removed_keypoints": before_keypoints - len(artifacts.keypoints),
        "removed_observations": before_observations - len(artifacts.observations),
        "removed_semantic_signals": before_signals - len([
            item for item in artifacts.semantic_signals if item.kind != "reflection"
        ]),
        "canonicalized_observations": canonicalized_observations,
        "canonicalized_semantic_signals": canonicalized_signals,
        "rejected_self_handovers": rejected_self_handovers,
    }
