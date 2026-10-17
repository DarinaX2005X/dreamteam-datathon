"""Validate visible people and object boxes; keep identity only across shared frames.

No object is remembered from location alone. Two matching observations at the
same source timestamps are required to carry a cash identity into another window.
"""
from __future__ import annotations

import math
from collections import defaultdict

from navidu.contracts.domain import Observation, Track, TrackSample
from navidu.contracts.units import frame_id
from navidu.pipeline.interfaces import PerceptionArtifacts, SemanticSignal


def _indices(value, count, *, minimum=2):
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        return None
    result = sorted(set(value))
    if len(result) < minimum or any(not 0 <= item < count for item in result):
        return None
    return result


def visible_people(payload, frame_tracks, timestamps, artifacts, model_id, prompt_version):
    """Only an explicit clear physical person claim can contribute to counting."""
    # Even a reflection claim with an unverified physical source contradicts
    # certainty that this track is physical in those exact frames. Preserve the
    # uncertainty across overlapping windows; this never hides a detector box
    # or establishes a confirmed reflection link.
    for item in payload.get("reflections", [])[:24]:
        if not isinstance(item, dict) or not isinstance(item.get("reflected_track"), str):
            continue
        indices = _indices(item.get("frames"), len(timestamps), minimum=1)
        if indices is None:
            continue
        for index in indices:
            if item["reflected_track"] in frame_tracks[index]:
                artifacts.semantic_signals.append(SemanticSignal(kind="uncertain_person",
                    start_sec=timestamps[index], end_sec=timestamps[index],
                    subjects=(item["reflected_track"],), model_id=model_id, prompt_version=prompt_version,
                    frame_ids=(frame_id(timestamps[index]),)))
    for item in payload.get("visible_people", [])[:24]:
        if not isinstance(item, dict) or item.get("clear") is not True \
                or item.get("physical") is not True or not isinstance(item.get("track"), str):
            continue
        source_indices = item.get("frames")
        if source_indices is None:
            start, end = item.get("start_frame"), item.get("end_frame")
            if type(start) is not int or type(end) is not int or not 0 <= start < end < len(timestamps):
                continue
            source_indices = list(range(start, end + 1))
        indices = _indices(source_indices, len(timestamps))
        if indices is None:
            continue
        # A detector can miss the first or last claimed frame. Retain only the
        # intersection of explicit model support and actual detections; never
        # fill the missing frame, and still require two supported frames.
        indices = [index for index in indices if item["track"] in frame_tracks[index]]
        if len(indices) < 2:
            continue
        for index in indices:
            artifacts.semantic_signals.append(SemanticSignal(kind="clear_person",
                start_sec=timestamps[index], end_sec=timestamps[index], subjects=(item["track"],),
                model_id=model_id, prompt_version=prompt_version,
                frame_ids=(frame_id(timestamps[index]),)))


def derive_simultaneous_people(artifacts: PerceptionArtifacts, *, max_gap_sec=1.5):
    """Run after reflection filtering; p3 over a video's lifetime means nothing."""
    known = {track.track_id: {sample.t for sample in track.samples}
             for track in artifacts.tracks if track.kind == "person"}
    uncertain = defaultdict(set)
    for signal in artifacts.semantic_signals:
        if signal.kind == "uncertain_person" and signal.start_sec == signal.end_sec:
            uncertain[signal.start_sec].update(signal.subjects)
    at_time = defaultdict(set)
    for signal in artifacts.semantic_signals:
        if signal.kind == "clear_person" and signal.start_sec == signal.end_sec:
            for person in signal.subjects:
                if signal.start_sec in known.get(person, set()) and person not in uncertain[signal.start_sec]:
                    at_time[signal.start_sec].add(person)
    groups = []
    for (left, people), (right, following) in zip(sorted(at_time.items()), sorted(at_time.items())[1:]):
        shared = tuple(sorted(people & following))
        if right - left > max_gap_sec or len(shared) < 3:
            continue
        if groups and groups[-1][2] == shared and groups[-1][1] == left:
            groups[-1] = (groups[-1][0], right, shared)
        else:
            groups.append((left, right, shared))
    next_id = max((int(item.obs_id[3:]) for item in artifacts.observations), default=0) + 1
    for left, right, people in groups:
        artifacts.observations.append(Observation(obs_id=f"ob-{next_id}",
            kind="extra_person_present", start_sec=left, end_sec=right, subjects=list(people),
            frame_ids=[frame_id(left), frame_id(right)], provenance="temporal_model"))
        next_id += 1
    artifacts.diagnostics["simultaneous_people"] = {"supported_frames": len(at_time),
        "uncertain_person_frames": sum(map(len, uncertain.values())),
        "maximum_clear_people": max(map(len, at_time.values()), default=0),
        "extra_person_intervals": len(groups)}


def _box(value):
    if not isinstance(value, list) or len(value) != 4 \
            or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1000
                   for x in value):
        return None
    left, top, right, bottom = [x / 1000 for x in value]
    if right - left < .002 or bottom - top < .002:
        return None
    return left, top, right, bottom


def _iou(left, right):
    intersection = max(0, min(left[2], right[2]) - max(left[0], right[0])) * \
        max(0, min(left[3], right[3]) - max(left[1], right[1]))
    union = (left[2] - left[0]) * (left[3] - left[1]) + \
        (right[2] - right[0]) * (right[3] - right[1]) - intersection
    return intersection / union if union > 0 else 0


def ground_objects(payload, timestamps, artifacts, model_id, prompt_version):
    """Return window-local to persistent object IDs, with actual source boxes."""
    mapping = {}
    previous_tracks = list(artifacts.tracks)
    claimed = set()
    next_id = max((int(track.track_id[1:]) for track in artifacts.tracks
                   if track.kind == "object"), default=1000000) + 1
    for item in payload.get("objects", [])[:12]:
        if not isinstance(item, dict) or item.get("kind") not in {"cash", "identity_document"} \
                or not isinstance(item.get("id"), str) or not 1 <= len(item["id"]) <= 16 \
                or item["id"] in mapping or not isinstance(item.get("samples"), list):
            continue
        samples = {}
        for sample in item["samples"][:16]:
            if not isinstance(sample, dict) or type(sample.get("frame")) is not int \
                    or not 0 <= sample["frame"] < len(timestamps):
                continue
            box = _box(sample.get("box"))
            if box is not None and (box[2] - box[0]) * (box[3] - box[1]) > .5:
                continue
            if sample.get("view", "full") == "detail":
                bounds = payload.get("_detail_bounds")
                if box is None or not isinstance(bounds, tuple) or len(bounds) != 4:
                    continue
                left, top, right, bottom = bounds
                box = (left + box[0] * (right - left), top + box[1] * (bottom - top),
                       left + box[2] * (right - left), top + box[3] * (bottom - top))
            elif sample.get("view", "full") != "full":
                continue
            if box is not None:
                samples[timestamps[sample["frame"]]] = box
        if len(samples) < 2:
            continue
        # Cash boxes covering a whole person or desk are not useful grounding.
        if any((box[2] - box[0]) * (box[3] - box[1]) > .25 for box in samples.values()):
            continue
        candidates = []
        for track in previous_tracks:
            if track.kind != "object" or track.object_class != item["kind"] or track.track_id in claimed:
                continue
            common = [sample for sample in track.samples if sample.t in samples]
            if len(common) >= 2 and all(_iou(sample.bbox, samples[sample.t]) >= .6 for sample in common):
                candidates.append(track)
        # Ambiguous identities remain separate, never pick a convenient one.
        track = candidates[0] if len(candidates) == 1 else None
        if track is None:
            track = Track(track_id=f"o{next_id}", kind="object", object_class=item["kind"],
                          first_sec=min(samples), last_sec=max(samples))
            next_id += 1
            artifacts.tracks.append(track)
        merged = {sample.t: sample for sample in track.samples}
        merged.update({timestamp: TrackSample(t=timestamp, bbox=box, score=None)
                       for timestamp, box in samples.items()})
        track.last_sec = max(track.last_sec, max(samples))
        track.first_sec = min(track.first_sec, min(samples))
        track.samples = [merged[key] for key in sorted(merged)]
        mapping[item["id"]] = (track, samples)
        claimed.add(track.track_id)
        artifacts.semantic_signals.append(SemanticSignal(kind=f"object:{item['kind']}",
            start_sec=min(samples), end_sec=max(samples), subjects=(track.track_id,),
            boxes=tuple(samples[key] for key in sorted(samples)), model_id=model_id,
            prompt_version=prompt_version, frame_ids=tuple(frame_id(key) for key in sorted(samples))))
    return mapping


def bind_grounded_events(payload, events, objects):
    for event in events:
        source = payload["events"][event["_source_index"]]
        ids = source.get("objects", [])
        if not isinstance(ids, list) or any(not isinstance(value, str) for value in ids):
            continue
        supported = []
        for local_id in dict.fromkeys(ids):
            if local_id not in objects:
                continue
            track, samples = objects[local_id]
            if sum(event["start_sec"] <= timestamp <= event["end_sec"] for timestamp in samples) < 2:
                continue
            if (event["kind"].startswith("cash_") or event["kind"] == "counting_motion") \
                    and track.object_class != "cash":
                continue
            if event["kind"] == "identity_document_visible" and track.object_class != "identity_document":
                continue
            supported.append(track)
        event["subjects"] = list(dict.fromkeys(event["subjects"] + [track.track_id for track in supported]))
        cash = [track for track in supported if track.object_class == "cash"]
        if len(cash) == 1:
            event["relation_id"] = f"cash:{cash[0].track_id}"
        elif len(cash) > 1:
            event.pop("relation_id", None)
