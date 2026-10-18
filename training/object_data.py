"""Reviewed spatial labels and source-disjoint splits for a small object detector."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import PurePosixPath

CLASSES = ("hand", "cash", "identity_document", "money_counter")
DETECTION_THRESHOLDS = {"hand": .1, "cash": .1, "identity_document": .1, "money_counter": .35}
CALIBRATION_THRESHOLDS = (.03, .05, .075, .1, .125, .15, .175, .2, .25, .3, .35, .4, .5, .6, .7)


def validate_annotations(payload, allowed_sources):
    if payload.get("schema_version") != 1 or payload.get("classes") != list(CLASSES):
        raise ValueError("Unsupported object annotation schema")
    records = payload.get("annotations")
    if not isinstance(records, list) or not records:
        raise ValueError("No reviewed object annotations")
    seen = set()
    for record in records:
        if record.get("source_sha256") not in allowed_sources:
            raise ValueError("Object annotation source is outside the training partition")
        if record.get("label_source") != "assistant_manual_review":
            raise ValueError("This training recipe requires reviewed boxes")
        if record.get("reviewed_classes") != list(CLASSES):
            raise ValueError("All target classes must be reviewed in each training frame")
        width, height, timestamp = record.get("width"), record.get("height"), record.get("timestamp_sec")
        if type(width) is not int or type(height) is not int or min(width, height) <= 0:
            raise ValueError("Invalid source dimensions")
        if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Invalid source timestamp")
        key = (record["source_sha256"], timestamp)
        if key in seen:
            raise ValueError("Duplicate source frame")
        seen.add(key)
        path = PurePosixPath(record.get("remote_frame", ""))
        if path.parts[:3] != ("/", "private", "object-labels") or ".." in path.parts:
            raise ValueError("Object frames must stay in the private annotation workspace")
        for obj in record.get("objects", []):
            box = obj.get("bbox_xyxy")
            if obj.get("kind") not in CLASSES or not isinstance(box, list) or len(box) != 4:
                raise ValueError("Invalid object class or box")
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in box):
                raise ValueError("Invalid box coordinates")
            if not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height:
                raise ValueError("Object box is outside the source image")
    return records


def split_by_source(records, *, seed=1729, validation_fraction=.2):
    sources = sorted({record["source_sha256"] for record in records})
    if len(sources) < 5:
        raise ValueError("At least five reviewed video sources are required")
    rng = random.Random(seed)
    class_sources = {kind: {r["source_sha256"] for r in records if any(o["kind"] == kind for o in r["objects"])}
                     for kind in CLASSES}
    validation_count = max(2, round(len(sources) * validation_fraction))
    class_counts = {kind: sum(o['kind'] == kind for r in records for o in r['objects']) for kind in CLASSES}
    best = None
    for _ in range(256):
        rng.shuffle(sources)
        validation = set(sources[:validation_count])
        if all(len(group) < 2 or (group & validation and group - validation) for group in class_sources.values()):
            counts = {kind: sum(o['kind'] == kind for r in records if r['source_sha256'] in validation
                               for o in r['objects']) for kind in CLASSES}
            imbalance = sum(abs(counts[kind] / total - validation_fraction)
                            for kind, total in class_counts.items() if total)
            imbalance += sum(.5 for group in class_sources.values() if len(group) >= 4 and len(group & validation) < 2)
            if best is None or imbalance < best[0]:
                best = imbalance, validation
    if best is None:
        raise ValueError("Cannot create source-disjoint class coverage; review more video sources")
    validation = best[1]
    train = [record for record in records if record["source_sha256"] not in validation]
    valid = [record for record in records if record["source_sha256"] in validation]
    assert {r["source_sha256"] for r in train}.isdisjoint(r["source_sha256"] for r in valid)
    return train, valid


def annotation_fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def crop_objects(objects, crop):
    """Clip visible object boxes to an image crop without inventing new labels."""
    x1, y1, x2, y2 = crop
    result = []
    for obj in objects:
        left, top, right, bottom = obj["bbox_xyxy"]
        clipped = [max(left, x1), max(top, y1), min(right, x2), min(bottom, y2)]
        area = max(0., clipped[2] - clipped[0]) * max(0., clipped[3] - clipped[1])
        if area < 9 or area < .3 * (right - left) * (bottom - top):
            continue
        result.append({"kind": obj["kind"], "bbox_xyxy": [clipped[0] - x1, clipped[1] - y1,
                                                            clipped[2] - x1, clipped[3] - y1]})
    return result


def object_views(width, height):
    from navidu.perception.object_detector import object_views as detector_views

    return detector_views(width, height)


def overlap(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(
        0., min(a[3], b[3]) - max(a[1], b[1]))
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return (intersection / union if union > 0 else 0.,
            intersection / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.)


def suppress_duplicates(detections):
    kept = []
    for item in sorted(detections, key=lambda d: d["score"], reverse=True):
        for old in kept:
            if old["kind"] != item["kind"]:
                continue
            iou, containment = overlap(old["bbox_xyxy"], item["bbox_xyxy"])
            if iou >= .45 or containment >= .85:
                break
        else:
            kept.append(item)
    return kept


def detection_metrics(predictions, targets, *, threshold=.35):
    result = {}
    for kind in CLASSES:
        cutoff = threshold[kind] if isinstance(threshold, dict) else threshold
        tp = fp = fn = 0
        for frame_predictions, frame_targets in zip(predictions, targets, strict=True):
            truth = [item for item in frame_targets if item["kind"] == kind]
            matched = set()
            for proposal in sorted((d for d in frame_predictions if d["kind"] == kind and d["score"] >= cutoff),
                                   key=lambda d: d["score"], reverse=True):
                candidates = [(overlap(proposal["bbox_xyxy"], target["bbox_xyxy"])[0], index)
                              for index, target in enumerate(truth) if index not in matched]
                best = max(candidates, default=(0., -1))
                if best[0] >= .5:
                    matched.add(best[1])
                    tp += 1
                else:
                    fp += 1
            fn += len(truth) - len(matched)
        precision = tp / (tp + fp) if tp + fp else 0.
        recall = tp / (tp + fn) if tp + fn else 0.
        result[kind] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
                        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.}
    return {"classes": result, "macro_f1": sum(v["f1"] for v in result.values()) / len(CLASSES),
            "threshold": threshold, "iou_threshold": .5}


def calibrated_metrics(predictions, targets):
    """Choose cutoffs from inner-validation predictions, retaining raw scores."""
    candidates = {value: detection_metrics(predictions, targets, threshold=value)
                  for value in CALIBRATION_THRESHOLDS}
    thresholds = {kind: max(CALIBRATION_THRESHOLDS, key=lambda value: (
        candidates[value]["classes"][kind]["f1"], candidates[value]["classes"][kind]["precision"], value))
        for kind in CLASSES}
    return detection_metrics(predictions, targets, threshold=thresholds)
