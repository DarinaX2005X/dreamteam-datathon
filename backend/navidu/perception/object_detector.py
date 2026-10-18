"""Reviewed small-object detector alongside the general person detector."""
from __future__ import annotations

import math

CLASSES = ("hand", "cash", "identity_document", "money_counter")
CLASS_OFFSET = 1000


def object_views(width, height):
    return [(0, 0, width, height)] + [
        (round(left * width), round(top * height), round((left + .45) * width), round((top + .35) * height))
        for top in (0, .65 / 3, .65 * 2 / 3, .65) for left in (0, .275, .55)]


def overlap(left, right):
    area = max(0., min(left[2], right[2]) - max(left[0], right[0])) * max(
        0., min(left[3], right[3]) - max(left[1], right[1]))
    a = (left[2] - left[0]) * (left[3] - left[1])
    b = (right[2] - right[0]) * (right[3] - right[1])
    return area / (a + b - area) if a + b > area else 0., area / min(a, b) if min(a, b) > 0 else 0.


def suppress_duplicates(detections):
    result = []
    for detection in sorted(detections, key=lambda item: item["score"], reverse=True):
        if any(item["kind"] == detection["kind"] and (lambda value: value[0] >= .45 or value[1] >= .85)(
                overlap(item["bbox_xyxy"], detection["bbox_xyxy"])) for item in result):
            continue
        result.append(detection)
    return result


def identity_views(width, height, detections):
    """Select existing tiles from observed work-area objects, without fixed ROIs."""
    views = object_views(width, height)
    anchors = [item for item in detections if item["kind"] in {"hand", "cash", "money_counter"}]
    if not anchors:
        return views
    ranked = []
    for index, view in enumerate(views[1:]):
        score = sum((2. if item["kind"] == "money_counter" else 1.) *
                    overlap(view, item["bbox_xyxy"])[1] for item in anchors)
        if score > 0:
            ranked.append((score, index, view))
    return [view for _, _, view in sorted(ranked, key=lambda item: (-item[0], item[1]))[:4]]


class ReviewedObjectDetector:
    def __init__(self, directory, entry, device):
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        from .assets import artifact_path

        self.model_id = entry["id"]
        self.device = "cuda" if device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
        if device == "cuda" and self.device != "cuda":
            raise RuntimeError("CUDA_UNAVAILABLE: object detector requires CUDA")
        if self.device == "cuda":
            torch.set_float32_matmul_precision("high")
        path = artifact_path(directory, entry["local_dir"])
        self.processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForObjectDetection.from_pretrained(path, local_files_only=True).to(self.device).eval()
        labels = tuple(self.model.config.id2label[index] for index in range(len(CLASSES)))
        if labels != CLASSES or self.model.config.num_labels != len(CLASSES):
            raise RuntimeError("MODEL_MANIFEST_INVALID: unexpected spatial classes")
        self.thresholds = {kind: float(entry.get("detection_thresholds", {}).get(kind, .35)) for kind in CLASSES}
        if any(not math.isfinite(value) or not 0 < value < 1 for value in self.thresholds.values()):
            raise RuntimeError("MODEL_MANIFEST_INVALID: invalid spatial threshold")
        self.labels = {str(CLASS_OFFSET + index): kind for index, kind in enumerate(CLASSES)}
        self.identity = None
        specialist = entry.get("identity_specialist")
        if specialist is not None:
            threshold = float(specialist["detection_threshold"])
            if not math.isfinite(threshold) or not 0 < threshold < 1:
                raise RuntimeError("MODEL_MANIFEST_INVALID: invalid identity threshold")
            identity_path = artifact_path(directory, specialist["local_dir"])
            identity_processor = AutoImageProcessor.from_pretrained(identity_path, local_files_only=True)
            identity_model = AutoModelForObjectDetection.from_pretrained(
                identity_path, local_files_only=True).to(self.device).eval()
            if tuple(identity_model.config.id2label[index] for index in range(len(CLASSES))) != CLASSES:
                raise RuntimeError("MODEL_MANIFEST_INVALID: unexpected identity specialist classes")
            self.identity = (identity_model, identity_processor, threshold)
            self.thresholds["identity_document"] = threshold

    def _predict_views(self, pixels, views, model, processor, thresholds):
        import torch

        detections = []
        for offset in range(0, len(views), 4):
            current = views[offset:offset + 4]
            crops = [pixels.crop(crop) for crop in current]
            inputs = processor(images=crops, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                outputs = model(**inputs)
            found = processor.post_process_object_detection(outputs, threshold=min(thresholds.values()),
                target_sizes=[image.size[::-1] for image in crops])
            for crop, result in zip(current, found, strict=True):
                for box, score, label in zip(result["boxes"].tolist(), result["scores"].tolist(),
                                             result["labels"].tolist(), strict=True):
                    if not 0 <= label < len(CLASSES) or not math.isfinite(score):
                        continue
                    kind = CLASSES[label]
                    if kind not in thresholds or score < thresholds[kind] or not all(map(math.isfinite, box)):
                        continue
                    full = [box[0] + crop[0], box[1] + crop[1], box[2] + crop[0], box[3] + crop[1]]
                    full = [min(pixels.size[i % 2], max(0., value)) for i, value in enumerate(full)]
                    if full[2] - full[0] >= 2 and full[3] - full[1] >= 2:
                        detections.append({"kind": kind, "bbox_xyxy": full, "score": score})
        return suppress_duplicates(detections)

    def predict(self, rgb):
        import numpy as np
        from PIL import Image

        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Object detector expects RGB uint8")
        pixels = Image.fromarray(rgb)
        views = object_views(*pixels.size)
        thresholds = {kind: score for kind, score in self.thresholds.items()
                      if self.identity is None or kind != "identity_document"}
        detections = self._predict_views(pixels, views, self.model, self.processor, thresholds)
        if self.identity is not None:
            model, processor, threshold = self.identity
            detections.extend(self._predict_views(pixels, identity_views(*pixels.size, detections),
                model, processor, {"identity_document": threshold}))
        detections = suppress_duplicates(detections)
        boxes = np.array([item["bbox_xyxy"] for item in detections], dtype=np.float32).reshape(-1, 4)
        boxes /= np.array([pixels.width, pixels.height, pixels.width, pixels.height])
        return (boxes, np.array([CLASS_OFFSET + CLASSES.index(item["kind"]) for item in detections], dtype=np.int64),
                np.array([item["score"] for item in detections], dtype=np.float32))


class CombinedDetector:
    def __init__(self, general, objects):
        if set(general.labels) & set(objects.labels):
            raise ValueError("Detector class namespaces overlap")
        self.general, self.objects = general, objects
        self.labels = {**general.labels, **objects.labels}
        self.model_id = general.model_id
        self.device = general.device if general.device == objects.device else general.device + "+" + objects.device

    def source_model(self, class_id):
        return self.objects.model_id if str(class_id) in self.objects.labels else self.general.model_id

    def association_scores(self, class_id, scores):
        import numpy as np

        # Object proposals have already passed their calibrated class cutoff.
        # Associate them geometrically; ByteTrack's COCO score fusion would
        # discard these lower-scoring DETR classes even at identical boxes.
        # The runtime keeps the original scores in tracks and raw detections.
        return np.ones_like(scores) if str(class_id) in self.objects.labels else scores

    def predict(self, rgb):
        import numpy as np

        return tuple(np.concatenate(values) for values in zip(self.general.predict(rgb), self.objects.predict(rgb), strict=True))
