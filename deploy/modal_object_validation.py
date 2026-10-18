"""Spatial evaluation on development videos excluded from detector training."""
from __future__ import annotations

import json
import re
from pathlib import Path

import modal

from deploy.modal_training import image, private

app = modal.App("lado-object-validation")


def valid_run_id(value):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", value):
        raise ValueError("Invalid run ID")


def validation_sources():
    from eval.training import load_training_partition

    root = Path("/private/dataset")
    sources = load_training_partition(root / "eval/training-splits.json",
        inventory=root / "research/dataset/inventory.csv", labels=root / "eval/labels.csv",
        folds=root / "eval/folds.csv", dataset_dir=root / "data/dataset", partition="validation")
    if len(sources) != 8:
        raise ValueError("Development validation partition changed")
    return sorted(sources, key=lambda source: source.source_sha256)


@app.function(image=image, cpu=(2, 2), memory=(4096, 4096), timeout=300,
              retries=0, max_containers=1, scaledown_window=2, volumes={"/private": private})
def extract(run_id):
    from navidu.ingest.frames import frame_at
    from navidu.ingest.probe import probe_video
    from PIL import Image

    valid_run_id(run_id)
    output = Path("/private/object-labels") / run_id
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for source_index, source in enumerate(validation_sources()):
        metadata = probe_video(source.video_path)
        for sample_index, fraction in enumerate((.2, .4, .6, .8)):
            timestamp = source.duration_sec * fraction
            frame = frame_at(source.video_path, timestamp, probe=metadata)
            pixels = Image.fromarray(frame.rgb)
            name = f"frame-{source_index}-{sample_index}.png"
            pixels.save(output / name, format="PNG")
            records.append({"source_index": source_index, "sample_index": sample_index,
                "source_sha256": source.source_sha256, "timestamp_sec": frame.timestamp_sec,
                "frame": name, "size": list(pixels.size), "objects": []})
    (output / "proposals.json").write_text(json.dumps(records) + "\n")
    private.commit()
    return records


@app.function(image=image, gpu="L4", cpu=(2, 2), memory=(8192, 8192), timeout=300,
              retries=0, max_containers=1, scaledown_window=2, volumes={"/private": private})
def evaluate(training_run_id, annotations):
    import time

    import numpy as np
    import torch
    from navidu.perception.object_detector import (
        CLASS_OFFSET,
        CLASSES,
        ReviewedObjectDetector,
    )
    from PIL import Image

    from training.object_data import (
        DETECTION_THRESHOLDS,
        detection_metrics,
        validate_annotations,
    )

    valid_run_id(training_run_id)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    records = validate_annotations(annotations, {source.source_sha256 for source in validation_sources()})
    records = [record for record in records if not record.get("ignore_boxes")]
    run = Path("/private/object-training") / training_run_id
    report = json.loads((run / "run-manifest.json").read_text())
    if report["state"] != "complete":
        raise ValueError("Training must finish before evaluating the selected checkpoint")
    calibration = run / "calibrated-thresholds.json"
    thresholds = json.loads(calibration.read_text())["detection_thresholds"] if calibration.is_file() else DETECTION_THRESHOLDS
    detector = ReviewedObjectDetector(run, {"id": training_run_id, "local_dir": "best",
        "detection_thresholds": thresholds}, "cuda")
    predictions, times = [], []
    for record in records:
        path = Path(record["remote_frame"]).resolve()
        if not path.is_relative_to(Path("/private/object-labels").resolve()):
            raise ValueError("Invalid annotation frame path")
        started = time.monotonic()
        with Image.open(path) as pixels:
            boxes, classes, scores = detector.predict(np.asarray(pixels.convert("RGB")))
            boxes *= np.array([pixels.width, pixels.height, pixels.width, pixels.height])
            predictions.append([{"kind": CLASSES[int(kind) - CLASS_OFFSET], "bbox_xyxy": box.tolist(), "score": float(score)}
                                for box, kind, score in zip(boxes, classes, scores, strict=True)])
        times.append(time.monotonic() - started)
    result = {"training_run_id": training_run_id, "partition": "development-validation",
        "frames": len(records), "sources": len({record["source_sha256"] for record in records}),
        "selected_epoch": report["selected_epoch"], "metrics": detection_metrics(predictions,
            [record["objects"] for record in records], threshold=thresholds), "max_frame_sec": max(times, default=0),
        "mean_frame_sec": sum(times) / len(times) if times else 0,
        "quality_approved": False, "peak_cuda_mib": torch.cuda.max_memory_allocated() / 1048576}
    result["class_support"] = {kind: values["tp"] + values["fn"] for kind, values in result["metrics"]["classes"].items()}
    (run / "external-spatial-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    (run / "external-spatial-annotations.json").write_text(json.dumps(annotations) + "\n")
    (run / "external-spatial-predictions.json").write_text(json.dumps(predictions) + "\n")
    private.commit()
    return result


@app.function(image=image, gpu="L4", cpu=(2, 2), memory=(8192, 8192), timeout=600,
              retries=0, max_containers=1, scaledown_window=2, volumes={"/private": private})
def diagnose(run_id):
    import random
    from collections import Counter

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    from training.object_data import CLASSES, detection_metrics
    from training.train_objects import make_training_input, predict_frame

    valid_run_id(run_id)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    directory = Path("/private/object-training") / run_id
    report = json.loads((directory / "run-manifest.json").read_text())
    if report["state"] != "complete":
        raise ValueError("Wait for completed training")
    records = [r for r in json.loads((directory / "annotations.json").read_text())["annotations"]
               if not r.get("ignore_boxes")]
    split = json.loads((directory / "split.json").read_text())
    result = {"checkpoints": {}}
    for checkpoint in ("best", "last"):
        processor = AutoImageProcessor.from_pretrained(directory / checkpoint, local_files_only=True)
        model = AutoModelForObjectDetection.from_pretrained(directory / checkpoint, local_files_only=True).to("cuda").eval()
        sets = {}
        for partition, field in (("train", "train_sources"), ("validation", "validation_sources")):
            selected = [r for r in records if r["source_sha256"] in split[field]]
            predictions = []
            for record in selected:
                with Image.open(record["remote_frame"]) as source:
                    predictions.append(predict_frame(model, processor, source.convert("RGB"), threshold=.01))
            sets[partition] = {"metrics_by_threshold": {str(threshold): detection_metrics(predictions,
                [r["objects"] for r in selected], threshold=threshold) for threshold in (.01, .05, .1, .2, .35)},
                "max_score": {kind: max((d["score"] for p in predictions for d in p if d["kind"] == kind), default=0.)
                              for kind in CLASSES}, "frames": len(selected)}
        pixels, annotation = make_training_input(records[0],
            (0, 0, records[0]["width"], records[0]["height"]), random.Random(0))
        batch = processor(images=pixels, annotations=annotation, return_tensors="pt")
        result["checkpoints"][checkpoint] = {"sets": sets, "pixel_shape": list(batch["pixel_values"].shape),
            "processed_categories": dict(Counter(batch["labels"][0]["class_labels"].tolist())),
            "processed_boxes": batch["labels"][0]["boxes"].tolist(), "config_num_labels": model.config.num_labels}
        del model
        torch.cuda.empty_cache()
    (directory / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    private.commit()
    return result


@app.local_entrypoint()
def main(stage: str = "extract", run_id: str = "objects-validation-20261018",
         annotations: str = "training/runs/manual-object-labels/manual-validation.json"):
    if stage == "extract":
        records = extract.remote(run_id)
        output = Path("training/runs/manual-object-labels") / run_id
        output.mkdir(parents=True, exist_ok=True)
        (output / "proposals.json").write_text(json.dumps(records) + "\n")
        result = {"frames": len(records), "sources": 8, "gpu_allocated": False}
    elif stage == "evaluate":
        result = evaluate.remote(run_id, json.loads(Path(annotations).read_text()))
        Path("eval/runs", run_id + "-spatial-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    elif stage == "diagnose":
        result = diagnose.remote(run_id)
        Path("eval/runs", run_id + "-diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    else:
        raise ValueError("Unknown validation stage")
    print(json.dumps(result))
