"""Bounded private training of the reviewed hand and small-object detector."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import modal

from deploy.modal_training import image, models, private

app = modal.App("lado-object-training")


@app.function(image=image, cpu=(2, 2), memory=(2048, 2048), timeout=120,
              retries=0, max_containers=1, scaledown_window=2, volumes={"/private": private})
def check_annotations(annotations: dict):
    from eval.training import load_training_partition
    from training.object_data import split_by_source, validate_annotations

    root = Path("/private/dataset")
    sources = load_training_partition(root / "eval/training-splits.json",
        inventory=root / "research/dataset/inventory.csv", labels=root / "eval/labels.csv",
        folds=root / "eval/folds.csv", dataset_dir=root / "data/dataset", partition="train")
    records = validate_annotations(annotations, {source.source_sha256 for source in sources})
    missing = [index for index, record in enumerate(records) if not Path(record["remote_frame"]).is_file()]
    outside = [index for index, record in enumerate(records) if not Path(record["remote_frame"]).resolve().is_relative_to(
        Path("/private/object-labels").resolve())]
    if missing or outside:
        raise ValueError(f"Missing frame indices: {missing}; invalid path indices: {outside}")
    train, valid = split_by_source([record for record in records if not record.get("ignore_boxes")])
    return {"checked_frames": len(records), "train_frames": len(train), "validation_frames": len(valid)}


@app.function(image=image, cpu=(2, 2), memory=(8192, 8192), timeout=300,
              retries=0, max_containers=1, scaledown_window=2, volumes={"/models": models})
def preflight():
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForObjectDetection

    from training.object_data import CLASSES
    from training.train_objects import freeze_batch_statistics, parameter_groups

    torch.set_num_threads(2)
    configuration = AutoConfig.from_pretrained("/models/rtdetr-v2-r18", local_files_only=True)
    configuration.id2label = dict(enumerate(CLASSES))
    configuration.label2id = {kind: index for index, kind in enumerate(CLASSES)}
    configuration.disable_custom_kernels = True
    model = AutoModelForObjectDetection.from_pretrained("/models/rtdetr-v2-r18", config=configuration,
        local_files_only=True, ignore_mismatched_sizes=True)
    groups = parameter_groups(model, 1e-4, 1e-5)
    processor = AutoImageProcessor.from_pretrained("/models/rtdetr-v2-r18", local_files_only=True)
    image = Image.new("RGB", (256, 256), color=(100, 110, 120))
    labels = {"image_id": 0, "annotations": [{"category_id": 1, "bbox": [80, 90, 25, 30],
                                              "area": 750, "iscrowd": 0}]}
    batch = processor(images=image, annotations=labels, size={"height": 256, "width": 256}, return_tensors="pt")
    model.train()
    freeze_batch_statistics(model)
    output = model(**batch)
    if not torch.isfinite(output.loss):
        raise RuntimeError("Non-finite synthetic loss")
    output.loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("Invalid object training gradients")
    if not any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in groups[0]["params"]):
        raise RuntimeError("Visual backbone did not receive gradients")
    model.eval()
    with torch.inference_mode():
        predictions = model(pixel_values=batch["pixel_values"])
    results = processor.post_process_object_detection(predictions, threshold=.01, target_sizes=[(256, 256)])
    return {"gpu_allocated": False, "synthetic_backward": True, "output_classes": predictions.logits.shape[-1],
            "finite_gradient_tensors": len(gradients), "backbone_gradients": True,
            "postprocessing": len(results) == 1}


@app.function(image=image, gpu="L4", cpu=(2, 2), memory=(16384, 16384),
              timeout=2100, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/models": models, "/private": private})
def train(run_id: str, annotations: dict):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("Invalid training run ID")
    output = Path("/private/object-training") / run_id
    if output.exists():
        raise ValueError("Run already exists")
    annotation_file = Path("/private/object-labels") / f"{run_id}-annotations.json"
    annotation_file.write_text(json.dumps(annotations) + "\n")
    private.commit()
    subprocess.run([sys.executable, "scripts/fetch_models.py", "--models-dir", "/models",
                    "--manifest", "/models/manifest.json", "--model", "rtdetr-v2-r18", "--verify-only"],
                   check=True, timeout=120)
    try:
        subprocess.run([sys.executable, "-m", "training.train_objects", "--annotations", str(annotation_file),
            "--output", str(output), "--epochs", "40", "--max-wall-minutes", "30"], check=True, timeout=1950)
    finally:
        private.commit()
    result = json.loads((output / "run-manifest.json").read_text())
    # The manifest contains aggregate metrics and a dataset fingerprint only.
    return result


@app.function(image=image, cpu=(1, 1), memory=(2048, 2048), timeout=120,
              retries=0, max_containers=1, scaledown_window=2, volumes={"/private": private})
def calibrate(run_id: str):
    from training.object_data import CLASSES, calibrated_metrics, detection_metrics

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("Invalid calibration run ID")
    run = Path("/private/object-training") / run_id
    report = json.loads((run / "run-manifest.json").read_text())
    if report["state"] != "complete":
        raise ValueError("Wait for training to complete")
    split = json.loads((run / "split.json").read_text())
    records = [r for r in json.loads((run / "annotations.json").read_text())["annotations"]
               if r["source_sha256"] in split["validation_sources"] and not r.get("ignore_boxes")]
    predictions = json.loads((run / "best-validation-predictions.json").read_text())
    targets = [r["objects"] for r in records]
    # Never select a threshold below the floor used to retain predictions.
    # All calibration uses inner validation sources only.
    if report.get("recipe") == "reviewed-spatial-v5-rare-class-crops":
        thresholds = calibrated_metrics(predictions, targets)["threshold"]
    else:
        grid = [round(.1 + index * .025, 3) for index in range(19)]
        candidates = {threshold: detection_metrics(predictions, targets, threshold=threshold) for threshold in grid}
        thresholds = {kind: max(grid, key=lambda threshold: (
            candidates[threshold]["classes"][kind]["f1"],
            candidates[threshold]["classes"][kind]["precision"], threshold)) for kind in CLASSES}
    result = {"partition": "inner-training-validation", "annotation_sha256": report["annotation_sha256"],
        "selected_epoch": report["selected_epoch"], "detection_thresholds": thresholds,
        "metrics": detection_metrics(predictions, targets, threshold=thresholds)}
    (run / "calibrated-thresholds.json").write_text(json.dumps(result, indent=2) + "\n")
    private.commit()
    return result


@app.function(image=image, cpu=(2, 2), memory=(2048, 2048), timeout=180,
              retries=0, max_containers=1, scaledown_window=2,
              volumes={"/models": models, "/private": private})
def export_model(run_id: str, model_id: str, identity_run_id: str = ""):
    if not all(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", value) for value in (run_id, model_id)):
        raise ValueError("Invalid export identifier")
    run = Path("/private/object-training") / run_id
    report = json.loads((run / "run-manifest.json").read_text())
    if report["state"] != "complete" or not report["selected_epoch"]:
        raise ValueError("Training has not produced a completed checkpoint")
    selected = next(epoch for epoch in report["epochs"] if epoch["epoch"] == report["selected_epoch"])
    calibration_file = run / "calibrated-thresholds.json"
    calibration = json.loads(calibration_file.read_text()) if calibration_file.is_file() else None
    if calibration and calibration["annotation_sha256"] != report["annotation_sha256"]:
        raise ValueError("Calibration does not belong to this training run")
    identity = None
    if identity_run_id:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", identity_run_id):
            raise ValueError("Invalid identity run identifier")
        identity_directory = run.parent / identity_run_id
        identity_report = json.loads((identity_directory / "run-manifest.json").read_text())
        identity_calibration = json.loads((identity_directory / "calibrated-thresholds.json").read_text())
        if identity_report["state"] != "complete" or (
                identity_calibration["annotation_sha256"] != identity_report["annotation_sha256"]):
            raise ValueError("Identity checkpoint or calibration is incomplete")
        identity = {"local_dir": f"{model_id}/identity-specialist", "training_run_id": identity_run_id,
            "annotation_sha256": identity_report["annotation_sha256"],
            "selected_epoch": identity_report["selected_epoch"],
            "detection_threshold": identity_calibration["detection_thresholds"]["identity_document"]}
    destination = Path("/models") / model_id
    if destination.exists():
        raise ValueError("Export destination already exists")
    destination.mkdir()
    files = []
    for name in ("config.json", "model.safetensors", "preprocessor_config.json"):
        source = run / "best" / name
        shutil.copyfile(source, destination / name)
        files.append({"path": f"{model_id}/{name}", "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                      "size_bytes": source.stat().st_size})
    if identity is not None:
        (destination / "identity-specialist").mkdir()
        for name in ("config.json", "model.safetensors", "preprocessor_config.json"):
            source = identity_directory / "best" / name
            shutil.copyfile(source, destination / "identity-specialist" / name)
            files.append({"path": f"{identity['local_dir']}/{name}",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "size_bytes": source.stat().st_size})
    entry = {"id": model_id, "name": "Lado hands, cash, identity documents and money counters",
        "runtime": "transformers-rt-detr-v2", "local_dir": model_id,
        "revision": next(file["sha256"] for file in files if file["path"].endswith("model.safetensors")),
        "base_model": report["base_model"], "base_revision": report["base_revision"],
        "training_run_id": run_id, "annotation_sha256": report["annotation_sha256"],
        "selected_epoch": report["selected_epoch"],
        "license": "Apache-2.0", "distribution": "private", "status": "experimental",
        "task": "object-detection", "files": files,
        "detection_thresholds": (calibration["detection_thresholds"] if calibration else
            selected["metrics"]["threshold"] if isinstance(selected["metrics"]["threshold"], dict)
            else dict.fromkeys(report["classes"], selected["metrics"]["threshold"]))}
    if identity is not None:
        entry["identity_specialist"] = identity
        entry["detection_thresholds"] = {**entry["detection_thresholds"],
            "identity_document": identity["detection_threshold"]}
    manifest_file = Path("/models/manifest.json")
    manifest = json.loads(manifest_file.read_text())
    if any(model["id"] == model_id for model in manifest["models"]):
        raise ValueError("Exported model ID already exists")
    manifest["models"].append(entry)
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    models.commit()
    return {"entry": entry, "selected_metrics": selected["metrics"], "calibration": calibration, "quality_approved": False}


@app.function(image=image, gpu="L4", cpu=(2, 2), memory=(8192, 8192), timeout=300,
              retries=0, max_containers=1, scaledown_window=2,
              volumes={"/models": models, "/private": private})
def verify_identity(model_id: str):
    import numpy as np
    import torch
    from navidu.perception.assets import model_entry, read_manifest, verify_model
    from navidu.perception.object_detector import CLASS_OFFSET, CLASSES, ReviewedObjectDetector
    from PIL import Image

    from training.object_data import annotation_fingerprint, detection_metrics

    if not re.fullmatch(r"lado-objects-[a-z0-9-]{1,40}", model_id):
        raise ValueError("Invalid private model identifier")
    entry = model_entry(read_manifest(Path("/models")), model_id)
    verify_model(Path("/models"), entry)
    specialist = entry["identity_specialist"]
    run = Path("/private/object-training") / specialist["training_run_id"]
    annotations = json.loads((run / "annotations.json").read_text())
    if annotation_fingerprint(annotations) != specialist["annotation_sha256"]:
        raise ValueError("Identity annotation fingerprint changed")
    split = json.loads((run / "split.json").read_text())
    records = [record for record in annotations["annotations"] if not record.get("ignore_boxes")
               and record["source_sha256"] in split["validation_sources"]]
    torch.set_num_threads(2)
    detector = ReviewedObjectDetector(Path("/models"), entry, "cuda")
    predictions, targets = [], []
    for record in records:
        path = Path(record["remote_frame"]).resolve()
        if not path.is_relative_to(Path("/private/object-labels").resolve()):
            raise ValueError("Invalid private frame path")
        with Image.open(path) as pixels:
            boxes, kinds, scores = detector.predict(np.asarray(pixels.convert("RGB")))
            boxes *= np.array([pixels.width, pixels.height, pixels.width, pixels.height])
        predictions.append([{"kind": "identity_document", "bbox_xyxy": box.tolist(), "score": float(score)}
                            for box, kind, score in zip(boxes, kinds, scores, strict=True)
                            if CLASSES[int(kind) - CLASS_OFFSET] == "identity_document"])
        targets.append([item for item in record["objects"] if item["kind"] == "identity_document"])
    result = {"model_id": model_id, "frames": len(records),
        "positive_frames": sum(bool(target) for target in targets),
        "scope": "Identity specialist inner validation. Primary detector partitions may overlap; not a final test.",
        "metrics": detection_metrics(predictions, targets, threshold=detector.thresholds)["classes"]["identity_document"],
        "quality_approved": False}
    (run / "composite-inner-identities.json").write_text(json.dumps(result, indent=2) + "\n")
    private.commit()
    return result


@app.local_entrypoint()
def main(stage: str = "preflight", run_id: str = "",
         annotations: str = "training/runs/manual-object-labels/manual-annotations.json", model_id: str = "",
         identity_run_id: str = ""):
    if stage == "preflight":
        result = preflight.remote()
    elif stage == "train":
        labels = json.loads(Path(annotations).read_text())
        print(json.dumps(check_annotations.remote(labels)), flush=True)
        result = train.remote(run_id, labels)
        Path("eval/runs", run_id + ".json").write_text(json.dumps(result, indent=2) + "\n")
    elif stage == "export":
        result = export_model.remote(run_id, model_id, identity_run_id)
        Path("eval/runs", model_id + "-export.json").write_text(json.dumps(result, indent=2) + "\n")
    elif stage == "calibrate":
        result = calibrate.remote(run_id)
        Path("eval/runs", run_id + "-calibration.json").write_text(json.dumps(result, indent=2) + "\n")
    elif stage == "verify-identity":
        result = verify_identity.remote(model_id)
        Path("eval/runs", model_id + "-identity-inner.json").write_text(json.dumps(result, indent=2) + "\n")
    else:
        raise ValueError("Unknown stage")
    print(json.dumps(result))
