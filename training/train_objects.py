"""Train RT-DETRv2 on reviewed spatial boxes, with video-source separation."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Mapping
from pathlib import Path

from training.object_data import (
    CLASSES,
    annotation_fingerprint,
    calibrated_metrics,
    crop_objects,
    object_views,
    split_by_source,
    suppress_duplicates,
    validate_annotations,
)


def move_to(value, device):
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: move_to(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to(item, device) for item in value]
    return value


def parameter_groups(model, learning_rate, backbone_learning_rate):
    """Adapt the visual backbone gently while learning new object categories."""
    backbone, head = [], []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = True
        (backbone if "backbone" in name else head).append(parameter)
    if not backbone or not head:
        raise ValueError("Expected both backbone and detector parameters")
    return [{"params": backbone, "lr": backbone_learning_rate},
            {"params": head, "lr": learning_rate}]


def freeze_batch_statistics(model):
    import torch

    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def training_views(record):
    width, height = record["width"], record["height"]
    candidates = object_views(width, height)
    views = [crop for crop in candidates if crop_objects(record['objects'], crop)]
    empty = [crop for crop in candidates if crop not in views]
    views.extend(empty[::max(1, len(empty) // 2)][:2])
    for obj in record["objects"]:
        if obj["kind"] not in {"hand", "cash", "identity_document"}:
            continue
        x1, y1, x2, y2 = obj["bbox_xyxy"]
        base = max(96, round(max(x2 - x1, y2 - y1) * 5))
        variants = [(1., 0., 0.)]
        if obj["kind"] in {"cash", "identity_document"}:
            variants.extend([(1.25, -.2, .15), (1.5, .2, -.15)])
        if obj["kind"] == "identity_document":
            variants.extend([(1., -.2, -.2), (1., .2, .2), (1.75, -.2, .2), (2., .2, -.2)])
        for scale, shift_x, shift_y in variants:
            side = min(width, height, round(base * scale))
            left = min(width - side, max(0, round((x1 + x2 - side) / 2 + side * shift_x)))
            top = min(height - side, max(0, round((y1 + y2 - side) / 2 + side * shift_y)))
            views.append((left, top, left + side, top + side))
    return list(dict.fromkeys(views))


def make_training_input(record, crop, rng):
    from PIL import Image, ImageEnhance, ImageOps

    with Image.open(record["remote_frame"]) as source:
        pixels = source.convert("RGB").crop(crop)
    objects = crop_objects(record["objects"], crop)
    if rng.random() < .5:
        pixels = ImageOps.mirror(pixels)
        objects = [{**obj, "bbox_xyxy": [pixels.width - obj["bbox_xyxy"][2], obj["bbox_xyxy"][1],
                                         pixels.width - obj["bbox_xyxy"][0], obj["bbox_xyxy"][3]]}
                   for obj in objects]
    pixels = ImageEnhance.Brightness(pixels).enhance(rng.uniform(.85, 1.15))
    pixels = ImageEnhance.Contrast(pixels).enhance(rng.uniform(.85, 1.15))
    annotations = []
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox_xyxy"]
        annotations.append({"category_id": CLASSES.index(obj["kind"]),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
    return pixels, {"image_id": 0, "annotations": annotations}


def predict_frame(model, processor, pixels, *, threshold=.03, crop_transform=None):
    import torch

    detections = []
    views = object_views(*pixels.size)
    for offset in range(0, len(views), 2):
        current = views[offset:offset + 2]
        crops = [pixels.crop(crop) for crop in current]
        transformed = [crop_transform(crop) for crop in crops] if crop_transform else crops
        inputs = move_to(dict(processor(images=transformed, return_tensors="pt")), "cuda")
        with torch.inference_mode():
            outputs = model(**inputs)
        found = processor.post_process_object_detection(outputs, threshold=threshold,
            target_sizes=[image.size[::-1] for image in crops])
        for crop, result in zip(current, found):
            for box, score, label in zip(result["boxes"].tolist(), result["scores"].tolist(), result["labels"].tolist()):
                class_id = int(label)
                if not 0 <= class_id < len(CLASSES):
                    continue
                local = box
                full = [local[0] + crop[0], local[1] + crop[1], local[2] + crop[0], local[3] + crop[1]]
                full = [min(pixels.size[i % 2], max(0., value)) for i, value in enumerate(full)]
                if full[2] - full[0] < 2 or full[3] - full[1] < 2:
                    continue
                detections.append({"kind": CLASSES[class_id], "bbox_xyxy": full, "score": float(score)})
    return suppress_duplicates(detections)


def evaluate(model, processor, records):
    from PIL import Image

    model.eval()
    predictions = []
    for record in records:
        with Image.open(record["remote_frame"]) as source:
            predictions.append(predict_frame(model, processor, source.convert("RGB")))
    return calibrated_metrics(predictions, [record["objects"] for record in records]), predictions


def save_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def train(args):
    import torch
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForObjectDetection

    from eval.training import load_training_partition

    if not torch.cuda.is_available():
        raise RuntimeError("Object training is supported only on the authorized remote GPU")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    dataset = Path("/private/dataset")
    sources = load_training_partition(dataset / "eval/training-splits.json",
        inventory=dataset / "research/dataset/inventory.csv", labels=dataset / "eval/labels.csv",
        folds=dataset / "eval/folds.csv", dataset_dir=dataset / "data/dataset", partition="train")
    payload = json.loads(Path(args.annotations).read_text())
    reviewed = validate_annotations(payload, {source.source_sha256 for source in sources})
    # Partially reviewed frames remain available for later correction. Standard
    # DETR negative losses must not treat an unresolved object as background.
    records = [record for record in reviewed if not record.get("ignore_boxes")]
    train_records, valid_records = split_by_source(records, seed=args.seed)
    for record in records:
        path = Path(record["remote_frame"]).resolve()
        if not path.is_relative_to(Path("/private/object-labels").resolve()) or not path.is_file():
            raise ValueError("A reviewed frame is missing from the private workspace")
    manifest = {"recipe": "reviewed-spatial-v5-rare-class-crops", "annotation_sha256": annotation_fingerprint(payload),
        "base_model": "rtdetr-v2-r18", "base_revision": "5650961749fa93567c0d46fc7f43ea4f9e914107",
        "classes": list(CLASSES), "seed": args.seed, "epochs_requested": args.epochs,
        "learning_rate": args.learning_rate, "backbone_learning_rate": args.backbone_learning_rate,
        "train_sources": len({r["source_sha256"] for r in train_records}),
        "validation_sources": len({r["source_sha256"] for r in valid_records}),
        "train_frames": len(train_records), "validation_frames": len(valid_records),
        "excluded_ambiguous_frames": len(reviewed) - len(records), "epochs": [], "selected_epoch": None,
        "quality_approved": False, "state": "running"}
    save_json(output / "run-manifest.json", manifest)
    save_json(output / "annotations.json", payload)
    save_json(output / "split.json", {"train_sources": sorted({r["source_sha256"] for r in train_records}),
                                     "validation_sources": sorted({r["source_sha256"] for r in valid_records})})
    configuration = AutoConfig.from_pretrained(args.base_model, local_files_only=True)
    configuration.id2label = dict(enumerate(CLASSES))
    configuration.label2id = {kind: index for index, kind in enumerate(CLASSES)}
    configuration.disable_custom_kernels = True
    processor = AutoImageProcessor.from_pretrained(args.base_model, local_files_only=True)
    model = AutoModelForObjectDetection.from_pretrained(args.base_model, config=configuration,
        ignore_mismatched_sizes=True, local_files_only=True).to("cuda")
    groups = parameter_groups(model, args.learning_rate, args.backbone_learning_rate)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    views = [(record, crop) for record in train_records for crop in training_views(record)]
    rng = random.Random(args.seed)
    start = time.monotonic()
    best = -1.
    no_gain = 0
    optimizer_steps = 0
    baseline, _ = evaluate(model, processor, valid_records)
    manifest["initial_metrics"] = baseline
    manifest["trainable_parameters"] = sum(p.numel() for p in parameters)
    save_json(output / "run-manifest.json", manifest)
    print(json.dumps({"stage": "baseline", "train_frames": len(train_records), "validation_frames": len(valid_records),
                      "training_views": len(views), "metrics": baseline}), flush=True)
    for epoch in range(1, args.epochs + 1):
        if time.monotonic() - start > args.max_wall_minutes * 60:
            manifest["stop_reason"] = "wall_budget"
            break
        rng.shuffle(views)
        model.train()
        # Batch size two cannot estimate stable running statistics. Convolutions
        # and affine parameters still receive gradients in evaluation-mode BN.
        freeze_batch_statistics(model)
        losses = []
        for offset in range(0, len(views), 2):
            if time.monotonic() - start > args.max_wall_minutes * 60 - 60:
                manifest["stop_reason"] = "wall_budget"
                break
            samples = [make_training_input(record, crop, rng) for record, crop in views[offset:offset + 2]]
            inputs = move_to(dict(processor(images=[item[0] for item in samples],
                annotations=[item[1] for item in samples], return_tensors="pt")), "cuda")
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**inputs)
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite object training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            optimizer_steps += 1
            losses.append(float(loss.detach()))
            if optimizer_steps % 20 == 0:
                progress = {"stage": "training", "epoch": epoch, "optimizer_steps": optimizer_steps,
                    "loss": round(sum(losses) / len(losses), 6),
                    "elapsed_sec": round(time.monotonic() - start, 2),
                    "peak_cuda_mib": round(torch.cuda.max_memory_allocated() / 1048576, 1)}
                save_json(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        if not losses:
            break
        metrics, predictions = evaluate(model, processor, valid_records)
        record = {"epoch": epoch, "loss": sum(losses) / len(losses), "metrics": metrics,
                  "optimizer_steps": optimizer_steps, "elapsed_sec": round(time.monotonic() - start, 2)}
        manifest["epochs"].append(record)
        if metrics["macro_f1"] > best + 1e-6:
            best, no_gain = metrics["macro_f1"], 0
            model.save_pretrained(output / "best", safe_serialization=True)
            processor.save_pretrained(output / "best")
            save_json(output / "best-validation-predictions.json", predictions)
            manifest["selected_epoch"] = epoch
        else:
            no_gain += 1
        save_json(output / "run-manifest.json", manifest)
        print(json.dumps({"stage": "validation", **record, "selected_epoch": manifest["selected_epoch"]}), flush=True)
        if epoch >= 24 and no_gain >= 12:
            manifest["stop_reason"] = "validation_plateau"
            break
        # Decay after the head has learned the new classes; no validation labels
        # are used in gradient updates or image sampling.
        if epoch in {20, 30}:
            for group in optimizer.param_groups:
                group["lr"] *= .3
    model.save_pretrained(output / "last", safe_serialization=True)
    processor.save_pretrained(output / "last")
    manifest["state"] = "complete"
    manifest["elapsed_sec"] = round(time.monotonic() - start, 2)
    manifest["peak_cuda_mib"] = round(torch.cuda.max_memory_allocated() / 1048576, 1)
    manifest["optimizer_steps"] = optimizer_steps
    save_json(output / "run-manifest.json", manifest)
    print(json.dumps({"stage": "complete", "selected_epoch": manifest["selected_epoch"],
                      "best_macro_f1": best, "elapsed_sec": manifest["elapsed_sec"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--base-model", default="/models/rtdetr-v2-r18")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--backbone-learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-wall-minutes", type=float, default=30)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 60 or not 1 <= args.max_wall_minutes <= 40 \
            or not math.isfinite(args.learning_rate) or not 1e-6 <= args.learning_rate <= 1e-3 \
            or not math.isfinite(args.backbone_learning_rate) \
            or not 1e-7 <= args.backbone_learning_rate <= args.learning_rate:
        parser.error("Object training recipe exceeds its bounds")
    train(args)


if __name__ == "__main__":
    main()
