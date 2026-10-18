"""Compare original, conventional and learned enlargement on reviewed crops."""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import modal

from deploy.modal_training import image, models, private

app = modal.App("lado-object-upscale")
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
MODEL_SIZE = 4885111
MODEL_SHA256 = "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292"


@app.function(image=image, cpu=(1, 1), memory=(1024, 1024), timeout=180,
              retries=0, max_containers=1, scaledown_window=2, volumes={"/models": models})
def prepare():
    import urllib.request

    directory = Path("/models/realesr-general-x4v3")
    directory.mkdir(exist_ok=True)
    path = directory / "model.pth"
    if not path.exists():
        with urllib.request.urlopen(MODEL_URL, timeout=90) as response:
            content = response.read(MODEL_SIZE + 1)
        if len(content) != MODEL_SIZE:
            raise ValueError("Unexpected upstream model size")
        path.write_bytes(content)
    result = {"url": MODEL_URL, "size_bytes": path.stat().st_size,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "gpu_allocated": False}
    if result["size_bytes"] != MODEL_SIZE or result["sha256"] != MODEL_SHA256:
        raise ValueError("Downloaded upscaler does not match the pinned release")
    (directory / "pin.json").write_text(json.dumps(result) + "\n")
    models.commit()
    return result


@app.function(image=image, gpu="L4", cpu=(2, 2), memory=(8192, 8192), timeout=600,
              retries=0, max_containers=1, scaledown_window=2,
              volumes={"/private": private, "/models": models})
def compare(run_id, expected_sha256):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    from training.object_data import detection_metrics
    from training.super_resolution import CropUpscaler
    from training.train_objects import predict_frame

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id) or not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
        raise ValueError("Invalid comparison identifiers")
    path = Path("/models/realesr-general-x4v3/model.pth")
    if path.stat().st_size != MODEL_SIZE or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Upscaler integrity check failed")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    run = Path("/private/object-training") / run_id
    report = json.loads((run / "run-manifest.json").read_text())
    if report["state"] != "complete":
        raise ValueError("Training must finish before upscale comparison")
    split = json.loads((run / "split.json").read_text())
    records = [r for r in json.loads((run / "annotations.json").read_text())["annotations"]
               if r["source_sha256"] in split["validation_sources"] and not r.get("ignore_boxes")]
    processor = AutoImageProcessor.from_pretrained(run / "best", local_files_only=True)
    detector = AutoModelForObjectDetection.from_pretrained(run / "best", local_files_only=True).to("cuda").eval()
    upscaler = CropUpscaler(path)
    methods = {"original": None, "bicubic_x4": lambda crop: crop.resize((crop.width * 4, crop.height * 4), Image.Resampling.BICUBIC),
               "realesrgan_x4": upscaler}
    result = {"run_id": run_id, "upscaler_sha256": expected_sha256, "frames": len(records), "methods": {},
              "evidence_uses_original_frames": True}
    for name, transform in methods.items():
        predictions, durations = [], []
        for record in records:
            started = time.monotonic()
            with Image.open(record["remote_frame"]) as source:
                predictions.append(predict_frame(detector, processor, source.convert("RGB"),
                                                  threshold=.01, crop_transform=transform))
            durations.append(time.monotonic() - started)
        result["methods"][name] = {"metrics_by_threshold": {str(t): detection_metrics(predictions,
            [r["objects"] for r in records], threshold=t) for t in (.1, .2, .35)},
            "mean_frame_sec": sum(durations) / len(durations), "max_frame_sec": max(durations)}
        (run / ("upscale-" + name + "-predictions.json")).write_text(json.dumps(predictions) + "\n")
        print(json.dumps({"method": name, **result["methods"][name]}), flush=True)
    (run / "upscale-comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    private.commit()
    return result


@app.local_entrypoint()
def main(stage: str = "prepare", run_id: str = "", sha256: str = ""):
    result = prepare.remote() if stage == "prepare" else compare.remote(run_id, sha256)
    name = "realesr-general-x4v3-pin.json" if stage == "prepare" else run_id + "-upscale.json"
    Path("eval/runs", name).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
