"""CPU checks and the explicitly invoked bounded Modal L4 onset-probe pilot."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import modal

from deploy.modal_training import ROOT, models, private
from deploy.modal_training import image as base_image

app = modal.App("lado-onset-probe")
image = (base_image
         .add_local_file(ROOT / "backend/pyproject.toml",
                         "/workspace/backend/pyproject.toml", copy=True)
         .add_local_file(ROOT / "deploy/modal_onset_probe.py",
                         "/workspace/deploy/modal_onset_probe.py", copy=True)
         .add_local_file(ROOT / "training/qwen_onset_probe.py",
                         "/workspace/training/qwen_onset_probe.py", copy=True)
         .add_local_file(ROOT / "training/train_qwen_onset_probe.py",
                         "/workspace/training/train_qwen_onset_probe.py", copy=True)
         .add_local_file(ROOT / "training/tests/test_qwen_onset_probe.py",
                         "/workspace/training/tests/test_qwen_onset_probe.py", copy=True)
         .pip_install("pytest>=8,<9", "ruff>=0.11,<1"))


@app.function(image=image, cpu=(2, 2), memory=(4096, 4096), timeout=300,
              retries=0, max_containers=1, min_containers=0, scaledown_window=2,
              volumes={"/models": models})
def cpu_tests() -> dict:
    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained("/models/qwen3-vl-4b", local_files_only=True)
    synthetic = Image.new("RGB", (320, 240), color=(17, 29, 41))
    batch = processor.image_processor(images=[synthetic], return_tensors="pt",
                                      min_pixels=65536, max_pixels=262144)
    if "pixel_values" not in batch or "image_grid_thw" not in batch:
        raise RuntimeError("vision-only Qwen processor smoke failed")
    commands = [
        [sys.executable, "-m", "pytest", "training/tests/test_qwen_onset_probe.py", "-q"],
        [sys.executable, "-m", "ruff", "check", "--config", "backend/pyproject.toml",
         "training/qwen_onset_probe.py", "training/train_qwen_onset_probe.py",
         "training/tests/test_qwen_onset_probe.py"],
    ]
    summaries = []
    for command in commands:
        result = subprocess.run(command, text=True, encoding="utf-8", errors="replace",
                                capture_output=True, timeout=240)
        print(result.stdout, flush=True)
        if result.returncode:
            print(result.stderr[-5000:], flush=True)
            raise RuntimeError("onset probe CPU check failed")
        summaries.append(result.stdout.splitlines()[-1])
    return {"gpu_allocated": False, "passed": True, "processor_smoke": {
        "pixel_values_shape": list(batch["pixel_values"].shape),
        "image_grid_thw": batch["image_grid_thw"].tolist()}, "summaries": summaries}


@app.function(image=image, gpu="L4", cpu=(2, 2), memory=(24576, 24576),
              timeout=1650, retries=0, max_containers=1, min_containers=0,
              scaledown_window=2, volumes={"/models": models, "/private": private})
def train(run_id: str) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("invalid run id")
    output = Path("/private/runs") / run_id
    if output.exists():
        raise RuntimeError("run already exists")
    command = [sys.executable, "training/train_qwen_onset_probe.py",
               "--base-model", "/models/qwen3-vl-4b", "--models-dir", "/models",
               "--dataset-dir", "/private/dataset/data/dataset",
               "--output-dir", str(output), "--cache-dir", "/private/cache/qwen-onset",
               "--training-manifest", "/private/dataset/eval/training-splits.json",
               "--inventory", "/private/dataset/research/dataset/inventory.csv",
               "--labels", "/private/dataset/eval/labels.csv",
               "--folds", "/private/dataset/eval/folds.csv"]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "2"}
    telemetry = Path("/private/runs") / f"{run_id}-gpu.csv"
    started = time.monotonic()
    with telemetry.open("w") as stream:
        monitor = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits", "-l", "5"], stdout=stream)
        process = subprocess.Popen(command, env=env)
        try:
            try:
                code = process.wait(timeout=1500)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=30)
                raise RuntimeError("hard 25-minute GPU budget reached") from None
        finally:
            if process.poll() is None:
                process.kill()
            monitor.terminate()
            monitor.wait(timeout=10)
            private.commit()
    if code:
        raise RuntimeError(f"onset probe exited with status {code}")
    report = json.loads((output / "run-manifest.json").read_text(encoding="utf-8"))
    return {"run_id": run_id, "selected": report["selected"],
            "oof_metrics": report["oof_metrics"],
            "validation_metrics": report["validation_metrics"],
            "elapsed_seconds": round(time.monotonic() - started, 3)}


@app.local_entrypoint()
def main(stage: str = "test", run_id: str = ""):
    if stage == "test":
        print(json.dumps(cpu_tests.remote()))
        return
    if stage != "train":
        raise ValueError("stage must be test or train")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("train requires a unique valid --run-id")
    try:
        print(json.dumps(train.remote(run_id)))
    finally:
        destination = ROOT / "training/runs"
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-X", "utf8", "-m", "modal", "volume", "get",
                        "lado-training-v1", f"/runs/{run_id}", str(destination)], check=True)
