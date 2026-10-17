"""Bounded Modal L4 training; public setup precedes GPU allocation.

Run from the repository root with the isolated .modal-venv operator environment.
Private archives require explicit authorization for processing in this workspace.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("lado-training")
setup_app = modal.App("lado-setup")
models = modal.Volume.from_name("lado-models-v1", create_if_missing=True, version=2)
private = modal.Volume.from_name("lado-training-v1", create_if_missing=True, version=2)

# Explicit code paths: never send a checkout, .env, virtualenv, or local dataset.
image = (
    modal.Image.from_registry("pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime")
    .entrypoint([])
    .apt_install("ca-certificates", "ffmpeg")
    .add_local_file(ROOT / "backend/pyproject.toml", "/tmp/backend/pyproject.toml", copy=True)
    .add_local_dir(ROOT / "backend/navidu", "/tmp/backend/navidu", copy=True,
                   ignore=["**/__pycache__/**", "**/*.pyc"])
    .run_commands("python -m pip install --no-cache-dir '/tmp/backend[service,runtime,cpu,adapters]' && rm -rf /tmp/backend")
    .add_local_file(ROOT / "models/manifest.json", "/workspace/model-manifest.json", copy=True)
    .add_local_file(ROOT / "scripts/fetch_models.py", "/workspace/scripts/fetch_models.py", copy=True)
    .env({"PYTHONPATH": "/workspace", "USE_TF": "0", "USE_FLAX": "0",
          "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"})
    .workdir("/workspace")
)
for directory in ("training", "eval"):
    for source in sorted((ROOT / directory).glob("*.py")):
        image = image.add_local_file(source, f"/workspace/{directory}/{source.name}", copy=True)


@setup_app.function(image=image, cpu=(2, 2), memory=(4096, 4096), timeout=1800,
              retries=0, max_containers=1, min_containers=0, scaledown_window=2,
              volumes={"/models": models})
def prepare_models(profile: str = "baseline") -> dict:
    selections = {"baseline": ["internvl3_5-4b", "rfdetr-nano-onnx"],
                  "qwen": ["qwen3-vl-4b"]}
    if profile not in selections:
        raise ValueError("Unknown model preparation profile")
    shutil.copyfile("/workspace/model-manifest.json", "/models/manifest.json")
    selection_args = [part for model_id in selections[profile] for part in ("--model", model_id)]
    subprocess.run([sys.executable, "scripts/fetch_models.py", "--models-dir", "/models",
                    "--manifest", "/models/manifest.json", *selection_args,
                    "--workers", "8", "--retries", "2"],
                   check=True, timeout=1700)
    models.commit()
    return {"models_ready": True, "gpu_allocated": False}


@setup_app.function(image=image, cpu=(2, 2), memory=(4096, 4096), timeout=600,
              retries=0, max_containers=1, min_containers=0, scaledown_window=2,
              volumes={"/private": private})
def prepare_data() -> dict:
    from eval.prepare_transfer import extract, verify
    archive = Path("/private/development.tar")
    verified = verify(archive)
    if (verified["unique_video_sources"] != 32 or verified["reserved_final_holdout_video_sources"]
            or verified["metadata_scope_unique_sources"] != 32):
        raise RuntimeError("Only the fixed 32 development videos may enter this run")
    extract(archive, Path("/private/dataset"))
    private.commit()
    return {"development_sources": 32, "holdout_sources": 0, "gpu_allocated": False}


@app.function(image=image, gpu="L4", cpu=(4, 4), memory=(32768, 32768),
              timeout=6000, retries=0, max_containers=1, min_containers=0,
              scaledown_window=2, volumes={"/models": models, "/private": private})
def train(run_id: str) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("Invalid run id")
    output = Path("/private/runs") / run_id
    if output.exists():
        raise RuntimeError("Run exists; download its artifacts instead of starting it twice")
    output.mkdir(parents=True)
    # The trainer claims an empty output directory. Keep telemetry alongside it.
    telemetry = Path("/private/runs") / (run_id + "-gpu.csv")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    command = [sys.executable, "training/train_proposal.py", "--host-profile", "modal-l4",
               "--base-model", "/models/internvl3_5-4b", "--models-dir", "/models",
               "--output-dir", str(output), "--dataset-dir", "/private/dataset/data/dataset",
               "--training-manifest", "/private/dataset/eval/training-splits.json",
               "--inventory", "/private/dataset/research/dataset/inventory.csv",
               "--labels", "/private/dataset/eval/labels.csv",
               "--folds", "/private/dataset/eval/folds.csv",
               "--epochs", "3", "--minimum-epochs", "3", "--learning-rate", "0.00005",
               "--max-wall-minutes", "90", "--cuda-memory-fraction", "0.94"]
    started = time.monotonic()
    with telemetry.open("w") as stream:
        monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw",
                                    "--format=csv,noheader,nounits", "-l", "5"], stdout=stream)
        process = subprocess.Popen(command, env=env)
        try:
            try:
                code = process.wait(timeout=5700)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise RuntimeError("Hard training time limit reached") from None
        finally:
            if process.poll() is None:
                process.terminate()
            monitor.terminate()
            monitor.wait(timeout=10)
            private.commit()
    if code:
        raise RuntimeError(f"Training exited with status {code}; persisted artifacts remain available")
    report = json.loads((output / "run-manifest.json").read_text())
    return {"run_id": run_id, "elapsed_seconds": round(time.monotonic() - started, 1),
            "epochs": len(report["epochs"]), "selected_epoch": report["selected_epoch"],
            "baseline": report["baseline_validation"]["balanced_accuracy"],
            "validation": [{"epoch": item["epoch"],
                            "balanced_accuracy": item["validation"]["balanced_accuracy"],
                            "mean_loss": item["training"]["mean_loss"]} for item in report["epochs"]]}


@setup_app.local_entrypoint()
def setup(stage: str = "models"):
    if stage in {"models", "qwen"}:
        print(json.dumps(prepare_models.remote("qwen" if stage == "qwen" else "baseline")))
    elif stage == "data":
        print(json.dumps(prepare_data.remote()))
    else:
        raise ValueError("Setup stage must be models, qwen or data")


@app.local_entrypoint()
def main(run_id: str = ""):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("An explicit valid unique --run-id is required")
    try:
        print(json.dumps(train.remote(run_id)))
    finally:
        # Retain artifacts on the user's machine even when training fails.
        destination = ROOT / "training/runs" / run_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-X", "utf8", "-m", "modal", "volume", "get",
                        "lado-training-v1", f"/runs/{run_id}", str(destination)], check=True)
