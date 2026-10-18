"""Run the fixed CPU-only A2 cache refinement in the authorized Modal workspace."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import modal

from deploy.modal_training import ROOT, private
from deploy.modal_training import image as base_image

app = modal.App("lado-onset-refinement")
image = (base_image
         .add_local_file(ROOT / "backend/pyproject.toml",
                         "/workspace/backend/pyproject.toml", copy=True)
         .add_local_file(ROOT / "training/qwen_onset_probe.py",
                         "/workspace/training/qwen_onset_probe.py", copy=True)
         .add_local_file(ROOT / "training/refine_qwen_onset_probe.py",
                         "/workspace/training/refine_qwen_onset_probe.py", copy=True)
         .add_local_file(ROOT / "training/tests/test_refine_qwen_onset_probe.py",
                         "/workspace/training/tests/test_refine_qwen_onset_probe.py", copy=True)
         .pip_install("pytest>=8,<9", "ruff>=0.11,<1"))


@app.function(image=image, cpu=(2, 2), memory=(8192, 8192), timeout=600,
              retries=0, max_containers=1, min_containers=0, scaledown_window=2,
              volumes={"/private": private})
def refine(run_id: str) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("invalid run id")
    output = Path("/private/runs") / run_id
    if output.exists():
        raise RuntimeError("run already exists")
    checks = [
        [sys.executable, "-m", "pytest", "training/tests/test_refine_qwen_onset_probe.py", "-q"],
        [sys.executable, "-m", "ruff", "check", "--config", "backend/pyproject.toml",
         "training/refine_qwen_onset_probe.py", "training/tests/test_refine_qwen_onset_probe.py"],
    ]
    for command in checks:
        subprocess.run(command, check=True, timeout=120)
    command = [sys.executable, "training/refine_qwen_onset_probe.py",
               "--original-report",
               "/private/runs/modal-qwen-onset-probe-20261018-a2/run-manifest.json",
               "--cache-dir", "/private/cache/qwen-onset",
               "--training-manifest", "/private/dataset/eval/training-splits.json",
               "--output-dir", str(output)]
    try:
        subprocess.run(command, check=True, timeout=540)
    finally:
        private.commit()
    report = json.loads((output / "run-manifest.json").read_text(encoding="utf-8"))
    return {"run_id": run_id, "outcome": report["outcome"],
            "chosen_variant": report["chosen_variant"],
            "validation_metrics": report["validation_metrics"],
            "elapsed_seconds": report["elapsed_seconds"], "gpu_allocated": False}


@app.local_entrypoint()
def main(run_id: str = ""):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("a unique --run-id is required")
    try:
        print(json.dumps(refine.remote(run_id)))
    finally:
        destination = ROOT / "training/runs"
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-X", "utf8", "-m", "modal", "volume", "get",
                        "lado-training-v1", f"/runs/{run_id}", str(destination)], check=True)
