"""Run code and synthetic model tests remotely; never load local video frames."""
from __future__ import annotations

import json
import subprocess
import sys

import modal

from deploy.modal_training import ROOT
from deploy.modal_training import image as base_image

app = modal.App("lado-code-checks")
image = base_image.pip_install("pytest>=8,<9", "httpx>=0.28,<1", "ruff>=0.11,<1")
for local, remote in (("backend/navidu", "backend/navidu"), ("backend/tests", "backend/tests"),
                      ("rules", "rules"), ("contracts", "contracts"),
                      ("training/tests", "training/tests"), ("deploy/tests", "deploy/tests"),
                      ("research/spike/tests", "research/spike/tests")):
    image = image.add_local_dir(ROOT / local, "/workspace/" + remote, copy=True,
                                ignore=["**/__pycache__/**", "**/*.pyc"])
image = image.add_local_file(ROOT / "backend/pyproject.toml", "/workspace/backend/pyproject.toml", copy=True)
image = image.add_local_file(ROOT / "models/manifest.json", "/workspace/models/manifest.json", copy=True)
image = image.add_local_file(ROOT / "research/dataset/extract_archive.py",
                             "/workspace/research/dataset/extract_archive.py", copy=True)
for source in sorted((ROOT / "scripts").glob("*.py")):
    image = image.add_local_file(source, f"/workspace/scripts/{source.name}", copy=True)


@app.function(image=image, cpu=(2, 2), memory=(8192, 8192), timeout=600,
              retries=0, max_containers=1, scaledown_window=2)
def check() -> dict:
    suites = ["backend/tests", "training/tests", "eval", "deploy/tests", "research/spike/tests"]
    # These three tests inspect all 40 source metadata rows, including holdout.
    # Those files are intentionally absent from the authorized remote workspace.
    selection = ("not ActualDatasetPartitionTests and not test_actual_inner_split_is_disjoint_and_omits_all_holdout_examples "
                 "and not test_reader_rejects_holdout_and_tampered_supervision_before_opening_video")
    result = subprocess.run([sys.executable, "-m", "pytest", *suites, "-q", "-k", selection],
                            text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=500)
    print(result.stdout, flush=True)
    if result.returncode:
        print(result.stderr[-5000:], flush=True)
        raise RuntimeError("Remote code checks failed")
    return {"gpu_allocated": False, "passed": True, "summary": result.stdout.splitlines()[-1]}


@app.local_entrypoint()
def main():
    print(json.dumps(check.remote()))
