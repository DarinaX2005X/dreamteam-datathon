"""Run code and synthetic model tests remotely; never load local video frames."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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


@app.function(image=image, cpu=(1, 1), memory=(2048, 2048), timeout=120,
              retries=0, max_containers=1, scaledown_window=2)
def generate_contracts():
    from navidu.contracts import generate

    generate.ROOT = Path("/tmp/lado-generated-contracts")
    generate.generate()
    return {path.relative_to(generate.ROOT).as_posix(): path.read_text(encoding="utf-8")
            for path in generate.ROOT.rglob("*") if path.is_file()}


@app.local_entrypoint()
def main(stage: str = "check"):
    if stage == "contracts":
        files = generate_contracts.remote()
        for relative, content in files.items():
            destination = (ROOT / "contracts" / relative).resolve()
            if not destination.is_relative_to((ROOT / "contracts").resolve()):
                raise ValueError("Generated contract path escaped its directory")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
        print(json.dumps({"generated_contract_files": len(files), "gpu_allocated": False}))
    elif stage == "check":
        print(json.dumps(check.remote()))
    else:
        raise ValueError("Unknown code-check stage")
