"""Pinned CPU preparation and explicitly invoked bounded VideoPrism L4 pilot."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import modal

from deploy.modal_training import ROOT, models, private
from deploy.modal_training import image as base_image

CODE_REVISION = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
MODEL_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
MODEL_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
MODEL_FILE = "flax_base_f16r288_repeated.npz"

app = modal.App("lado-videoprism-probe")
image = (base_image
         .apt_install("git")
         .env({"USE_TF": "1", "USE_FLAX": "1", "TF_CPP_MIN_LOG_LEVEL": "2",
               "TF_FORCE_GPU_ALLOW_GROWTH": "false"})
         .pip_install("jax[cuda12]==0.5.3", "flax==0.10.6", "tensorflow-cpu==2.18.0",
                      "scikit-learn==1.6.1", "einshape==1.0", "einops==0.8.1",
                      "sentencepiece==0.2.0",
                      f"git+https://github.com/google-deepmind/videoprism.git@{CODE_REVISION}")
         .add_local_file(ROOT / "training/videoprism_probe.py",
                         "/workspace/training/videoprism_probe.py", copy=True)
         .add_local_file(ROOT / "training/train_videoprism_probe.py",
                         "/workspace/training/train_videoprism_probe.py", copy=True)
         .add_local_file(ROOT / "training/tests/test_videoprism_probe.py",
                         "/workspace/training/tests/test_videoprism_probe.py", copy=True)
         .add_local_file(ROOT / "deploy/modal_videoprism_probe.py",
                         "/workspace/deploy/modal_videoprism_probe.py", copy=True)
         .add_local_file(ROOT / "deploy/modal_training.py",
                         "/workspace/deploy/modal_training.py", copy=True))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(image=image, cpu=(4, 4), memory=(12288, 12288), timeout=900,
              retries=0, max_containers=1, min_containers=0, scaledown_window=2,
              volumes={"/models": models})
def prepare() -> dict:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import jax
    import jax.numpy as jnp
    from huggingface_hub import hf_hub_download
    from videoprism import models as vp

    directory = Path("/models/videoprism-base-f16r288")
    path = Path(hf_hub_download(repo_id="google/videoprism-base-f16r288",
                                filename=MODEL_FILE, revision=MODEL_REVISION,
                                local_dir=directory))
    if path.stat().st_size != 457475296 or _sha256(path) != MODEL_SHA256:
        raise RuntimeError("pinned VideoPrism weight verification failed")
    models.commit()
    model = vp.get_model("videoprism_public_v1_base")
    state = vp.load_pretrained_weights("videoprism_public_v1_base", str(path))
    synthetic = jnp.zeros((1, 16, 288, 288, 3), dtype=jnp.float32)
    output = jax.jit(model.apply, static_argnames=("train",))(state, synthetic, train=False)
    array = output[0] if isinstance(output, tuple) else output
    if tuple(array.shape) != (1, 4096, 768):
        raise RuntimeError(f"synthetic forward shape mismatch: {array.shape}")
    subprocess.run([sys.executable, "-m", "unittest",
                    "training/tests/test_videoprism_probe.py"], check=True, timeout=120)
    models.commit()
    return {"gpu_allocated": False, "weights_sha256": MODEL_SHA256,
            "model_revision": MODEL_REVISION, "code_revision": CODE_REVISION,
            "synthetic_shape": list(array.shape)}


@app.function(image=image, gpu="L4", cpu=(4, 4), memory=(24576, 24576), timeout=270,
              retries=0, max_containers=1, min_containers=0, scaledown_window=2,
              volumes={"/models": models, "/private": private})
def train(run_id: str) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("invalid run id")
    output = Path("/private/runs") / run_id
    if output.exists():
        raise RuntimeError("run already exists")
    command = [sys.executable, "training/train_videoprism_probe.py",
               "--weights", f"/models/videoprism-base-f16r288/{MODEL_FILE}",
               "--dataset-dir", "/private/dataset/data/dataset",
               "--output-dir", str(output),
               "--training-manifest", "/private/dataset/eval/training-splits.json",
               "--inventory", "/private/dataset/research/dataset/inventory.csv",
               "--labels", "/private/dataset/eval/labels.csv",
               "--folds", "/private/dataset/eval/folds.csv"]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TF_CPP_MIN_LOG_LEVEL": "2",
           "CUDA_VISIBLE_DEVICES": "0", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
           "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90", "OMP_NUM_THREADS": "4"}
    try:
        result = subprocess.run(command, env=env, timeout=240)
    finally:
        private.commit()
    if result.returncode:
        raise RuntimeError(f"VideoPrism probe exited {result.returncode}")
    report = json.loads((output / "run-manifest.json").read_text(encoding="utf-8"))
    return {"run_id": run_id, "outcome": report["outcome"],
            "oof_metrics": report.get("oof_metrics"),
            "validation_metrics": report.get("validation_metrics")}


@app.local_entrypoint()
def main(stage: str = "prepare", run_id: str = ""):
    if stage == "prepare":
        print(json.dumps(prepare.remote()))
        return
    if stage != "train" or not run_id:
        raise ValueError("stage must be prepare or train; train requires --run-id")
    try:
        print(json.dumps(train.remote(run_id)))
    finally:
        destination = ROOT / "training/runs"
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-X", "utf8", "-m", "modal", "volume", "get",
                        "lado-training-v1", f"/runs/{run_id}", str(destination)], check=True)
