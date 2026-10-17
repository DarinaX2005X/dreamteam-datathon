"""Bounded comparisons on the fixed eight development-validation sources."""
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

app = modal.App("lado-evaluation")
image = base_image.add_local_dir(ROOT / "rules", "/workspace/rules", copy=True)
data = modal.Volume.from_name("lado-data-v1", version=2)


@app.function(image=image, cpu=(2, 2), memory=(4096, 4096), timeout=180,
              retries=0, max_containers=1, min_containers=0, scaledown_window=2,
              volumes={"/models": models, "/data": data})
def cpu_preflight() -> dict:
    """Check processor input and single-container SQLite without allocating a GPU."""
    import secrets
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import torch
    from navidu.perception.temporal import visual_request
    from navidu.service.config import Settings
    from navidu.service.store import Store
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained("/models/qwen3-vl-4b", local_files_only=True)
    frames = [(float(index), np.zeros((720, 1280, 3), dtype=np.uint8)) for index in range(3)]
    images, content = visual_request(frames, [{}] * 3)
    prompt = processor.apply_chat_template([{"role": "user", "content": content}],
                                           add_generation_prompt=True, tokenize=False)
    inputs = processor(images=images, text=prompt, return_tensors="pt",
                       min_pixels=65536, max_pixels=451584).to("cpu", dtype=torch.bfloat16)
    assert inputs["input_ids"].dtype == inputs["image_grid_thw"].dtype == torch.int64
    assert len(inputs["image_grid_thw"]) == 3 and inputs["pixel_values"].dtype == torch.bfloat16
    token_count = inputs["input_ids"].shape[1]
    store = Store(Settings(data_dir=Path("/data/verification") / secrets.token_hex(8)))

    def write_state(index):
        store.worker_state(str(index), model_status="loading", device="cpu",
                           limitations=[], pipeline_version="storage-preflight")

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(write_state, range(20)))
    with store.connection() as db:
        count = db.execute("SELECT COUNT(*) FROM worker_state").fetchone()[0]
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok" and count == 20
    data.commit()
    result = {"gpu_allocated": False, "qwen_processor_ok": True, "input_tokens": token_count,
              "sqlite_concurrent_writes": count, "sqlite_integrity": "ok"}
    (store.settings.data_dir / "preflight.json").write_text(json.dumps(result) + "\n")
    data.commit()
    print(json.dumps(result), flush=True)
    return result


@app.function(image=image, gpu="L4", cpu=(4, 4), memory=(32768, 32768),
              timeout=3000, retries=0, max_containers=1, min_containers=0,
              scaledown_window=2, volumes={"/models": models, "/private": private})
def evaluate(run_id: str, model_id: str, prompt_variant: str = "v5", max_pixels: int = 451584) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("Invalid run ID")
    if model_id not in {"internvl3_5-4b", "qwen3-vl-4b"} or prompt_variant not in {"v5", "v6-evidence"}:
        raise ValueError("Unknown comparison profile")
    if not 65536 <= max_pixels <= 1048576:
        raise ValueError("Invalid image budget")
    from eval.dataset import load_sources

    dataset = Path("/private/dataset")
    sources = load_sources(dataset / "research/dataset/inventory.csv",
                           dataset / "eval/labels.csv", dataset / "eval/folds.csv")
    if len(sources) != 32 or any(source.fold == 0 for source in sources):
        raise RuntimeError("Only authorized development sources may be used")
    selected = sorted((source for source in sources if source.fold == 4), key=lambda source: source.sha256)
    if len(selected) != 8:
        raise RuntimeError("Validation split changed")
    output = Path("/private/evaluation") / run_id
    output.mkdir(parents=True, exist_ok=False)
    aggregate = output / "aggregate.json"
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "NAVIDU_DETECTOR_MODEL": "rfdetr-nano-onnx", "NAVIDU_SEMANTIC_MODEL": model_id,
           "NAVIDU_SEMANTIC_WINDOW_FRAMES": "16", "NAVIDU_SEMANTIC_FPS": "1",
           "NAVIDU_SEMANTIC_MAX_NEW_TOKENS": "1024", "NAVIDU_SEMANTIC_MAX_PATCHES": "4",
           "NAVIDU_SEMANTIC_DTYPE": "bf16", "NAVIDU_SEMANTIC_STRUCTURED_OUTPUT": "0",
           "NAVIDU_QWEN_MAX_PIXELS": str(max_pixels), "NAVIDU_SEMANTIC_PROMPT_VARIANT": prompt_variant,
           "NAVIDU_CPU_THREADS": "4", "NAVIDU_PROPOSAL_ADAPTER": ""}
    command = [sys.executable, "eval/run_pipeline.py", "run", "--device", "auto",
               "--analysis-fps", "3", "--models-dir", "/models", "--rules-dir", "/workspace/rules",
               "--dataset-dir", str(dataset / "data/dataset"),
               "--inventory", str(dataset / "research/dataset/inventory.csv"),
               "--labels", str(dataset / "eval/labels.csv"), "--folds", str(dataset / "eval/folds.csv"),
               "--output-root", str(output), "--aggregate-report", str(aggregate),
               "--runtime-label", f"modal-l4-{model_id}-{prompt_variant}", "--split", "development"]
    for source in selected:
        command.extend(["--video-id", source.rows[0].video_id])
    started = time.monotonic()
    with (output / "driver.log").open("w") as log, (output / "gpu.csv").open("w") as telemetry:
        monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw",
                                    "--format=csv,noheader,nounits", "-l", "5"], stdout=telemetry)
        process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        try:
            try:
                code = process.wait(timeout=2700)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                code = 124
        finally:
            if process.poll() is None:
                process.terminate()
            monitor.terminate()
            monitor.wait(timeout=10)
            private.commit()
    if not aggregate.is_file():
        raise RuntimeError(f"Evaluation exited {code} without aggregate; private diagnostics persisted")
    report = json.loads(aggregate.read_text())
    return {"run_id": run_id, "exit_code": code, "elapsed_seconds": round(time.monotonic() - started, 1),
            "metrics": report["metrics"]}


@app.local_entrypoint()
def main(run_id: str, model_id: str = "qwen3-vl-4b", prompt_variant: str = "v5", max_pixels: int = 451584):
    try:
        print(json.dumps(evaluate.remote(run_id, model_id, prompt_variant, max_pixels)))
    finally:
        destination = ROOT / "eval/runs/modal" / run_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-X", "utf8", "-m", "modal", "volume", "get", "lado-training-v1",
                        f"/evaluation/{run_id}", str(destination)], check=True)
