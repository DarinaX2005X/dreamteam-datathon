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
              retries=0, max_containers=1, scaledown_window=2, volumes={"/private": private})
def repair_projection(run_id: str) -> dict:
    """Recheck persisted evidence after the evaluator path correction; no inference."""
    import shutil
    from types import SimpleNamespace

    from navidu.contracts.domain import AnalysisRun, FindingSet, VideoMeta

    from eval.dataset import load_sources
    from eval.run_pipeline import RunStore, _artifact_projection, read_json, write_report

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("Invalid run ID")
    root = Path("/private/evaluation") / run_id
    report = read_json(root / "aggregate.json")
    store = RunStore(root, report["fingerprint"])
    profile = read_json(store.directory / "profile.json")
    dataset = Path("/private/dataset")
    sources = load_sources(dataset / "research/dataset/inventory.csv",
                           dataset / "eval/labels.csv", dataset / "eval/folds.csv")
    selected = [source for source in sources if (store.source_dir(source) / "latest.json").exists()]
    repaired, records = 0, {}
    for source in selected:
        record = store.latest(source)
        if record["status"] == "failed" and record.get("failure_code") == "FileNotFoundError":
            attempt = store.source_dir(source) / record["artifact_directory"]
            exception = (attempt / "exception.txt").read_text()
            if "_artifact_projection" not in exception or "/evidence/original/" not in exception:
                raise RuntimeError("Failure is not the known evaluator path mismatch")
            artifacts = SimpleNamespace(
                run=AnalysisRun.model_validate(read_json(attempt / "run.json")),
                video=VideoMeta.model_validate(read_json(attempt / "video.json")),
                finding_set=FindingSet.model_validate(read_json(attempt / "finding_set.json")))
            replacement = _artifact_projection(artifacts, source,
                fingerprint=profile["fingerprint"], pipeline_version=profile["pipeline_version"],
                latency=record["latency_sec"], attempt=attempt)
            replacement.update(previous_failed_attempts=record.get("previous_failed_attempts", 0),
                projection_repaired=True, inference_repeated=False)
            shutil.copyfile(attempt / "prediction.json", attempt / "prediction-before-projection-repair.json")
            store.save(source, attempt, replacement)
            record = replacement
            repaired += 1
        records[source.sha256] = record
    original = root / "aggregate-before-projection-repair.json"
    if not original.exists():
        shutil.copyfile(root / "aggregate.json", original)
    updated = write_report(store, selected, records, profile=profile, split=report["split"],
        dataset_hash=report["dataset_fingerprint"], run_stats={"executed": 0,
            "report_only": True, "projection_repaired": repaired, "inference_repeated": False},
        output=root / "aggregate.json")
    private.commit()
    return {"gpu_allocated": False, "repaired": repaired, "metrics": updated["metrics"]}


@app.local_entrypoint()
def repair(run_id: str):
    print(json.dumps(repair_projection.remote(run_id)))
    destination = ROOT / "eval/runs/modal-repaired" / run_id
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-X", "utf8", "-m", "modal", "volume", "get", "lado-training-v1",
                    f"/evaluation/{run_id}", str(destination.parent)], check=True)


@app.function(image=image, cpu=(2, 2), memory=(4096, 4096), timeout=180,
              retries=0, max_containers=1, min_containers=0, scaledown_window=2,
              volumes={"/models": models, "/data": data})
def cpu_preflight() -> dict:
    """Check processor input and single-container SQLite without allocating a GPU."""
    import secrets
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import torch
    from navidu.perception.proposal_task import LORA_TARGET_MODULES, encode_proposal
    from navidu.perception.qwen import prepare_qwen_inputs
    from navidu.perception.temporal import visual_request
    from navidu.service.config import Settings
    from navidu.service.store import Store
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration

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
    video, _, _ = prepare_qwen_inputs(processor, frames, [{"p1": (.2, .1, .7, .9)}] * 3,
        variant="v7-context", max_pixels=451584, mode="video", detail=(.1, .25, .9, .95))
    assert len(video["video_grid_thw"]) == 2 and video["input_ids"].dtype == torch.int64
    detail_video_tokens = video["input_ids"].shape[1]
    training = encode_proposal(processor, frames, model_id="qwen3-vl-4b",
                               target={"candidate": 1, "frame": 1})
    supervised_tokens = int((training["labels"] != -100).sum())
    assert 0 < supervised_tokens < 32
    assert int((training["labels"] == -100).sum()) > 100
    # A tiny native architecture checks actual PEFT module matching and backward
    # without loading pretrained weights or allocating a GPU.
    tiny_config = Qwen3VLConfig(
        text_config={"vocab_size": 32, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2,
                     "head_dim": 16, "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3]}},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                       "num_heads": 2, "out_hidden_size": 32, "num_position_embeddings": 16,
                       "deepstack_visual_indexes": []},
        image_token_id=1, video_token_id=2, vision_start_token_id=3, vision_end_token_id=4)
    tiny = get_peft_model(Qwen3VLForConditionalGeneration(tiny_config), LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", target_modules=LORA_TARGET_MODULES))
    trainable = [(name, parameter) for name, parameter in tiny.named_parameters() if parameter.requires_grad]
    assert len(trainable) == 4 and all("lora_" in name and "language_model" in name for name, _ in trainable)
    tokens = torch.tensor([[5, 6, 7, 8, 9, 10]])
    loss = tiny(input_ids=tokens, labels=tokens).loss
    loss.backward()
    assert torch.isfinite(loss) and all(parameter.grad is not None for _, parameter in trainable)
    assert all(not parameter.requires_grad for name, parameter in tiny.named_parameters() if ".visual." in name)
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
              "qwen_two_video_input_ok": True, "qwen_two_video_tokens": detail_video_tokens,
              "qwen_supervised_tokens": supervised_tokens, "qwen_lora_backward_ok": True,
              "sqlite_concurrent_writes": count, "sqlite_integrity": "ok"}
    (store.settings.data_dir / "preflight.json").write_text(json.dumps(result) + "\n")
    data.commit()
    print(json.dumps(result), flush=True)
    return result


@app.function(image=image, gpu="L4", cpu=(4, 4), memory=(32768, 32768),
              timeout=3000, retries=0, max_containers=1, min_containers=0,
              scaledown_window=2, volumes={"/models": models, "/private": private})
def evaluate(run_id: str, model_id: str, prompt_variant: str = "v5", max_pixels: int = 451584,
             input_mode: str = "images", window_frames: int = 16, max_sources: int = 8,
             detail_view: bool = False) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("Invalid run ID")
    if model_id not in {"internvl3_5-4b", "qwen3-vl-4b", "qwen3-vl-8b"} or prompt_variant not in {"v5", "v6-evidence", "v7-context", "v8-focused"}:
        raise ValueError("Unknown comparison profile")
    if not 65536 <= max_pixels <= 1048576:
        raise ValueError("Invalid image budget")
    if input_mode not in {"images", "video"} or max_sources not in {2, 8}:
        raise ValueError("Invalid evaluation mode or source count")
    if input_mode == "video" and not model_id.startswith("qwen3-vl-"):
        raise ValueError("Native video requires Qwen")
    if not 3 <= window_frames <= (64 if input_mode == "video" else 16):
        raise ValueError("Invalid frame budget")
    from eval.dataset import load_sources

    dataset = Path("/private/dataset")
    sources = load_sources(dataset / "research/dataset/inventory.csv",
                           dataset / "eval/labels.csv", dataset / "eval/folds.csv")
    if len(sources) != 32 or any(source.fold == 0 for source in sources):
        raise RuntimeError("Only authorized development sources may be used")
    selected = sorted((source for source in sources if source.fold == 4), key=lambda source: source.sha256)
    if len(selected) != 8:
        raise RuntimeError("Validation split changed")
    if max_sources == 2:
        selected = [max((source for source in selected if bool(source.has_violation) == positive),
                        key=lambda source: source.duration_sec) for positive in (True, False)]
    output = Path("/private/evaluation") / run_id
    output.mkdir(parents=True, exist_ok=False)
    aggregate = output / "aggregate.json"
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "NAVIDU_DETECTOR_MODEL": "rfdetr-nano-onnx", "NAVIDU_SEMANTIC_MODEL": model_id,
           "NAVIDU_SEMANTIC_WINDOW_FRAMES": str(window_frames), "NAVIDU_SEMANTIC_FPS": "1",
           "NAVIDU_SEMANTIC_MAX_NEW_TOKENS": "2048" if prompt_variant in {"v7-context", "v8-focused"} else "1024",
           "NAVIDU_SEMANTIC_MAX_PATCHES": "4",
           "NAVIDU_SEMANTIC_DTYPE": "bf16", "NAVIDU_SEMANTIC_STRUCTURED_OUTPUT": "0",
           "NAVIDU_QWEN_MAX_PIXELS": str(max_pixels), "NAVIDU_SEMANTIC_PROMPT_VARIANT": prompt_variant,
           "NAVIDU_QWEN_INPUT_MODE": input_mode,
           "NAVIDU_QWEN_DETAIL_VIEW": "1" if detail_view else "0",
           "NAVIDU_MODEL_INPUT_CAPTURE_DIR": str(output / "private-model-inputs"),
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
def main(run_id: str, model_id: str = "qwen3-vl-4b", prompt_variant: str = "v5", max_pixels: int = 451584,
         input_mode: str = "images", window_frames: int = 16, max_sources: int = 8,
         detail_view: bool = False):
    try:
        print(json.dumps(evaluate.remote(run_id, model_id, prompt_variant, max_pixels,
                                         input_mode, window_frames, max_sources, detail_view)))
    finally:
        destination = ROOT / "eval/runs/modal" / run_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-X", "utf8", "-m", "modal", "volume", "get", "lado-training-v1",
                        f"/evaluation/{run_id}", str(destination.parent)], check=True)
