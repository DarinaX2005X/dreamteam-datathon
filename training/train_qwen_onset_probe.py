#!/usr/bin/env python3
"""Train a bounded frozen-Qwen onset-transition proposer on the fixed development split."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for import_root in (ROOT, BACKEND):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

MODEL_ID = "qwen3-vl-4b"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
MODEL_NAME = f"Qwen/Qwen3-VL-4B-Instruct@{MODEL_REVISION}"
SEED = 1729
FPS = 2.0
MAX_PIXELS = 262144
EPOCHS = 20
LEARNING_RATE = 1e-3
HARD_WALL_SECONDS = 25 * 60
CACHE_BUDGET_SECONDS = 20 * 60
REQUIRED_POSITIVE_IDS = {
    "69caf59620fcd80bedcb664514730ab38f2f800da27000a8e037d7440be5dc7a",  # X-15
    "90607b0b0cb772fcf29b0c7e0e56fa6af36da6fac0efe93b8a616b5de94a156c",  # X-18
}
ADDITIONAL_POSITIVE_IDS = {
    "a37930cf119bb691be77d8624b18333b04d9c67066a7f4627d63e3208dfa6057",  # X-13
    "66fc5d28c2371592f9260305589bdc86636fdd2ebe7a7ec976eec00a23d43df4",  # X-46
}
REQUIRED_NEGATIVE_IDS = {
    "901f61cf1a1b8754dda56466308c19fc203a1a4c0e85d1e7827849d34e2d52f0",
    "ac4cf04844c0aa08a5e19f5f7fe984f1477943294922d6ef77bdc5b88b97160e",
    "deeeee54539fb894b17447296bd7f0807b9dc662bad9af4dcd27f4fbdd677187",
    "efd876a04224cac781ac64bf5d6e107ded67340b7b7715fd9ca3408b240a1075",
}


class ProbeTrainingError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def enforce_modal_l4(torch: Any) -> dict[str, Any]:
    if platform.system() != "Linux" or not os.getenv("MODAL_TASK_ID") \
            or not torch.cuda.is_available():
        raise ProbeTrainingError("the probe requires the authorized Modal CUDA task")
    properties = torch.cuda.get_device_properties(0)
    if str(properties.name).upper() not in {"NVIDIA L4", "L4"} \
            or int(properties.total_memory) < 22 * 1024**3:
        raise ProbeTrainingError("the probe is restricted to one NVIDIA L4 24 GB")
    return {"gpu_name": str(properties.name), "gpu_memory_bytes": int(properties.total_memory),
            "cpu_count": os.cpu_count(), "profile": "modal-l4"}


def cache_identity(dataset_fingerprint: str, model_hash: str, samples: list[Any],
                   torch_version: str, extractor_source_sha256: str,
                   decoder_source_sha256: str) -> dict[str, Any]:
    return {"schema_version": 1, "dataset_fingerprint": dataset_fingerprint,
            "model_id": MODEL_NAME, "model_aggregate_sha256": model_hash,
            "transformers_version": importlib.metadata.version("transformers"),
            "torch_version": torch_version,
            "extractor": {"schema": "qwen-final-image-embeddings-mean-v1",
                          "layer": "get_image_features final output", "pool": "token mean",
                          "extractor_source_sha256": extractor_source_sha256,
                          "decoder_source_sha256": decoder_source_sha256},
            "preprocessing": {"fps": FPS, "min_pixels": 65536,
                              "max_pixels": MAX_PIXELS, "per_frame": True},
            "sources": [{"sample_id": sample.sample_id,
                         "source_sha256": sample.source_sha256,
                         "duration_sec": sample.duration_sec} for sample in samples]}


def _image_summary(model: Any, processor: Any, rgb: Any, torch: Any) -> Any:
    from training.qwen_onset_probe import pool_qwen_image_features

    batch = processor.image_processor(images=[rgb], return_tensors="pt", min_pixels=65536,
                                      max_pixels=MAX_PIXELS)
    accepted = set(inspect.signature(model.get_image_features).parameters)
    image_inputs = {}
    for name in ("pixel_values", "image_grid_thw"):
        if name in batch and name in accepted:
            value = batch[name].to("cuda")
            image_inputs[name] = value.to(torch.bfloat16) if name == "pixel_values" else value
    if "pixel_values" not in image_inputs:
        raise ProbeTrainingError("Qwen processor did not produce native image inputs")
    output = model.get_image_features(**image_inputs)
    return pool_qwen_image_features(output).cpu()


def load_or_build_cache(model: Any, processor: Any, samples: list[Any], identity: dict[str, Any],
                        cache_dir: Path, setup_elapsed: float, progress_path: Path,
                        torch: Any) -> tuple[dict[str, Any], str, dict[str, float | int]]:
    from navidu.ingest import iter_frames
    from safetensors.torch import load_file, save_file

    identity_bytes = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    key = hashlib.sha256(identity_bytes).hexdigest()
    tensor_path = cache_dir / f"{key}.safetensors"
    manifest_path = cache_dir / f"{key}.json"
    if tensor_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != identity or manifest.get("sha256") != file_sha256(tensor_path):
            raise ProbeTrainingError("strict cache identity or hash mismatch")
        tensors = load_file(str(tensor_path))
        if set(tensors) != {f"source_{i:02d}_{suffix}" for i in range(len(samples))
                            for suffix in ("features", "times")}:
            raise ProbeTrainingError("cached tensor inventory mismatch")
        return tensors, manifest["sha256"], {"reused": 1,
                                              "frame_count": manifest["frame_count"]}
    cache_dir.mkdir(parents=True, exist_ok=True)
    tensors, completed = {}, 0
    expected = sum(max(2, math.ceil(sample.duration_sec * FPS)) for sample in samples)
    encode_started = time.monotonic()
    first_100: dict[str, float | int] = {}
    with torch.inference_mode():
        for index, sample in enumerate(samples):
            features, times = [], []
            for frame in iter_frames(sample.video_path, analysis_fps=FPS):
                features.append(_image_summary(model, processor, frame.rgb, torch))
                times.append(float(frame.timestamp_sec))
                completed += 1
                encode_elapsed = time.monotonic() - encode_started
                if completed == 100:
                    projected_encode = encode_elapsed / completed * expected
                    projected_total = setup_elapsed + projected_encode
                    first_100 = {"reused": 0, "frames": 100,
                                 "encode_elapsed_seconds": round(encode_elapsed, 3),
                                 "frames_per_second": round(completed / encode_elapsed, 4),
                                 "projected_encode_seconds": round(projected_encode, 3),
                                 "setup_elapsed_seconds": round(setup_elapsed, 3),
                                 "projected_total_seconds": round(projected_total, 3)}
                    write_json(progress_path, {"state": "encoding", "first_100": first_100})
                if completed == 100 and projected_total > CACHE_BUDGET_SECONDS:
                    raise ProbeTrainingError("first-100-frame ETA exceeds the 20-minute cache budget")
                if setup_elapsed + encode_elapsed > CACHE_BUDGET_SECONDS:
                    raise ProbeTrainingError("feature cache exceeded its 20-minute budget")
            if len(features) < 5:
                raise ProbeTrainingError("source is too short for two-second transitions")
            tensors[f"source_{index:02d}_features"] = torch.stack(features).contiguous()
            tensors[f"source_{index:02d}_times"] = torch.tensor(times, dtype=torch.float32)
            write_json(progress_path, {"state": "encoding", "sources_completed": index + 1,
                                       "sources_total": len(samples), "frames_completed": completed,
                                       "first_100": first_100 or None})
    temporary = tensor_path.with_suffix(".tmp")
    save_file(tensors, str(temporary), metadata={"schema_version": "1", "private": "true"})
    os.replace(temporary, tensor_path)
    digest = file_sha256(tensor_path)
    write_json(manifest_path, {"identity": identity, "sha256": digest,
                               "frame_count": completed})
    return tensors, digest, {**first_100, "frame_count": completed}


def prepare_items(samples: list[Any], tensors: dict[str, Any], fold_by_id: dict[str, int],
                  torch: Any) -> list[dict[str, Any]]:
    from training.qwen_onset_probe import transition_pairs

    items = []
    for index, sample in enumerate(samples):
        pairs, times = transition_pairs(tensors[f"source_{index:02d}_features"],
                                        tensors[f"source_{index:02d}_times"])
        items.append({"sample_id": sample.sample_id, "fold": fold_by_id[sample.sample_id],
                      "truth": bool(sample.has_violation),
                      "onset_sec": sample.time_sec if sample.time_supervision else None,
                      "pairs": pairs.float(), "times": times.float()})
    return items


def fit_probe(items: list[dict[str, Any]], torch: Any, seed_offset: int = 0) -> tuple[Any, Any, Any]:
    from training.qwen_onset_probe import balanced_source_loss, build_probe, source_loss

    torch.manual_seed(SEED + seed_offset)
    dimension = items[0]["pairs"].shape[1] // 4
    stacked = torch.cat([item["pairs"] for item in items])
    mean = stacked.mean(0).to("cuda")
    std = stacked.std(0).clamp_min(1e-5).to("cuda")
    head = build_probe(dimension).to("cuda")
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    for _epoch in range(EPOCHS):
        head.train()
        source_losses = []
        for item in items:
            values = (item["pairs"].to("cuda") - mean) / std
            logits = head(values).flatten()
            loss, _ = source_loss(logits, item["times"].to("cuda"), item["truth"],
                                  item["onset_sec"])
            source_losses.append((item["truth"], loss))
        loss = balanced_source_loss(source_losses)
        if not torch.isfinite(loss):
            raise ProbeTrainingError("non-finite probe loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return head.eval(), mean, std


def score_items(head: Any, mean: Any, std: Any, items: list[dict[str, Any]],
                torch: Any) -> list[dict[str, Any]]:
    from training.qwen_onset_probe import normalized_log_mean_exp

    rows = []
    with torch.inference_mode():
        for item in items:
            logits = head((item["pairs"].to("cuda") - mean) / std).flatten()
            score = float(torch.sigmoid(normalized_log_mean_exp(logits)))
            index = int(logits.argmax())
            rows.append({"sample_id": item["sample_id"], "truth": item["truth"],
                         "score": score, "time_sec": float(item["times"][index])})
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    if args.output_dir.exists():
        raise ProbeTrainingError("output directory already exists")
    args.output_dir.mkdir(parents=True)
    import torch
    from navidu.ingest import iter_frames
    from navidu.perception.assets import model_entry, read_manifest, verify_model
    from safetensors.torch import save_file
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from eval.training import load_training_partition
    from training.qwen_onset_probe import (
        binary_metrics,
        calibrate_threshold,
        strict_validation_gate,
    )

    host = enforce_modal_l4(torch)
    torch.cuda.set_per_process_memory_fraction(0.94, 0)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    common = {"manifest_path": args.training_manifest, "inventory": args.inventory,
              "labels": args.labels, "folds": args.folds, "dataset_dir": args.dataset_dir,
              "verify_hashes": True}
    train = load_training_partition(**common, partition="train")
    validation = load_training_partition(**common, partition="validation")
    if (len(train), sum(x.has_violation for x in train), len(validation),
            sum(x.has_violation for x in validation)) != (24, 12, 8, 4):
        raise ProbeTrainingError("fixed split must be train 12+/12- and validation 4+/4-")
    split = json.loads(args.training_manifest.read_text(encoding="utf-8"))
    entry = model_entry(read_manifest(args.models_dir), MODEL_ID)
    if entry["revision"] != MODEL_REVISION \
            or args.base_model.resolve() != (args.models_dir / entry["local_dir"]).resolve():
        raise ProbeTrainingError("base model does not match the pinned Qwen entry")
    model_hash = verify_model(args.models_dir, entry)
    processor = AutoProcessor.from_pretrained(args.base_model, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    write_json(args.output_dir / "progress.json", {
        "state": "model_loaded", "elapsed_seconds": round(time.monotonic() - started, 3)})
    samples = train + validation
    decoder_source = inspect.getsourcefile(iter_frames)
    if decoder_source is None:
        raise ProbeTrainingError("cannot identify the frame decoder source")
    identity = cache_identity(
        split["dataset_fingerprint"], model_hash, samples, str(torch.__version__),
        file_sha256(Path(__file__)), file_sha256(Path(decoder_source)))
    setup_elapsed = time.monotonic() - started
    tensors, cache_hash, cache_stats = load_or_build_cache(
        model, processor, samples, identity, args.cache_dir, setup_elapsed,
        args.output_dir / "progress.json", torch)
    del model
    torch.cuda.empty_cache()
    fold_by_id = {row["sample_id"]: int(row["fold"]) for row in split["samples"]}
    items = prepare_items(samples, tensors, fold_by_id, torch)
    train_items, validation_items = items[:24], items[24:]
    oof_rows = []
    for fold in (1, 2, 3):
        fit = [item for item in train_items if item["fold"] != fold]
        score = [item for item in train_items if item["fold"] == fold]
        if (len(fit), len(score), sum(x["truth"] for x in score)) != (16, 8, 4):
            raise ProbeTrainingError("unexpected inner-fold composition")
        head, mean, std = fit_probe(fit, torch, seed_offset=fold)
        oof_rows.extend(score_items(head, mean, std, score, torch))
        write_json(args.output_dir / "progress.json", {
            "state": "oof", "folds_completed": fold, "folds_total": 3,
            "elapsed_seconds": round(time.monotonic() - started, 3)})
    try:
        calibration = calibrate_threshold(oof_rows)
    except ValueError as error:
        fallback_threshold = 0.5
        fallback_metrics = binary_metrics(
            [row["truth"] for row in oof_rows],
            [row["score"] >= fallback_threshold for row in oof_rows])
        elapsed = time.monotonic() - started
        report = {"schema_version": 1, "state": "complete", "outcome": "rejected",
                  "rejection_reason": str(error), "model_id": MODEL_NAME,
                  "role": "binary_proposer_not_normative_verdict", "host": host,
                  "seed": SEED, "epochs": EPOCHS, "learning_rate": LEARNING_RATE,
                  "cache_identity": identity, "cache_sha256": cache_hash,
                  "cache_stats": cache_stats, "oof_rows": oof_rows,
                  "oof_threshold": None, "oof_metrics": fallback_metrics,
                  "oof_diagnostic_threshold": fallback_threshold,
                  "validation_rows": [], "validation_metrics": None,
                  "selected": False, "checkpoint_sha256": None,
                  "holdout_used": False, "production_improvement_claimed": False,
                  "elapsed_seconds": round(elapsed, 3)}
        write_json(args.output_dir / "run-manifest.json", report)
        write_json(args.output_dir / "progress.json", {
            "state": "complete", "outcome": "rejected",
            "reason": "oof_anti_collapse_guard", "elapsed_seconds": round(elapsed, 3)})
        return report
    head, mean, std = fit_probe(train_items, torch, seed_offset=10)
    validation_rows = score_items(head, mean, std, validation_items, torch)
    for row in validation_rows:
        row["candidate"] = row["score"] >= calibration.threshold
    metrics = binary_metrics([row["truth"] for row in validation_rows],
                             [row["candidate"] for row in validation_rows])
    selected = strict_validation_gate(validation_rows, REQUIRED_POSITIVE_IDS,
                                      REQUIRED_NEGATIVE_IDS, ADDITIONAL_POSITIVE_IDS)
    checkpoint_hash = None
    if selected:
        checkpoint = args.output_dir / "qwen-onset-probe.safetensors"
        state = {f"head.{key}": value.detach().cpu() for key, value in head.state_dict().items()}
        state.update({"feature_mean": mean.cpu(), "feature_std": std.cpu()})
        save_file(state, str(checkpoint), metadata={"schema_version": "1", "role": "proposer"})
        checkpoint_hash = file_sha256(checkpoint)
    elapsed = time.monotonic() - started
    if elapsed > HARD_WALL_SECONDS:
        raise ProbeTrainingError("hard 25-minute budget exceeded; checkpoint is invalid")
    report = {"schema_version": 1, "state": "complete",
              "outcome": "selected" if selected else "rejected", "model_id": MODEL_NAME,
              "role": "binary_proposer_not_normative_verdict", "host": host,
              "seed": SEED, "epochs": EPOCHS, "learning_rate": LEARNING_RATE,
              "cache_identity": identity, "cache_sha256": cache_hash,
              "cache_stats": cache_stats,
              "oof_rows": oof_rows, "oof_threshold": calibration.threshold,
              "oof_metrics": calibration.metrics, "validation_rows": validation_rows,
              "validation_metrics": metrics, "selected": selected,
              "checkpoint_sha256": checkpoint_hash, "holdout_used": False,
              "production_improvement_claimed": False, "elapsed_seconds": round(elapsed, 3)}
    write_json(args.output_dir / "run-manifest.json", report)
    write_json(args.output_dir / "progress.json", {
        "state": "complete", "outcome": report["outcome"],
        "elapsed_seconds": report["elapsed_seconds"]})
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"selected": result["selected"],
                      "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
