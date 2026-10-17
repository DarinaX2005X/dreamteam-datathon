#!/usr/bin/env python3
"""Cache frozen InternVL image features and train a bounded temporal MIL head."""

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
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
for import_root in (ROOT, BACKEND):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from eval.training import load_training_partition  # noqa: E402
from training.temporal_head import build_temporal_head, mil_loss, paired_improves  # noqa: E402

BASE_MODEL_ID = "OpenGVLab/InternVL3_5-4B-HF@6bd4487402110ef9889ba50eb7aefeb302526fed"
MIN_CUDA_MEMORY_BYTES = 40 * 1024**3
REQUIRED_DEPENDENCIES = {
    "transformers": "4.57.6", "peft": "0.17.1",
    "accelerate": "1.10.1", "safetensors": "0.5.3",
}
_STOP = False
_OWNED_OUTPUT_DIR: Path | None = None


class TemporalTrainingError(RuntimeError):
    pass


def request_stop(_signum: int, _frame: Any) -> None:
    global _STOP
    _STOP = True


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class Progress:
    def __init__(self, output: Path, max_wall_minutes: float) -> None:
        if not math.isfinite(max_wall_minutes) or not 1 <= max_wall_minutes <= 90:
            raise TemporalTrainingError("temporal run budget must be between 1 and 90 minutes")
        self.output, self.started = output, time.time()
        self.deadline = self.started + max_wall_minutes * 60
        self.stage, self.stage_started = "initializing", self.started

    def begin(self, stage: str, torch: Any | None = None) -> None:
        self.stage, self.stage_started = stage, time.time()
        if torch is not None:
            torch.cuda.reset_peak_memory_stats()

    def stopping(self) -> bool:
        return _STOP or time.time() >= self.deadline - 60

    def write(self, torch: Any | None = None, **values: Any) -> None:
        now = time.time()
        record = {"schema_version": 1, "state": values.pop("state", "running"),
                  "stage": self.stage, "elapsed_seconds": round(now - self.started, 3),
                  "budget_remaining_seconds": round(max(0.0, self.deadline - now), 3), **values}
        if torch is not None:
            record["cuda_memory_mib"] = {
                "allocated": round(torch.cuda.memory_allocated() / 1024**2, 3),
                "reserved": round(torch.cuda.memory_reserved() / 1024**2, 3),
                "peak_allocated": round(torch.cuda.max_memory_allocated() / 1024**2, 3),
            }
        write_json_atomic(self.output / "progress.json", record)
        with (self.output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def initialize_progress(output_dir: Path, max_wall_minutes: float) -> Progress:
    global _OWNED_OUTPUT_DIR
    if output_dir.exists() and any(output_dir.iterdir()):
        raise TemporalTrainingError("output directory must be new or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    _OWNED_OUTPUT_DIR = output_dir.resolve()
    progress = Progress(output_dir, max_wall_minutes)
    progress.write(state="running", completed=0, total=0)
    return progress


@dataclass(frozen=True)
class Recipe:
    seed: int = 1729
    fps: float = 2.0
    projection_dim: int = 256
    epochs: int = 30
    patience: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    threshold: float = 0.5
    localization_sigma_sec: float = 1.0

    def validate(self) -> None:
        expected = (1729, 2.0, 256, 30, 5, 0.2, 0.5, 1.0)
        actual = (self.seed, self.fps, self.projection_dim, self.epochs, self.patience,
                  self.dropout, self.threshold, self.localization_sigma_sec)
        if actual != expected or self.learning_rate != 1e-3 or self.weight_decay != 1e-4:
            raise TemporalTrainingError("temporal recipe is fixed; hyperparameter sweeps are disabled")


def enforce_host(torch: Any) -> dict:
    if platform.system() != "Linux" or not torch.cuda.is_available():
        raise TemporalTrainingError("temporal training requires the authorized Linux CUDA host")
    props = torch.cuda.get_device_properties(0)
    if "L40S" not in str(props.name).upper() or int(props.total_memory) < MIN_CUDA_MEMORY_BYTES:
        raise TemporalTrainingError("temporal training requires the authorized L40S 48 GB class GPU")
    return {"gpu_name": str(props.name), "gpu_memory_bytes": int(props.total_memory),
            "cuda": torch.version.cuda}


def validate_dependencies() -> dict[str, str]:
    versions = {name: importlib.metadata.version(name) for name in ("torch", *REQUIRED_DEPENDENCIES)}
    for name, expected in REQUIRED_DEPENDENCIES.items():
        if versions[name] != expected:
            raise TemporalTrainingError(f"requires {name} {expected}, found {versions[name]}")
    if versions["torch"].split("+")[0] != "2.6.0":
        raise TemporalTrainingError("requires torch 2.6.0")
    return versions


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_safetensors(path: Path, tensors: dict[str, Any],
                       metadata: dict[str, str] | None = None) -> str:
    from safetensors.torch import save_file

    temporary = path.with_suffix(path.suffix + ".tmp")
    cpu_tensors = {name: value.detach().cpu().contiguous() for name, value in tensors.items()}
    save_file(cpu_tensors, str(temporary), metadata=metadata)
    os.replace(temporary, path)
    return file_sha256(path)


def spatial_summary(tokens: Any) -> Any:
    """Return global mean plus a 2x2 spatial pool when patch layout is square."""
    import torch
    if hasattr(tokens, "last_hidden_state"):
        tokens = tokens.last_hidden_state
    if isinstance(tokens, (tuple, list)):
        if len(tokens) != 1:
            raise TemporalTrainingError("ambiguous native image feature output")
        tokens = tokens[0]
    if tokens.ndim == 3:
        # Some processor configurations return multiple tiles for one frame.
        # Pool each tile spatially, then average tiles without making tile count
        # an input feature.
        return torch.stack([spatial_summary(tile) for tile in tokens]).mean(0)
    if tokens.ndim != 2:
        raise TemporalTrainingError("unexpected image feature shape")
    patch = tokens
    side = math.isqrt(patch.shape[0])
    if side * side != patch.shape[0] and patch.shape[0] > 1:
        candidate = patch[1:]
        side = math.isqrt(candidate.shape[0])
        if side * side == candidate.shape[0]:
            patch = candidate
    global_mean = patch.mean(dim=0)
    side = math.isqrt(patch.shape[0])
    if side * side != patch.shape[0] or side < 2:
        return torch.cat([global_mean] * 5)
    grid = patch.reshape(side, side, -1)
    cut = side // 2
    quadrants = (grid[:cut, :cut], grid[:cut, cut:], grid[cut:, :cut], grid[cut:, cut:])
    return torch.cat((global_mean, *(part.reshape(-1, part.shape[-1]).mean(0) for part in quadrants)))


def projection_matrix(input_dim: int, output_dim: int, torch: Any, device: str) -> Any:
    generator = torch.Generator(device="cpu").manual_seed(1729)
    matrix = torch.randn(input_dim, output_dim, generator=generator, dtype=torch.float32)
    matrix /= math.sqrt(output_dim)
    return matrix.to(device)


def cache_features(model: Any, processor: Any, samples: list, cache_path: Path,
                   progress: Progress, torch: Any) -> tuple[list[dict], str, str]:
    from navidu.ingest import iter_frames

    tensors, records, projection = {}, [], None
    total_frames = sum(max(2, math.ceil(sample.duration_sec * 2.0)) for sample in samples)
    completed_frames = 0
    model.eval()
    with torch.inference_mode():
        for index, sample in enumerate(samples):
            if progress.stopping():
                raise TemporalTrainingError("feature caching exceeded its bounded budget")
            frame_vectors, times = [], []
            for frame in iter_frames(sample.video_path, analysis_fps=2.0):
                if progress.stopping():
                    raise TemporalTrainingError("feature caching exceeded its bounded budget")
                batch = processor.image_processor(
                    images=[frame.rgb], crop_to_patches=False, return_tensors="pt")
                accepted = inspect.signature(model.get_image_features).parameters
                image_inputs = {}
                for name in ("pixel_values", "pixel_attention_mask", "spatial_shapes"):
                    if name in batch and name in accepted:
                        value = batch[name].to("cuda")
                        image_inputs[name] = value.to(dtype=torch.bfloat16) if name == "pixel_values" else value
                if "pixel_values" not in image_inputs:
                    raise TemporalTrainingError("processor did not produce native image features")
                tokens = model.get_image_features(**image_inputs)
                summary = spatial_summary(tokens).float()
                if projection is None:
                    projection = projection_matrix(summary.numel(), 256, torch, "cuda")
                frame_vectors.append((summary @ projection).cpu())
                times.append(float(frame.timestamp_sec))
                completed_frames += 1
                progress.write(torch, completed=completed_frames, total=total_frames,
                               sources_completed=index, sources_total=len(samples))
            if len(frame_vectors) < 2:
                raise TemporalTrainingError("a development source has fewer than two sampled frames")
            key = f"source_{index:02d}"
            tensors[key] = torch.stack(frame_vectors).contiguous()
            tensors[key + "_times"] = torch.tensor(times, dtype=torch.float32)
            records.append({"sample_id": sample.sample_id, "feature_key": key,
                            "has_violation": bool(sample.has_violation),
                            "time_sec": sample.time_sec if sample.time_supervision else None})
    cache_hash = atomic_safetensors(
        cache_path, tensors, {"schema_version": "1", "fps": "2.0"})
    projection_path = cache_path.with_name("projection.safetensors")
    projection_hash = atomic_safetensors(
        projection_path, {"projection": projection},
        {"schema_version": "1", "seed": "1729"})
    return records, cache_hash, projection_hash


def binary_metrics(rows: list[dict]) -> dict:
    tp = sum(row["truth"] and row["candidate"] for row in rows)
    tn = sum(not row["truth"] and not row["candidate"] for row in rows)
    fp = sum(not row["truth"] and row["candidate"] for row in rows)
    fn = sum(row["truth"] and not row["candidate"] for row in rows)
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "recall": recall, "specificity": specificity,
            "balanced_accuracy": (recall + specificity) / 2}


def evaluate(head: Any, items: list[dict], features: dict, mean: Any, std: Any,
             torch: Any) -> dict:
    rows, errors = [], []
    head.eval()
    with torch.inference_mode():
        for item in items:
            values = (features[item["feature_key"]].to("cuda") - mean) / std
            times = features[item["feature_key"] + "_times"].to("cuda")
            logits = head(values)
            score = float(torch.sigmoid(logits.max()))
            candidate = score >= 0.5
            time_sec = float(times[logits.argmax()]) if candidate else None
            error = abs(time_sec - item["time_sec"]) if candidate and item["time_sec"] is not None else None
            if error is not None:
                errors.append(error)
            rows.append({"sample_id": item["sample_id"], "truth": item["has_violation"],
                         "candidate": candidate, "score": score, "time_sec": time_sec,
                         "correct": candidate == item["has_violation"],
                         "localization_error_sec": error})
    result = binary_metrics(rows)
    result.update({"per_source": rows, "localization_mae_sec": sum(errors) / len(errors) if errors else None,
                   "localization_match_count": len(errors), "threshold": 0.5})
    return result


def run(args: argparse.Namespace) -> dict:
    global _OWNED_OUTPUT_DIR, _STOP
    _OWNED_OUTPUT_DIR = None
    _STOP = False
    progress = initialize_progress(args.output_dir, args.max_wall_minutes)

    import torch
    from navidu.perception.assets import model_entry, read_manifest, verify_model
    from safetensors.torch import load_file
    from transformers import AutoProcessor, InternVLForConditionalGeneration

    recipe = Recipe()
    recipe.validate()
    dependencies, host = validate_dependencies(), enforce_host(torch)
    torch.cuda.set_per_process_memory_fraction(0.70, device=0)
    host["cuda_allocator_fraction"] = 0.70
    random.seed(recipe.seed)
    torch.manual_seed(recipe.seed)
    torch.cuda.manual_seed_all(recipe.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    common = {"manifest_path": args.training_manifest, "inventory": args.inventory,
              "labels": args.labels, "folds": args.folds, "dataset_dir": args.dataset_dir,
              "verify_hashes": True}
    train = load_training_partition(**common, partition="train")
    validation = load_training_partition(**common, partition="validation")
    if (len(train), len(validation)) != (24, 8):
        raise TemporalTrainingError("fixed split must contain 24 train and 8 validation sources")
    models_dir = args.models_dir or args.base_model.parent
    base_entry = model_entry(read_manifest(models_dir), "internvl3_5-4b")
    if args.base_model.resolve() != (models_dir / str(base_entry["local_dir"])).resolve():
        raise TemporalTrainingError("--base-model must select the pinned 4B manifest entry")
    progress.begin("base_model_verification", torch)
    base_model_hash = verify_model(models_dir, base_entry)
    progress.write(torch, completed=1, total=1)
    baseline_sha256 = file_sha256(args.native_baseline)
    baseline = json.loads(args.native_baseline.read_text(encoding="utf-8"))
    baseline_rows = baseline.get("per_source", [])
    validation_truth = {sample.sample_id: bool(sample.has_violation) for sample in validation}
    baseline_ids = [row.get("sample_id") for row in baseline_rows]
    if (len(baseline_rows) != 8 or len(set(baseline_ids)) != 8
            or set(baseline_ids) != set(validation_truth)):
        raise TemporalTrainingError("native baseline does not match the fixed validation sources")
    for row in baseline_rows:
        row["truth"] = validation_truth[row["sample_id"]]
        if row.get("correct") is not (row.get("candidate") == row["truth"]):
            raise TemporalTrainingError("native baseline truth differs from fixed validation labels")
        row["score"] = None
    baseline["invalid_predictions"] = int(baseline.get("invalid_output_count", 0)) + int(
        baseline.get("no_valid_source_output_count", 0))
    progress.begin("model_loading", torch)
    processor = AutoProcessor.from_pretrained(args.base_model, local_files_only=True)
    model = InternVLForConditionalGeneration.from_pretrained(
        args.base_model, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    progress.write(torch, completed=1, total=1)
    progress.begin("feature_cache", torch)
    cache_path = args.output_dir / "features.safetensors"
    records, cache_hash, projection_hash = cache_features(
        model, processor, train + validation, cache_path, progress, torch)
    write_json_atomic(args.output_dir / "feature-cache-manifest.json", {
        "schema_version": 1, "cache_sha256": cache_hash,
        "dataset_fingerprint": json.loads(args.training_manifest.read_text(encoding="utf-8"))["dataset_fingerprint"],
        "fps": recipe.fps, "projection_seed": recipe.seed, "projection_dim": recipe.projection_dim,
        "projection_sha256": projection_hash,
        "sources": records,
    })
    del model
    torch.cuda.empty_cache()
    features = load_file(str(cache_path))
    train_items, val_items = records[:24], records[24:]
    stacked = torch.cat([features[item["feature_key"]] for item in train_items]).float()
    mean, std = stacked.mean(0).to("cuda"), stacked.std(0).clamp_min(1e-5).to("cuda")
    head = build_temporal_head().to("cuda")
    optimizer = torch.optim.AdamW(head.parameters(), lr=recipe.learning_rate,
                                  weight_decay=recipe.weight_decay)
    best, best_epoch, stale, history = baseline, None, 0, []
    progress.begin("head_training", torch)
    stop_reason = "epochs_complete"
    for epoch in range(1, recipe.epochs + 1):
        if progress.stopping():
            break
        order = list(range(len(train_items)))
        random.Random(recipe.seed + epoch).shuffle(order)
        head.train()
        losses = []
        for completed, item_index in enumerate(order, 1):
            if progress.stopping():
                break
            item = train_items[item_index]
            values = (features[item["feature_key"]].to("cuda") - mean) / std
            times = features[item["feature_key"] + "_times"].to("cuda")
            optimizer.zero_grad(set_to_none=True)
            loss, _parts = mil_loss(head(values), item["has_violation"], times, item["time_sec"])
            if not torch.isfinite(loss):
                raise TemporalTrainingError("non-finite temporal loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            progress.write(torch, epoch=epoch, completed=completed, total=len(order),
                           mean_loss=sum(losses) / len(losses))
        if len(losses) != len(order):
            stop_reason = "wall_budget_or_signal"
            break
        checkpoint_path = args.output_dir / f"epoch-{epoch}.safetensors"
        checkpoint_hash = atomic_safetensors(
            checkpoint_path, head.state_dict(), {"epoch": str(epoch)})
        metrics = evaluate(head, val_items, features, mean, std, torch)
        metrics["invalid_predictions"] = 0
        accepted = paired_improves(metrics, best)
        if accepted:
            best, best_epoch, stale = metrics, epoch, 0
            atomic_safetensors(
                args.output_dir / "temporal_head.safetensors", head.state_dict(),
                {"selected_epoch": str(epoch)})
        else:
            stale += 1
        history.append({"epoch": epoch, "mean_loss": sum(losses) / len(losses),
                        "validation": metrics, "accepted": accepted,
                        "checkpoint": checkpoint_path.name,
                        "checkpoint_sha256": checkpoint_hash})
        write_json_atomic(args.output_dir / "run-manifest.json", {
            "schema_version": 1, "state": "running", "base_model_id": BASE_MODEL_ID,
            "base_model_aggregate_sha256": base_model_hash,
            "recipe": asdict(recipe), "cache_sha256": cache_hash, "best_epoch": best_epoch,
            "baseline_validation": baseline, "native_baseline_sha256": baseline_sha256,
            "epochs": history, "holdout_used": False})
        if stale >= recipe.patience:
            stop_reason = "early_stopping_patience"
            break
    else:
        stop_reason = "epochs_complete"
    if progress.stopping() and stop_reason == "epochs_complete":
        stop_reason = "wall_budget_or_signal"
    selected = best_epoch is not None
    selected_hash = file_sha256(args.output_dir / "temporal_head.safetensors") if selected else None
    config = {"schema_version": 1, "model": "frozen_vision_temporal_mil",
              "base_model_id": BASE_MODEL_ID, "base_model_aggregate_sha256": base_model_hash,
              "input": "projected global+2x2 patch means and adjacent delta", "recipe": asdict(recipe),
              "feature_mean": mean.cpu().tolist(), "feature_std": std.cpu().tolist(),
              "projection_seed": 1729, "cache_sha256": cache_hash, "selected": selected}
    config["projection_sha256"] = projection_hash
    config["checkpoint_sha256"] = selected_hash
    config["native_baseline_sha256"] = baseline_sha256
    write_json_atomic(args.output_dir / "temporal-config.json", config)
    result = {"schema_version": 1, "state": "complete", "base_model_id": BASE_MODEL_ID,
              "base_model_aggregate_sha256": base_model_hash,
              "dependencies": dependencies, "host": host, "recipe": asdict(recipe),
              "cache_sha256": cache_hash, "projection_sha256": projection_hash,
              "checkpoint_sha256": selected_hash, "native_baseline_sha256": baseline_sha256,
              "baseline_validation": baseline, "epochs": history,
              "selected_epoch": best_epoch, "selected_validation": best if selected else None,
              "selection_metric": "paired_no_regression_then_localization_gain_at_least_0.5s",
              "stop_reason": stop_reason, "holdout_used": False,
              "elapsed_seconds": round(time.time() - progress.started, 3)}
    write_json_atomic(args.output_dir / "run-manifest.json", result)
    progress.begin("complete", torch)
    progress.write(
        torch, state="complete", completed=len(history), total=recipe.epochs,
        selected_epoch=best_epoch,
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--native-baseline", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, default=ROOT / "eval/training-splits.json")
    parser.add_argument("--inventory", type=Path, default=ROOT / "research/dataset/inventory.csv")
    parser.add_argument("--labels", type=Path, default=ROOT / "eval/labels.csv")
    parser.add_argument("--folds", type=Path, default=ROOT / "eval/folds.csv")
    parser.add_argument("--max-wall-minutes", type=float, default=90)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    signal.signal(signal.SIGTERM, request_stop)
    args = parse_args(argv)
    try:
        result = run(args)
    except BaseException as error:
        if _OWNED_OUTPUT_DIR == args.output_dir.resolve():
            write_json_atomic(args.output_dir / "progress.json", {
                "schema_version": 1, "state": "failed", "stage": "failed",
                "failure_type": type(error).__name__,
            })
        raise
    print(json.dumps({"event": "temporal_training_complete", "selected_epoch": result["selected_epoch"],
                      "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

