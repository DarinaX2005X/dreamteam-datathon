#!/usr/bin/env python3
"""Train and select the native InternVL proposal LoRA on an L40S-class host."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import signal
import sys
import time
from collections.abc import Iterable, Sequence
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

BASE_MODEL_ID = "OpenGVLab/InternVL3_5-4B-HF@6bd4487402110ef9889ba50eb7aefeb302526fed"
TRAIN_FOLDS = (1, 2, 3)
VALIDATION_FOLDS = (4,)
MIN_CUDA_MEMORY_BYTES = 40 * 1024**3
REQUIRED_DEPENDENCIES = {
    "transformers": "4.57.6",
    "peft": "0.17.1",
    "accelerate": "1.10.1",
    "safetensors": "0.5.3",
}


class TrainingError(RuntimeError):
    """Raised when training would violate the fixed recipe or artifact contract."""


_STOP_REQUESTED = False
_OWNED_OUTPUT_DIR: Path | None = None


def _request_stop(_signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


class ProgressReporter:
    """Persist aggregate, sanitized progress for detached training supervision."""

    def __init__(self, output_dir: Path, max_wall_minutes: float = 165) -> None:
        self.output_dir = output_dir
        self.started = time.time()
        if not math.isfinite(max_wall_minutes) or not 1 <= max_wall_minutes <= 180:
            raise TrainingError("wall time budget must be between 1 and 180 minutes")
        self.deadline = self.started + max_wall_minutes * 60
        self.stage_started = self.started
        self.stage = "initializing"

    def should_stop(self) -> bool:
        # Leave time for an adapter-only checkpoint; the launch supervisor also
        # enforces the user's hard wall-clock ceiling.
        return time.time() >= self.deadline - 60

    def begin(self, stage: str, torch_module: Any | None = None) -> None:
        self.stage = stage
        self.stage_started = time.time()
        if torch_module is not None:
            torch_module.cuda.reset_peak_memory_stats()

    def write(self, torch_module: Any | None = None, **fields: Any) -> None:
        now = time.time()
        completed = int(fields.get("completed", 0))
        total = int(fields.get("total", 0))
        stage_elapsed = max(0.0, now - self.stage_started)
        eta = None
        if completed > 0 and total >= completed:
            eta = stage_elapsed / completed * (total - completed)
        record = {
            "schema_version": 1,
            "state": fields.pop("state", "running"),
            "stage": self.stage,
            "epoch": fields.pop("epoch", None),
            "started_at_unix": round(self.started, 3),
            "elapsed_seconds": round(now - self.started, 3),
            "budget_remaining_seconds": round(max(0.0, self.deadline - now), 3),
            "stage_elapsed_seconds": round(stage_elapsed, 3),
            "completed": completed,
            "total": total,
            "stage_eta_seconds": round(eta, 3) if eta is not None else None,
            **fields,
        }
        if torch_module is not None:
            record["cuda_memory_mib"] = {
                "allocated": round(torch_module.cuda.memory_allocated() / 1024**2, 3),
                "reserved": round(torch_module.cuda.memory_reserved() / 1024**2, 3),
                "peak_allocated": round(torch_module.cuda.max_memory_allocated() / 1024**2, 3),
            }
        destination = self.output_dir / "progress.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
        with (self.output_dir / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


@dataclass(frozen=True)
class Recipe:
    seed: int = 1729
    epochs: int = 3
    frame_count: int = 16
    fps: float = 2.0
    windows_per_video: int = 6
    inference_stride_frames: int = 8
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 8
    max_new_tokens: int = 32

    def validate(self) -> None:
        if not 1 <= self.epochs <= 3:
            raise TrainingError("epochs must be between 1 and 3")
        if (self.frame_count, self.fps, self.windows_per_video) != (16, 2.0, 6):
            raise TrainingError("the approved recipe requires 16 frames, 2 fps, and 6 windows")
        if (self.rank, self.alpha, self.dropout) != (8, 16, 0.05):
            raise TrainingError("the approved LoRA recipe is rank 8, alpha 16, dropout 0.05")
        if self.gradient_accumulation_steps < 1:
            raise TrainingError("gradient accumulation must be positive")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enforce_training_host(torch_module: Any) -> dict[str, Any]:
    if platform.system() != "Linux":
        raise TrainingError("proposal training is allowed only on the authorized Linux GPU host")
    if not torch_module.cuda.is_available():
        raise TrainingError("proposal training requires CUDA")
    properties = torch_module.cuda.get_device_properties(0)
    name = str(properties.name)
    memory = int(properties.total_memory)
    if "L40S" not in name.upper() or memory < MIN_CUDA_MEMORY_BYTES:
        raise TrainingError("proposal training requires the authorized NVIDIA L40S 48 GB class GPU")
    return {"gpu_name": name, "gpu_memory_bytes": memory, "cuda": torch_module.version.cuda}


def seed_everything(seed: int, torch_module: Any) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)
    torch_module.use_deterministic_algorithms(True, warn_only=True)


def binary_metrics(truth: Sequence[bool], predicted: Sequence[bool]) -> dict[str, float | int]:
    if len(truth) != len(predicted) or not truth:
        raise TrainingError("validation predictions must match a nonempty truth set")
    tp = sum(t and p for t, p in zip(truth, predicted))
    tn = sum((not t) and (not p) for t, p in zip(truth, predicted))
    fp = sum((not t) and p for t, p in zip(truth, predicted))
    fn = sum(t and (not p) for t, p in zip(truth, predicted))
    positive = tp + fn
    negative = tn + fp
    recall = tp / positive if positive else 0.0
    specificity = tn / negative if negative else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def choose_video_proposal(proposals: Iterable[Any]) -> tuple[bool, float | None]:
    """Aggregate windows deterministically: any candidate, earliest valid proposed time."""
    positive_times = [
        float(proposal.time_sec)
        for proposal in proposals
        if bool(proposal.candidate) and proposal.time_sec is not None
    ]
    return (bool(positive_times), min(positive_times) if positive_times else None)


def checkpoint_improves(candidate: dict[str, Any], reference: dict[str, Any]) -> bool:
    """Accept a paired gain without sacrificing any previously correct source."""
    if candidate.get("invalid_output_count", 0) or candidate.get("no_valid_source_output_count", 0):
        return False
    current = {item["sample_id"]: item for item in candidate.get("per_source", [])}
    previous = {item["sample_id"]: item for item in reference.get("per_source", [])}
    if not current or current.keys() != previous.keys():
        return False
    if any(previous[key]["correct"] and not current[key]["correct"] for key in current):
        return False
    if any(current[key]["correct"] and not previous[key]["correct"] for key in current):
        return True
    # With identical decisions, compare time errors on the same true positives.
    current_errors = {key: row["localization_error_sec"] for key, row in current.items()
                      if row.get("localization_error_sec") is not None}
    previous_errors = {key: row["localization_error_sec"] for key, row in previous.items()
                       if row.get("localization_error_sec") is not None}
    return bool(current_errors) and current_errors.keys() == previous_errors.keys() \
        and (sum(previous_errors.values()) - sum(current_errors.values())) / len(current_errors) >= 0.5


def adapter_state(model: Any, torch_module: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().to(device="cpu", dtype=torch_module.float32).clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def count_changed(before: dict[str, Any], model: Any, torch_module: Any) -> tuple[int, int]:
    after = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(after) != set(before):
        raise TrainingError("trainable parameter set changed during training")
    changed = sum(
        not torch_module.equal(value, after[name].detach().to(device="cpu", dtype=torch_module.float32))
        for name, value in before.items()
    )
    return changed, len(before)


def move_batch(batch: Any, device: str) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def _decode_generated(processor: Any, generated: Any, prompt_length: int) -> str:
    tokens = generated[0][prompt_length:]
    return processor.decode(tokens, skip_special_tokens=True)


def evaluate(
    model: Any,
    processor: Any,
    samples: Sequence[Any],
    recipe: Recipe,
    torch_module: Any,
    reporter: ProgressReporter | None = None,
    epoch: int | None = None,
) -> dict[str, Any]:
    from navidu.perception.proposal_task import decode_proposal, encode_proposal

    from training.data import decode_window, make_inference_windows

    windows = make_inference_windows(
        samples,
        frame_count=recipe.frame_count,
        fps=recipe.fps,
        stride_frames=recipe.inference_stride_frames,
    )
    by_sample: dict[str, list[Any]] = {sample.sample_id: [] for sample in samples}
    cache: dict[Any, Any] = {}
    invalid_outputs = 0
    model.eval()
    with torch_module.inference_mode():
        for completed, spec in enumerate(windows, start=1):
            if _STOP_REQUESTED or (reporter is not None and reporter.should_stop()):
                raise TrainingError("validation stopped before completion; checkpoint is not selected")
            frames = decode_window(
                spec, frame_count=recipe.frame_count, fps=recipe.fps, cache=cache
            )
            timestamps = [float(timestamp) for timestamp, _ in frames]
            inputs = move_batch(encode_proposal(processor, frames), "cuda")
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=recipe.max_new_tokens,
                use_cache=True,
            )
            try:
                proposal = decode_proposal(
                    _decode_generated(processor, generated, inputs["input_ids"].shape[-1]),
                    timestamps,
                )
                by_sample[spec.sample.sample_id].append(proposal)
            except RuntimeError:
                invalid_outputs += 1

            if reporter is not None:
                reporter.write(
                    torch_module,
                    epoch=epoch,
                    completed=completed,
                    total=len(windows),
                    validation_completed=completed,
                    validation_total=len(windows),
                )

    truth: list[bool] = []
    predicted: list[bool] = []
    localization_errors: list[float] = []
    no_valid_output_count = 0
    per_source = []
    for sample in samples:
        proposals = by_sample[sample.sample_id]
        if not proposals:
            # A source with no parseable output receives no true-positive or
            # true-negative credit; this is an evaluation penalty, not inference.
            no_valid_output_count += 1
            truth.append(bool(sample.has_violation))
            predicted.append(not bool(sample.has_violation))
            per_source.append({"sample_id": sample.sample_id, "correct": False,
                               "candidate": None, "time_sec": None,
                               "localization_error_sec": None})
            continue
        candidate, time_sec = choose_video_proposal(proposals)
        truth.append(bool(sample.has_violation))
        predicted.append(candidate)
        error = None
        if candidate and sample.has_violation and sample.time_supervision and time_sec is not None:
            error = abs(time_sec - float(sample.time_sec))
            localization_errors.append(error)
        per_source.append({"sample_id": sample.sample_id,
                           "correct": candidate == bool(sample.has_violation),
                           "candidate": candidate, "time_sec": time_sec,
                           "localization_error_sec": error})
    metrics = binary_metrics(truth, predicted)
    metrics.update(
        {
            "window_count": len(windows),
            "invalid_output_count": invalid_outputs,
            "no_valid_source_output_count": no_valid_output_count,
            "per_source": per_source,
            "localization_mae_sec": (
                sum(localization_errors) / len(localization_errors)
                if localization_errors
                else None
            ),
            "localization_match_count": len(localization_errors),
            "aggregation": "candidate if any valid window is positive; earliest proposed time",
        }
    )
    return metrics


def train_epoch(
    model: Any,
    processor: Any,
    windows: Sequence[Any],
    recipe: Recipe,
    epoch: int,
    optimizer: Any,
    torch_module: Any,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    from navidu.perception.proposal_task import encode_proposal

    from training.data import decode_window, supervised_target

    order = list(range(len(windows)))
    random.Random(recipe.seed + epoch).shuffle(order)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    losses: list[float] = []
    optimizer_step_losses: list[dict[str, float | int]] = []
    accumulated_losses: list[float] = []
    optimizer_steps = 0
    skipped_unsupervised = 0
    steps_per_epoch = math.ceil(len(windows) / recipe.gradient_accumulation_steps)
    cache: dict[Any, Any] = {}
    for index in order:
        spec = windows[index]
        frames = decode_window(spec, frame_count=recipe.frame_count, fps=recipe.fps, cache=cache)
        target = supervised_target(spec, frames)
        if target is None:
            skipped_unsupervised += 1
            continue
        batch = move_batch(encode_proposal(processor, frames, target=target), "cuda")
        labels = batch.get("labels")
        if labels is None or not bool((labels != -100).any()):
            raise TrainingError("proposal encoder did not produce response-only supervised labels")
        loss = model(**batch).loss
        if not torch_module.isfinite(loss):
            raise TrainingError("non-finite training loss")
        (loss / recipe.gradient_accumulation_steps).backward()
        numeric_loss = float(loss.detach().cpu())
        losses.append(numeric_loss)
        accumulated_losses.append(numeric_loss)
        budget_exhausted = reporter is not None and reporter.should_stop()
        if len(accumulated_losses) == recipe.gradient_accumulation_steps or budget_exhausted or _STOP_REQUESTED:
            rescale_partial_gradients(model, len(accumulated_losses), recipe.gradient_accumulation_steps)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            optimizer_step_losses.append(
                {
                    "optimizer_step": optimizer_steps,
                    "mean_microbatch_loss": sum(accumulated_losses) / len(accumulated_losses),
                    "microbatch_count": len(accumulated_losses),
                }
            )
            if reporter is not None:
                reporter.write(
                    torch_module,
                    epoch=epoch,
                    completed=optimizer_steps,
                    total=steps_per_epoch,
                    optimizer_steps_completed=(epoch - 1) * steps_per_epoch + optimizer_steps,
                    optimizer_steps_total=recipe.epochs * steps_per_epoch,
                    mean_loss=sum(losses) / len(losses),
                    latest_loss=numeric_loss,
                    skipped_unsupervised_windows=skipped_unsupervised,
                    stop_requested=_STOP_REQUESTED or budget_exhausted,
                )
            accumulated_losses.clear()
            if _STOP_REQUESTED or budget_exhausted:
                break
    if accumulated_losses:
        rescale_partial_gradients(
            model, len(accumulated_losses), recipe.gradient_accumulation_steps
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        optimizer_step_losses.append(
            {
                "optimizer_step": optimizer_steps,
                "mean_microbatch_loss": sum(accumulated_losses) / len(accumulated_losses),
                "microbatch_count": len(accumulated_losses),
            }
        )
        if reporter is not None:
            reporter.write(
                torch_module,
                epoch=epoch,
                completed=optimizer_steps,
                total=steps_per_epoch,
                optimizer_steps_completed=(epoch - 1) * steps_per_epoch + optimizer_steps,
                optimizer_steps_total=recipe.epochs * steps_per_epoch,
                mean_loss=sum(losses) / len(losses),
                latest_loss=losses[-1],
                skipped_unsupervised_windows=skipped_unsupervised,
                stop_requested=_STOP_REQUESTED,
            )
    if not losses or not optimizer_steps:
        raise TrainingError("training produced no supervised optimizer steps")
    return {
        "epoch": epoch,
        "mean_loss": sum(losses) / len(losses),
        "min_loss": min(losses),
        "max_loss": max(losses),
        "supervised_steps": len(losses),
        "optimizer_steps": optimizer_steps,
        "skipped_unsupervised_windows": skipped_unsupervised,
        "optimizer_step_metrics": optimizer_step_losses,
        "stop_requested": _STOP_REQUESTED or (reporter is not None and reporter.should_stop()),
    }


def artifact_hashes(directory: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            artifacts.append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return artifacts


def package_versions() -> dict[str, str]:
    result = {}
    for package in ("torch", "transformers", "peft", "accelerate", "safetensors"):
        result[package] = importlib.metadata.version(package)
    return result


def validate_versions(versions: dict[str, str]) -> None:
    if not versions["torch"].startswith("2.6.0"):
        raise TrainingError(f"torch 2.6.0 is required, found {versions['torch']}")
    for package, expected in REQUIRED_DEPENDENCIES.items():
        if versions[package] != expected:
            raise TrainingError(f"{package} {expected} is required, found {versions[package]}")


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def initialize_progress(output_dir: Path, max_wall_minutes: float = 165) -> ProgressReporter:
    """Claim a new/empty run directory without altering an existing run."""
    global _OWNED_OUTPUT_DIR
    if output_dir.exists() and any(output_dir.iterdir()):
        raise TrainingError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _OWNED_OUTPUT_DIR = output_dir.resolve()
    reporter = ProgressReporter(output_dir, max_wall_minutes)
    reporter.write(state="running", completed=0, total=0)
    return reporter


def rescale_partial_gradients(model: Any, microbatch_count: int, target_count: int) -> None:
    """Undo fixed accumulation division for a final short optimizer group."""
    if not 0 < microbatch_count <= target_count:
        raise TrainingError("invalid partial gradient accumulation count")
    if microbatch_count == target_count:
        return
    factor = target_count / microbatch_count
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(factor)


def mark_failed_status(output_dir: Path, error: BaseException) -> None:
    """Finish detached status without persisting exception text or private paths."""
    status_path = output_dir / "progress.json"
    if not status_path.is_file():
        return
    try:
        record = json.loads(status_path.read_text(encoding="utf-8"))
        if record.get("state") == "interrupted":
            return
        record.update(
            {
                "state": "failed",
                "elapsed_seconds": round(
                    time.time() - float(record.get("started_at_unix", time.time())), 3
                ),
                "stage_eta_seconds": None,
                "failure_type": type(error).__name__,
            }
        )
        write_json_atomic(status_path, record)
        with (output_dir / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, TypeError, ValueError):
        pass


def run(args: argparse.Namespace) -> dict[str, Any]:
    global _OWNED_OUTPUT_DIR, _STOP_REQUESTED
    _STOP_REQUESTED = False
    _OWNED_OUTPUT_DIR = None
    reporter = initialize_progress(args.output_dir, args.max_wall_minutes)
    recipe = Recipe(
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    recipe.validate()
    dependencies = package_versions()
    validate_versions(dependencies)

    import torch
    from navidu.perception.assets import model_entry, read_manifest, verify_model
    from navidu.perception.proposal_task import LORA_TARGET_MODULES, PROMPT_VERSION
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, InternVLForConditionalGeneration

    from eval.training import load_training_partition
    from training.data import make_training_windows

    host = enforce_training_host(torch)
    if not math.isfinite(args.cuda_memory_fraction) or not 0 < args.cuda_memory_fraction <= 1:
        raise TrainingError("CUDA allocator fraction must be in (0, 1]")
    torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device=0)
    host["cuda_allocator_fraction"] = args.cuda_memory_fraction
    seed_everything(recipe.seed, torch)

    models_dir = args.models_dir or args.base_model.parent
    model_manifest = read_manifest(models_dir)
    base_entry = model_entry(model_manifest, "internvl3_5-4b")
    expected_base = (models_dir / str(base_entry["local_dir"])).resolve()
    if args.base_model.resolve() != expected_base:
        raise TrainingError("--base-model must select the pinned internvl3_5-4b manifest entry")
    reporter.begin("base_model_verification")
    reporter.write(completed=0, total=1)
    base_model_aggregate_sha256 = verify_model(models_dir, base_entry)
    reporter.write(completed=1, total=1)

    reporter.begin("dataset_loading")
    reporter.write(completed=0, total=2)
    load_kwargs = {
        "manifest_path": args.training_manifest,
        "inventory": args.inventory,
        "labels": args.labels,
        "folds": args.folds,
        "dataset_dir": args.dataset_dir,
        "verify_hashes": True,
    }
    train_samples = load_training_partition(**load_kwargs, partition="train")
    reporter.write(completed=1, total=2)
    validation_samples = load_training_partition(**load_kwargs, partition="validation")
    reporter.write(completed=2, total=2)
    if (len(train_samples), len(validation_samples)) != (24, 8):
        raise TrainingError("fixed inner split must contain 24 training and 8 validation videos")
    split_manifest = json.loads(args.training_manifest.read_text(encoding="utf-8"))

    reporter.begin("model_loading", torch)
    reporter.write(torch, completed=0, total=1)
    processor = AutoProcessor.from_pretrained(args.base_model, local_files_only=True)
    model = InternVLForConditionalGeneration.from_pretrained(
        args.base_model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reporter.write(torch, completed=1, total=1)
    reporter.begin("baseline_validation", torch)
    baseline = evaluate(
        model, processor, validation_samples, recipe, torch, reporter=reporter, epoch=0
    )
    baseline_seconds = time.time() - reporter.stage_started
    write_json_atomic(args.output_dir / "baseline-validation.json", baseline)

    lora = LoraConfig(
        r=recipe.rank,
        lora_alpha=recipe.alpha,
        lora_dropout=recipe.dropout,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise TrainingError("only LoRA parameters may remain trainable")
    before = adapter_state(model, torch)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=recipe.learning_rate,
        weight_decay=recipe.weight_decay,
    )
    windows = make_training_windows(
        train_samples,
        frame_count=recipe.frame_count,
        fps=recipe.fps,
        windows_per_video=recipe.windows_per_video,
        seed=recipe.seed,
    )
    history = []
    best_epoch = None
    best_metrics = baseline
    stop_reason = "epochs_complete"
    for epoch in range(1, recipe.epochs + 1):
        epoch_started = time.time()
        reporter.begin("training", torch)
        training_metrics = train_epoch(
            model, processor, windows, recipe, epoch, optimizer, torch, reporter=reporter
        )
        if training_metrics["stop_requested"]:
            checkpoint = args.output_dir / f"interrupted-epoch-{epoch}"
            model.save_pretrained(checkpoint, safe_serialization=True)
            interruption = {
                "schema_version": 1,
                "state": "interrupted",
                "epoch": epoch,
                "restart_from_completed_epoch": epoch - 1,
                "exact_mid_epoch_resume_supported": False,
                "checkpoint": checkpoint.name,
                "training": training_metrics,
                "artifacts": artifact_hashes(checkpoint),
            }
            write_json_atomic(args.output_dir / "resume-metadata.json", interruption)
            reporter.write(
                torch,
                state="interrupted",
                epoch=epoch,
                completed=training_metrics["optimizer_steps"],
                total=math.ceil(len(windows) / recipe.gradient_accumulation_steps),
                mean_loss=training_metrics["mean_loss"],
                latest_loss=training_metrics["optimizer_step_metrics"][-1][
                    "mean_microbatch_loss"
                ],
            )
            raise TrainingError("training interrupted safely at an optimizer boundary")
        changed, trainable_count = count_changed(before, model, torch)
        if changed == 0:
            raise TrainingError("training did not change any adapter tensors")
        reporter.begin("checkpoint", torch)
        reporter.write(torch, epoch=epoch, completed=0, total=1)
        checkpoint = args.output_dir / f"epoch-{epoch}"
        model.save_pretrained(checkpoint, safe_serialization=True)
        artifacts = artifact_hashes(checkpoint)
        if not any(item["path"].endswith(".safetensors") for item in artifacts):
            raise TrainingError("adapter checkpoint did not contain safetensors weights")
        reporter.write(torch, epoch=epoch, completed=1, total=1)
        write_json_atomic(
            args.output_dir / "run-manifest.json",
            {
                "schema_version": 1,
                "state": "validation_pending",
                "base_model_id": BASE_MODEL_ID,
                "base_model_aggregate_sha256": base_model_aggregate_sha256,
                "base_model_revision": base_entry["revision"],
                "prompt_version": PROMPT_VERSION,
                "dataset_fingerprint": split_manifest["dataset_fingerprint"],
                "folds": {
                    "train": list(TRAIN_FOLDS),
                    "validation": list(VALIDATION_FOLDS),
                },
                "holdout_used": False,
                "recipe": asdict(recipe),
                "dependencies": dependencies,
                "host": host,
                "sample_counts": {
                    "train": len(train_samples),
                    "validation": len(validation_samples),
                },
                "training_window_count": len(windows),
                "baseline_validation": baseline,
                "epochs": history,
                "pending_epoch": {
                    "epoch": epoch,
                    "training": training_metrics,
                    "changed_trainable_tensors": changed,
                    "trainable_tensor_count": trainable_count,
                    "checkpoint": checkpoint.name,
                    "artifacts": artifacts,
                },
                "selected_epoch": best_epoch,
                "selection_metric": "paired_no_regression_then_localization",
                "elapsed_seconds": round(time.time() - reporter.started, 3),
            },
        )
        reporter.begin("validation", torch)
        validation_metrics = evaluate(
            model,
            processor,
            validation_samples,
            recipe,
            torch,
            reporter=reporter,
            epoch=epoch,
        )
        if checkpoint_improves(validation_metrics, best_metrics):
            best_metrics = validation_metrics
            best_epoch = epoch
        history.append(
            {
                "epoch": epoch,
                "training": training_metrics,
                "validation": validation_metrics,
                "changed_trainable_tensors": changed,
                "trainable_tensor_count": trainable_count,
                "checkpoint": checkpoint.name,
                "artifacts": artifacts,
            }
        )

        partial_manifest = {
            "schema_version": 1,
            "state": "running" if epoch < recipe.epochs else "complete",
            "base_model_id": BASE_MODEL_ID,
            "base_model_aggregate_sha256": base_model_aggregate_sha256,
            "base_model_revision": base_entry["revision"],
            "prompt_version": PROMPT_VERSION,
            "dataset_fingerprint": split_manifest["dataset_fingerprint"],
            "folds": {"train": list(TRAIN_FOLDS), "validation": list(VALIDATION_FOLDS)},
            "holdout_used": False,
            "recipe": asdict(recipe),
            "dependencies": dependencies,
            "host": host,
            "sample_counts": {"train": len(train_samples), "validation": len(validation_samples)},
            "training_window_count": len(windows),
            "baseline_validation": baseline,
            "epochs": history,
            "selected_epoch": best_epoch,
            "selection_metric": "paired_no_regression_then_localization",
            "elapsed_seconds": round(time.time() - reporter.started, 3),
        }
        write_json_atomic(args.output_dir / "run-manifest.json", partial_manifest)
        epoch_seconds = time.time() - epoch_started
        # Continue only if there is evidence of improvement and room for a
        # complete next epoch plus validation and checkpoint reserve.
        if best_epoch is None:
            stop_reason = "no_paired_validation_gain"
            break
        if epoch < recipe.epochs and time.time() + max(epoch_seconds, baseline_seconds) + 60 >= reporter.deadline:
            stop_reason = "next_epoch_exceeds_budget"
            break

    result = {
        "schema_version": 1,
        "state": "complete",
        "base_model_id": BASE_MODEL_ID,
        "base_model_aggregate_sha256": base_model_aggregate_sha256,
        "base_model_revision": base_entry["revision"],
        "prompt_version": PROMPT_VERSION,
        "dataset_fingerprint": split_manifest["dataset_fingerprint"],
        "folds": {"train": list(TRAIN_FOLDS), "validation": list(VALIDATION_FOLDS)},
        "holdout_used": False,
        "recipe": asdict(recipe),
        "dependencies": dependencies,
        "host": host,
        "sample_counts": {"train": len(train_samples), "validation": len(validation_samples)},
        "training_window_count": len(windows),
        "baseline_validation": baseline,
        "epochs": history,
        "selected_epoch": best_epoch,
        "selection_metric": "paired_no_regression_then_localization",
        "stop_reason": stop_reason,
        "baseline_validation_seconds": round(baseline_seconds, 3),
        "elapsed_seconds": round(time.time() - reporter.started, 3),
    }
    write_json_atomic(args.output_dir / "run-manifest.json", result)
    reporter.begin("complete", torch)
    reporter.write(
        torch,
        state="complete",
        epoch=len(history),
        completed=len(history),
        total=recipe.epochs,
        selected_epoch=best_epoch,
        selected_balanced_accuracy=best_metrics["balanced_accuracy"] if best_epoch else None,
        stop_reason=stop_reason,
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, default=ROOT / "eval/training-splits.json")
    parser.add_argument("--inventory", type=Path, default=ROOT / "research/dataset/inventory.csv")
    parser.add_argument("--labels", type=Path, default=ROOT / "eval/labels.csv")
    parser.add_argument("--folds", type=Path, default=ROOT / "eval/folds.csv")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-wall-minutes", type=float, default=165)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.70)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    signal.signal(signal.SIGTERM, _request_stop)
    args = parse_args(argv)
    try:
        result = run(args)
    except BaseException as error:
        if _OWNED_OUTPUT_DIR == args.output_dir.resolve():
            mark_failed_status(args.output_dir, error)
        raise
    print(
        json.dumps(
            {
                "event": "proposal_training_complete",
                "selected_epoch": result["selected_epoch"],
                "baseline_balanced_accuracy": result["baseline_validation"]["balanced_accuracy"],
                "selected_balanced_accuracy": result["epochs"][result["selected_epoch"] - 1][
                    "validation"
                ]["balanced_accuracy"] if result["selected_epoch"] else None,
                "output_dir": str(args.output_dir),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TrainingError as error:
        print(json.dumps({"event": "proposal_training_failed", "reason": str(error)}), file=sys.stderr)
        raise SystemExit(1)
    except Exception as error:  # noqa: BLE001 - detached jobs require a terminal status
        print(
            json.dumps(
                {"event": "proposal_training_failed", "failure_type": type(error).__name__}
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)
