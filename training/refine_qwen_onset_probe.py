#!/usr/bin/env python3
"""CPU-only linear refinements over the immutable A2 Qwen feature cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 1729
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
HARD_WALL_SECONDS = 10 * 60
VARIANTS = ("delta_only_linear", "full_transition_linear")
REQUIRED_POSITIVE_IDS = {
    "69caf59620fcd80bedcb664514730ab38f2f800da27000a8e037d7440be5dc7a",
    "90607b0b0cb772fcf29b0c7e0e56fa6af36da6fac0efe93b8a616b5de94a156c",
}
ADDITIONAL_POSITIVE_IDS = {
    "a37930cf119bb691be77d8624b18333b04d9c67066a7f4627d63e3208dfa6057",
    "66fc5d28c2371592f9260305589bdc86636fdd2ebe7a7ec976eec00a23d43df4",
}
REQUIRED_NEGATIVE_IDS = {
    "901f61cf1a1b8754dda56466308c19fc203a1a4c0e85d1e7827849d34e2d52f0",
    "ac4cf04844c0aa08a5e19f5f7fe984f1477943294922d6ef77bdc5b88b97160e",
    "deeeee54539fb894b17447296bd7f0807b9dc662bad9af4dcd27f4fbdd677187",
    "efd876a04224cac781ac64bf5d6e107ded67340b7b7715fd9ca3408b240a1075",
}


class RefinementError(RuntimeError):
    pass


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def variant_features(pairs: Any, variant: str) -> Any:
    if pairs.ndim != 2 or pairs.shape[1] % 4:
        raise ValueError("transition feature width must contain four equal blocks")
    if variant == "delta_only_linear":
        return pairs[:, pairs.shape[1] // 2:]
    if variant == "full_transition_linear":
        return pairs
    raise ValueError("unknown refinement variant")


def choose_variant(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    feasible = [row for row in results if row.get("feasible")]
    if not feasible:
        return None
    order = {name: -index for index, name in enumerate(VARIANTS)}
    return max(feasible, key=lambda row: (
        row["oof_metrics"]["balanced_accuracy"], row["oof_metrics"]["specificity"],
        row["oof_metrics"]["recall"], order[row["variant"]]))


def load_inputs(original_report: Path, cache_dir: Path, split_path: Path,
                torch: Any) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    from safetensors.torch import load_file
    original = json.loads(original_report.read_text(encoding="utf-8"))
    if original.get("state") != "complete" or original.get("holdout_used") is not False \
            or not isinstance(original.get("cache_identity"), dict):
        raise RefinementError("original A2 manifest is incomplete")
    identity = original["cache_identity"]
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    cache_path = cache_dir / f"{hashlib.sha256(encoded).hexdigest()}.safetensors"
    if not cache_path.is_file() or file_sha256(cache_path) != original.get("cache_sha256"):
        raise RefinementError("private cache is missing or differs from the A2 manifest")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("dataset_fingerprint") != identity.get("dataset_fingerprint") \
            or split.get("reserved_holdout_examples_included") is not False:
        raise RefinementError("split does not match the cache identity")
    metadata = {row["sample_id"]: row for row in split["samples"]}
    sources = identity.get("sources", [])
    if len(sources) != 32 or set(metadata) != {row.get("sample_id") for row in sources}:
        raise RefinementError("cache source inventory differs from fixed development split")
    tensors = load_file(str(cache_path))
    expected = {f"source_{i:02d}_{suffix}" for i in range(32)
                for suffix in ("features", "times")}
    if set(tensors) != expected:
        raise RefinementError("cache tensor inventory is invalid")
    items = []
    for index, source in enumerate(sources):
        sample_id = source["sample_id"]
        row = metadata[sample_id]
        target = row["targets"]
        items.append({"sample_id": sample_id, "partition": row["partition"],
                      "fold": int(row["fold"]), "truth": bool(target["has_violation"]),
                      "onset_sec": target["time_sec"] if target["time_supervision"] else None,
                      "frame_features": tensors[f"source_{index:02d}_features"],
                      "frame_times": tensors[f"source_{index:02d}_times"]})
    return items, original, file_sha256(cache_path)


def materialize(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create transition features only for the partition currently being used."""
    from training.qwen_onset_probe import transition_pairs

    result = []
    for item in items:
        pairs, times = transition_pairs(item["frame_features"], item["frame_times"])
        result.append({key: value for key, value in item.items()
                       if key not in {"frame_features", "frame_times"}} |
                      {"pairs": pairs.float(), "times": times.float()})
    return result


def fit_linear(items: list[dict[str, Any]], variant: str, seed_offset: int,
               torch: Any) -> tuple[Any, Any, Any, list[float]]:
    from training.qwen_onset_probe import balanced_source_loss, source_loss

    torch.manual_seed(SEED + seed_offset)
    selected = [variant_features(item["pairs"], variant) for item in items]
    stacked = torch.cat(selected)
    mean, std = stacked.mean(0), stacked.std(0).clamp_min(1e-5)
    head = torch.nn.Linear(stacked.shape[1], 1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    history = []
    for _epoch in range(EPOCHS):
        source_losses = []
        for item, values in zip(items, selected):
            logits = head((values - mean) / std).flatten()
            loss, _ = source_loss(logits, item["times"], item["truth"], item["onset_sec"])
            source_losses.append((item["truth"], loss))
        total = balanced_source_loss(source_losses)
        if not torch.isfinite(total):
            raise RefinementError("non-finite linear probe loss")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        history.append(float(total.detach()))
    return head.eval(), mean, std, history


def score(head: Any, mean: Any, std: Any, items: list[dict[str, Any]], variant: str,
          torch: Any) -> list[dict[str, Any]]:
    from training.qwen_onset_probe import normalized_log_mean_exp

    rows = []
    with torch.inference_mode():
        for item in items:
            values = variant_features(item["pairs"], variant)
            logits = head((values - mean) / std).flatten()
            rows.append({"sample_id": item["sample_id"], "truth": item["truth"],
                         "score": float(torch.sigmoid(normalized_log_mean_exp(logits))),
                         "time_sec": float(item["times"][int(logits.argmax())])})
    return rows


def metrics_at(rows: list[dict[str, Any]], threshold: float) -> dict[str, float | int]:
    from training.qwen_onset_probe import binary_metrics

    return binary_metrics([row["truth"] for row in rows],
                          [row["score"] >= threshold for row in rows])


def evaluate_variant(train_items: list[dict[str, Any]], variant: str,
                     torch: Any) -> dict[str, Any]:
    from training.qwen_onset_probe import calibrate_threshold

    oof_rows, folds = [], []
    for fold in (1, 2, 3):
        fit = [item for item in train_items if item["fold"] != fold]
        held = [item for item in train_items if item["fold"] == fold]
        head, mean, std, history = fit_linear(fit, variant, fold, torch)
        train_rows = score(head, mean, std, fit, variant, torch)
        held_rows = score(head, mean, std, held, variant, torch)
        oof_rows.extend(held_rows)
        folds.append({"fold": fold, "fit_sources": 16, "held_sources": 8,
                      "train_metrics_at_0_5": metrics_at(train_rows, 0.5),
                      "oof_metrics_at_0_5": metrics_at(held_rows, 0.5),
                      "initial_loss": history[0], "final_loss": history[-1]})
    result = {"variant": variant, "oof_rows": oof_rows, "per_fold": folds,
              "diagnostic_metrics_at_0_5": metrics_at(oof_rows, 0.5)}
    try:
        calibration = calibrate_threshold(oof_rows)
    except ValueError as error:
        result.update({"feasible": False, "rejection_reason": str(error),
                       "threshold": None, "oof_metrics": None})
        return result
    result.update({"feasible": True, "threshold": calibration.threshold,
                   "oof_metrics": calibration.metrics})
    for fold in folds:
        held = [row for row in oof_rows if next(
            item["fold"] for item in train_items if item["sample_id"] == row["sample_id"]
        ) == fold["fold"]]
        fold["oof_metrics_at_selected_threshold"] = metrics_at(held, calibration.threshold)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    if platform.system() != "Linux" or os.getenv("MODAL_TASK_ID") is None:
        raise RefinementError("refinement runs only in the authorized Modal task")
    if args.output_dir.exists():
        raise RefinementError("output directory already exists")
    args.output_dir.mkdir(parents=True)
    recipe = {"schema_version": 1, "variants": list(VARIANTS), "seed": SEED,
              "epochs": EPOCHS, "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
              "weight_decay": WEIGHT_DECAY, "scaler": "fit-fold-only standardization",
              "bag": "normalized_log_mean_exp_tau_0.25",
              "positive_supervision": "onset_support_only_radius_1.5_sigma_1.0",
              "oof_guard": {"specificity": 0.75, "recall": 0.5},
              "selection": "OOF balanced_accuracy then specificity then recall; A wins ties"}
    write_json(args.output_dir / "recipe.json", recipe)
    import torch
    from safetensors.torch import save_file

    from training.qwen_onset_probe import strict_validation_gate

    torch.set_num_threads(2)
    random.seed(SEED)
    torch.manual_seed(SEED)
    items, original, cache_hash = load_inputs(
        args.original_report, args.cache_dir, args.training_manifest, torch)
    train_items = materialize([item for item in items if item["partition"] == "train"])
    if (len(train_items), sum(item["truth"] for item in train_items)) != (24, 12):
        raise RefinementError("train split is not the fixed 12+/12-")
    variants = []
    for name in VARIANTS:
        variants.append(evaluate_variant(train_items, name, torch))
        write_json(args.output_dir / "progress.json", {
            "state": "oof", "variants_completed": len(variants), "variants_total": 2,
            "elapsed_seconds": round(time.monotonic() - started, 3)})
    chosen = choose_variant(variants)
    common = {"schema_version": 1, "state": "complete", "recipe": recipe,
              "original_report_sha256": file_sha256(args.original_report),
              "cache_sha256": cache_hash, "variant_results": variants,
              "holdout_used": False, "gpu_allocated": False,
              "validation_used_for_selection": False,
              "role": "binary_proposer_not_normative_verdict"}
    if chosen is None:
        report = {**common, "outcome": "rejected", "selected": False,
                  "rejection_reason": "neither fixed variant has a feasible OOF threshold",
                  "chosen_variant": None, "validation_rows": [],
                  "validation_metrics": None, "checkpoint_sha256": None,
                  "elapsed_seconds": round(time.monotonic() - started, 3)}
        write_json(args.output_dir / "run-manifest.json", report)
        write_json(args.output_dir / "progress.json", {"state": "complete",
                   "outcome": "rejected", "elapsed_seconds": report["elapsed_seconds"]})
        return report
    head, mean, std, full_history = fit_linear(train_items, chosen["variant"], 10, torch)
    validation = materialize([item for item in items if item["partition"] == "validation"])
    validation_rows = score(head, mean, std, validation, chosen["variant"], torch)
    for row in validation_rows:
        row["candidate"] = row["score"] >= chosen["threshold"]
    validation_metrics = metrics_at(validation_rows, chosen["threshold"])
    selected = strict_validation_gate(validation_rows, REQUIRED_POSITIVE_IDS,
                                      REQUIRED_NEGATIVE_IDS, ADDITIONAL_POSITIVE_IDS)
    checkpoint_hash = None
    if selected:
        checkpoint = args.output_dir / "refined-onset-probe.safetensors"
        state = {f"head.{key}": value.detach() for key, value in head.state_dict().items()}
        state.update({"feature_mean": mean, "feature_std": std})
        save_file(state, str(checkpoint), metadata={"variant": chosen["variant"],
                                                    "role": "proposer"})
        checkpoint_hash = file_sha256(checkpoint)
    elapsed = time.monotonic() - started
    if elapsed > HARD_WALL_SECONDS:
        raise RefinementError("hard ten-minute CPU budget exceeded")
    report = {**common, "outcome": "selected" if selected else "rejected",
              "selected": selected, "chosen_variant": chosen["variant"],
              "chosen_threshold": chosen["threshold"],
              "full_train_initial_loss": full_history[0],
              "full_train_final_loss": full_history[-1],
              "validation_rows": validation_rows, "validation_metrics": validation_metrics,
              "checkpoint_sha256": checkpoint_hash, "elapsed_seconds": round(elapsed, 3),
              "production_improvement_claimed": False}
    write_json(args.output_dir / "run-manifest.json", report)
    write_json(args.output_dir / "progress.json", {"state": "complete",
               "outcome": report["outcome"], "elapsed_seconds": report["elapsed_seconds"]})
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-report", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"outcome": result["outcome"],
                      "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
