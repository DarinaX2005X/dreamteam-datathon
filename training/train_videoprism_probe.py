#!/usr/bin/env python3
"""Bounded frozen VideoPrism-B source-level proposer experiment."""

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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "backend"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from training.videoprism_probe import (  # noqa: E402
    FPS,
    binary_metrics,
    calibrate,
    clean_training_starts,
    coverage_starts,
    letterbox_rgb,
    normalized_lme_numpy,
    onset_start,
    select_clip,
)

MODEL_REPO = "google/videoprism-base-f16r288"
MODEL_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
MODEL_FILE = "flax_base_f16r288_repeated.npz"
MODEL_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
CODE_REVISION = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
SEED = 1729
HARD_WALL_SECONDS = 210
ETA_LIMIT_SECONDS = 180
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


class ProbeError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def enforce_modal_l4(jax: Any) -> dict[str, Any]:
    devices = jax.devices("gpu") if platform.system() == "Linux" and os.getenv("MODAL_TASK_ID") else []
    if len(devices) != 1 or "L4" not in str(getattr(devices[0], "device_kind", "")).upper():
        raise ProbeError("experiment requires exactly one authorized Modal L4")
    return {"profile": "modal-l4", "device": str(devices[0]),
            "device_kind": str(devices[0].device_kind)}


def _strict_gate(rows: list[dict[str, Any]]) -> bool:
    expected = REQUIRED_POSITIVE_IDS | ADDITIONAL_POSITIVE_IDS | REQUIRED_NEGATIVE_IDS
    if len(rows) != 8 or {row["sample_id"] for row in rows} != expected \
            or any(not math.isfinite(float(row["score"])) for row in rows):
        return False
    by_id = {row["sample_id"]: row for row in rows}
    predicted = {key for key, row in by_id.items() if row["candidate"]}
    return (REQUIRED_POSITIVE_IDS <= predicted
            and not (REQUIRED_NEGATIVE_IDS & predicted)
            and bool(ADDITIONAL_POSITIVE_IDS & predicted))


def _decode(sample: Any) -> list[tuple[float, Any]]:
    from navidu.ingest import iter_frames
    frames = [(float(frame.timestamp_sec), frame.rgb)
              for frame in iter_frames(sample.video_path, analysis_fps=FPS)]
    if len(frames) < 16:
        raise ProbeError(f"insufficient frames: {sample.sample_id}")
    return frames


def _planned_starts(sample: Any, mode: str) -> list[float]:
    if mode == "coverage":
        return coverage_starts(sample.duration_sec)
    if sample.has_violation:
        if not sample.time_supervision:
            raise ProbeError("positive fit source lacks reviewed onset")
        return [onset_start(sample.duration_sec, sample.time_sec)]
    return clean_training_starts(sample.duration_sec)


def _embed_sources(samples: list[Any], mode: str, forward: Any, state: Any,
                   jnp: Any, counter: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for sample in samples:
        frames = _decode(sample)
        starts = _planned_starts(sample, mode)
        vectors, actual_bounds = [], []
        for start in starts:
            selected = select_clip(frames, start)
            clip = np.stack([letterbox_rgb(rgb) for _, rgb in selected])
            actual_bounds.append((float(selected[0][0]), float(selected[-1][0])))
            outputs = forward(state, jnp.asarray(clip[None]), train=False)
            array = np.asarray(outputs[0] if isinstance(outputs, tuple) else outputs)
            if array.shape != (1, 4096, 768):
                raise ProbeError(f"unexpected VideoPrism output {array.shape}")
            # Official output is 16 time steps x 16 x 16 spatial tokens.
            vectors.append(array.reshape(1, 16, 16, 16, 768).mean(axis=(2, 3)).mean(axis=1)[0])
            counter["clips"] += 1
            if counter["clips"] == 32:
                elapsed = time.monotonic() - counter["started"]
                projected = elapsed / 32 * counter["planned_total"]
                counter["eta"] = {"first_32_seconds": round(elapsed, 3),
                                  "planned_clips": counter["planned_total"],
                                  "projected_all_seconds": round(projected, 3)}
                write_json(counter["progress"], {"state": "embedding", "eta": counter["eta"]})
                if projected > ETA_LIMIT_SECONDS:
                    raise ProbeError("first-32 ETA projects above the 8-minute limit")
            if time.monotonic() - counter["wall_started"] > HARD_WALL_SECONDS:
                raise ProbeError("10-minute hard wall exceeded")
        items.append({"sample_id": sample.sample_id, "truth": bool(sample.has_violation),
                      "starts": starts, "actual_bounds": actual_bounds,
                      "x": np.stack(vectors).astype(np.float32)})
    return items


def _fit(items: list[dict[str, Any]], seed: int) -> tuple[Any, np.ndarray, np.ndarray]:
    from sklearn.linear_model import LogisticRegression
    x = np.concatenate([item["x"] for item in items])
    mean = x.mean(0)
    std = x.std(0).clip(1e-5)
    labels, weights = [], []
    class_counts = {truth: sum(item["truth"] == truth for item in items) for truth in (False, True)}
    for item in items:
        labels.extend([item["truth"]] * len(item["x"]))
        weights.extend([1.0 / (class_counts[item["truth"]] * len(item["x"]))] * len(item["x"]))
    head = LogisticRegression(C=0.01, solver="liblinear", random_state=seed,
                              max_iter=1000, fit_intercept=True)
    head.fit((x - mean) / std, labels, sample_weight=np.asarray(weights) * len(items))
    return head, mean, std


def _score(head: Any, mean: np.ndarray, std: np.ndarray,
           items: list[dict[str, Any]], fold: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        logits = head.decision_function((item["x"] - mean) / std)
        bag_logit = normalized_lme_numpy(logits)
        best = int(np.argmax(logits))
        actual_start, actual_end = item["actual_bounds"][best]
        rows.append({"sample_id": item["sample_id"], "truth": item["truth"],
                     "score": 1 / (1 + math.exp(-bag_logit)),
                     "fold": fold,
                     "requested_clip_start_sec": float(item["starts"][best]),
                     "candidate_clip_start_sec": actual_start,
                     "candidate_clip_end_sec": actual_end})
    return rows


def _save_cache(output_dir: Path, identity: dict[str, Any], phase: str,
                groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    inventory = []
    for group, items in groups.items():
        for index, item in enumerate(items):
            key = f"{group}_{index:02d}"
            arrays[f"{key}_x"] = item["x"]
            arrays[f"{key}_starts"] = np.asarray(item["starts"], dtype=np.float32)
            arrays[f"{key}_actual_bounds"] = np.asarray(item["actual_bounds"], dtype=np.float32)
            inventory.append({"key": key, "sample_id": item["sample_id"],
                              "clips": len(item["starts"])})
    identity_bytes = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    cache_key = hashlib.sha256(identity_bytes).hexdigest()
    path = output_dir / f"videoprism-embeddings-{phase}-{cache_key}.npz"
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)
    manifest = {"cache_key": cache_key, "phase": phase, "identity": identity,
                "inventory": inventory, "file": path.name, "sha256": sha256(path)}
    write_json(path.with_suffix(".json"), manifest)
    return {"key": cache_key, "phase": phase, "sha256": manifest["sha256"],
            "file": path.name, "clips": sum(x["clips"] for x in inventory)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    wall_started = time.monotonic()
    if args.output_dir.exists():
        raise ProbeError("output directory already exists")
    args.output_dir.mkdir(parents=True)
    import jax
    import jax.numpy as jnp
    from navidu.ingest import iter_frames
    from videoprism import models as vp

    from eval.training import load_training_partition

    host = enforce_modal_l4(jax)
    random.seed(SEED)
    np.random.seed(SEED)
    if sha256(args.weights) != MODEL_SHA256:
        raise ProbeError("pinned VideoPrism checkpoint hash mismatch")
    common = {"manifest_path": args.training_manifest, "inventory": args.inventory,
              "labels": args.labels, "folds": args.folds, "dataset_dir": args.dataset_dir,
              "verify_hashes": True}
    split = json.loads(args.training_manifest.read_text(encoding="utf-8"))
    decoder_source = inspect.getsourcefile(iter_frames)
    if decoder_source is None:
        raise ProbeError("cannot identify decoder source")
    train = load_training_partition(**common, partition="train")
    validation = load_training_partition(**common, partition="validation")
    if split.get("reserved_holdout_examples_included") is not False:
        raise ProbeError("training manifest must explicitly exclude reserved holdout")
    if (len(train), sum(x.has_violation for x in train), len(validation),
            sum(x.has_violation for x in validation)) != (24, 12, 8, 4):
        raise ProbeError("fixed split mismatch")
    fold_by_id = {row["sample_id"]: int(row["fold"]) for row in split["samples"]}
    identity = {"schema_version": 1, "dataset_fingerprint": split["dataset_fingerprint"],
                "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
                "model_file": MODEL_FILE, "model_sha256": MODEL_SHA256,
                "official_code_revision": CODE_REVISION, "recipe": {
                    "frames": 16, "resolution": 288, "fps": FPS, "stride_sec": 4,
                    "adaptation": "aspect-preserving RGB letterbox with zero fill",
                    "pool": "spatial-per-time mean then temporal mean",
                    "fit_positive": "single reviewed-onset-centered clip",
                    "fit_clean": "six fixed clips", "held": "label-blind full coverage"},
                "helper_sha256": sha256(ROOT / "training/videoprism_probe.py"),
                "trainer_sha256": sha256(Path(__file__)),
                "decoder_source_sha256": sha256(Path(decoder_source)),
                "versions": {name: importlib.metadata.version(name) for name in
                             ("jax", "flax", "numpy", "scikit-learn")},
                "fixed_sources": [{"sample_id": x.sample_id, "source_sha256": x.source_sha256,
                                   "duration_sec": x.duration_sec} for x in train + validation],
                "holdout_used": False}
    model = vp.get_model("videoprism_public_v1_base")
    state = vp.load_pretrained_weights("videoprism_public_v1_base", str(args.weights))
    forward = jax.jit(model.apply, static_argnames=("train",))
    planned = sum(len(_planned_starts(x, "coverage")) + len(_planned_starts(x, "fit")) for x in train)
    planned += sum(len(_planned_starts(x, "coverage")) for x in validation)
    counter = {"clips": 0, "started": time.monotonic(), "wall_started": wall_started,
               "planned_total": planned, "progress": args.output_dir / "progress.json", "eta": None}
    # Fit clips and held clips are separate by construction; held schedules never consult labels.
    fit_items = _embed_sources(train, "fit", forward, state, jnp, counter)
    held_items = _embed_sources(train, "coverage", forward, state, jnp, counter)
    cache = _save_cache(args.output_dir, identity, "train", {"fit": fit_items,
                                                              "coverage": held_items})
    oof = []
    fold_diagnostics = []
    for fold in (1, 2, 3):
        fit = [x for x in fit_items if fold_by_id[x["sample_id"]] != fold]
        held = [x for x in held_items if fold_by_id[x["sample_id"]] == fold]
        if len(fit) != 16 or len(held) != 8:
            raise ProbeError("source-grouped fold mismatch")
        head, mean, std = _fit(fit, SEED + fold)
        fold_rows = _score(head, mean, std, held, fold=fold)
        oof.extend(fold_rows)
        fold_diagnostics.append({"fold": fold, "fit_sources": len(fit),
                                 "held_sources": len(held),
                                 "held_metrics_at_0_5": binary_metrics(
                                     [x["truth"] for x in fold_rows],
                                     [x["score"] >= 0.5 for x in fold_rows])})
    try:
        threshold, oof_metrics = calibrate(oof)
    except ValueError as error:
        diagnostic = binary_metrics([x["truth"] for x in oof],
                                    [x["score"] >= 0.5 for x in oof])
        report = {"state": "complete", "outcome": "rejected", "reason": str(error),
                  "identity": identity, "host": host, "oof_rows": oof,
                  "oof_threshold": None, "oof_metrics": diagnostic,
                  "oof_diagnostic_threshold": 0.5, "fold_diagnostics": fold_diagnostics,
                  "validation_rows": [], "validation_metrics": None,
                  "selected": False, "cache": cache, "clips_encoded": counter["clips"],
                  "holdout_used": False, "production_improvement_claimed": False,
                  "role": "source-level review proposer", "eta": counter["eta"],
                  "elapsed_seconds": round(time.monotonic() - wall_started, 3)}
        write_json(args.output_dir / "run-manifest.json", report)
        return report
    # Validation is decoded only after the predeclared OOF guard passes.
    val_items = _embed_sources(validation, "coverage", forward, state, jnp, counter)
    cache = _save_cache(args.output_dir, identity, "fixed32", {
        "fit": fit_items, "coverage": held_items, "validation": val_items})
    head, mean, std = _fit(fit_items, SEED + 10)
    rows = _score(head, mean, std, val_items)
    for row in rows:
        row["candidate"] = row["score"] >= threshold
    metrics = binary_metrics([x["truth"] for x in rows], [x["candidate"] for x in rows])
    selected = _strict_gate(rows)
    checkpoint_sha256 = None
    if selected:
        checkpoint = args.output_dir / "videoprism-linear-proposer.npz"
        temporary = checkpoint.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, coef=head.coef_, intercept=head.intercept_,
                            feature_mean=mean, feature_std=std,
                            threshold=np.asarray([threshold], dtype=np.float64))
        os.replace(temporary, checkpoint)
        checkpoint_sha256 = sha256(checkpoint)
    by_id = {row["sample_id"]: row for row in rows}
    report = {"state": "complete", "outcome": "selected" if selected else "rejected",
              "identity": identity, "host": host, "oof_rows": oof,
              "oof_threshold": threshold, "oof_metrics": oof_metrics,
              "fold_diagnostics": fold_diagnostics,
              "validation_rows": rows, "validation_metrics": metrics,
              "strict_gate": {
                  "x15_x18": all(by_id[x]["candidate"] for x in REQUIRED_POSITIVE_IDS),
                  "all_four_clean": all(not by_id[x]["candidate"] for x in REQUIRED_NEGATIVE_IDS),
                  "one_of_x13_x46": any(by_id[x]["candidate"] for x in ADDITIONAL_POSITIVE_IDS)},
              "selected": selected, "holdout_used": False,
              "checkpoint_sha256": checkpoint_sha256, "cache": cache,
              "clips_encoded": counter["clips"],
              "production_improvement_claimed": False,
              "role": "source-level review proposer; candidate clips are not confirmed onsets",
              "eta": counter["eta"], "elapsed_seconds": round(time.monotonic() - wall_started, 3)}
    write_json(args.output_dir / "run-manifest.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "dataset-dir", "output-dir", "training-manifest",
                 "inventory", "labels", "folds"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args())), flush=True)
