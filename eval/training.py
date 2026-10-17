"""Duplicate-safe organizer supervision; training readers cannot select holdout.

Only frames and label-blind temporal coordinates may become model inputs. Paths,
sample IDs, fold IDs and targets are metadata/loss inputs, never prompt features.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.dataset import EvaluationSource, dataset_fingerprint, file_sha256, load_sources  # noqa: E402

TRAIN_FOLDS = frozenset({1, 2, 3})
VALIDATION_FOLDS = frozenset({4})


@dataclass(frozen=True)
class TrainingSample:
    sample_id: str
    source_sha256: str
    video_path: Path
    duration_sec: float
    has_violation: bool
    time_sec: float | None
    time_supervision: bool


def targets_for(source: EvaluationSource) -> dict:
    time_supervision = bool(source.has_violation and not source.timing_ambiguous and len(source.label_secs) == 1)
    return {"has_violation": bool(source.has_violation),
            "time_sec": source.label_secs[0] if time_supervision else None,
            "time_supervision": time_supervision}


def _counts(sources: list[EvaluationSource]) -> dict:
    return {"unique_sources": len(sources), "source_rows": sum(len(source.rows) for source in sources),
            "positive_sources": sum(source.has_violation for source in sources),
            "clean_sources": sum(1 - source.has_violation for source in sources),
            "timestamp_supervised_sources": sum(targets_for(source)["time_supervision"] for source in sources),
            "ambiguous_timing_sources": sum(source.timing_ambiguous for source in sources)}


def build_training_manifest(sources: list[EvaluationSource], table_hash: str) -> dict:
    if len({source.sha256 for source in sources}) != len(sources):
        raise ValueError("Training source identities must be unique SHA256 groups")
    training = [source for source in sources if source.fold in TRAIN_FOLDS]
    validation = [source for source in sources if source.fold in VALIDATION_FOLDS]
    holdout = [source for source in sources if source.fold == 0]
    development = training + validation
    if len(development) + len(holdout) != len(sources):
        raise ValueError("Unknown source fold")
    return {
        "schema_version": 1, "dataset_fingerprint": table_hash,
        "split_rule": {"group_key": "sha256", "train_folds": sorted(TRAIN_FOLDS),
                       "validation_folds": sorted(VALIDATION_FOLDS), "reserved_final_holdout_folds": [0]},
        "counts": {"train": _counts(training), "validation": _counts(validation),
                   "development": _counts(development), "reserved_final_holdout": _counts(holdout)},
        "target_policy": {
            "source": "Organizer binary and timestamp annotations only; no inferred action or role labels.",
            "ambiguous_timing": "Preserve the positive binary target but mask localization when byte-identical rows disagree on timestamp; these are not two confirmed events.",
            "negative_timing": "Negative sources have no event timestamp and no localization loss.",
            "model_inputs": "Video frames and label-blind temporal coordinates only. Never include source path, filename, sample ID, SHA, fold, label, or target-derived crop location in an inference prompt.",
            "validation_sampling": "Validation and final inference frame/window selection must not use labels or organizer timestamps.",
            "refit": "The development partition may be used for a final 32-source refit only after hyperparameters are selected using train/validation. Freeze the resulting model before the reserved final holdout.",
        },
        "reserved_holdout_examples_included": False,
        "samples": [{
            "sample_id": source.sha256, "source_sha256": source.sha256,
            "partition": "train" if source.fold in TRAIN_FOLDS else "validation",
            "fold": source.fold, "video_relative_file": source.rows[0].relative_file,
            "duration_sec": source.duration_sec, "source_row_count": len(source.rows),
            "annotation_time_secs": list(source.label_secs), "timing_ambiguous": source.timing_ambiguous,
            "targets": targets_for(source),
        } for source in sorted(development, key=lambda source: source.sha256)],
    }


def load_training_partition(
    manifest_path: Path, *, inventory: Path, labels: Path, folds: Path,
    dataset_dir: Path, partition: str, verify_hashes: bool = True,
) -> list[TrainingSample]:
    """Return only the named development partition after label/group validation."""
    if partition not in {"train", "validation", "development"}:
        raise ValueError("Training readers may select train, validation, or final development refit; never holdout")
    tables = [inventory.resolve(), labels.resolve(), folds.resolve()]
    sources = load_sources(*tables)
    expected = build_training_manifest(sources, dataset_fingerprint(tables))
    actual = json.loads(manifest_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError("Training manifest differs from the fixed organizer tables and grouping policy")
    allowed_folds = TRAIN_FOLDS if partition == "train" else VALIDATION_FOLDS if partition == "validation" else TRAIN_FOLDS | VALIDATION_FOLDS
    result = []
    for source in sorted(sources, key=lambda source: source.sha256):
        if source.fold not in allowed_folds:
            continue
        path = source.video_path(dataset_dir)
        if not path.is_file():
            raise ValueError("A selected development source is unavailable")
        if verify_hashes and file_sha256(path) != source.sha256:
            raise ValueError("A selected development source differs from its organizer SHA256")
        target = targets_for(source)
        result.append(TrainingSample(source.sha256, source.sha256, path, source.duration_sec,
                                     target["has_violation"], target["time_sec"], target["time_supervision"]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=ROOT / "research/dataset/inventory.csv")
    parser.add_argument("--labels", type=Path, default=ROOT / "eval/labels.csv")
    parser.add_argument("--folds", type=Path, default=ROOT / "eval/folds.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "eval/training-splits.json")
    args = parser.parse_args()
    tables = [args.inventory.resolve(), args.labels.resolve(), args.folds.resolve()]
    result = build_training_manifest(load_sources(*tables), dataset_fingerprint(tables))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "training_splits_prepared", "counts": result["counts"],
                      "dataset_fingerprint": result["dataset_fingerprint"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
