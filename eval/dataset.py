"""Exact organizer labels, source grouping and the fixed evaluation partitions."""
from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceRow:
    video_id: str
    relative_file: str
    label_sec: float | None
    type_code: str


@dataclass(frozen=True)
class EvaluationSource:
    sha256: str
    has_violation: int
    fold: int
    duration_sec: float
    rows: tuple[SourceRow, ...]

    @property
    def label_secs(self) -> tuple[float, ...]:
        return tuple(sorted({row.label_sec for row in self.rows if row.label_sec is not None}))

    @property
    def timing_ambiguous(self) -> bool:
        return len(self.label_secs) > 1

    def video_path(self, dataset_dir: Path) -> Path:
        root = dataset_dir.resolve()
        path = (root / self.rows[0].relative_file).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Video path escapes the organizer dataset directory")
        return path


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty input table: {path.name}")
    mapped = {row["video_id"]: row for row in rows}
    if len(mapped) != len(rows):
        raise ValueError(f"Duplicate video_id in {path.name}")
    return mapped


def load_sources(inventory: Path, labels: Path, folds: Path) -> list[EvaluationSource]:
    metadata, annotations, assignments = (_read_csv(path) for path in (inventory, labels, folds))
    if not metadata.keys() == annotations.keys() == assignments.keys():
        raise ValueError("Inventory, labels and folds must cover the same exact video identifiers")
    grouped: dict[str, list[tuple[dict, dict, dict]]] = {}
    for video_id, row in metadata.items():
        label, fold = annotations[video_id], assignments[video_id]
        sha256 = row["sha256"]
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise ValueError("Invalid source SHA256 in inventory")
        if fold["sha256"] != sha256:
            raise ValueError("Fold identity does not match the inventory")
        if not row["has_violation"] == label["has_violation"] == fold["has_violation"]:
            raise ValueError("Binary annotations disagree between evaluation tables")
        grouped.setdefault(sha256, []).append((row, label, fold))
    sources = []
    for sha256, members in grouped.items():
        group_folds = {int(fold["fold"]) for _, _, fold in members}
        verdicts = {int(label["has_violation"]) for _, label, _ in members}
        durations = {float(row["duration_sec"]) for row, _, _ in members}
        if len(group_folds) != 1 or not group_folds.issubset(set(range(5))):
            raise ValueError("A source must belong to exactly one of the five fixed folds")
        if len(verdicts) != 1 or not verdicts.issubset({0, 1}):
            raise ValueError("Byte-identical sources have inconsistent binary labels")
        if len(durations) != 1:
            raise ValueError("Byte-identical sources have inconsistent durations")
        verdict, duration = next(iter(verdicts)), next(iter(durations))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Invalid source duration")
        source_rows = []
        for row, label, _ in members:
            timestamp = float(label["label_sec"]) if label["label_sec"] else None
            if bool(verdict) != (timestamp is not None):
                raise ValueError("Binary annotation and timestamp are inconsistent")
            if timestamp is not None and (not math.isfinite(timestamp) or not 0 <= timestamp < duration):
                raise ValueError("Source timestamp is outside the video")
            source_rows.append(SourceRow(row["video_id"], row["file"], timestamp, label.get("type_code", "")))
        sources.append(EvaluationSource(sha256, verdict, next(iter(group_folds)), duration, tuple(source_rows)))
    return sources


def select_sources(sources: list[EvaluationSource], split: str, video_ids: list[str] | None = None) -> list[EvaluationSource]:
    if split not in {"development", "holdout"}:
        raise ValueError("Choose development or holdout explicitly")
    allowed = [source for source in sources if (source.fold == 0) == (split == "holdout")]
    if not video_ids:
        return allowed
    requested = set(video_ids)
    all_ids = {row.video_id: source for source in sources for row in source.rows}
    if not requested.issubset(all_ids):
        raise ValueError("An explicitly selected video identifier is absent from the organizer tables")
    if any((all_ids[video_id].fold == 0) != (split == "holdout") for video_id in requested):
        raise ValueError("A selected video belongs to a different partition; holdout fold 0 is reserved")
    return [source for source in allowed if any(row.video_id in requested for row in source.rows)]


def dataset_fingerprint(paths: list[Path]) -> str:
    result = hashlib.sha256()
    for path in paths:
        result.update(path.name.encode("utf-8") + b"\0" + path.read_bytes())
    return result.hexdigest()
