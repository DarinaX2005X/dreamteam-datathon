"""Produce deterministic source-grouped folds and retain organizer annotations.

Splits group byte-identical files together. This protects against known exact
duplicates; it does not claim independent sites, actors or camera viewpoints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def group_sources(rows: list[dict[str, str]]) -> list[dict]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        sha256 = row["sha256"]
        if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
            raise ValueError(f"Invalid source hash for {row['video_id']}")
        grouped[sha256].append(row)
    result = []
    for sha256, members in grouped.items():
        verdicts = {int(member["has_violation"]) for member in members}
        if len(verdicts) != 1 or not verdicts.issubset({0, 1}):
            raise ValueError(f"Contradictory binary labels for source {sha256}")
        events = sorted({float(row["label_sec"]) for row in members if row["label_sec"] != ""})
        result.append({
            "source_group": sha256[:16], "sha256": sha256,
            "video_ids": [row["video_id"] for row in members],
            "has_violation": verdicts.pop(), "label_secs": events,
            "timing_ambiguous": len(events) > 1,
            "label_provenance": [{"video_id": row["video_id"], "label_sec": float(row["label_sec"]) if row["label_sec"] else None} for row in members],
            "annotation_note": "Inconsistent timestamps on the same video bytes; preserve source annotations, exclude from headline timing, and do not treat them as two confirmed events." if len(events) > 1 else "",
        })
    return result


def assign_folds(groups: list[dict], n_folds: int = 5, seed: str = "datathon26-2026-10-17") -> dict[str, int]:
    if n_folds < 2 or n_folds > len(groups):
        raise ValueError("Fold count must be between 2 and source group count")
    assignments: dict[str, int] = {}
    class_counts = [[0, 0] for _ in range(n_folds)]
    totals = [0] * n_folds
    for verdict in (1, 0):
        selected = sorted(
            (group for group in groups if group["has_violation"] == verdict),
            key=lambda group: hashlib.sha256(f"{seed}|{group['sha256']}".encode()).hexdigest(),
        )
        for group in selected:
            fold = min(range(n_folds), key=lambda index: (class_counts[index][verdict], totals[index], index))
            assignments[group["sha256"]] = fold
            class_counts[fold][verdict] += 1
            totals[fold] += 1
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=Path("research/dataset/inventory.csv"))
    parser.add_argument("--output", type=Path, default=Path("eval/folds.csv"))
    parser.add_argument("--sources", type=Path, default=Path("eval/source-labels.json"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", default="datathon26-2026-10-17")
    args = parser.parse_args()
    with args.inventory.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = group_sources(rows)
    assignment = assign_folds(groups, args.folds, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video_id", "source_group", "sha256", "fold", "has_violation"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**{name: row[name] for name in ("video_id", "source_group", "sha256", "has_violation")}, "fold": assignment[row["sha256"]]})
    for group in groups:
        group["fold"] = assignment[group["sha256"]]
    args.sources.parent.mkdir(parents=True, exist_ok=True)
    args.sources.write_text(json.dumps({"schema_version": 1, "seed": args.seed, "n_folds": args.folds, "sources": groups}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = [sum(1 for group in groups if assignment[group["sha256"]] == index) for index in range(args.folds)]
    print(f"Assigned {len(rows)} files / {len(groups)} sources to folds: {counts}")


if __name__ == "__main__":
    main()
