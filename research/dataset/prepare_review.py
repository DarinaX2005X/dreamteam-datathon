"""Prepare local review windows and frame candidates, never invented annotations."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=Path("research/dataset/inventory.csv"))
    parser.add_argument("--sources", type=Path, default=Path("eval/source-labels.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("research/dataset"))
    args = parser.parse_args()
    with args.inventory.open(encoding="utf-8", newline="") as stream:
        inventory = {row["video_id"]: row for row in csv.DictReader(stream)}
    source_labels = json.loads(args.sources.read_text(encoding="utf-8"))
    windows = []
    candidates = []
    for source in source_labels["sources"]:
        video_id = source["video_ids"][0]
        row = inventory[video_id]
        duration = float(row["duration_sec"])
        end = max(0, duration - 0.1)
        requested = []
        for label in source["label_secs"]:
            for offset in (0, -1, 1, -3, 3):
                candidate = round(min(end, max(0, label + offset)), 3)
                if candidate not in requested:
                    requested.append(candidate)
        for index in range(12):
            candidate = round(end * index / 11, 3)
            if candidate not in requested:
                requested.append(candidate)
        requested = sorted(requested[:12])
        events = [{"label_sec": label, "start_sec": round(max(0, label - 5), 3), "end_sec": round(min(duration, label + 5), 3)} for label in source["label_secs"]]
        windows.append({
            "video_id": video_id, "all_video_ids": source["video_ids"], "file": row["file"],
            "source_sha256": source["sha256"], "has_violation": source["has_violation"],
            "duration_sec": duration, "label_windows": events,
            "suggested_frame_times_sec": requested, "fold": source["fold"],
            "observation_status": "awaiting_local_model_or_human_review",
            "sampling_note": "Review candidates only. Sampled images cannot establish absence of an event between samples.",
        })
        for index, requested_sec in enumerate(requested):
            candidates.append({
                "candidate_id": f"{source['source_group']}-{index:02d}", "video_id": video_id,
                "source_sha256": source["sha256"], "requested_sec": requested_sec,
                "fold": source["fold"], "annotation_status": "unannotated",
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "annotations").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "review-windows.json").write_text(json.dumps({"schema_version": 1, "source_count": len(windows), "windows": windows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "annotations/frame-candidates.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    print(f"Prepared {len(windows)} source review windows and {len(candidates)} unannotated frame candidates. No frames or observations fabricated.")


if __name__ == "__main__":
    main()
