"""Inventory organizer videos with ffprobe and cryptographic duplicate grouping."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eval.convert_labels import read_labels  # noqa: E402


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        check=True, capture_output=True, encoding="utf-8",
    )
    return json.loads(result.stdout)


def collect(videos_dir: Path, labels_path: Path) -> list[dict]:
    rows = []
    for label in read_labels(labels_path, videos_dir):
        video = videos_dir / f"{label.video_id}.mov"
        metadata = probe(video)
        streams = metadata["streams"]
        stream = next(s for s in streams if s["codec_type"] == "video")
        rotation = stream.get("tags", {}).get("rotate", 0)
        for side_data in stream.get("side_data_list", []):
            rotation = side_data.get("rotation", rotation)
        rotation = float(rotation)
        sha256 = digest(video)
        duration = float(metadata["format"]["duration"])
        if label.label_sec is not None and not 0 <= label.label_sec < duration:
            raise ValueError(f"Label outside video duration: {label.video_id}")
        rows.append({
            "video_id": label.video_id, "file": f"open/videos/{video.name}",
            "duration_sec": round(duration, 6), "fps": round(float(Fraction(stream["avg_frame_rate"])), 6),
            "width": stream["width"], "height": stream["height"], "codec": stream["codec_name"],
            "rotation": rotation, "size_mb": round(video.stat().st_size / 1_000_000, 6),
            "has_violation": label.has_violation, "label_sec": label.label_sec if label.label_sec is not None else "",
            "sha256": sha256, "source_group": sha256[:16],
            "fps_rational": stream["avg_frame_rate"], "r_frame_rate": stream["r_frame_rate"],
            "time_base": stream["time_base"], "start_time": stream.get("start_time", metadata["format"].get("start_time", "0")),
            "frame_count": stream.get("nb_frames", ""),
            "audio_streams": sum(s.get("codec_type") == "audio" for s in streams),
            "video_streams": sum(s.get("codec_type") == "video" for s in streams),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Folder containing open/videos and open/open_labels.xlsx")
    parser.add_argument("--output", type=Path, default=Path("research/dataset/inventory.csv"))
    parser.add_argument("--summary", type=Path, default=Path("research/dataset/inventory-summary.json"))
    args = parser.parse_args()
    rows = collect(args.dataset_dir / "open/videos", args.dataset_dir / "open/open_labels.xlsx")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["sha256"]].append(row)
    durations = [row["duration_sec"] for row in rows]
    summary = {
        "video_files": len(rows), "unique_video_bytes": len(groups),
        "positive_rows": sum(row["has_violation"] for row in rows),
        "negative_rows": sum(1 - row["has_violation"] for row in rows),
        "duration_sec": {"min": min(durations), "median": statistics.median(durations), "max": max(durations), "sum": round(sum(durations), 6)},
        "fps": {"min": min(row["fps"] for row in rows), "max": max(row["fps"] for row in rows)},
        "dimensions": sorted({f'{row["width"]}x{row["height"]}' for row in rows}),
        "codecs": sorted({row["codec"] for row in rows}),
        "rotations": sorted({row["rotation"] for row in rows}),
        "audio_streams": sum(row["audio_streams"] for row in rows),
        "total_size_mb": round(sum(row["size_mb"] for row in rows), 6),
        "duplicate_groups": [{"sha256": key, "members": [{"video_id": row["video_id"], "has_violation": row["has_violation"], "label_sec": row["label_sec"]} for row in members]} for key, members in groups.items() if len(members) > 1],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
