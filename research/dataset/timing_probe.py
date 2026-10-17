"""Read decoded frame PTS locally; no frames or descriptions leave the machine."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def inspect(path: Path) -> dict:
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "frame=best_effort_timestamp_time", "-of", "json", str(path),
    ], check=True, capture_output=True, encoding="utf-8")
    times = [float(frame["best_effort_timestamp_time"]) for frame in json.loads(result.stdout)["frames"] if "best_effort_timestamp_time" in frame]
    differences = [round(end - start, 6) for start, end in zip(times, times[1:])]
    if not differences:
        raise ValueError(f"At least two decoded frames required: {path}")
    return {
        "video_id": path.stem, "decoded_frames": len(times),
        "first_pts_sec": times[0], "last_pts_sec": times[-1],
        "min_delta_sec": min(differences), "max_delta_sec": max(differences),
        "gaps_over_0_04_sec": sum(delta > 0.04 for delta in differences),
        "strictly_increasing": all(delta > 0 for delta in differences),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, default=Path("data/dataset/open/videos"))
    parser.add_argument("--video-ids", nargs="+", default=["X-1", "Х-3", "Х-14", "Х-28"])
    parser.add_argument("--output", type=Path, default=Path("research/dataset/timing-check.json"))
    args = parser.parse_args()
    records = [inspect(args.videos_dir / f"{video_id}.mov") for video_id in args.video_ids]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"method": "ffprobe decoded best_effort_timestamp_time", "records": records}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Decoded PTS of {sum(row['decoded_frames'] for row in records)} frames in {len(records)} videos. Pixel content was not exported.")


if __name__ == "__main__":
    main()
