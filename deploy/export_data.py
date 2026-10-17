#!/usr/bin/env python3
"""Create a portable, hashed Lado result/model archive at an operator-selected path."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIDEO_NAMES = {
    "video.mov",
    "video.mp4",
    "video.m4v",
    "video.avi",
    "video.mkv",
    "video.webm",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export Lado results/models without external services."
    )
    parser.add_argument(
        "destination",
        type=Path,
        help="Existing operator-selected local or mounted directory",
    )
    parser.add_argument("--include-models", action="store_true")
    parser.add_argument(
        "--include-videos",
        action="store_true",
        help="Include sensitive uploaded bank videos in the local archive",
    )
    args = parser.parse_args()
    destination = args.destination.expanduser().resolve()
    if not destination.is_dir():
        parser.error(f"destination must be an existing directory: {destination}")
    data_dir = (ROOT / "data").resolve()
    models_dir = (ROOT / "models").resolve()
    if destination == data_dir or data_dir in destination.parents:
        parser.error("destination must be outside the project data directory")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive = destination / f"lado-export-{stamp}.tar.gz"

    def exclude(path: Path) -> bool:
        return not args.include_videos and path.name.lower() in VIDEO_NAMES

    with tarfile.open(archive, "x:gz") as output:
        for source, prefix in [(data_dir, "data"), (models_dir, "models")]:
            if source == models_dir and not args.include_models:
                continue
            if not source.exists():
                continue
            for path in sorted(source.rglob("*")):
                if path.is_file() and not exclude(path):
                    output.add(
                        path,
                        arcname=str(Path(prefix) / path.relative_to(source)),
                        recursive=False,
                    )
    hasher = hashlib.sha256()
    with archive.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    record = {
        "archive": archive.name,
        "sha256": digest,
        "created_at": datetime.now(UTC).isoformat(),
        "included_models": args.include_models,
        "included_source_videos": args.include_videos,
    }
    sidecar = archive.with_suffix(archive.suffix + ".json")
    sidecar.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Created {archive}")
    print(f"SHA256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
