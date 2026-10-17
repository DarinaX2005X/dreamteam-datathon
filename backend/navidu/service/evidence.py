"""Locate extracted media without accepting paths from request or result payloads."""

import re
from pathlib import Path


def artifact_frame_path(directory: Path, frame_id: str, overlay: bool, *, fallback=True):
    if not re.fullmatch(r"fr-[0-9]+", frame_id):
        return None
    root = directory.resolve()
    kind = "overlay" if overlay else "original"
    candidates = [
        root / "frames" / kind / f"{frame_id}.jpg",
        root / "evidence" / kind / f"{frame_id}.jpg",
        root / "frames" / (frame_id + (".overlay.jpg" if overlay else ".jpg")),
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_relative_to(root) and not candidate.is_symlink() and resolved.is_file():
            return resolved
    if overlay and fallback:
        return artifact_frame_path(root, frame_id, False)
    return None
