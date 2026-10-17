"""Offline development-window probe. Emits only typed local model predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", action="append", nargs=3, metavar=("FILE", "START", "END"), required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=ROOT / "research/spike/artifacts/semantic")
    args = parser.parse_args()
    from research.spike.probe import deny_network

    deny_network()
    from navidu.contracts.domain import Track
    from navidu.ingest import iter_frames
    from navidu.perception.assets import model_entry, read_manifest, verify_model
    from navidu.perception.temporal import PROMPT_VERSION, TemporalVerifier
    from navidu.pipeline.interfaces import PerceptionArtifacts

    source_hash = hashlib.sha256((ROOT / "backend/navidu/perception/temporal.py").read_bytes()).hexdigest()
    model_dir = ROOT / "models"
    entry = model_entry(read_manifest(model_dir), "internvl3_5-1b")
    model_hash = verify_model(model_dir, entry)
    started = time.perf_counter()
    model = TemporalVerifier(model_dir, entry, args.device)
    load_sec = round(time.perf_counter() - started, 3)
    print(json.dumps({"event": "loaded", "device": model.device, "load_sec": load_sec}), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    for window_index, (file, start, end) in enumerate(args.window):
        video = ROOT / "data/dataset/open/videos" / file
        start, end = float(start), float(end)
        tracks_path = ROOT / "research/spike/artifacts" / video.stem / "rfdetr-nano-onnx-cpu-3fps/tracks.jsonl"
        tracks = [Track.model_validate(json.loads(line)) for line in tracks_path.read_text(encoding="utf-8").splitlines()]
        frames = [(round(frame.timestamp_sec, 3), frame.rgb) for frame in iter_frames(video, analysis_fps=model.sample_fps)
                  if start <= frame.timestamp_sec <= end]
        artifacts = PerceptionArtifacts()
        started = time.perf_counter()
        error = None
        try:
            model.observe(frames, tracks, artifacts)
        except RuntimeError as exc:
            error = str(exc)
        result = {"video_file": video.name, "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "window": [start, end], "sample_pts": [timestamp for timestamp, _ in frames],
            "device": model.device, "load_sec": load_sec,
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "model_hash": model_hash, "adapter_code_sha256": source_hash,
            "prompt_version": PROMPT_VERSION, "metrics": model.last_metrics,
            "observations": [item.model_dump() for item in artifacts.observations],
            "signals": [asdict(item) for item in artifacts.semantic_signals], "error": error}
        path = args.output / f"{video.stem}-{start:g}-{end:g}-{PROMPT_VERSION}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"window_index": window_index, "elapsed_sec": result["elapsed_sec"],
            "metrics": model.last_metrics,
            "error_code": error.split(":", 1)[0] if error else None}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
