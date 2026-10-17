"""CPU-only native Qwen video processor check using public metadata and blank frames."""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main():
    import numpy as np
    import torch
    from navidu.perception.qwen import prepare_qwen_inputs
    from transformers import AutoProcessor

    entry = next(item for item in json.loads((ROOT / "models/manifest.json").read_text())["models"]
                 if item["id"] == "qwen3-vl-4b")
    cache = ROOT / "training/runs/public-qwen-processor"

    def fetch(item):
        path = cache / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with urllib.request.urlopen(item["url"], timeout=90) as response:
                path.write_bytes(response.read())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(fetch, [item for item in entry["files"] if item["size_bytes"] < 20_000_000]))
    processor = AutoProcessor.from_pretrained(cache / entry["local_dir"], local_files_only=True)
    frames = [(10.0 + index, np.zeros((360, 640, 3), dtype=np.uint8)) for index in range(64)]
    inputs, _, prompt = prepare_qwen_inputs(processor, frames, [{}] * len(frames),
        variant="v6-evidence", max_pixels=262144, mode="video")
    assert "pixel_values_videos" in inputs and "pixel_values" not in inputs
    assert "0=0.000s" in prompt and "63=63.000s" in prompt
    assert inputs["video_grid_thw"][0, 0] == 32
    count = inputs["input_ids"].shape[1]
    assert 4000 < count < 11000
    decoded = processor.decode(inputs["input_ids"][0], skip_special_tokens=False)
    assert "<0.5 seconds>" in decoded and "<62.5 seconds>" in decoded
    inputs = inputs.to("cpu", dtype=torch.bfloat16)
    assert inputs["input_ids"].dtype == inputs["video_grid_thw"].dtype == torch.int64
    print(json.dumps({"native_video_frames": 64, "input_tokens": count,
                      "grid": inputs["video_grid_thw"].tolist(), "timestamps_verified": True,
                      "gpu_used": False, "pretrained_model_loaded": False}))


if __name__ == "__main__":
    main()
