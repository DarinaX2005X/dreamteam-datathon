"""Reconstruct private training inputs on CPU without model inference or image output to logs."""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))


def main() -> None:
    import numpy as np
    from navidu.ingest import iter_frames
    from navidu.perception.proposal_task import encode_proposal, proposal_messages
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    from transformers import AutoProcessor

    from eval.training import load_training_partition
    from training.data import decode_window, make_training_windows, supervised_target

    output = ROOT / "training/runs/input-examples-20261017"
    output.mkdir(parents=True, exist_ok=True)
    entry = next(item for item in json.loads((ROOT / "models/manifest.json").read_text())["models"]
                 if item["id"] == "internvl3_5-4b")
    cache = output / "public-processor"

    def fetch(item):
        path = cache / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with urllib.request.urlopen(item["url"], timeout=90) as response:
                path.write_bytes(response.read())
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError("Public processor hash mismatch")

    # Download public processor metadata only, never model weights or private inputs.
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(fetch, [item for item in entry["files"] if item["size_bytes"] < 20_000_000]))
    processor = AutoProcessor.from_pretrained(cache / entry["local_dir"], local_files_only=True)
    samples = load_training_partition(ROOT / "eval/training-splits.json",
        inventory=ROOT / "research/dataset/inventory.csv", labels=ROOT / "eval/labels.csv",
        folds=ROOT / "eval/folds.csv", dataset_dir=ROOT / "data/dataset", partition="train")
    windows = make_training_windows(samples)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    counts = {}
    for label, positive in (("positive", True), ("clean", False)):
        spec = next(item for item in windows if item.sample.has_violation == positive)
        frames = decode_window(spec)
        target = supervised_target(spec, frames)
        if target is None:
            raise RuntimeError("Selected window has no target")
        inputs = encode_proposal(processor, frames, target=target)
        pixels = inputs["pixel_values"].float().numpy()
        mean = np.asarray(processor.image_processor.image_mean).reshape(1, 3, 1, 1)
        std = np.asarray(processor.image_processor.image_std).reshape(1, 3, 1, 1)
        visible = np.clip(np.rint((pixels * std + mean) * 255), 0, 255).astype(np.uint8)
        assert visible.shape == (16, 3, 448, 448)
        destination = output / label
        destination.mkdir(exist_ok=True)
        sheet = Image.new("RGB", (4 * 448, 4 * 478), "#141821")
        draw = ImageDraw.Draw(sheet)
        for index, (timestamp, _) in enumerate(frames):
            image = Image.fromarray(visible[index].transpose(1, 2, 0))
            image.save(destination / f"frame-{index:02d}.png")
            x, y = (index % 4) * 448, (index // 4) * 478
            sheet.paste(image, (x, y))
            draw.text((x + 8, y + 452), f"Frame {index} | +{timestamp - frames[0][0]:.3f} s", font=font, fill="white")
        sheet.save(output / f"{label}-16-frames.png")
        messages = proposal_messages(frames)
        readable = "\n".join(item.get("text", "[IMAGE: next frame]") for item in messages[0]["content"])
        (output / f"{label}-prompt.txt").write_text(readable, encoding="utf-8")
        (destination / "chat-template.txt").write_text(processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True), encoding="utf-8")
        (destination / "training-target.json").write_text(json.dumps(target) + "\n", encoding="utf-8")
        (destination / "provenance.json").write_text(json.dumps({
            "source_sha256": spec.sample.source_sha256, "window_ordinal": spec.ordinal,
            "requested_start_sec": spec.start_sec, "source_timestamps": [item[0] for item in frames],
            "pixel_tensor_shape": list(pixels.shape), "model_revision": entry["revision"],
            "note": "Actual deterministic training crop and native processor pixels reconstructed on CPU; captions are outside model pixels."
        }, indent=2) + "\n", encoding="utf-8")
        full = list(iter_frames(spec.sample.video_path, analysis_fps=2))
        indices = np.linspace(0, len(full) - 1, min(32, len(full)), dtype=int)
        overview = Image.new("RGB", (4 * 640, ((len(indices) + 3) // 4) * 390), "#141821")
        overview_draw = ImageDraw.Draw(overview)
        for position, index in enumerate(indices):
            frame = full[index]
            image = ImageOps.contain(Image.fromarray(frame.rgb), (640, 360))
            x, y = (position % 4) * 640, (position // 4) * 390
            overview.paste(image, (x + (640 - image.width) // 2, y + (360 - image.height) // 2))
            overview_draw.text((x + 8, y + 363), f"Source time {frame.timestamp_sec:.3f} s", font=font, fill="white")
        overview.save(output / f"{label}-full-context.png")
        (destination / "original-file.txt").write_text(str(spec.sample.video_path.resolve()), encoding="utf-8")
        counts[label] = {"frames": len(frames), "input_tokens": inputs["input_ids"].shape[1]}
    print(json.dumps({"examples": counts, "gpu_used": False, "output": str(output)}))


if __name__ == "__main__":
    main()
