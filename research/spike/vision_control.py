"""Offline closed-answer visual controls; never emits private generated prose."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")


def closed_answer(verifier, images, question, allowed, *, preencode=False):
    import torch

    content = []
    for index, _ in enumerate(images):
        content.extend([{"type": "text", "text": f"Frame {index}: "}, {"type": "image"}])
    content.append({"type": "text", "text": question})
    prompt = verifier.processor.apply_chat_template([{"role": "user", "content": content}],
        add_generation_prompt=True, tokenize=False)
    inputs = verifier.processor(images=images, text=prompt, return_tensors="pt",
                                crop_to_patches=False).to(verifier.device, dtype=verifier.dtype)
    started = time.perf_counter()
    vision_sec = None
    with torch.inference_mode():
        if preencode:
            cache = getattr(verifier, "_control_features", {})
            parts = []
            for image, patch in zip(images, inputs["pixel_values"]):
                key = hashlib.sha256(image.tobytes()).hexdigest()
                if key not in cache:
                    cache[key] = verifier.model.get_image_features(patch.unsqueeze(0))
                parts.append(cache[key])
            verifier._control_features = cache
            features = torch.cat(parts, dim=0)
            embeddings = verifier.model.get_input_embeddings()(inputs["input_ids"])
            mask = verifier.model.model.get_placeholder_mask(inputs["input_ids"],
                inputs_embeds=embeddings, image_features=features)
            inputs["inputs_embeds"] = embeddings.masked_scatter(mask, features)
            inputs.pop("pixel_values")
            if verifier.device == "cuda":
                torch.cuda.synchronize()
            vision_sec = round(time.perf_counter() - started, 4)
        output = verifier.model.generate(**inputs, do_sample=False, max_new_tokens=24,
            pad_token_id=verifier.processor.tokenizer.eos_token_id)
    generated = output[0, inputs["input_ids"].shape[1]:]
    value = verifier.processor.decode(generated, skip_special_tokens=True).strip().lower()
    value = value.strip("\"'.!。 \r\n")
    # Deliberately exclude all unconstrained text from the returned record.
    return {"answer": value if value in allowed else None,
            "valid_closed_answer": value in allowed, "generated_tokens": len(generated),
            "elapsed_sec": round(time.perf_counter() - started, 4),
            "vision_sec": vision_sec,
            "path": "preencoded" if preencode else "native"}


def main():
    from research.spike.probe import deny_network

    deny_network()
    import numpy as np
    from navidu.ingest import iter_frames
    from navidu.perception.assets import model_entry, read_manifest, verify_model
    from navidu.perception.temporal import TemporalVerifier
    from PIL import Image, ImageDraw

    directory = ROOT / "models"
    entry = model_entry(read_manifest(directory), "internvl3_5-1b")
    model_hash = verify_model(directory, entry)
    started = time.perf_counter()
    verifier = TemporalVerifier(directory, entry, "auto")
    report = {"model_hash": model_hash, "device": verifier.device,
              "load_sec": round(time.perf_counter() - started, 4), "controls": []}
    print(json.dumps({key: value for key, value in report.items() if key != "controls"}), flush=True)
    color_images = []
    for color in ("red", "blue"):
        fixture = Image.new("RGB", (448, 448), "white")
        ImageDraw.Draw(fixture).rectangle((70, 70, 378, 378), fill=color)
        image = np.array(fixture, copy=True)
        color_images.append(image)
        for preencode in (False, True):
            result = closed_answer(verifier, [image],
                "What is the color of the large square? Reply with exactly one word: red or blue.",
                {"red", "blue"}, preencode=preencode)
            result.update({"control": "synthetic_color", "expected": color,
                           "correct": result["answer"] == color})
            report["controls"].append(result)
            print(json.dumps(result), flush=True)
    result = closed_answer(verifier, color_images,
        "Look at the two images in order. Which color is shown in the SECOND image? "
        "Reply with exactly one word: red or blue.", {"red", "blue"}, preencode=True)
    result.update({"control": "synthetic_temporal_order", "expected": "blue",
                   "correct": result["answer"] == "blue"})
    report["controls"].append(result)
    print(json.dumps(result), flush=True)
    video = ROOT / "data/dataset/open/videos/X-1.mov"
    tracks_path = ROOT / "research/spike/artifacts/X-1/rfdetr-nano-onnx-cpu-3fps/tracks.jsonl"
    tracks = [json.loads(line) for line in tracks_path.read_text(encoding="utf-8").splitlines()]
    for frame in iter_frames(video, analysis_fps=1):
        if round(frame.timestamp_sec, 3) not in {0.0, 10.0, 20.0}:
            continue
        result = closed_answer(verifier, [frame.rgb],
            "How many people are visible in this image? Count only visible people. "
            "Reply with exactly one digit from 0 to 9.", set("0123456789"), preencode=True)
        detector_count = sum(track["kind"] == "person" and any(
            abs(sample["t"] - frame.timestamp_sec) <= 0.002 for sample in track["samples"])
            for track in tracks)
        result.update({"control": "private_person_count", "video": video.name,
                       "source_pts": round(frame.timestamp_sec, 3),
                       "detector_tracked_person_count": detector_count})
        report["controls"].append(result)
        print(json.dumps({"control": "private_person_count", "valid_closed_answer": result["valid_closed_answer"],
                          "elapsed_sec": result["elapsed_sec"]}), flush=True)
    target = ROOT / "research/spike/artifacts/semantic/vision-control.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
