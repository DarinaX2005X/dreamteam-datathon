"""Short local closed-label hypotheses on bounded development video windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/spike/.venv-quant-deps"))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

QUESTIONS = {
    "banknotes_visible": "Are recognizable banknotes visible in these images? Receipts and plain paper do not count.",
    "counting_motion": "Does a person visibly count banknotes by repeatedly separating notes or feeding a banknote counting machine? Simple hand movement does not count.",
    "cash_handover": "Is there a visible transfer of banknotes from one person to another person across these ordered frames?",
    "cash_on_desk": "Are recognizable banknotes lying on the counter or desk?",
    "phone_visible": "Is a mobile phone clearly visible? Do not mistake paper or banknotes for a phone.",
    "phone_use": "Is someone visibly holding or interacting with a mobile phone?",
    "bag_opened": "Does someone visibly open a bag across these ordered frames? An already-open bag is not an opening action.",
    "bag_integrity_damage": "Is a bag visibly torn, cut or broken? An open zipper is not damage.",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", action="append", nargs=3, metavar=("FILE", "START", "END"), required=True)
    parser.add_argument("--model-id", default="internvl3_5-1b")
    parser.add_argument("--quantization", choices=["none", "nf4", "int8"], default="none")
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    from research.spike.probe import deny_network

    deny_network()
    import numpy as np
    import torch
    from navidu.contracts.domain import Track
    from navidu.ingest import iter_frames
    from navidu.perception.assets import model_entry, read_manifest, verify_model
    from navidu.perception.temporal import TemporalVerifier, _samples_at
    from PIL import Image, ImageDraw, ImageFont

    from research.spike.vision_control import closed_answer

    torch.set_num_threads(6)
    directory = ROOT / "models"
    entry = model_entry(read_manifest(directory), args.model_id)
    model_hash = verify_model(directory, entry)
    started = time.perf_counter()
    if args.quantization == "none":
        verifier = TemporalVerifier(directory, entry, "auto")
    else:
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            InternVLForConditionalGeneration,
        )

        quant = BitsAndBytesConfig(load_in_4bit=args.quantization == "nf4",
            load_in_8bit=args.quantization == "int8", bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "lm_head"])
        path = str(directory / entry["local_dir"])
        model = InternVLForConditionalGeneration.from_pretrained(path, local_files_only=True,
            trust_remote_code=False, dtype=torch.float16, attn_implementation="sdpa",
            device_map={"": "cuda:0"}, quantization_config=quant).eval()
        verifier = SimpleNamespace(model=model, processor=AutoProcessor.from_pretrained(path,
            local_files_only=True, trust_remote_code=False), device="cuda", dtype=torch.float16)
    report = {"model_id": args.model_id, "model_hash": model_hash, "quantization": args.quantization,
        "device": verifier.device, "load_sec": round(time.perf_counter() - started, 4),
        "model_memory_mib": round(verifier.model.get_memory_footprint() / 1048576, 1), "windows": []}
    print(json.dumps({key: value for key, value in report.items() if key != "windows"}), flush=True)
    fixture = np.full((448, 448, 3), 255, dtype=np.uint8)
    fixture[70:378, 70:378] = [255, 0, 0]
    report["control"] = closed_answer(verifier, [fixture],
        "What is the color of the large square? Reply with exactly one word: red or blue.",
        {"red", "blue"}, preencode=True)
    print(json.dumps({"control": report["control"]}), flush=True)
    for file, start, end in args.window:
        video = ROOT / "data/dataset/open/videos" / file
        start, end = float(start), float(end)
        tracks_path = ROOT / "research/spike/artifacts" / video.stem / "rfdetr-nano-onnx-cpu-3fps/tracks.jsonl"
        tracks = [Track.model_validate(json.loads(line)) for line in tracks_path.read_text(encoding="utf-8").splitlines()]
        frames = [frame for frame in iter_frames(video, analysis_fps=1)
                  if start <= frame.timestamp_sec <= end][::args.stride]
        images = []
        known_tracks = set()
        for frame in frames:
            image = Image.fromarray(frame.rgb)
            draw = ImageDraw.Draw(image)
            people = _samples_at(tracks, round(frame.timestamp_sec, 3))
            known_tracks.update(people)
            for track_id, box in people.items():
                x1, y1, x2, y2 = [round(value * size) for value, size in
                    zip(box, (image.width, image.height, image.width, image.height))]
                draw.rectangle((x1, y1, x2, y2), outline=(255, 225, 55), width=2)
                draw.text((x1 + 3, y1 + 3), track_id,
                    font=ImageFont.load_default(size=max(24, round(image.height / 28))),
                    fill=(255, 225, 55), stroke_width=1, stroke_fill=(0, 0, 0))
            images.append(np.array(image, copy=True))
        result = {"video": file, "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "sample_pts": [round(frame.timestamp_sec, 3) for frame in frames],
            "known_tracks": sorted(known_tracks), "answers": {}}
        for question_index, (key, question) in enumerate(QUESTIONS.items()):
            answer = closed_answer(verifier, images,
                question + " Reply with exactly one word: yes, no, or unclear.",
                {"yes", "no", "unclear"}, preencode=True)
            result["answers"][key] = answer
            print(json.dumps({"question_index": question_index,
                "valid_closed_answer": answer["valid_closed_answer"],
                "elapsed_sec": answer["elapsed_sec"], "vision_sec": answer["vision_sec"]}), flush=True)
        for track in sorted(known_tracks):
            answer = closed_answer(verifier, images,
                f"The yellow box marked {track} identifies one person. What role does this person visibly have? "
                "Reply with exactly one word: cashier, client, or unknown.",
                {"cashier", "client", "unknown"}, preencode=True)
            result["answers"][f"role:{track}"] = answer
        report["windows"].append(result)
    target = ROOT / f"research/spike/artifacts/semantic/action-control-{args.model_id}-{args.quantization}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    answers = [answer for window in report["windows"] for answer in window["answers"].values()]
    print(json.dumps({"completed_windows": len(report["windows"]),
        "valid_answers": sum(answer["valid_closed_answer"] for answer in answers),
        "total_answers": len(answers), "query_elapsed_sec": sum(answer["elapsed_sec"] for answer in answers)}), flush=True)


if __name__ == "__main__":
    main()
