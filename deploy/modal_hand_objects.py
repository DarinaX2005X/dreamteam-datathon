"""Private hand/object localization pilot. Only aggregate diagnostics leave Modal."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import modal

from deploy.modal_training import image, models, private

app = modal.App("lado-hand-objects")
KINDS = ("hand", "cash", "identity_document", "money_counter")
QUERIES = "a human hand. banknotes. an identity card. a banknote counting machine."


def box_iou(a, b):
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def category(label):
    if "machine" in label or "counter" in label:
        return "money_counter"
    if "hand" in label:
        return "hand"
    if "card" in label or "identity" in label:
        return "identity_document"
    if "banknote" in label or "money" in label:
        return "cash"
    return None


def merge_candidates(candidates):
    merged = []
    for candidate in sorted(candidates, key=lambda x: x["score"], reverse=True):
        duplicate = next((old for old in merged if old["kind"] == candidate["kind"]
                          and box_iou(old["box"], candidate["box"]) >= .4), None)
        if duplicate is None:
            merged.append({**candidate, "views": [candidate["view"]]})
        elif candidate["view"] not in duplicate["views"]:
            duplicate["views"].append(candidate["view"])
    return merged


@app.function(image=image, cpu=(2, 2), memory=(4096, 4096), timeout=1800,
              retries=0, max_containers=1, scaledown_window=2, volumes={"/models": models})
def prepare(model_id: str = "grounding-dino-base") -> dict:
    if model_id not in {"grounding-dino-base", "rtdetr-v2-r18"}:
        raise ValueError("Unsupported model")
    shutil.copyfile("/workspace/model-manifest.json", "/models/manifest.json")
    subprocess.run([sys.executable, "scripts/fetch_models.py", "--models-dir", "/models",
                    "--manifest", "/models/manifest.json", "--model", model_id,
                    "--workers", "4", "--retries", "2"], check=True, timeout=1700)
    models.commit()
    return {"model": model_id, "ready": True, "gpu_allocated": False}


@app.function(image=image, gpu="L4", cpu=(2, 2), memory=(32768, 32768),
              timeout=2400, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/models": models, "/private": private})
def probe(run_id: str, max_sources: int = 2, samples_per_source: int = 6, verification_cap: int = 12) -> dict:
    import gc
    import time

    import torch
    from navidu.ingest.frames import frame_at
    from navidu.ingest.probe import probe_video
    from PIL import Image
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        Qwen3VLForConditionalGeneration,
    )

    from eval.training import load_training_partition

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("Invalid run ID")
    if max_sources not in {2, 8, 24} or not 4 <= samples_per_source <= 8 or not 4 <= verification_cap <= 12:
        raise ValueError("Invalid bounded annotation recipe")
    output = Path("/private/object-labels") / run_id
    output.mkdir(parents=True, exist_ok=False)
    dataset = Path("/private/dataset")
    sources = load_training_partition(dataset / "eval/training-splits.json",
        inventory=dataset / "research/dataset/inventory.csv", labels=dataset / "eval/labels.csv",
        folds=dataset / "eval/folds.csv", dataset_dir=dataset / "data/dataset", partition="train")
    selected = [s for positive in (True, False) for s in
                sorted((s for s in sources if bool(s.has_violation) == positive),
                       key=lambda s: (-s.duration_sec, s.source_sha256))[:max_sources // 2]]
    fractions = ((.1, .25, .4, .6, .75, .9) if samples_per_source == 6 else
                 tuple((index + 1) / (samples_per_source + 1) for index in range(samples_per_source)))
    processor = AutoProcessor.from_pretrained("/models/grounding-dino-base", local_files_only=True)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(
        "/models/grounding-dino-base", local_files_only=True).to("cuda").eval()
    records = []
    started = time.monotonic()
    for source_index, source in enumerate(selected):
        metadata = probe_video(source.video_path)
        for sample_index, fraction in enumerate(fractions):
            frame = frame_at(source.video_path, source.duration_sec * fraction, probe=metadata)
            pixels = Image.fromarray(frame.rgb)
            width, height = pixels.size
            frame_name = f"frame-{source_index}-{sample_index}.png"
            pixels.save(output / frame_name)
            views = [(0., 0., 1., 1.), (0., 0., .65, .65), (.35, 0., 1., .65),
                     (0., .35, .65, 1.), (.35, .35, 1., 1.)]
            candidates = []
            for view_index, view in enumerate(views):
                crop_box = tuple(round(v * (width if i % 2 == 0 else height)) for i, v in enumerate(view))
                crop = pixels.crop(crop_box)
                inputs = processor(images=crop, text=QUERIES, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    predictions = detector(**inputs)
                found = processor.post_process_grounded_object_detection(predictions, inputs.input_ids,
                    threshold=.22, text_threshold=.2, target_sizes=[crop.size[::-1]])[0]
                labels = found.get("text_labels", found.get("labels"))
                for box, score, label in zip(found["boxes"], found["scores"], labels):
                    kind = category(label)
                    if kind is None:
                        continue
                    raw = box.tolist()
                    full = [(raw[i] + crop_box[i % 2]) / (width if i % 2 == 0 else height)
                            for i in range(4)]
                    full = [min(1., max(0., value)) for value in full]
                    if full[0] >= full[2] or full[1] >= full[3]:
                        continue
                    candidates.append({"kind": kind, "box": full, "score": float(score), "view": view_index})
            objects = merge_candidates(candidates)
            record = {"source_index": source_index, "sample_index": sample_index,
                      "source_sha256": source.source_sha256, "timestamp_sec": frame.timestamp_sec,
                      "frame": frame_name, "size": list(pixels.size), "objects": objects}
            records.append(record)
            print(json.dumps({"stage": "localize", "source_index": source_index, "sample_index": sample_index,
                "counts": {kind: sum(o["kind"] == kind for o in objects) for kind in KINDS},
                "elapsed_sec": round(time.monotonic() - started, 2)}), flush=True)
    (output / "proposals.json").write_text(json.dumps(records) + "\n")
    private.commit()
    del detector, processor, predictions, inputs
    gc.collect()
    torch.cuda.empty_cache()
    processor = AutoProcessor.from_pretrained("/models/qwen3-vl-8b", local_files_only=True)
    verifier = Qwen3VLForConditionalGeneration.from_pretrained("/models/qwen3-vl-8b",
        local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    question = ('Identify the central object in this crop. Return only JSON with kind chosen from '
                'hand,cash,identity_document,money_counter,other,uncertain and visible=true or false. '
                'Cash means real banknotes, not ordinary paper or a receipt. An identity_document is an '
                'identification card or passport, not any rectangular paper. A money_counter must be a '
                'banknote counting machine, not a printer or keyboard. A hand includes visible fingers '
                'or palm, not a forearm alone. Do not read personal information. Use uncertain if unclear.')
    prompt = processor.apply_chat_template([{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": question}]}], tokenize=False, add_generation_prompt=True)
    checked = 0
    for record in records:
        pixels = Image.open(output / record["frame"]).convert("RGB")
        width, height = pixels.size
        # Reserve room for all classes; many spurious card/machine proposals must
        # not crowd hands or banknotes out of the verification budget.
        ranked = sorted(record["objects"], key=lambda o: (len(o["views"]), o["score"]), reverse=True)
        candidates = [item for kind in KINDS for item in [o for o in ranked if o["kind"] == kind][:2]]
        candidates += [item for item in ranked if item not in candidates][:max(0, verification_cap - len(candidates))]
        candidates = candidates[:verification_cap]
        for object_index, obj in enumerate(candidates):
            x1, y1, x2, y2 = obj["box"]
            margin_x, margin_y = max(.015, (x2 - x1) * .25), max(.015, (y2 - y1) * .25)
            crop = pixels.crop((max(0, int((x1 - margin_x) * width)), max(0, int((y1 - margin_y) * height)),
                                min(width, round((x2 + margin_x) * width)), min(height, round((y2 + margin_y) * height))))
            inputs = processor(images=crop, text=prompt, return_tensors="pt",
                min_pixels=262144, max_pixels=451584).to("cuda", dtype=torch.bfloat16)
            with torch.inference_mode():
                generated = verifier.generate(**inputs, do_sample=False, max_new_tokens=80,
                    max_time=15., pad_token_id=processor.tokenizer.eos_token_id)
            answer = processor.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            (output / f"answer-{record['source_index']}-{record['sample_index']}-{object_index}.txt").write_text(answer)
            try:
                value = json.loads(answer[answer.find("{"):answer.rfind("}") + 1])
                obj["verifier_kind"] = value.get("kind") if value.get("kind") in {*KINDS, "other", "uncertain"} else "invalid"
                obj["agreement"] = value.get("kind") == obj["kind"] and value.get("visible") is True
            except (ValueError, TypeError, AttributeError):
                obj["verifier_kind"], obj["agreement"] = "invalid", False
            checked += 1
        (output / "verified-proposals.json").write_text(json.dumps(records) + "\n")
        private.commit()
        print(json.dumps({"stage": "verify", "source_index": record["source_index"],
            "sample_index": record["sample_index"], "checked": checked,
            "agreements": {kind: sum(o["kind"] == kind and o.get("agreement", False)
                                   for o in candidates) for kind in KINDS},
            "elapsed_sec": round(time.monotonic() - started, 2)}), flush=True)
    result = {"run_id": run_id, "frames": len(records), "sources": len(selected), "partition": "train",
        "quality_approved": False, "human_labels": 0, "checked": checked,
        "samples_per_source": samples_per_source, "verification_cap": verification_cap,
        "proposal_counts": {kind: sum(o["kind"] == kind for r in records for o in r["objects"]) for kind in KINDS},
        "agreement_counts": {kind: sum(o["kind"] == kind and o.get("agreement", False)
                                     for r in records for o in r["objects"]) for kind in KINDS},
        "elapsed_sec": round(time.monotonic() - started, 2),
        "peak_cuda_mib": round(torch.cuda.max_memory_allocated() / 1048576, 1)}
    (output / "verified-proposals.json").write_text(json.dumps(records) + "\n")
    (output / "numeric.json").write_text(json.dumps(result) + "\n")
    private.commit()
    return result


@app.local_entrypoint()
def main(run_id: str = "", stage: str = "probe", max_sources: int = 2,
         samples_per_source: int = 6, verification_cap: int = 12, selected_frames: str = "0:3,1:2",
         model_id: str = "grounding-dino-base"):
    if stage == "prepare":
        result = prepare.remote(model_id)
    elif stage == "probe":
        result = probe.remote(run_id, max_sources, samples_per_source, verification_cap)
        Path("eval/runs", f"{run_id}.json").write_text(json.dumps(result, indent=2) + "\n")
    elif stage == "review":
        packet = review.remote(run_id, selected_frames)
        destination = Path("training/runs/manual-object-labels") / run_id
        destination.mkdir(parents=True, exist_ok=True)
        for name, content in packet.items():
            if Path(name).name != name:
                raise ValueError("Invalid review filename")
            (destination / name).write_bytes(content)
        result = {"review_files": len(packet), "directory": str(destination)}
    else:
        raise ValueError("Unknown stage")
    print(json.dumps(result))


@app.function(image=image, cpu=(2, 2), memory=(2048, 2048), timeout=180,
              retries=0, max_containers=1, scaledown_window=2, volumes={"/private": private})
def review(run_id: str, selected_frames: str) -> dict:
    import io

    from PIL import Image

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("Invalid run ID")
    selections = selected_frames.split(",")
    if not 1 <= len(selections) <= 12 or any(not re.fullmatch(r"\d{1,2}:\d", value) for value in selections):
        raise ValueError("Invalid bounded frame selection")
    output = Path("/private/object-labels") / run_id
    provenance = []
    packet = {}
    for selection in selections:
        source, sample = map(int, selection.split(":"))
        name = f"frame-{source}-{sample}.png"
        with Image.open(output / name) as original:
            pixels = original.convert("RGB")
            width, height = pixels.size
            crop_box = (0, round(height * .4), round(width * .68), round(height * .64))
            crop = pixels.crop(crop_box)
            crop = crop.resize((crop.width * 3, crop.height * 3), Image.Resampling.NEAREST)
            stream = io.BytesIO()
            crop.save(stream, format="PNG")
            packet[f"detail-{source}-{sample}.png"] = stream.getvalue()
            packet[name] = (output / name).read_bytes()
            provenance.append({"frame": name, "source_size": [width, height], "crop_pixels": list(crop_box), "scale": 3})
    packet["review-provenance.json"] = (json.dumps(provenance, indent=2) + "\n").encode()
    return packet
