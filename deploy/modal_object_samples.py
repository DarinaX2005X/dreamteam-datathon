"""Extract bounded frames from an authorized development partition for review."""
from __future__ import annotations

import io
import json
import math
import re
from pathlib import Path

import modal

from deploy.modal_training import image, private

app = modal.App("lado-object-extra-samples")


@app.function(image=image, cpu=(2, 2), memory=(4096, 4096), timeout=300,
              retries=0, max_containers=1, scaledown_window=2, volumes={"/private": private})
def extract(run_id: str, selections: str, partition: str = "train"):
    from navidu.ingest.frames import frame_at
    from navidu.ingest.probe import probe_video
    from PIL import Image

    from eval.training import load_training_partition

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("Invalid review run identifier")
    if partition not in {"train", "validation"}:
        raise ValueError("Only authorized development partitions are supported")
    choices = selections.split(",")
    if not 1 <= len(choices) <= 12 or any(not re.fullmatch(r"\d{1,2}:\d+(?:\.\d+)?", item) for item in choices):
        raise ValueError("At most twelve explicit source/time choices are supported")
    dataset = Path("/private/dataset")
    sources = load_training_partition(dataset / "eval/training-splits.json",
        inventory=dataset / "research/dataset/inventory.csv", labels=dataset / "eval/labels.csv",
        folds=dataset / "eval/folds.csv", dataset_dir=dataset / "data/dataset", partition=partition)
    allowed = {source.source_sha256: source for source in sources}
    if partition == "train":
        previous = json.loads(Path("/private/object-labels/hands-objects-train24-20261018/proposals.json").read_text())
        indices = {record["source_index"]: record["source_sha256"] for record in previous}
    else:
        indices = {index: source.source_sha256 for index, source in
                   enumerate(sorted(sources, key=lambda source: source.source_sha256))}
    output = Path("/private/object-labels") / run_id
    output.mkdir(parents=True, exist_ok=False)
    packet, records = {}, []
    for sample_index, choice in enumerate(choices):
        source_index, seconds = choice.split(":")
        source_index, seconds = int(source_index), float(seconds)
        source = allowed.get(indices.get(source_index))
        if source is None or not math.isfinite(seconds) or not 0 <= seconds < source.duration_sec:
            raise ValueError("Requested frame is outside the selected development partition")
        metadata = probe_video(source.video_path)
        frame = frame_at(source.video_path, seconds, probe=metadata)
        pixels = Image.fromarray(frame.rgb)
        name = f"frame-{source_index}-{sample_index}.png"
        pixels.save(output / name, format="PNG")
        packet[name] = (output / name).read_bytes()
        crop = pixels.crop((0, round(pixels.height * .4), round(pixels.width * .68), round(pixels.height * .64)))
        crop = crop.resize((crop.width * 3, crop.height * 3), Image.Resampling.NEAREST)
        buffer = io.BytesIO()
        crop.save(buffer, format="PNG")
        packet[f"detail-{source_index}-{sample_index}.png"] = buffer.getvalue()
        records.append({"source_index": source_index, "sample_index": sample_index,
            "source_sha256": source.source_sha256, "timestamp_sec": frame.timestamp_sec,
            "frame": name, "size": list(pixels.size), "objects": []})
    (output / "proposals.json").write_text(json.dumps(records, indent=2) + "\n")
    private.commit()
    packet["proposals.json"] = (output / "proposals.json").read_bytes()
    return packet


@app.local_entrypoint()
def main(run_id: str, selections: str, partition: str = "train"):
    packet = extract.remote(run_id, selections, partition)
    destination = Path("training/runs/manual-object-labels") / run_id
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in packet.items():
        if Path(name).name != name:
            raise ValueError("Invalid packet filename")
        (destination / name).write_bytes(data)
    print(json.dumps({"review_files": len(packet), "gpu_allocated": False}))
