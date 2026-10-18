"""Retrieve a bounded private annotation packet using a CPU-only image."""
from __future__ import annotations

import json
import re
from pathlib import Path

import modal

app = modal.App("lado-object-review")
image = modal.Image.debian_slim(python_version="3.11").pip_install("Pillow==11.2.1")
private = modal.Volume.from_name("lado-training-v1", version=2)


@app.function(image=image, cpu=(1, 1), memory=(1024, 1024), timeout=180,
              retries=0, max_containers=1, scaledown_window=2, volumes={"/private": private})
def review(run_id: str, selections: str):
    import io

    from PIL import Image

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("Invalid run ID")
    choices = selections.split(",")
    if not 1 <= len(choices) <= 12 or any(not re.fullmatch(r"\d{1,2}:\d", choice) for choice in choices):
        raise ValueError("Invalid bounded selection")
    output, provenance = {}, []
    directory = Path("/private/object-labels") / run_id
    for choice in choices:
        source, sample = map(int, choice.split(":"))
        filename = f"frame-{source}-{sample}.png"
        output[filename] = (directory / filename).read_bytes()
        with Image.open(directory / filename) as pixels:
            crop_box = (0, round(pixels.height * .4), round(pixels.width * .68), round(pixels.height * .64))
            crop = pixels.convert("RGB").crop(crop_box)
            crop = crop.resize((crop.width * 3, crop.height * 3), Image.Resampling.NEAREST)
            buffer = io.BytesIO()
            crop.save(buffer, format="PNG")
            output[f"detail-{source}-{sample}.png"] = buffer.getvalue()
            provenance.append({"frame": filename, "source_size": list(pixels.size), "crop_pixels": crop_box, "scale": 3})
    output["review-provenance.json"] = (json.dumps(provenance, indent=2) + "\n").encode()
    return output


@app.local_entrypoint()
def main(run_id: str, selected_frames: str):
    packet = review.remote(run_id, selected_frames)
    destination = Path("training/runs/manual-object-labels") / run_id
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in packet.items():
        if Path(name).name != name:
            raise ValueError("Invalid packet filename")
        (destination / name).write_bytes(content)
    print(json.dumps({"review_files": len(packet)}))
