"""Bounded object-grounding diagnostic; bank frames and raw answers stay private."""
from __future__ import annotations

import json
import re
from pathlib import Path

import modal

from deploy.modal_training import image, models, private

app = modal.App("lado-object-probe")


@app.function(image=image, gpu="L4", cpu=(2, 2), memory=(32768, 32768),
              timeout=600, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/models": models, "/private": private})
def probe(run_id: str) -> dict:
    import time

    import torch
    from navidu.ingest.frames import frame_at
    from navidu.ingest.probe import probe_video
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from eval.dataset import load_sources

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("Invalid run ID")
    output = Path("/private/evaluation") / run_id
    output.mkdir(parents=True, exist_ok=False)
    dataset = Path("/private/dataset")
    sources = load_sources(dataset / "research/dataset/inventory.csv",
                           dataset / "eval/labels.csv", dataset / "eval/folds.csv")
    selected = [max((s for s in sources if s.fold == 4 and bool(s.has_violation) == positive),
                    key=lambda s: s.duration_sec) for positive in (True, False)]
    processor = AutoProcessor.from_pretrained("/models/qwen3-vl-8b", local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained("/models/qwen3-vl-8b",
        local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    prompt = processor.apply_chat_template([{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text":
        "Locate visible banknotes, stacks of banknotes, and identity documents in this image. "
        "Return only JSON: {\"objects\":[{\"kind\":\"cash\",\"box\":[x1,y1,x2,y2]}]}. "
        "kind must be cash or identity_document. Coordinates are integers 0-1000, "
        "relative to this full image. Give tight boxes. A receipt or ordinary paper is "
        "not cash or an identity document. Do not read any personal data. "
        "Return an empty objects array if none is recognizable."}]}],
        add_generation_prompt=True, tokenize=False)
    records = []
    for source_index, source in enumerate(selected):
        path = source.video_path(dataset / "data/dataset")
        metadata = probe_video(path)
        for sample_index, fraction in enumerate((.1, .25, .4, .6, .75, .9)):
            frame = frame_at(path, source.duration_sec * fraction, probe=metadata)
            pixels = Image.fromarray(frame.rgb)
            inputs = processor(images=pixels, text=prompt, return_tensors="pt",
                min_pixels=1048576, max_pixels=1048576).to("cuda", dtype=torch.bfloat16)
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=512,
                    max_time=45.0, pad_token_id=processor.tokenizer.eos_token_id)
            text = processor.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            (output / f"response-{source_index}-{sample_index}.txt").write_text(text)
            try:
                answer = json.loads(text[text.find("{"):text.rfind("}") + 1])
                objects = answer["objects"]
                valid = [o for o in objects if isinstance(o, dict) and o.get("kind") in {"cash", "identity_document"}
                    and isinstance(o.get("box"), list) and len(o["box"]) == 4
                    and all(type(v) in (int, float) and 0 <= v <= 1000 for v in o["box"])
                    and o["box"][0] < o["box"][2] and o["box"][1] < o["box"][3]]
                counts = {kind: sum(o["kind"] == kind for o in valid) for kind in ("cash", "identity_document")}
            except (ValueError, KeyError, TypeError):
                counts = {"invalid_response": 1}
            record = {"source_index": source_index, "sample_index": sample_index,
                "counts": counts, "input_tokens": inputs["input_ids"].shape[1],
                "source_size": list(pixels.size), "image_grid_thw": inputs["image_grid_thw"].tolist(),
                "seconds": round(time.monotonic() - started, 2),
                "peak_cuda_mib": round(torch.cuda.max_memory_allocated() / 1048576, 1)}
            records.append(record)
            print(json.dumps(record), flush=True)
    result = {"run_id": run_id, "quality_approved": False, "records": records}
    (output / "numeric.json").write_text(json.dumps(result) + "\n")
    private.commit()
    return result


@app.local_entrypoint()
def main(run_id: str):
    result = probe.remote(run_id)
    path = Path("eval/runs") / f"{run_id}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
