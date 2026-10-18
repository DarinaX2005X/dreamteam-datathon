"""Bounded factual-action diagnostic; never trains or assigns legal verdicts."""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import modal

from deploy.modal_training import image, models, private

app = modal.App("lado-action-diagnostic")

ACTIONS = frozenset({"drinking", "eating", "food_package", "cash_on_desk",
    "cash_in_hands", "cash_to_storage", "cash_into_clothing", "storage_opened",
    "storage_closed", "storage_key_left", "person_exit", "smartphone_use",
    "collection_bag_on_desk", "collection_bag_left_unattended"})
INSTRUCTIONS = """Inspect the visible actions in this bank workplace video.
Report observable facts, not violations, intentions or permissions. Include
non-cash actions as well as cash handling. Allowed labels:
collection_bag_on_desk: a recognizable collection pouch/bag lies on the work desk;
collection_bag_left_unattended: the worker visibly walks away through the doorway
while the SAME collection pouch stays on the desk, and no other visible physical
worker takes over its supervision. Require the visible departure AND the bag
remaining after departure. A book, personal handbag, empty desk, a worker merely
standing up, or a bag already put away does not establish this event. Reflections
in glass are images of the worker, not a second supervising person;
drinking: a visible cup/bottle is brought to the person's mouth;
eating: recognizable food is brought to the mouth;
food_package: recognizable food packaging, not banknotes or paper;
cash_on_desk, cash_in_hands: recognizable banknotes, not food wrappers or cards;
cash_to_storage: visible banknotes are put INSIDE a cabinet/safe/drawer;
cash_into_clothing: visible banknotes are put into a clothing pocket;
storage_opened, storage_closed: a storage cabinet/safe/drawer door visibly changes state;
storage_key_left: a recognizable KEY stays visibly in the storage lock after the
person releases it. A knob, dark dot or failure to see removal is insufficient;
person_exit: a person visibly exits the room through a doorway;
smartphone_use: recognizable handheld smartphone use, not desk phone/calculator.
Keep distinct actions separate, at most eight events. Use zero-based frame indices
from the supplied time mapping, not video seconds. Require at least two supporting
frames. Return ONLY JSON: {"events":[{"kind":"allowed_label","start_frame":0,
"end_frame":1}]}. Omit uncertain facts. No explanations or personal data.
"""


@app.function(image=image, gpu="L4", cpu=(2, 2), memory=(24576, 24576),
              timeout=900, retries=0, max_containers=1, min_containers=0,
              scaledown_window=2, volumes={"/models": models, "/private": private})
def probe(run_id: str, selections: str):
    import numpy as np
    import torch
    from navidu.ingest.frames import iter_frames
    from navidu.ingest.probe import probe_video
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from eval.training import load_training_partition

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("Invalid run identifier")
    choices = selections.split(",")
    if not 1 <= len(choices) <= 3 or any(not re.fullmatch(r"\d:\d+(?:\.\d+)?:\d+(?:\.\d+)?", c) for c in choices):
        raise ValueError("Specify up to three development source:start:end selections")
    dataset = Path("/private/dataset")
    sources = sorted(load_training_partition(dataset / "eval/training-splits.json",
        inventory=dataset / "research/dataset/inventory.csv", labels=dataset / "eval/labels.csv",
        folds=dataset / "eval/folds.csv", dataset_dir=dataset / "data/dataset", partition="validation"),
        key=lambda source: source.source_sha256)
    requests = []
    for choice in choices:
        source_index, start, end = map(float, choice.split(":"))
        if int(source_index) >= len(sources):
            raise ValueError("Unknown development source")
        source = sources[int(source_index)]
        if not all(map(math.isfinite, (start, end))) or not 0 <= start < end <= source.duration_sec or end - start > 31:
            raise ValueError("The diagnostic interval must fit within 31 seconds of its source")
        requests.append((source, start, end))
    output = Path("/private/action-diagnostics") / run_id
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    processor = AutoProcessor.from_pretrained("/models/qwen3-vl-8b", local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained("/models/qwen3-vl-8b",
        local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    reports = []
    for index, (source, start, end) in enumerate(requests):
        metadata = probe_video(source.video_path)
        frames = [frame for frame in iter_frames(source.video_path, analysis_fps=1., probe=metadata)
                  if start <= frame.timestamp_sec <= end]
        if not 2 <= len(frames) <= 32:
            raise ValueError("Invalid sampled interval")
        times = [frame.timestamp_sec for frame in frames]
        clock = [round((t - times[0]) * 1000) for t in times]
        mapping = ", ".join(f"{i}={t:.3f}s" for i, t in enumerate(times))
        content = [{"type": "video"}, {"type": "text", "text": INSTRUCTIONS +
            f"\nThis clip has {len(frames)} frames. Allowed indices: 0 through {len(frames) - 1}. " +
            "Frame-to-source-time mapping: " + mapping}]
        prompt = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
        inputs = processor(videos=[np.stack([frame.rgb for frame in frames])], text=prompt, return_tensors="pt",
            videos_kwargs={"do_sample_frames": False, "video_metadata": [{"fps": 1000.,
                "total_num_frames": clock[-1] + 1, "frames_indices": clock}],
                "size": {"shortest_edge": len(frames) * 65536, "longest_edge": len(frames) * 451584}}).to("cuda")
        started = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=512, max_time=120., use_cache=True)
        response = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        (output / f"response-{index}.txt").write_text(response)
        clean = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            raw_events = json.loads(clean).get("events", [])
        except (ValueError, AttributeError):
            raw_events = []
        events = []
        for item in raw_events[:8] if isinstance(raw_events, list) else []:
            if not isinstance(item, dict) or item.get("kind") not in ACTIONS:
                continue
            first, last = item.get("start_frame"), item.get("end_frame")
            if type(first) is int and type(last) is int and 0 <= first < last < len(times):
                events.append({"kind": item["kind"], "start_sec": times[first], "end_sec": times[last]})
        reports.append({"source_sha256": source.source_sha256, "frames": len(frames), "events": events,
                        "inference_sec": round(time.monotonic() - started, 2), "quality_approved": False})
        (output / "results.json").write_text(json.dumps(reports, indent=2) + "\n")
        private.commit()
    return reports


@app.local_entrypoint()
def main(run_id: str, selections: str):
    result = probe.remote(run_id, selections)
    path = Path("eval/runs") / f"{run_id}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": run_id, "sources": len(result), "quality_approved": False}))
