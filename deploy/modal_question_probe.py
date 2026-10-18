"""Ask independent factual questions on private development clips, without verdicts."""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import modal

from deploy.modal_training import image, models, private

app = modal.App("lado-question-diagnostic")

QUESTIONS = {
    "bag_on_desk": "Is a recognizable cash-collection bag or flat fabric pouch lying on the work desk? A sheet of paper, book, tray or rectangular desk pad is not a bag. Identify only frames where its bag/pouch structure is visible.",
    "worker_exit": "Does the worker actually walk away from the desk and exit through the doorway? Standing up or leaning away is insufficient. Mark frames showing the departure through the doorway.",
    "same_bag_after_exit": "Compare before departure and the final frames. Does the SAME recognizable cash-collection bag or flat fabric pouch remain on TOP OF THE DESK after the worker has left through the doorway? Require both the visible departure and the bag visibly remaining afterward. If the worker stays, the bag is removed, or its final location is not visible, do not answer yes. For yes, give the frames AFTER departure where the bag remains.",
    "cash_after_exit": "Do recognizable BANKNOTES remain openly on the desk after the worker visibly leaves through the doorway? Do not infer money inside a closed bag. For yes, give frames AFTER departure with the notes still visible.",
    "cash_into_storage": "Does the worker visibly put recognizable BANKNOTES inside a cabinet, safe or drawer? An empty hand, ordinary papers or merely opening the door is insufficient. Mark the frames showing actual placement of the notes inside.",
    "storage_closed": "Does the door of that cabinet, safe or drawer visibly move from OPEN to CLOSED? A door that stays closed throughout is not a closing action. Mark the action interval; do not claim whether it is locked.",
    "another_person_after_exit": "After the worker visibly exits, does ANOTHER clearly visible physical person remain at the work desk? Exclude the departing worker's mirror/glass reflection and unclear silhouettes. If there is no visible departure, answer unknown.",
}

DEPARTURE_QUESTIONS = {
    "bag_on_desk": QUESTIONS["bag_on_desk"],
    "worker_at_exit": "Follow the worker who initially works at the cash desk. In which frames has that worker walked away from the work area and reached or passed through the ROOM EXIT DOORWAY? The nearby cash cabinet, safe or drawer is part of the work area: opening/closing that storage, standing up, or turning around beside the desk is not reaching the exit. Answer yes only for frames where the worker is at or through the room exit after leaving the work area.",
    "same_bag_after_departure": "Does the SAME recognizable cash-collection bag or flat fabric pouch stay on TOP OF THE WORK DESK while the worker has walked away to the ROOM EXIT DOORWAY? The nearby safe/cabinet is not the exit. Compare the bag before and after departure. For yes, list only frames after the worker has reached the exit while this bag remains on the desk. Do not infer hidden contents.",
    "workstation_unattended": "After the initial worker walks from the cash work area to or through the ROOM EXIT DOORWAY, is the work desk visibly left without any physical person attending it? Do not count a mirror/glass reflection as a person. Someone still using the adjacent cash cabinet/safe is still working in the area, not absent. For yes, list only frames with the worker at/through the room exit and no other visible physical person at the work desk. Use unknown if that area is obscured.",
}

TIMELINE_QUESTION = """For EVERY supplied video frame, answer these three concrete questions:
1. Where is the actual worker who initially works behind the cash desk?
   at_desk = physically working at/reaching from that desk;
   leaving = visibly moving away from the desk toward/through the exit;
   away = visibly outside that workstation after departing, including outside
   the room, or the workstation is clearly empty after the visible departure;
   unknown = location cannot be determined. Do not count a mirror/glass image
   as the actual worker at the workstation.
2. Is a recognizable cash-collection bag or flat fabric pouch resting ON TOP
   OF THE DESK in this frame? yes/no/unknown. A book, desk pad, paper sheet,
   or a bag on the floor is not a bag on the desk. Do not infer its contents.
3. Is ANOTHER clearly visible physical person at that desk in this frame?
   yes/no/unknown. Ignore reflections; do not guess from unclear silhouettes.
Return ONLY JSON with one row per frame:
{"timeline":[[0,"at_desk","yes","no"],[1,"leaving","yes","no"]]}.
The example is only the schema. Rows contain frame index, worker position,
bag-on-desk answer, another-person-at-desk answer. Read each actual frame;
do not copy the example states or extend a state past a visible change.
"""


@app.function(image=image, gpu="L4", cpu=(2, 2), memory=(24576, 24576),
              timeout=900, retries=0, max_containers=1, min_containers=0,
              scaledown_window=2, volumes={"/models": models, "/private": private})
def probe(run_id: str, selections: str, mode: str = "questions"):
    import numpy as np
    import torch
    from navidu.ingest.frames import iter_frames
    from navidu.ingest.probe import probe_video
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from eval.training import load_training_partition

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("Invalid run identifier")
    if mode not in {"questions", "timeline", "departure", "positions"}:
        raise ValueError("Invalid diagnostic mode")
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
    output = Path("/private/question-diagnostics") / run_id
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    processor = AutoProcessor.from_pretrained("/models/qwen3-vl-8b", local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained("/models/qwen3-vl-8b",
        local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    reports = []
    for source_index, (source, start, end) in enumerate(requests):
        metadata = probe_video(source.video_path)
        frames = [frame for frame in iter_frames(source.video_path, analysis_fps=1., probe=metadata)
                  if start <= frame.timestamp_sec <= end]
        if not 2 <= len(frames) <= 32:
            raise ValueError("Invalid sampled interval")
        times = [frame.timestamp_sec for frame in frames]
        clock = [round((t - times[0]) * 1000) for t in times]
        mapping = ", ".join(f"{i}={t:.3f}s" for i, t in enumerate(times))
        video = np.stack([frame.rgb for frame in frames])
        answers = {}
        started = time.monotonic()
        if mode == "positions":
            from PIL import Image

            positions = []
            question = ("Where is the ACTUAL cash-desk worker's body in this image? Choose ONE label. "
                "work_area = at the desk OR using/standing beside the nearby cash cabinet/safe/drawer; "
                "exit_doorway = at or passing through the FULL-HEIGHT ROOM EXIT used by people; "
                "outside = clearly on the other side of that room exit; "
                "unclear = the actual worker cannot be located confidently. "
                "A low cabinet door is NOT the room exit. Ignore mirror/glass reflections. "
                "Return ONLY one label: work_area, exit_doorway, outside, unclear.")
            for index in range(0, len(frames), 2):
                prompt = processor.apply_chat_template([{"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": question}]}],
                    add_generation_prompt=True, tokenize=False)
                inputs = processor(images=[Image.fromarray(frames[index].rgb)], text=prompt,
                    return_tensors="pt", min_pixels=65536, max_pixels=451584).to("cuda")
                with torch.inference_mode():
                    generated = model.generate(**inputs, do_sample=False, max_new_tokens=32,
                                               max_time=30., use_cache=True)
                response = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:],
                                                  skip_special_tokens=True)[0].strip()
                (output / f"position-{source_index}-{index}.txt").write_text(response, encoding="utf-8")
                label = response.lower().strip('". \n')
                if label not in {"work_area", "exit_doorway", "outside", "unclear"}:
                    label = "invalid"
                positions.append([times[index], label])
                del inputs, generated
            reports.append({"source_sha256": source.source_sha256, "frames": len(frames),
                "positions": positions, "inference_sec": round(time.monotonic() - started, 2),
                "quality_approved": False})
            (output / "results.json").write_text(json.dumps(reports, indent=2) + "\n")
            private.commit()
            continue
        questions = (QUESTIONS if mode == "questions" else DEPARTURE_QUESTIONS
                     if mode == "departure" else {"timeline": TIMELINE_QUESTION})
        for code, question in questions.items():
            prompt_text = (question + "\nAnswer only the question from visible evidence. "
                'Return ONLY JSON: {"answer":"yes|no|unknown","frames":[0,1]}. '
                "Use unknown when visibility is insufficient; absence of evidence is not no. "
                "For yes, list 2-5 distinct supporting frame indices. For no or unknown, use []. "
                if mode != "timeline" else question)
            prompt_text += f"\nAllowed indices: 0 through {len(frames)-1}. Do not use video seconds. Mapping: " + mapping
            prompt = processor.apply_chat_template([{"role": "user", "content": [
                {"type": "video"}, {"type": "text", "text": prompt_text}]}],
                add_generation_prompt=True, tokenize=False)
            inputs = processor(videos=[video], text=prompt, return_tensors="pt",
                videos_kwargs={"do_sample_frames": False, "video_metadata": [{"fps": 1000.,
                    "total_num_frames": clock[-1] + 1, "frames_indices": clock}],
                    "size": {"shortest_edge": len(frames) * 65536,
                             "longest_edge": len(frames) * 451584}}).to("cuda")
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False,
                                           max_new_tokens=256 if mode != "timeline" else 1536,
                                           max_time=90., use_cache=True)
            response = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:],
                                              skip_special_tokens=True)[0]
            (output / f"response-{source_index}-{code}.txt").write_text(response, encoding="utf-8")
            clean = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            if mode == "timeline":
                try:
                    rows = json.loads(clean).get("timeline", [])
                except (ValueError, AttributeError):
                    rows = []
                valid = []
                for row in rows if isinstance(rows, list) else []:
                    if isinstance(row, list) and len(row) == 4 and type(row[0]) is int \
                            and 0 <= row[0] < len(times) and row[1] in {"at_desk", "leaving", "away", "unknown"} \
                            and row[2] in {"yes", "no", "unknown"} and row[3] in {"yes", "no", "unknown"}:
                        valid.append([times[row[0]], *row[1:]])
                answers[code] = valid
                del inputs, generated
                continue
            answer, support = "invalid", []
            try:
                value = json.loads(clean)
                candidate = value.get("answer")
                indices = value.get("frames")
                if candidate in {"yes", "no", "unknown"} and isinstance(indices, list):
                    valid = all(type(i) is int and 0 <= i < len(times) for i in indices)
                    if valid and ((candidate == "yes" and 2 <= len(set(indices)) <= len(times))
                                  or (candidate != "yes" and not indices)):
                        answer, support = candidate, [times[i] for i in sorted(set(indices))]
            except (ValueError, AttributeError, TypeError):
                pass
            answers[code] = {"answer": answer, "support_sec": support}
            del inputs, generated
        reports.append({"source_sha256": source.source_sha256, "frames": len(frames),
            "answers": answers, "inference_sec": round(time.monotonic() - started, 2),
            "quality_approved": False})
        (output / "results.json").write_text(json.dumps(reports, indent=2) + "\n")
        private.commit()
    return reports


@app.local_entrypoint()
def main(run_id: str, selections: str, mode: str = "questions"):
    result = probe.remote(run_id, selections, mode)
    path = Path("eval/runs") / f"{run_id}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": run_id, "sources": len(result), "quality_approved": False}))
