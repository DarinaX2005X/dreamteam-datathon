"""Bound factual questions and join independently supported observations in time."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

QUESTION_PROMPTS = {
    "bag_on_desk": "Is a recognizable cash-collection bag or flat fabric pouch lying on the work desk? A sheet of paper, book, tray or rectangular desk pad is not a bag. Identify only frames where its bag/pouch structure is visible.",
    "worker_at_exit": "Follow the worker who initially works at the cash desk. In which frames has that worker walked away from the work area and reached or passed through the ROOM EXIT DOORWAY? The nearby cash cabinet, safe or drawer is part of the work area: opening/closing that storage, standing up, or turning around beside the desk is not reaching the exit. Answer yes only for frames where the worker is at or through the room exit after leaving the work area.",
    "same_bag_after_departure": "Does the SAME recognizable cash-collection bag or flat fabric pouch stay on TOP OF THE WORK DESK while the worker has walked away to the ROOM EXIT DOORWAY? The nearby safe/cabinet is not the exit. Compare the bag before and after departure. For yes, list only frames after the worker has reached the exit while this bag remains on the desk. Do not infer hidden contents.",
    "workstation_unattended": "After the initial worker walks from the cash work area to or through the ROOM EXIT DOORWAY, is the work desk visibly left without any physical person attending it? Do not count a mirror/glass reflection as a person. Someone still using the adjacent cash cabinet/safe is still working in the area, not absent. For yes, list only frames with the worker at/through the room exit and no other visible physical person at the work desk. Use unknown if that area is obscured.",
}


@dataclass(frozen=True)
class QuestionAnswer:
    answer: str
    frames: tuple[int, ...]


def parse_answer(text: str, frame_count: int) -> QuestionAnswer:
    """Keep all valid support; the number of returned frames is not a truth test."""
    clean = text.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        if lines[-1].strip() == "```":
            clean = "\n".join(lines[1:-1])
    try:
        value = json.loads(clean)
    except (TypeError, ValueError) as error:
        raise ValueError("Question response is not JSON") from error
    if not isinstance(value, dict) or not isinstance(value.get("answer"), str) \
            or value["answer"] not in {"yes", "no", "unknown"}:
        raise ValueError("Question response needs yes/no/unknown")
    frames = value.get("frames")
    if not isinstance(frames, list) or any(type(index) is not int or not 0 <= index < frame_count
                                        for index in frames):
        raise ValueError("Question support is outside the supplied frames")
    support = tuple(sorted(set(frames)))
    if (value["answer"] == "yes" and len(support) < 2) or (value["answer"] != "yes" and support):
        raise ValueError("Question answer and supporting frames disagree")
    return QuestionAnswer(value["answer"], support)


def unattended_bag_intervals(
    bag: QuestionAnswer,
    departure: QuestionAnswer,
    same_bag_after_departure: QuestionAnswer,
    unattended: QuestionAnswer,
    timestamps: list[float],
) -> list[tuple[int, int]]:
    """A bag plus movement is insufficient: require a visible departure and clear desk."""
    answers = (bag, departure, same_bag_after_departure, unattended)
    if any(item.answer != "yes" for item in answers):
        return []
    support = set.intersection(*(set(item.frames) for item in answers))
    supported = sorted(index for index in support if 0 <= index < len(timestamps))
    groups: list[list[int]] = []
    for index in supported:
        if groups and index == groups[-1][-1] + 1 \
                and timestamps[index] - timestamps[groups[-1][-1]] <= 2.1:
            groups[-1].append(index)
        else:
            groups.append([index])
    return [(group[0], group[-1]) for group in groups if len(group) >= 2
            and len([before for before in bag.frames if before < group[0]]) >= 2]


def verify_bag_questions(model, processor, frames, *, device, dtype,
                         should_stop, capture: Path | None = None):
    """Run the same short questions on a fixed overlapping sweep, without video labels."""
    import numpy as np
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class Cancelled(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return should_stop()

    if capture is not None:
        capture.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    metrics = {"question_windows": 0, "question_calls": 0, "question_unknown": 0,
               "question_generated_tokens": 0}
    intervals: dict[str, list[tuple[float, float]]] = {"bag_present": [], "bag_left_unattended": []}
    for offset in range(0, len(frames), 6):
        window = frames[offset:offset + 9]
        if len(window) < 4:
            continue
        metrics["question_windows"] += 1
        timestamps = [item[0] for item in window]
        clock = [round((t - timestamps[0]) * 1000) for t in timestamps]
        video = np.stack([item[1] for item in window])
        mapping = ", ".join(f"{i}={t - timestamps[0]:.3f}s" for i, t in enumerate(timestamps))
        answers = {}
        for name, question in QUESTION_PROMPTS.items():
            if should_stop():
                raise RuntimeError("ANALYSIS_CANCELLED")
            text = (question + "\nAnswer only the question from visible evidence. "
                    'Return ONLY JSON: {"answer":"yes|no|unknown","frames":[0,1]}. '
                    "Use unknown when visibility is insufficient; absence of evidence is not no. "
                    "For yes, list 2-5 distinct supporting frame indices. For no or unknown, use []. "
                    "\nVideo timestamps are relative to this window and start at zero. "
                    f"Allowed indices: 0 through {len(window)-1}. Return frame indices, not seconds. "
                    "Index-to-video-time mapping: " + mapping)
            prompt = processor.apply_chat_template([{"role": "user", "content": [
                {"type": "video"}, {"type": "text", "text": text}]}],
                add_generation_prompt=True, tokenize=False)
            inputs = processor(videos=[video], text=prompt, return_tensors="pt",
                videos_kwargs={"do_sample_frames": False, "video_metadata": [{"fps": 1000.,
                    "total_num_frames": clock[-1] + 1, "frames_indices": clock}],
                    "size": {"shortest_edge": len(window) * 65536,
                             "longest_edge": len(window) * 451584}}).to(device, dtype=dtype)
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=256,
                    max_time=90., use_cache=True,
                    stopping_criteria=StoppingCriteriaList([Cancelled()]))
            if should_stop():
                raise RuntimeError("ANALYSIS_CANCELLED")
            generated = output[:, inputs.input_ids.shape[1]:]
            raw = processor.batch_decode(generated, skip_special_tokens=True)[0]
            metrics["question_calls"] += 1
            metrics["question_generated_tokens"] += generated.shape[1]
            if capture is not None:
                (capture / f"{offset:03d}-{name}.txt").write_text(raw, encoding="utf-8")
            answer = parse_answer(raw, len(window))
            answers[name] = answer
            del inputs, output, generated
            if answer.answer != "yes":
                metrics["question_unknown"] += int(answer.answer == "unknown")
                break
            if name == "bag_on_desk":
                intervals["bag_present"].append((timestamps[answer.frames[0]], timestamps[answer.frames[-1]]))
        if len(answers) == 4:
            for first, last in unattended_bag_intervals(*answers.values(), timestamps):
                intervals["bag_left_unattended"].append((timestamps[first], timestamps[last]))
    events = []
    for kind, ranges in intervals.items():
        merged: list[list[float]] = []
        for first, last in sorted(ranges):
            if merged and first <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], last)
            else:
                merged.append([first, last])
        events.extend({"kind": kind, "start_sec": first, "end_sec": last, "subjects": [],
                       "actor_id": None, "recipient_id": None} for first, last in merged)
    metrics["question_seconds"] = round(time.perf_counter() - started, 4)
    metrics["question_unattended_intervals"] = sum(event["kind"] == "bag_left_unattended" for event in events)
    return events, metrics
