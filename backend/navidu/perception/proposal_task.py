"""The shared supervised task exposes only video frames and relative times.

A learned candidate schedules closer visual verification. It is never a fact,
an actor label, a visibility guarantee, or a normative conclusion by itself.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

PROMPT_VERSION = "cash-candidate-v1"
LORA_TARGET_MODULES = r"model\.language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)"
_INSTRUCTIONS = """Locate a candidate irregular cash-operation episode in this ordered
video sequence using the learned examples. This is a proposal for closer visual
verification, not a legal conclusion. Return only JSON with two fields:
{"candidate":1,"frame":5} if an episode is visible, using the best supporting
zero-based frame index; otherwise {"candidate":0,"frame":null}.
Do not describe people, actions, reasons or regulations. Do not copy the example.
"""


@dataclass(frozen=True)
class Proposal:
    candidate: bool
    frame_index: int | None
    time_sec: float | None


def _timestamps(frames: list[tuple[float, Any]]) -> list[float]:
    times = [float(timestamp) for timestamp, _ in frames]
    if len(times) < 2 or any(not math.isfinite(timestamp) or timestamp < 0 for timestamp in times) \
            or any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("Proposal frames must have finite, increasing source timestamps")
    return times


def proposal_messages(frames: list[tuple[float, Any]]) -> list[dict]:
    times = _timestamps(frames)
    content = []
    for index, timestamp in enumerate(times):
        content.extend([{"type": "text", "text":
                         f"Frame {index}, relative time {timestamp - times[0]:.3f} seconds.\n"},
                        {"type": "image"}])
    content.append({"type": "text", "text": _INSTRUCTIONS})
    return [{"role": "user", "content": content}]


def _validate_target(value: Any, frame_count: int) -> tuple[bool, int | None]:
    if not isinstance(value, dict) or set(value) != {"candidate", "frame"} \
            or type(value.get("candidate")) is not int or value["candidate"] not in {0, 1}:
        raise ValueError("Invalid proposal target")
    index = value["frame"]
    if value["candidate"] == 0:
        if index is not None:
            raise ValueError("A clean proposal cannot name an event frame")
        return False, None
    if type(index) is not int or not 0 <= index < frame_count:
        raise ValueError("A candidate must name an actual input frame")
    return True, index


def encode_proposal(processor, frames: list[tuple[float, Any]], *, target: dict | None = None):
    """Return native processor tensors; optionally mask all prompt tokens in loss."""
    messages = proposal_messages(frames)
    images = [rgb for _, rgb in frames]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(images=images, text=prompt, return_tensors="pt", crop_to_patches=False)
    if target is None:
        return inputs
    _validate_target(target, len(frames))
    response = json.dumps(target, separators=(",", ":"))
    full = processor.apply_chat_template(messages + [{"role": "assistant", "content": response}],
                                         add_generation_prompt=False, tokenize=False)
    combined = processor(images=images, text=full, return_tensors="pt", crop_to_patches=False)
    prefix = inputs["input_ids"][0].tolist()
    complete = combined["input_ids"][0].tolist()
    prefix_length = next((index for index, pair in enumerate(zip(prefix, complete)) if pair[0] != pair[1]),
                         min(len(prefix), len(complete)))
    # One boundary token may merge with the response under a BPE tokenizer.
    # Earlier divergence would mean prompt/image processing was inconsistent.
    if prefix_length < len(prefix) - 1 or prefix_length >= len(complete):
        raise RuntimeError("TRAINING_INPUT_INVALID: response token boundary is inconsistent")
    labels = combined["input_ids"].clone()
    labels[:, :prefix_length] = -100
    labels[combined["attention_mask"] == 0] = -100
    combined["labels"] = labels
    return combined


def decode_proposal(text: str, timestamps: list[float]) -> Proposal:
    """Reject malformed output without exposing generated text in diagnostics."""
    try:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines[-1].strip() == "```":
                stripped = "\n".join(lines[1:-1])
        value = json.loads(stripped)
        candidate, index = _validate_target(value, len(timestamps))
        if any(not math.isfinite(timestamp) or timestamp < 0 for timestamp in timestamps):
            raise ValueError("Invalid source timestamp")
    except (TypeError, ValueError, IndexError) as error:
        raise RuntimeError("MODEL_OUTPUT_INVALID: обучаемая модель не вернула проверяемое предложение.") from error
    return Proposal(candidate=candidate, frame_index=index,
                    time_sec=float(timestamps[index]) if index is not None else None)
