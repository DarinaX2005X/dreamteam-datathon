"""Local multi-frame verification with bounded, source-linked observations.

The model's generated prose is never printed, sent to a service, or used as a
normative explanation. Only validated event fields can leave this adapter.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

from navidu.contracts.domain import Observation, Track, TrackRole
from navidu.contracts.units import frame_id
from navidu.pipeline.interfaces import PerceptionArtifacts, SemanticSignal

from .reflections import validate_reflections

PROMPT_VERSION = "cash-actions-v4-batch-schema"
EVENT_KINDS = frozenset({
    "cash_in_hands", "cash_on_desk", "cash_handover", "counting_motion",
    "cash_sorting", "cash_packing", "bag_opened", "bag_integrity_damage",
    "personal_item_identified", "camera_device_visible", "phone_visible", "bag_present",
})
_ACTION_KINDS = EVENT_KINDS - {"cash_in_hands", "cash_on_desk", "phone_visible", "bag_present",
                              "camera_device_visible", "personal_item_identified"}
_ROLES = frozenset({"cashier", "client", "unknown"})
_INSTRUCTIONS = """Label the visible actions in this ordered video sequence. Yellow p1, p2,
etc. are tracked people. Use their IDs. Give first/last supporting frame indices.
Labels: cash_in_hands, cash_on_desk, cash_handover, counting_motion, cash_sorting,
cash_packing, bag_opened, bag_integrity_damage, personal_item_identified,
camera_device_visible, phone_visible, bag_present.
Cash means recognizable banknotes, not receipts or plain paper. Counting means
separating notes repeatedly or putting notes through a counting machine. Handover
means visible transfer between two people. A hand movement alone is not counting.
Report visible facts; do not judge legality or permission. Use at least two frames
per event. Omit unclear actions. Assign cashier/client/unknown roles only when clear.
Detected boxes can be mirror or glass reflections. Only when clearly visible, report
the reflected track and its visible physical source with at least three simultaneous
supporting frames. List every clearly supported frame index as one contiguous run;
do not span uncertain frames. Do not infer reflections from similarity or proximity alone.
If two actions visibly involve the SAME banknote batch in this sequence, link their
zero-based event indices in same_cash_batches and give supporting frame indices.
Each same_cash_batches item must use these exact keys:
{"events":[0,1],"frames":[0,1]}. The numbers are examples only. Include at least
two distinct event indices and two distinct supporting frames. Every linked event
must have a supporting frame within its own start_frame/end_frame interval.
Being performed by the same person is insufficient for this cash-batch link.
Return only JSON. Schema example (do not copy these example events):
{"events":[{"kind":"counting_motion","start_frame":0,"end_frame":1,
"subjects":["p1"],"actor":"p1","recipient":null}],
"roles":[{"track":"p1","role":"cashier"}],"same_cash_batches":[],
"reflections":[{"reflected_track":"p3","source_track":"p1","frames":[0,1,2]}]}
Use [] for missing observations. Handover needs distinct actor and recipient IDs.
Other recipient fields are null. Never invent a person ID. At most six events.
"""


def resolve_dtype(device: str, requested: str, *, bf16_supported: bool = False) -> str:
    """Keep existing defaults; explicit precision never silently falls back."""
    if requested not in {"auto", "bf16", "fp16", "fp32"}:
        raise RuntimeError("INVALID_CONFIG: NAVIDU_SEMANTIC_DTYPE должен быть auto, bf16, fp16 или fp32.")
    selected = ("fp16" if device == "cuda" else "fp32") if requested == "auto" else requested
    if device == "cpu" and selected != "fp32":
        raise RuntimeError("INVALID_CONFIG: CPU-профиль временной модели поддерживает только fp32.")
    if selected == "bf16" and not bf16_supported:
        raise RuntimeError("INVALID_CONFIG: выбранный CUDA-адаптер не поддерживает bf16.")
    return selected


def _samples_at(tracks: list[Track], timestamp: float) -> dict[str, tuple]:
    found = {}
    for track in tracks:
        if track.kind != "person":
            continue
        for sample in reversed(track.samples):
            if abs(sample.t - timestamp) <= 0.002:
                found[track.track_id] = sample.bbox
                break
            if sample.t < timestamp - 0.002:
                break
    return found


def _decode_payload(text: str) -> dict:
    # Some instruction-tuned checkpoints wrap otherwise valid JSON in a fence.
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1])
    try:
        value = json.loads(stripped)
    except (TypeError, ValueError) as error:
        raise RuntimeError("MODEL_OUTPUT_INVALID: временная модель не вернула проверяемый JSON.") from error
    if not isinstance(value, dict) or not isinstance(value.get("events"), list) \
            or not isinstance(value.get("roles", []), list) \
            or not isinstance(value.get("same_cash_batches", []), list) \
            or not isinstance(value.get("reflections", []), list):
        raise RuntimeError("MODEL_OUTPUT_INVALID: неверная схема временных наблюдений.")
    return value


def validate_events(payload: dict, frame_tracks: list[dict], timestamps: list[float]) -> tuple[list[dict], list[dict]]:
    """Reject invented IDs, time ranges and unsupported actions without guessing."""
    events, roles = [], []
    for source_index, value in enumerate(payload["events"][:24]):
        if not isinstance(value, dict) or not isinstance(value.get("kind"), str) \
                or value["kind"] not in EVENT_KINDS:
            continue
        start, end = value.get("start_frame"), value.get("end_frame")
        if type(start) is not int or type(end) is not int or not 0 <= start < end < len(timestamps):
            continue
        # An actor must be actually detected in two or more supporting frames.
        present_counts = {}
        for people in frame_tracks[start:end + 1]:
            for track_id in people:
                present_counts[track_id] = present_counts.get(track_id, 0) + 1
        allowed = {key for key, count in present_counts.items() if count >= 2}
        subjects = value.get("subjects")
        if not isinstance(subjects, list) or any(not isinstance(item, str) or item not in allowed for item in subjects):
            continue
        actor, recipient = value.get("actor"), value.get("recipient")
        if (actor is not None and not isinstance(actor, str)) \
                or (recipient is not None and not isinstance(recipient, str)):
            continue
        if actor is not None and actor not in allowed:
            continue
        if value["kind"] in _ACTION_KINDS and actor is None:
            continue
        if value["kind"] == "cash_handover":
            if recipient not in allowed or recipient == actor:
                continue
        elif recipient is not None:
            continue
        bindings = list(dict.fromkeys(subjects + [item for item in (actor, recipient) if item is not None]))
        events.append({"kind": value["kind"], "start_sec": timestamps[start],
                       "end_sec": timestamps[end], "subjects": bindings,
                       "actor_id": actor, "recipient_id": recipient,
                       "_source_index": source_index})
    known = {key for people in frame_tracks for key in people}
    for value in payload.get("roles", [])[:24]:
        if isinstance(value, dict) and isinstance(value.get("track"), str) \
                and isinstance(value.get("role"), str) \
                and value["track"] in known and value["role"] in _ROLES:
            roles.append({"track": value["track"], "role": value["role"]})
    return events, roles


def bind_cash_relations(payload: dict, events: list[dict], frame_tracks: list[dict],
                        timestamps: list[float], model_id: str) -> None:
    by_index = {event["_source_index"]: event for event in events}
    cash_kinds = {"cash_in_hands", "cash_on_desk", "cash_handover", "counting_motion",
                  "cash_sorting", "cash_packing"}
    claimed = set()
    for index, batch in enumerate(payload.get("same_cash_batches", [])[:12]):
        if not isinstance(batch, dict):
            continue
        indices, support = batch.get("events"), batch.get("frames")
        if not isinstance(indices, list) or len(indices) < 2 or not isinstance(support, list) \
                or any(type(value) is not int for value in indices + support):
            continue
        if len(set(indices)) != len(indices) or len(set(support)) < 2:
            continue
        if any(value not in by_index or value in claimed for value in indices) \
                or any(not 0 <= value < len(timestamps) for value in support):
            continue
        selected = [by_index[value] for value in indices]
        if any(event["kind"] not in cash_kinds for event in selected):
            continue
        # Every linked event needs an actual supporting frame, and each of its
        # bound people must occur in the model's declared evidence for the batch.
        if any(not any(event["start_sec"] <= timestamps[value] <= event["end_sec"] for value in support)
               for event in selected):
            continue
        known = {person for value in support for person in frame_tracks[value]}
        if any(person not in known for event in selected for person in event["subjects"]):
            continue
        window_hash = hashlib.sha256(json.dumps(timestamps, separators=(",", ":")).encode("ascii")).hexdigest()[:12]
        relation = f"rel:{model_id}:{round(timestamps[0] * 1000):09d}-{round(timestamps[-1] * 1000):09d}:{window_hash}:{index}"
        for event in selected:
            event["relation_id"] = relation
        claimed.update(indices)


class TemporalVerifier:
    def __init__(self, directory: Path, entry: dict, device: str):
        import torch
        from transformers import AutoProcessor, InternVLForConditionalGeneration

        torch.set_num_threads(max(1, int(os.getenv("NAVIDU_CPU_THREADS", "6"))))
        self.model_id = entry["id"]
        self.window_frames = int(os.getenv("NAVIDU_SEMANTIC_WINDOW_FRAMES", "8"))
        self.sample_fps = float(os.getenv("NAVIDU_SEMANTIC_FPS", "1"))
        self.max_new_tokens = int(os.getenv("NAVIDU_SEMANTIC_MAX_NEW_TOKENS", "512"))
        self.max_patches = int(os.getenv("NAVIDU_SEMANTIC_MAX_PATCHES", "1"))
        if not 3 <= self.window_frames <= 16 or not 0.5 <= self.sample_fps <= 5 \
                or not 64 <= self.max_new_tokens <= 2048 or not 1 <= self.max_patches <= 4:
            raise RuntimeError("INVALID_CONFIG: неверный размер временного окна или бюджет модели.")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA_UNAVAILABLE: временная модель не видит CUDA.")
        self.device = "cuda" if device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
        requested_dtype = os.getenv("NAVIDU_SEMANTIC_DTYPE", "auto")
        self.dtype_name = resolve_dtype(self.device, requested_dtype,
            bf16_supported=(self.device == "cuda" and requested_dtype == "bf16"
                            and torch.cuda.is_bf16_supported(including_emulation=False)))
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.dtype_name]
        path = (directory / entry["local_dir"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise RuntimeError("MODEL_MANIFEST_INVALID: неверный каталог временной модели.")
        self.processor = AutoProcessor.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        self.model = InternVLForConditionalGeneration.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False,
            dtype=self.dtype, attn_implementation="sdpa",
        ).to(self.device).eval()
        self._image_features = OrderedDict()
        self.proposal = None
        self.last_metrics: dict = {}
        # Runtime enables each feature only from a matching measured validation
        # report. Loading weights does not establish semantic accuracy.
        self.capabilities = {
            "cash_detection": False, "handover_direction": False, "counting_action": False,
            "bag_damage": False, "bag_opening": False, "personal_item_semantics": False,
            "cash_on_desk": False,
        }

    def _infer(self, frames: list[tuple], frame_tracks: list[dict], should_stop: Callable[[], bool]) -> dict:
        import numpy as np
        import torch
        from PIL import Image, ImageDraw, ImageFont
        from transformers import StoppingCriteria, StoppingCriteriaList

        if self.proposal is not None:
            self.proposal.disable()

        class Cancelled(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return should_stop()

        content = []
        images = []
        for index, ((timestamp, rgb), people) in enumerate(zip(frames, frame_tracks)):
            image = Image.fromarray(rgb)
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default(size=max(28, round(image.height / 26)))
            for track_id, box in people.items():
                x1, y1, x2, y2 = [round(value * size) for value, size in
                                   zip(box, (image.width, image.height, image.width, image.height))]
                draw.rectangle((x1, y1, x2, y2), outline=(255, 225, 55), width=2)
                draw.text((x1 + 2, y1 + 2), track_id, font=font, fill=(255, 225, 55),
                          stroke_width=1, stroke_fill=(0, 0, 0))
            images.append(np.array(image, copy=True))
            content.extend([{"type": "text", "text": f"\nFrame {index}, source time {timestamp:.3f} seconds.\n"},
                            {"type": "image"}])
        content.append({"type": "text", "text": "\n" + _INSTRUCTIONS})
        prompt = self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                   add_generation_prompt=True, tokenize=False)
        image_options = {"crop_to_patches": self.max_patches > 1,
                         "min_patches": 1, "max_patches": self.max_patches}
        inputs = self.processor(images=images, text=prompt, return_tensors="pt", **image_options)
        patch_counts = [self.processor.image_processor.get_number_of_image_patches(
            image.shape[0], image.shape[1], image_options) for image in images]
        if sum(patch_counts) != inputs["pixel_values"].shape[0]:
            raise RuntimeError("MODEL_INPUT_INVALID: количество тайлов не совпало с процессором.")
        inputs = inputs.to(self.device, dtype=self.dtype)
        if should_stop():
            raise RuntimeError("ANALYSIS_CANCELLED")
        started = time.perf_counter()
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            # A vision batch of eight frames can exceed a 4 GB laptop's working
            # set. Encode each frame with the same published model and preserve
            # all tokens, then concatenate in their original temporal order.
            features = []
            offset = 0
            for image, count in zip(images, patch_counts):
                if should_stop():
                    raise RuntimeError("ANALYSIS_CANCELLED")
                key = hashlib.sha256(image.tobytes()).hexdigest()
                if key in self._image_features:
                    feature = self._image_features.pop(key)
                else:
                    feature = torch.cat([self.model.get_image_features(inputs["pixel_values"][part:part + 1])
                                         for part in range(offset, offset + count)], dim=0)
                offset += count
                self._image_features[key] = feature
                while len(self._image_features) > 32:
                    self._image_features.popitem(last=False)
                features.append(feature)
            image_features = torch.cat(features, dim=0)
            inputs_embeds = self.model.get_input_embeddings()(inputs["input_ids"])
            image_mask = self.model.model.get_placeholder_mask(
                inputs["input_ids"], inputs_embeds=inputs_embeds, image_features=image_features)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
            if self.device == "cuda":
                torch.cuda.synchronize()
            vision_sec = time.perf_counter() - started
            output = self.model.generate(input_ids=inputs["input_ids"], inputs_embeds=inputs_embeds,
                attention_mask=inputs["attention_mask"], do_sample=False, num_beams=1,
                max_new_tokens=self.max_new_tokens, use_cache=True, max_time=120.0,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([Cancelled()]))
        if should_stop():
            raise RuntimeError("ANALYSIS_CANCELLED")
        text = self.processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        self.last_metrics = {"vision_sec": round(vision_sec, 4),
            "generation_sec": round(time.perf_counter() - started - vision_sec, 4),
            "input_tokens": inputs["input_ids"].shape[1],
            "image_patches": sum(patch_counts),
            "generated_tokens": output.shape[1] - inputs["input_ids"].shape[1],
            "peak_cuda_mib": round(torch.cuda.max_memory_allocated() / 1048576, 1)
                             if self.device == "cuda" else None}
        payload = _decode_payload(text)
        self.last_metrics["raw_event_count"] = len(payload["events"])
        self.last_metrics["raw_role_count"] = len(payload.get("roles", []))
        self.last_metrics["raw_cash_batch_count"] = len(payload.get("same_cash_batches", []))
        return payload

    def observe(self, frames: list[tuple], tracks: list[Track], artifacts: PerceptionArtifacts,
                *, should_stop: Callable[[], bool] = lambda: False) -> None:
        frame_tracks = [_samples_at(tracks, timestamp) for timestamp, _ in frames]
        payload = self._infer(frames, frame_tracks, should_stop)
        timestamps = [timestamp for timestamp, _ in frames]
        events, roles = validate_events(payload, frame_tracks, timestamps)
        bind_cash_relations(payload, events, frame_tracks, timestamps, self.model_id)
        reflections = validate_reflections(payload, frame_tracks, timestamps)
        self.last_metrics.update({"accepted_event_count": len(events), "accepted_role_count": len(roles),
                                  "accepted_reflection_count": len(reflections),
                                  "bound_cash_batch_count": len({event["relation_id"] for event in events
                                      if event.get("relation_id") is not None})})
        artifacts.diagnostics.setdefault("semantic_windows", []).append({
            "start_sec": timestamps[0], "end_sec": timestamps[-1],
            "frame_count": len(frames), **self.last_metrics,
        })
        artifacts.semantic_windows_completed += 1
        artifacts.capabilities.update(self.capabilities)
        for claim in reflections:
            artifacts.semantic_signals.append(SemanticSignal(
                kind="reflection", start_sec=claim["start_sec"], end_sec=claim["end_sec"],
                subjects=(claim["reflected_track"], claim["source_track"]),
                model_id=self.model_id, prompt_version=PROMPT_VERSION, confidence=None,
                frame_ids=(frame_id(claim["start_sec"]), frame_id(claim["end_sec"]))))
        for event in events:
            event.pop("_source_index", None)
            event_frames = (frame_id(event["start_sec"]), frame_id(event["end_sec"]))
            artifacts.semantic_signals.append(SemanticSignal(
                kind=event["kind"], start_sec=event["start_sec"], end_sec=event["end_sec"],
                subjects=tuple(event["subjects"]), model_id=self.model_id,
                prompt_version=PROMPT_VERSION, confidence=None,
                relation_id=event.get("relation_id"), frame_ids=event_frames))
            existing = next((item for item in artifacts.observations if event.get("relation_id") is not None
                and item.relation_id == event.get("relation_id") and item.kind == event["kind"]
                and item.provenance == "temporal_model" and item.subjects == event["subjects"]
                and item.actor_id == event["actor_id"] and item.recipient_id == event["recipient_id"]
                and item.start_sec <= event["end_sec"] and event["start_sec"] <= item.end_sec), None)
            if existing is not None:
                existing.end_sec = max(existing.end_sec, event["end_sec"])
                existing.frame_ids = list(dict.fromkeys(existing.frame_ids + [frame_id(event["end_sec"])]))[-3:]
                continue
            artifacts.observations.append(Observation(
                obs_id=f"ob-{len(artifacts.observations) + 1}", **event,
                score=None, provenance="temporal_model", model_id=self.model_id,
                frame_ids=list(event_frames)))
        for role in roles:
            artifacts.semantic_signals.append(SemanticSignal(
                kind=f"role:{role['role']}", start_sec=timestamps[0], end_sec=timestamps[-1],
                subjects=(role["track"],), model_id=self.model_id,
                prompt_version=PROMPT_VERSION, confidence=None,
                frame_ids=(frame_id(timestamps[0]), frame_id(timestamps[-1]))))


def apply_semantic_roles(tracks: list[Track], signals: list[SemanticSignal]) -> None:
    for track in tracks:
        votes = {signal.kind.removeprefix("role:") for signal in signals
                 if signal.kind.startswith("role:") and track.track_id in signal.subjects}
        if track.kind == "person" and len(votes) == 1 and "unknown" not in votes:
            track.role = TrackRole(value=next(iter(votes)), source="auto")
