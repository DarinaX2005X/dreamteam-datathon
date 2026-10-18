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

PROMPT_VERSION = "cash-actions-v5-local-frame-indices"
EVENT_KINDS = frozenset({
    "cash_in_hands", "cash_on_desk", "cash_handover", "counting_motion",
    "cash_sorting", "cash_packing", "bag_opened", "bag_integrity_damage",
    "personal_item_identified", "camera_device_visible", "phone_visible", "bag_present",
    "identity_document_visible",
})
_ACTION_KINDS = EVENT_KINDS - {"cash_in_hands", "cash_on_desk", "phone_visible", "bag_present",
                              "camera_device_visible", "personal_item_identified", "identity_document_visible"}
_ROLES = frozenset({"cashier", "client", "unknown"})
_INSTRUCTIONS = """Label the visible actions in this ordered video sequence. Yellow p1, p2,
etc. are tracked people. Use their IDs. Give first/last supporting frame indices
from THIS window only. They are zero-based integer indices, not video seconds.
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

_EVIDENCE_INSTRUCTIONS = """Extract visible cash-handling facts from these ordered frames.
Yellow boxes labelled p1, p2, etc. identify tracked people, not their roles.

Check the sequence in this order:
1. Identify recognizable banknotes and their visible holder. Receipts, documents,
   an empty hand, or an obscured object do not establish cash.
2. Compare successive frames. A transfer needs the same visible cash moving from
   one person's control to a DIFFERENT person's control. Reaching is insufficient.
   Counting needs repeated separation of banknotes or visible use of a counting
   machine. Merely moving a stack, sorting papers, or touching cash is insufficient.
3. Assign cashier/client roles only from visible operational positions and actions.
   Use unknown if the role is ambiguous. Do not infer authorization or legality.
4. Keep only events supported by at least two distinct frames and known person IDs.
   Report one interval per continuing action. Prefer clear cash actions to incidental
   objects when more than six observations are available.

Allowed kinds: cash_in_hands, cash_on_desk, cash_handover, counting_motion,
cash_sorting, cash_packing, bag_opened, bag_integrity_damage,
personal_item_identified, camera_device_visible, phone_visible, bag_present.
A visible phone alone does not establish that it is a prohibited personal item.
Bag damage needs visible physical damage; opening a bag is a separate observation.

Return ONLY a JSON object with these four arrays:
- events: objects with kind, start_frame, end_frame, subjects, actor, recipient.
  start_frame/end_frame are zero-based INTEGER indices in this window, NOT seconds.
  subjects is an array of visible p-IDs. Actions require an actor. cash_handover
  requires a distinct recipient; all other recipients are null.
- roles: objects with track and role (cashier, client, unknown).
- same_cash_batches: objects with events (indices into your events array) and frames
  (supporting frame indices). Link at least two events only if the SAME physical
  banknote batch is visibly traceable between them. Same person or location is not
  enough. Give at least two frames, including support inside every linked interval.
- reflections: objects with reflected_track, source_track, frames. Require a visible
  physical person and its mirror/glass image, supported by at least three simultaneous
  frames in one contiguous run. Similar appearance or nearby positions are not enough.
Use empty arrays when evidence is absent. Do not fill fields from assumptions.
At most six events. Never invent a track, a hidden action, or supporting frame.
"""


_CONTEXT_INSTRUCTIONS = """Read this ordered bank-counter video as a sequence of visible facts.
Yellow p1, p2 etc. identify tracked people. Their numbers NEVER establish roles.
The first person is not necessarily the cashier. Identify cashier/client only from
their working position and interactions. A later client can have a new p-ID without
being an extra person: simultaneous presence matters, not total IDs over the clip.

Follow each clearly recognizable banknote batch: holder, visible transfer, desk,
counting and packing. Compare before and after, including client changes. A hand
reaching, ordinary paper or a rectangular card does not establish cash. Counting
requires repeated note separation or visible counting-machine use, not moving a
stack. Do not infer that an unobserved action never happened. A document is an
identity document only when its visible appearance supports that specific class;
do not read or reproduce any personal data. Report facts, never legal conclusions.

Return ONLY JSON with these six arrays. All frame references are integer indices
in this window, NOT seconds. Use [] when there is no clear evidence.
- events: at most 10 objects with kind, start_frame, end_frame, subjects, actor,
  recipient, objects. Kinds: cash_in_hands, cash_on_desk, cash_handover,
  counting_motion, cash_sorting, cash_packing, identity_document_visible,
  bag_opened, bag_integrity_damage, personal_item_identified, camera_device_visible,
  phone_visible, bag_present. Use at least two supporting frames per event.
  subjects contains known p-IDs; objects contains local object IDs from objects
  below. Cash handover requires DIFFERENT visible actor and recipient p-IDs.
  Other recipients are null. Actions require an actor. Continuing action = one
  interval. Keep actions for separate clients and separate banknote batches separate.
- roles: objects with track and role (cashier, client, unknown).
- same_cash_batches: objects with events (indices in events) and frames. Link at
  least two events only if the SAME physical cash batch is visibly traceable.
  Include two supporting frames and evidence inside every linked event interval.
  The same holder or desk location alone is NOT proof of the same cash batch.
- reflections: objects with reflected_track, source_track, frames. Require a
  clearly visible physical source and its mirror/glass image in at least three
  simultaneous contiguous frames. Similarity or proximity alone is insufficient.
- visible_people: objects with track, clear, physical, frames. Include only known
  p-IDs that are clearly recognizable physical people in each listed frame, with
  clear=true and physical=true. Omit dark silhouettes, heavily obscured people,
  unclear figures behind glass, images and reflections. Glass alone is not a reason
  to omit a clearly visible actual person. At least two supported frames per item.
- objects: at most 6 objects with id, kind, samples. id is local, e.g. c1; kind is
  cash or identity_document. Each sample has frame and box=[x1,y1,x2,y2], coordinates
  from 0 to 1000 relative to the full image. Tight boxes around the visible object,
  never the whole person or desk. Give 2 to 6 samples at useful action boundaries,
  including the first and last two frames when the same object is visible there.
  Reuse an id only when its physical continuity is visible. Different clients'
  cash stays distinct even if placed together. Omit objects too small to recognize.
Never invent hidden people, boxes, transfers, identities or supporting frames.
"""

_FOCUSED_INSTRUCTIONS = """Describe the visible participants and cash operations in this video.
The yellow p-IDs identify people, not roles. Identify cashier/client from their
position and actions. Describe what IS visible even when no violation is present.
Follow banknotes from one holder to another, onto the desk, through counting and
packing. Distinguish money from ordinary paper and identity cards. Counting means
separating notes repeatedly or using a counting machine; a hand reach is not a transfer.

Return JSON with six arrays:
1. roles: {track, role}, role is cashier/client/unknown.
2. visible_people: {track, clear:true, physical:true, start_frame, end_frame}.
   Give contiguous intervals of clearly visible REAL people, excluding reflections,
   dark silhouettes and heavily obscured figures. A later client is a different ID,
   not an additional simultaneous person. Never assign a role from a track number.
3. objects: {id, kind, samples:[{frame, box, view}]}. kind is cash or identity_document.
   id is c1..c6. Give 2-6 tight boxes per object, with box=[x1,y1,x2,y2] on a 0-1000
   coordinate scale; view is full (or detail when a detail view is supplied).
   Reuse an object ID only for the same visible physical batch; keep clients' batches separate.
4. events: {kind,start_frame,end_frame,subjects,actor,recipient,objects}.
   Kinds: cash_in_hands,cash_on_desk,cash_handover,counting_motion,cash_sorting,
   cash_packing,identity_document_visible,phone_visible,bag_present,bag_opened,
   bag_integrity_damage,personal_item_identified,camera_device_visible.
   subjects are p-IDs, objects are c-IDs. Cash handover needs distinct actor and
   recipient p-IDs; other recipients are null. Actions need an actor. At most 10 events.
5. same_cash_batches: {events:[event indices],frames:[supporting frame indices]}.
   Link two or more events only for visibly continuous handling of the SAME notes.
6. reflections: {reflected_track,source_track,frames}. Require the visible physical
   source and its reflected image together in at least three consecutive frames.

Frame indices refer to THIS window, not seconds. Intervals require start_frame <
end_frame and at least two visible supporting frames. Report useful visible facts;
use empty arrays only for categories that are not visible. Do not invent hidden
actions, read personal data, or judge legality. Do not treat missing evidence as
proof that a required action never happened.
"""


def prompt_profile(variant: str) -> tuple[str, str]:
    if variant == "v5":
        return PROMPT_VERSION, _INSTRUCTIONS
    if variant == "v6-evidence":
        return "cash-actions-v6-evidence-order", _EVIDENCE_INSTRUCTIONS
    if variant == "v7-context":
        return "cash-actions-v7-grounded-context", _CONTEXT_INSTRUCTIONS
    if variant == "v8-focused":
        return "cash-actions-v8-focused-context", _FOCUSED_INSTRUCTIONS
    raise RuntimeError("INVALID_CONFIG: unknown semantic prompt variant.")


class TemporalOutputError(RuntimeError):
    code = "MODEL_OUTPUT_INVALID"
    retryable = False

    def __init__(self, message: str, *, numeric_metrics: dict | None = None):
        super().__init__(message)
        self.numeric_metrics = numeric_metrics or {}


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
        raise TemporalOutputError(
            "MODEL_OUTPUT_INVALID: временная модель не вернула проверяемый JSON.",
            numeric_metrics={"response_chars": len(stripped),
                             "json_error_pos": getattr(error, "pos", None)},
        ) from error
    if not isinstance(value, dict) or not isinstance(value.get("events"), list) \
            or not isinstance(value.get("roles", []), list) \
            or not isinstance(value.get("same_cash_batches", []), list) \
            or not isinstance(value.get("reflections", []), list) \
            or not isinstance(value.get("visible_people", []), list) \
            or not isinstance(value.get("objects", []), list):
        raise TemporalOutputError("MODEL_OUTPUT_INVALID: неверная схема временных наблюдений.",
                                  numeric_metrics={"response_chars": len(stripped)})
    return value


def validate_events(payload: dict, frame_tracks: list[dict], timestamps: list[float], *,
                    diagnostics: dict[str, int] | None = None) -> tuple[list[dict], list[dict]]:
    """Reject invented IDs, time ranges and unsupported actions without guessing."""
    def count(key: str) -> None:
        if diagnostics is not None:
            diagnostics[key] = diagnostics.get(key, 0) + 1

    events, roles = [], []
    if diagnostics is not None:
        diagnostics["events_over_processing_limit"] = max(0, len(payload["events"]) - 24)
    for source_index, value in enumerate(payload["events"][:24]):
        if not isinstance(value, dict) or not isinstance(value.get("kind"), str) \
                or value["kind"] not in EVENT_KINDS:
            count("rejected_kind_or_shape")
            continue
        # Only the fixed vocabulary becomes a diagnostic key. Generated text,
        # subject IDs and unknown property values are never copied into it.
        count(f"raw_kind_{value['kind']}")
        start, end = value.get("start_frame"), value.get("end_frame")
        if type(start) is not int or type(end) is not int:
            count("rejected_frame_type")
            continue
        if not 0 <= start < len(timestamps) or not 0 <= end < len(timestamps):
            count("rejected_frame_range")
            continue
        if start >= end:
            count("rejected_frame_order_or_single")
            continue
        # An actor must be actually detected in two or more supporting frames.
        present_counts = {}
        for people in frame_tracks[start:end + 1]:
            for track_id in people:
                present_counts[track_id] = present_counts.get(track_id, 0) + 1
        allowed = {key for key, count in present_counts.items() if count >= 2}
        subjects = value.get("subjects")
        if not isinstance(subjects, list) or any(not isinstance(item, str) for item in subjects):
            count("rejected_subject_type")
            continue
        if any(item not in allowed for item in subjects):
            count("rejected_subject_support")
            continue
        actor, recipient = value.get("actor"), value.get("recipient")
        if (actor is not None and not isinstance(actor, str)) \
                or (recipient is not None and not isinstance(recipient, str)):
            count("rejected_actor_or_recipient_type")
            continue
        if actor is not None and actor not in allowed:
            count("rejected_actor_support")
            continue
        if value["kind"] in _ACTION_KINDS and actor is None:
            count("rejected_action_without_actor")
            continue
        if value["kind"] == "cash_handover":
            if recipient not in allowed or recipient == actor:
                count("rejected_handover_recipient")
                continue
        elif recipient is not None:
            count("rejected_unexpected_recipient")
            continue
        bindings = list(dict.fromkeys(subjects + [item for item in (actor, recipient) if item is not None]))
        events.append({"kind": value["kind"], "start_sec": timestamps[start],
                       "end_sec": timestamps[end], "subjects": bindings,
                       "actor_id": actor, "recipient_id": recipient,
                       "_source_index": source_index})
        count(f"accepted_kind_{value['kind']}")
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


def visual_request(frames: list[tuple], frame_tracks: list[dict], variant: str = "v5") -> tuple[list, list[dict]]:
    """The same marked frames and question are used by each candidate model."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    _, instructions = prompt_profile(variant)
    content, images = [], []
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
        content.extend([{"type": "text", "text": f"\nFrame {index}:\n"}, {"type": "image"}])
    content.append({"type": "text", "text": "\n" + instructions +
        f"\nThis window has {len(frames)} frames. The only allowed frame indices are "
        f"0 through {len(frames) - 1}. Every start_frame, end_frame and frames entry "
        "must use these indices. Use start_frame < end_frame for every event. "
        "Combine a continuing action into one interval; do not emit one event per frame.\n"})
    if variant == "v8-focused":
        content[-1]["text"] += "Known person IDs in this window: " + ", ".join(
            sorted({person for people in frame_tracks for person in people})) + ".\n"
    return images, content


class TemporalVerifier:
    prompt_version = PROMPT_VERSION
    prompt_variant = "v5"
    max_window_frames = 16
    @staticmethod
    def model_class():
        from transformers import InternVLForConditionalGeneration

        return InternVLForConditionalGeneration

    def __init__(self, directory: Path, entry: dict, device: str):
        import torch
        from transformers import AutoProcessor

        torch.set_num_threads(max(1, int(os.getenv("NAVIDU_CPU_THREADS", "6"))))
        self.model_id = entry["id"]
        self.prompt_variant = os.getenv("NAVIDU_SEMANTIC_PROMPT_VARIANT", "v5")
        self.prompt_version, _ = prompt_profile(self.prompt_variant)
        self.window_frames = int(os.getenv("NAVIDU_SEMANTIC_WINDOW_FRAMES", "8"))
        self.sample_fps = float(os.getenv("NAVIDU_SEMANTIC_FPS", "1"))
        self.max_new_tokens = int(os.getenv("NAVIDU_SEMANTIC_MAX_NEW_TOKENS", "512"))
        self.max_patches = int(os.getenv("NAVIDU_SEMANTIC_MAX_PATCHES", "1"))
        self.structured_output = os.getenv("NAVIDU_SEMANTIC_STRUCTURED_OUTPUT", "0").lower() in {
            "1", "true", "yes",
        }
        if not 3 <= self.window_frames <= self.max_window_frames or not 0.5 <= self.sample_fps <= 5 \
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
        self._structured_tokenizer_data = None
        if self.structured_output:
            try:
                from lmformatenforcer.integrations.transformers import (
                    build_token_enforcer_tokenizer_data,
                )
            except ImportError as error:
                raise RuntimeError("INVALID_CONFIG: установите optional dependency structured.") from error
            self._structured_tokenizer_data = build_token_enforcer_tokenizer_data(self.processor.tokenizer)
        self.model = self.model_class().from_pretrained(
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
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        if self.proposal is not None:
            self.proposal.disable()

        class Cancelled(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return should_stop()

        images, content = visual_request(frames, frame_tracks, self.prompt_variant)
        prompt = self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                   add_generation_prompt=True, tokenize=False)
        image_options = {"crop_to_patches": self.max_patches > 1,
                         "min_patches": 1, "max_patches": self.max_patches}
        inputs = self.processor(images=images, text=prompt, return_tensors="pt", **image_options)
        generation_options = {}
        known_track_ids = sorted({track_id for people in frame_tracks for track_id in people})
        if self.structured_output and known_track_ids:
            from .structured_output import build_prefix, build_schema

            generation_options["prefix_allowed_tokens_fn"] = build_prefix(
                self.processor.tokenizer, build_schema(len(frames), known_track_ids, self.prompt_variant),
                tokenizer_data=self._structured_tokenizer_data,
            )
        # LMFE 0.11.3 cannot enforce maxItems=0. Windows without known people
        # use the ordinary verifier and its unchanged post-validation.
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
                stopping_criteria=StoppingCriteriaList([Cancelled()]), **generation_options)
        if should_stop():
            raise RuntimeError("ANALYSIS_CANCELLED")
        text = self.processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        self.last_metrics = {"vision_sec": round(vision_sec, 4),
            "generation_sec": round(time.perf_counter() - started - vision_sec, 4),
            "input_tokens": inputs["input_ids"].shape[1],
            "image_patches": sum(patch_counts),
            "structured_output_active": bool(generation_options),
            "generated_tokens": output.shape[1] - inputs["input_ids"].shape[1],
            "max_new_tokens": self.max_new_tokens,
            "token_limit_reached": output.shape[1] - inputs["input_ids"].shape[1] >= self.max_new_tokens,
            "peak_cuda_mib": round(torch.cuda.max_memory_allocated() / 1048576, 1)
                             if self.device == "cuda" else None}
        try:
            payload = _decode_payload(text)
        except TemporalOutputError as error:
            error.numeric_metrics.update(self.last_metrics)
            raise
        self.last_metrics["raw_event_count"] = len(payload["events"])
        self.last_metrics["raw_role_count"] = len(payload.get("roles", []))
        self.last_metrics["raw_cash_batch_count"] = len(payload.get("same_cash_batches", []))
        return payload

    def observe(self, frames: list[tuple], tracks: list[Track], artifacts: PerceptionArtifacts,
                *, should_stop: Callable[[], bool] = lambda: False) -> None:
        frame_tracks = [_samples_at(tracks, timestamp) for timestamp, _ in frames]
        self._spatial_tracks = [track for track in tracks if track.kind == "object"]
        self._object_bindings = {}
        try:
            payload = self._infer(frames, frame_tracks, should_stop)
            detected_objects = self._object_bindings
        finally:
            self._spatial_tracks = []
            self._object_bindings = {}
        if self.prompt_variant in {"v7-context", "v8-focused"} and any(
            not isinstance(payload.get(key), list)
            for key in ("events", "roles", "same_cash_batches", "reflections", "visible_people", "objects")
        ):
            raise TemporalOutputError("MODEL_OUTPUT_INVALID: отсутствует часть наблюдений контекста.")
        timestamps = [timestamp for timestamp, _ in frames]
        event_validation = {}
        events, roles = validate_events(payload, frame_tracks, timestamps, diagnostics=event_validation)
        bind_cash_relations(payload, events, frame_tracks, timestamps, self.model_id)
        reflections = validate_reflections(payload, frame_tracks, timestamps)
        if self.prompt_variant in {"v7-context", "v8-focused"}:
            from .grounding import bind_grounded_events, ground_objects, visible_people

            objects = ground_objects(payload, timestamps, artifacts, self.model_id, self.prompt_version,
                                     detected_objects=detected_objects)
            bind_grounded_events(payload, events, objects)
            visible_people(payload, frame_tracks, timestamps, artifacts, self.model_id, self.prompt_version)
            self.last_metrics.update({"grounded_objects": len(objects),
                "grounded_cash_objects": sum(track.object_class == "cash" for track, _ in objects.values()),
                "grounded_identity_documents": sum(track.object_class == "identity_document" for track, _ in objects.values()),
                "raw_visible_people": len(payload["visible_people"])})
        self.last_metrics.update({"accepted_event_count": len(events), "accepted_role_count": len(roles),
                                  "accepted_reflection_count": len(reflections),
                                  "bound_cash_batch_count": len({event["relation_id"] for event in events
                                      if event.get("relation_id") is not None})})
        artifacts.diagnostics.setdefault("semantic_windows", []).append({
            "start_sec": timestamps[0], "end_sec": timestamps[-1],
            "frame_count": len(frames), **self.last_metrics, "event_validation": event_validation,
        })
        artifacts.semantic_windows_completed += 1
        artifacts.capabilities.update(self.capabilities)
        if self.prompt_variant not in {"v7-context", "v8-focused"}:
            artifacts.capabilities["clear_person_count"] = False
        for claim in reflections:
            artifacts.semantic_signals.append(SemanticSignal(
                kind="reflection", start_sec=claim["start_sec"], end_sec=claim["end_sec"],
                subjects=(claim["reflected_track"], claim["source_track"]),
                model_id=self.model_id, prompt_version=self.prompt_version, confidence=None,
                frame_ids=(frame_id(claim["start_sec"]), frame_id(claim["end_sec"]))))
        for event in events:
            event.pop("_source_index", None)
            event_frames = (frame_id(event["start_sec"]), frame_id(event["end_sec"]))
            artifacts.semantic_signals.append(SemanticSignal(
                kind=event["kind"], start_sec=event["start_sec"], end_sec=event["end_sec"],
                subjects=tuple(event["subjects"]), model_id=self.model_id,
                prompt_version=self.prompt_version, confidence=None,
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
                prompt_version=self.prompt_version, confidence=None,
                frame_ids=(frame_id(timestamps[0]), frame_id(timestamps[-1]))))


def apply_semantic_roles(tracks: list[Track], signals: list[SemanticSignal]) -> None:
    for track in tracks:
        votes = {signal.kind.removeprefix("role:") for signal in signals
                 if signal.kind.startswith("role:") and track.track_id in signal.subjects}
        if track.kind == "person" and len(votes) == 1 and "unknown" not in votes:
            track.role = TrackRole(value=next(iter(votes)), source="auto")
