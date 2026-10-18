"""Local Qwen3-VL candidate using the unchanged temporal evidence contract."""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from statistics import median

from .temporal import TemporalOutputError, TemporalVerifier, _decode_payload, visual_request


def detail_bounds(frame_tracks):
    """A stable person's observed box proposes a crop, never a role or a fact."""
    candidates = {}
    for people in frame_tracks:
        for person, box in people.items():
            candidates.setdefault(person, []).append(box)
    candidates = [boxes for boxes in candidates.values() if len(boxes) >= max(2, len(frame_tracks) // 2)]
    if not candidates:
        return None
    boxes = max(candidates, key=lambda boxes: len(boxes) * median(
        (box[2] - box[0]) * (box[3] - box[1]) for box in boxes))
    x1, y1, x2, y2 = [median(box[index] for box in boxes) for index in range(4)]
    width, height = x2 - x1, y2 - y1
    if width <= .02 or height <= .05:
        return None
    return max(0., x1 - width * .65), max(0., y1 + height * .2), \
        min(1., x2 + width * .65), min(1., y2 + height * .15)


def spatial_candidates(tracks, timestamps, thresholds=None):
    """Keep only repeated spatial detections at this window's exact source times."""
    indices = {timestamp: index for index, timestamp in enumerate(timestamps)}
    priority = {"cash": 0, "identity_document": 1, "hand": 2, "money_counter": 3}
    candidates = []
    for track in tracks:
        if track.object_class not in priority:
            continue
        samples = [sample for sample in track.samples if sample.t in indices and sample.score is not None
                   and sample.score >= (thresholds or {}).get(track.object_class, .35)]
        if len(samples) >= 2:
            candidates.append((track, samples))
    candidates.sort(key=lambda item: (priority[item[0].object_class], -len(item[1])))
    return candidates[:10]


def spatial_bindings(tracks, timestamps, thresholds=None):
    objects = [(track, samples) for track, samples in spatial_candidates(tracks, timestamps, thresholds)
               if track.object_class in {"cash", "identity_document"}][:6]
    return {f"c{index + 1}": (track, {sample.t: sample.bbox for sample in samples})
            for index, (track, samples) in enumerate(objects)}


def spatial_context(tracks, timestamps, thresholds=None):
    """Bounded detector hints and a crop derived only from this window's samples."""
    indices = {timestamp: index for index, timestamp in enumerate(timestamps)}
    candidates = spatial_candidates(tracks, timestamps, thresholds)
    ids = {track.track_id: local_id for local_id, (track, _) in spatial_bindings(tracks, timestamps, thresholds).items()}
    boxes = [sample.bbox for _, samples in candidates for sample in samples]
    detail = None
    if boxes:
        x1, y1 = min(box[0] for box in boxes), min(box[1] for box in boxes)
        x2, y2 = max(box[2] for box in boxes), max(box[3] for box in boxes)
        if .015 < x2 - x1 < .95 and .015 < y2 - y1 < .65:
            detail = max(0., x1 - .05), max(0., y1 - .04), min(1., x2 + .05), min(1., y2 + .04)
    hints = []
    for track, samples in candidates:
        selected = [samples[round(index * (len(samples) - 1) / 3)] for index in range(4)]
        selected = {sample.t: sample for sample in selected}
        hints.append({"kind": track.object_class, **({"id": ids[track.track_id]} if track.track_id in ids else {}),
            "samples": [{"frame": indices[sample.t],
            "box": [round(value * 1000) for value in sample.bbox]} for sample in selected.values()]})
    text = ("\nSpatial detector candidates in FULL-scene coordinates (0..1000), not verified facts: " +
            json.dumps(hints, separators=(",", ":")) +
            ". Inspect the pixels to accept or reject each candidate. A hand overlapping money "
            "does not by itself prove holding or handover. Follow the same notes over time; compare "
            "money on the desk, held in one or both hands, and entering or leaving the counting machine. "
            "Use provided c-IDs in events when visually confirmed. You may omit their repeated boxes "
            "from the objects array; their source boxes are already retained. Give new objects unused c-IDs. "
            "A machine's presence alone is not a counting action. Return only visibly supported events.\n") if hints else ""
    return detail, text


def prepare_qwen_inputs(processor, frames, frame_tracks, *, variant, max_pixels, mode,
                        detail=None, object_hints=""):
    """Preserve source PTS and bound the native video processor's total pixel budget."""
    import numpy as np

    images, content = visual_request(frames, frame_tracks, variant)
    content[-1]["text"] += object_hints
    if mode == "images":
        prompt = processor.apply_chat_template([{"role": "user", "content": content}],
                                               add_generation_prompt=True, tokenize=False)
        inputs = processor(images=images, text=prompt, return_tensors="pt",
                           min_pixels=65536, max_pixels=max_pixels)
    elif mode == "video":
        relative = [float(timestamp - frames[0][0]) for timestamp, _ in frames]
        if len(frames) < 2 or any(b <= a for a, b in zip(relative, relative[1:])):
            raise ValueError("Video inputs need increasing source timestamps")
        # A millisecond clock expresses the exact already-sampled source PTS.
        # These indices are metadata for the timebase, not a new frame sampler.
        clock_indices = [round(timestamp * 1000) for timestamp in relative]
        mapping = ", ".join(f"{index}={timestamp:.3f}s" for index, timestamp in enumerate(relative))
        text = ("The video contains the ordered source frames below. Video timestamps are relative "
                "to this window. Return start_frame/end_frame and supporting frames as INTEGER "
                "indices from this explicit index-to-time mapping, not seconds: " + mapping + ".\n" +
                content[-1]["text"])
        videos = [np.stack(images)]
        metadata = [{"fps": 1000.0, "total_num_frames": clock_indices[-1] + 1,
                     "frames_indices": clock_indices}]
        content = [{"type": "text", "text": "Full scene, view=full:\n"}, {"type": "video"}]
        if detail is not None:
            from PIL import Image

            crops = []
            for pixels in images:
                source = Image.fromarray(pixels)
                width, height = source.size
                crop = source.crop(tuple(round(value * size) for value, size in
                    zip(detail, (width, height, width, height))))
                crops.append(np.asarray(crop.resize((crop.width * 2, crop.height * 2), Image.Resampling.BICUBIC)))
            videos.append(np.stack(crops))
            metadata.append(dict(metadata[0]))
            content.extend([{"type": "text", "text": "\nDetail view of the SAME frames and times, view=detail:\n"},
                            {"type": "video"}])
            text += ("\nThe second video is a magnified region from the FIRST video, not another camera "
                     "or additional people. Compare the full scene for roles and simultaneity; inspect "
                     "the detail for hands, banknotes and cards. Each object sample must add view=full "
                     "or view=detail; box coordinates are relative to that selected view. Frame indices "
                     "are identical in both videos. Never count people in the two views twice.\n")
        content.append({"type": "text", "text": text})
        prompt = processor.apply_chat_template([{"role": "user", "content": content}],
                                               add_generation_prompt=True, tokenize=False)
        inputs = processor(videos=videos, text=prompt, return_tensors="pt",
            videos_kwargs={"do_sample_frames": False,
                "video_metadata": metadata,
                # Qwen3-VL video size is a TOTAL pixel budget across the clip.
                "size": {"shortest_edge": len(images) * 65536,
                         "longest_edge": len(images) * max_pixels}})
    else:
        raise ValueError("Unsupported Qwen input mode")
    return inputs, images, prompt


class QwenTemporalVerifier(TemporalVerifier):
    @staticmethod
    def model_class():
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration

    def __init__(self, directory, entry, device):
        self._capture_count = 0
        self.detail_view = os.getenv("NAVIDU_QWEN_DETAIL_VIEW", "0") == "1"
        self.input_mode = os.getenv("NAVIDU_QWEN_INPUT_MODE", "images")
        if self.input_mode not in {"images", "video"}:
            raise RuntimeError("INVALID_CONFIG: unsupported Qwen input mode.")
        if self.detail_view and self.input_mode != "video":
            raise RuntimeError("INVALID_CONFIG: detail view requires native video input.")
        self.max_window_frames = 64 if self.input_mode == "video" else 16
        self.max_pixels = int(os.getenv("NAVIDU_QWEN_MAX_PIXELS", "451584"))
        if not 65536 <= self.max_pixels <= 1048576:
            raise RuntimeError("INVALID_CONFIG: Qwen image pixel budget is outside the supported range.")
        super().__init__(directory, entry, device)
        self.prompt_version += f"-qwen-{self.input_mode}"

    def _infer(self, frames: list[tuple], frame_tracks: list[dict], should_stop: Callable[[], bool]) -> dict:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        if self.proposal is not None:
            self.proposal.disable()

        class Cancelled(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return should_stop()

        object_detail, object_hints = spatial_context(getattr(self, "_spatial_tracks", []),
                                                     [timestamp for timestamp, _ in frames],
                                                     getattr(self, "spatial_thresholds", None))
        self._object_bindings = spatial_bindings(getattr(self, "_spatial_tracks", []),
                                                [timestamp for timestamp, _ in frames],
                                                getattr(self, "spatial_thresholds", None))
        detail = (object_detail or detail_bounds(frame_tracks)) if self.detail_view else None
        inputs, images, prompt = prepare_qwen_inputs(self.processor, frames, frame_tracks,
            variant=self.prompt_variant, max_pixels=self.max_pixels, mode=self.input_mode, detail=detail,
            object_hints=object_hints)
        capture = None
        capture_root = os.getenv("NAVIDU_MODEL_INPUT_CAPTURE_DIR")
        if capture_root and self._capture_count < 4:
            import json

            from PIL import Image

            self._capture_count += 1
            capture = Path(capture_root) / f"window-{self._capture_count:02d}"
            capture.mkdir(parents=True, exist_ok=False)
            for index, pixels in enumerate(images):
                Image.fromarray(pixels).save(capture / f"frame-{index:02d}.png")
            (capture / "prompt.txt").write_text(prompt, encoding="utf-8")
            (capture / "input.json").write_text(json.dumps({
                "source_timestamps": [float(item[0]) for item in frames],
                "input_mode": self.input_mode, "max_pixels_per_frame": self.max_pixels,
                "input_tokens": inputs["input_ids"].shape[1],
                "detail_bounds": detail,
                "note": "Marked source frames before native aspect-preserving resize; private user review only."
            }, indent=2) + "\n", encoding="utf-8")
        # BatchFeature.to preserves integer token IDs and grid coordinates.
        inputs = inputs.to(self.device, dtype=self.dtype)
        if should_stop():
            raise RuntimeError("ANALYSIS_CANCELLED")
        options = {}
        known_ids = sorted({track for people in frame_tracks for track in people})
        if self.structured_output and known_ids:
            from .structured_output import build_prefix, build_schema

            options["prefix_allowed_tokens_fn"] = build_prefix(
                self.processor.tokenizer, build_schema(len(frames), known_ids, self.prompt_variant),
                tokenizer_data=self._structured_tokenizer_data,
            )
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, num_beams=1,
                max_new_tokens=self.max_new_tokens, use_cache=True, max_time=120.0,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([Cancelled()]), **options)
        if self.device == "cuda":
            torch.cuda.synchronize()
        if should_stop():
            raise RuntimeError("ANALYSIS_CANCELLED")
        input_count = inputs["input_ids"].shape[1]
        count = output.shape[1] - input_count
        self.last_metrics = {"inference_sec": round(time.perf_counter() - started, 4),
            "input_tokens": input_count, "image_count": len(images), "max_pixels": self.max_pixels,
            "input_mode": self.input_mode,
            "detail_view": detail is not None,
            "structured_output_active": bool(options), "generated_tokens": count,
            "max_new_tokens": self.max_new_tokens, "token_limit_reached": count >= self.max_new_tokens,
            "peak_cuda_mib": round(torch.cuda.max_memory_allocated() / 1048576, 1)
                             if self.device == "cuda" else None}
        text = self.processor.decode(output[0, input_count:], skip_special_tokens=True)
        if capture is not None:
            (capture / "model-response.txt").write_text(text, encoding="utf-8")
        try:
            payload = _decode_payload(text)
        except TemporalOutputError as error:
            error.numeric_metrics.update(self.last_metrics)
            raise
        if detail is not None:
            payload["_detail_bounds"] = detail
        self.last_metrics.update({"raw_event_count": len(payload["events"]),
            "raw_role_count": len(payload.get("roles", [])),
            "raw_cash_batch_count": len(payload.get("same_cash_batches", []))})
        return payload
