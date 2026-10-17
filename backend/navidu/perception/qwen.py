"""Local Qwen3-VL candidate using the unchanged temporal evidence contract."""
from __future__ import annotations

import os
import time
from collections.abc import Callable

from .temporal import TemporalOutputError, TemporalVerifier, _decode_payload, visual_request


class QwenTemporalVerifier(TemporalVerifier):
    @staticmethod
    def model_class():
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration

    def __init__(self, directory, entry, device):
        self.max_pixels = int(os.getenv("NAVIDU_QWEN_MAX_PIXELS", "451584"))
        if not 65536 <= self.max_pixels <= 1048576:
            raise RuntimeError("INVALID_CONFIG: Qwen image pixel budget is outside the supported range.")
        super().__init__(directory, entry, device)

    def _infer(self, frames: list[tuple], frame_tracks: list[dict], should_stop: Callable[[], bool]) -> dict:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        if self.proposal is not None:
            raise RuntimeError("INVALID_CONFIG: InternVL proposal weights cannot be used with Qwen.")

        class Cancelled(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return should_stop()

        images, content = visual_request(frames, frame_tracks, self.prompt_variant)
        prompt = self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                   add_generation_prompt=True, tokenize=False)
        inputs = self.processor(images=images, text=prompt, return_tensors="pt",
                                min_pixels=65536, max_pixels=self.max_pixels)
        # BatchFeature.to preserves integer token IDs and grid coordinates.
        inputs = inputs.to(self.device, dtype=self.dtype)
        if should_stop():
            raise RuntimeError("ANALYSIS_CANCELLED")
        options = {}
        known_ids = sorted({track for people in frame_tracks for track in people})
        if self.structured_output and known_ids:
            from .structured_output import build_prefix, build_schema

            options["prefix_allowed_tokens_fn"] = build_prefix(
                self.processor.tokenizer, build_schema(len(frames), known_ids),
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
            "structured_output_active": bool(options), "generated_tokens": count,
            "max_new_tokens": self.max_new_tokens, "token_limit_reached": count >= self.max_new_tokens,
            "peak_cuda_mib": round(torch.cuda.max_memory_allocated() / 1048576, 1)
                             if self.device == "cuda" else None}
        text = self.processor.decode(output[0, input_count:], skip_special_tokens=True)
        try:
            payload = _decode_payload(text)
        except TemporalOutputError as error:
            error.numeric_metrics.update(self.last_metrics)
            raise
        self.last_metrics.update({"raw_event_count": len(payload["events"]),
            "raw_role_count": len(payload.get("roles", [])),
            "raw_cash_batch_count": len(payload.get("same_cash_batches", []))})
        return payload
