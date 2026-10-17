"""A pinned LoRA proposes times; the unchanged base model verifies observations."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .assets import ModelAssetsError, artifact_path
from .proposal_task import LORA_TARGET_MODULES, PROMPT_VERSION, decode_proposal, encode_proposal

TASK = {"prompt_version": PROMPT_VERSION, "window_frames": 16, "fps": 2.0,
        "stride_frames": 8, "max_new_tokens": 32}


def validate_adapter(directory: Path, entry: dict, base_entry: dict, base_hash: str) -> Path:
    """Reject an adapter for another base or any recipe that changes base weights."""
    expected_base = {"id": base_entry["id"], "revision": base_entry.get("revision"),
                     "sha256": base_hash}
    if entry.get("runtime") != "peft-lora-proposal" or entry.get("base_model") != expected_base \
            or entry.get("task") != TASK:
        raise ModelAssetsError("MODEL_ADAPTER_INVALID: адаптер не соответствует базовой модели или задаче.")
    local_dir = entry.get("local_dir")
    if not isinstance(local_dir, str):
        raise ModelAssetsError("MODEL_ADAPTER_INVALID: отсутствует каталог адаптера.")
    path = artifact_path(directory, local_dir)
    expected_files = {path / "adapter_config.json", path / "adapter_model.safetensors"}
    declared_files = {artifact_path(directory, item["path"]) for item in entry["files"]}
    if not expected_files.issubset(declared_files) \
            or any(not item.is_relative_to(path) for item in declared_files):
        raise ModelAssetsError("MODEL_ADAPTER_INVALID: нужны проверяемые config и safetensors адаптера.")
    try:
        config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ModelAssetsError("MODEL_ADAPTER_INVALID: повреждена конфигурация адаптера.") from error
    if not isinstance(config, dict) or config.get("peft_type") != "LORA" \
            or config.get("target_modules") != LORA_TARGET_MODULES \
            or config.get("bias") != "none" or config.get("r") != 8 \
            or config.get("lora_alpha") != 16 or config.get("lora_dropout") != 0.05 \
            or config.get("modules_to_save") not in (None, []) \
            or config.get("task_type") is not None \
            or config.get("rank_pattern", {}) or config.get("alpha_pattern", {}) \
            or config.get("layer_replication") not in (None, []) \
            or config.get("trainable_token_indices") is not None \
            or config.get("use_dora", False) or config.get("use_rslora", False) \
            or config.get("lora_bias", False):
        raise ModelAssetsError("MODEL_ADAPTER_INVALID: разрешён только обученный q/v LoRA без изменения базовых слоёв.")
    return path


def refinement_window(frames: list[tuple], center: float, frame_count: int = 16) -> list[tuple]:
    """Choose dense context from already sampled source frames, without labels."""
    if len(frames) < 2 or not frames[0][0] <= center <= frames[-1][0]:
        raise ValueError("A proposal must refer to retained source frames")
    count = min(frame_count, len(frames))
    candidates = [index for index in range(len(frames) - count + 1)
                  if frames[index][0] <= center <= frames[index + count - 1][0]]
    first = min(candidates, key=lambda index:
                abs((frames[index][0] + frames[index + count - 1][0]) / 2 - center))
    return frames[first:first + count]


class ProposalVerifier:
    """One shared base checkpoint; adapter layers are disabled outside propose()."""

    window_frames = TASK["window_frames"]
    sample_fps = TASK["fps"]
    stride_frames = TASK["stride_frames"]

    def __init__(self, base, directory: Path, entry: dict, base_entry: dict, base_hash: str):
        path = validate_adapter(directory, entry, base_entry, base_hash)
        try:
            from peft import PeftModel
        except ImportError as error:
            raise RuntimeError("ADAPTER_DEPENDENCY_MISSING: установите зависимости backend[adapters].") from error
        self.model_id = entry["id"]
        self.base = base
        self.model = PeftModel.from_pretrained(base.model, str(path), adapter_name="proposal",
                                              is_trainable=False, local_files_only=True)
        self.model.eval()
        self.disable()
        self.last_metrics: dict = {}

    def disable(self) -> None:
        self.model.base_model.disable_adapter_layers()

    def propose(self, frames: list[tuple], *, should_stop=lambda: False):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        class Cancelled(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return should_stop()

        if should_stop():
            raise RuntimeError("ANALYSIS_CANCELLED")
        inputs = encode_proposal(self.base.processor, frames).to(self.base.device, dtype=self.base.dtype)
        started = time.perf_counter()
        self.model.base_model.enable_adapter_layers()
        try:
            with torch.inference_mode():
                output = self.model.generate(**inputs, do_sample=False, num_beams=1,
                    max_new_tokens=TASK["max_new_tokens"], use_cache=True, max_time=60.0,
                    pad_token_id=self.base.processor.tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([Cancelled()]))
        finally:
            # A malformed result, cancellation or CUDA failure must never leave
            # learned binary labels influencing the factual verification pass.
            self.disable()
        if should_stop():
            raise RuntimeError("ANALYSIS_CANCELLED")
        prompt_length = inputs["input_ids"].shape[1]
        generated = self.base.processor.decode(output[0, prompt_length:], skip_special_tokens=True)
        proposal = decode_proposal(generated, [timestamp for timestamp, _ in frames])
        self.last_metrics = {"generation_sec": round(time.perf_counter() - started, 4),
            "input_tokens": prompt_length, "generated_tokens": output.shape[1] - prompt_length,
            "candidate": proposal.candidate}
        return proposal
