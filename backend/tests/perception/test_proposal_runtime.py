"""Learned proposals may select context, never become facts or alter the verifier."""

import json
import math
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from navidu.contracts.pipeline import PipelineConfig
from navidu.perception.assets import ModelAssetsError
from navidu.perception.proposal import (
    TASK,
    ProposalVerifier,
    refinement_window,
    task_for_model,
    validate_adapter,
)
from navidu.perception.proposal_task import LORA_TARGET_MODULES, Proposal
from navidu.perception.runtime import LocalPerceptionRuntime, _selected_ids


def _adapter(tmp_path):
    directory = tmp_path / "proposal"
    directory.mkdir()
    config = {"peft_type": "LORA", "target_modules": LORA_TARGET_MODULES, "bias": "none",
              "r": 8, "lora_alpha": 16, "lora_dropout": .05}
    (directory / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "adapter_model.safetensors").write_bytes(b"synthetic-test-placeholder")
    base = {"id": "base", "revision": "pinned"}
    entry = {"id": "proposal", "runtime": "peft-lora-proposal", "local_dir": "proposal",
             "task": dict(TASK), "base_model": {**base, "sha256": "a" * 64},
             "files": [{"path": "proposal/adapter_config.json"},
                       {"path": "proposal/adapter_model.safetensors"}]}
    return entry, base, config


def test_adapter_requires_exact_base_hash_and_shared_task(tmp_path):
    entry, base, _ = _adapter(tmp_path)
    assert validate_adapter(tmp_path, entry, base, "a" * 64) == tmp_path / "proposal"
    with pytest.raises(ModelAssetsError, match="MODEL_ADAPTER_INVALID"):
        validate_adapter(tmp_path, entry, base, "b" * 64)
    entry["task"]["fps"] = 1
    with pytest.raises(ModelAssetsError, match="MODEL_ADAPTER_INVALID"):
        validate_adapter(tmp_path, entry, base, "a" * 64)


def test_qwen_adapter_binds_exact_base_and_training_pixel_budget(tmp_path):
    entry, base, _ = _adapter(tmp_path)
    base["id"] = "qwen3-vl-4b"
    entry["base_model"]["id"] = base["id"]
    with pytest.raises(ModelAssetsError, match="MODEL_ADAPTER_INVALID"):
        validate_adapter(tmp_path, entry, base, "a" * 64)
    entry["task"] = task_for_model(base["id"])
    assert validate_adapter(tmp_path, entry, base, "a" * 64).name == "proposal"
    entry["task"]["max_pixels"] = 451584
    with pytest.raises(ModelAssetsError, match="MODEL_ADAPTER_INVALID"):
        validate_adapter(tmp_path, entry, base, "a" * 64)


@pytest.mark.parametrize("changes", [
    {"target_modules": "all-linear"}, {"bias": "all"}, {"modules_to_save": ["lm_head"]},
    {"r": 16}, {"use_dora": True}, {"rank_pattern": {"q_proj": 32}},
    {"trainable_token_indices": [1]}, {"task_type": "CAUSAL_LM"},
])
def test_adapter_cannot_modify_other_base_weights(tmp_path, changes):
    entry, base, config = _adapter(tmp_path)
    (tmp_path / "proposal/adapter_config.json").write_text(json.dumps({**config, **changes}), encoding="utf-8")
    with pytest.raises(ModelAssetsError, match="MODEL_ADAPTER_INVALID"):
        validate_adapter(tmp_path, entry, base, "a" * 64)


def test_adapter_selection_is_explicit_and_requires_semantics(monkeypatch):
    monkeypatch.delenv("NAVIDU_PROPOSAL_ADAPTER", raising=False)
    assert _selected_ids("detector", "base", True) == ["detector", "base"]
    monkeypatch.setenv("NAVIDU_PROPOSAL_ADAPTER", "trained")
    assert _selected_ids("detector", "base", True) == ["detector", "base", "trained"]
    with pytest.raises(ModelAssetsError, match="INVALID_CONFIG"):
        _selected_ids("detector", "base", False)


def test_adapter_loading_wraps_the_existing_base_and_starts_disabled(tmp_path, monkeypatch):
    entry, base_entry, _ = _adapter(tmp_path)
    seen = {}
    wrapper = SimpleNamespace(base_model=SimpleNamespace(disable_adapter_layers=lambda: seen.update(disabled=True)))
    wrapper.eval = lambda: wrapper

    def load(model, path, **kwargs):
        seen.update(base=model, path=path, **kwargs)
        return wrapper

    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=load)))
    base = SimpleNamespace(model=object())
    adapter = ProposalVerifier(base, tmp_path, entry, base_entry, "a" * 64)
    assert seen["base"] is base.model
    assert seen["is_trainable"] is False and seen["local_files_only"] is True
    assert seen["disabled"] is True
    assert adapter.base is base


@pytest.mark.parametrize("failure", [None, "generation", "decode", "cancel"])
def test_adapter_is_disabled_even_when_generation_or_decoding_fails(monkeypatch, failure):
    import navidu.perception.proposal as module

    states = []
    layers = SimpleNamespace(enable_adapter_layers=lambda: states.append("enabled"),
                             disable_adapter_layers=lambda: states.append("disabled"))

    class Batch(dict):
        def to(self, *args, **kwargs):
            return self

    class Generated:
        shape = (1, 5)

        def __getitem__(self, _):
            return [9]

    def generate(**kwargs):
        assert states[-1] == "enabled"
        if failure == "generation":
            raise RuntimeError("synthetic failure")
        return Generated()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(StoppingCriteria=object, StoppingCriteriaList=list))
    monkeypatch.setattr(module, "encode_proposal", lambda *args: Batch(input_ids=SimpleNamespace(shape=(1, 4))))
    adapter = object.__new__(ProposalVerifier)
    adapter.model = SimpleNamespace(base_model=layers, generate=generate)
    adapter.base = SimpleNamespace(device="cpu", dtype="fp32", processor=SimpleNamespace(
        tokenizer=SimpleNamespace(eos_token_id=2),
        decode=lambda *args, **kwargs: "invalid" if failure == "decode" else '{"candidate":1,"frame":1}'))
    stop_checks = iter([False, failure == "cancel"])
    if failure:
        with pytest.raises(RuntimeError):
            adapter.propose([(0., None), (.5, None)], should_stop=lambda: next(stop_checks))
    else:
        assert adapter.propose([(0., None), (.5, None)]).time_sec == .5
    assert states == ["enabled", "disabled"]


def test_dense_refinement_always_contains_the_inferred_frame():
    frames = [(index / 2, None) for index in range(48)]
    for center in (0., 7.5, 23.5):
        window = refinement_window(frames, center)
        assert len(window) == 16 and window[0][0] <= center <= window[-1][0]
    with pytest.raises(ValueError, match="retained source"):
        refinement_window(frames, 24.)


@pytest.mark.parametrize("positive", [False, True])
def test_proposals_only_schedule_base_verification_and_keep_baseline_tail(monkeypatch, positive):
    # Numeric fakes exercise the real streaming scheduler. No model, actual
    # image, GPU, detector package or browser is loaded by this test.
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(isfinite=math.isfinite, array=lambda value: value))
    monkeypatch.setitem(sys.modules, "supervision", SimpleNamespace())

    class ClassIds(list):
        def tolist(self):
            return list(self)

    class Image:
        shape = (32, 32, 3)

        def copy(self):
            return self

    class Proposer:
        window_frames, sample_fps, stride_frames = 16, 2., 8

        def __init__(self):
            self.windows = []

        def propose(self, frames, *, should_stop):
            self.windows.append([item[0] for item in frames])
            return Proposal(positive, len(frames) - 1 if positive else None,
                            frames[-1][0] if positive else None)

    class Semantic:
        window_frames, sample_fps = 8, 1.

        def __init__(self):
            self.proposal, self.windows = Proposer(), []
            self.last_metrics = {}

        def observe(self, frames, tracks, artifacts, *, should_stop):
            self.windows.append([item[0] for item in frames])

    runtime = LocalPerceptionRuntime()
    runtime._semantic = Semantic()
    runtime._detector = SimpleNamespace(model_id="fake", labels={},
        predict=lambda rgb: ([], ClassIds(), []))
    monkeypatch.setattr(runtime, "prepare", lambda config: None)
    events = []
    result = runtime.perceive([(index / 3, Image()) for index in range(64)],
        config=PipelineConfig(check_id="0123456789ab", analysis_fps=3),
        emit=lambda event: events.append(event), should_stop=lambda: False)
    windows = runtime._semantic.windows
    assert list(range(8)) in windows  # The ordinary sweep is always retained.
    assert any(window[-1] == 21. and len(window) < 16 for window in windows)
    assert any(len(window) == 16 for window in windows) is positive
    assert runtime._semantic.proposal.windows[0][-1] < 8.  # Grid sampling does not drift to 1.5 fps.
    assert result.observations == [] and result.semantic_signals == []
    assert result.capabilities["cash_detection"] is False
    assert result.capabilities["continuous_other_positions_visibility"] is False
    assert [event.processed_until_sec for event in events] == sorted(event.processed_until_sec for event in events)
