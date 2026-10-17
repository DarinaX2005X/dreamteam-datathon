"""Cache identity follows the meaning of a loaded pipeline, not a fixed semver."""

from navidu.contracts.pipeline import PipelineConfig
from navidu.perception.configuration import configuration_snapshot, effective_version
from navidu.perception.validation import validation_binding


def _version(config, *, semantic=False, model_hash="a" * 64):
    snapshot = configuration_snapshot(config, [{"id": "test", "files": [
        {"path": "model.onnx", "size_bytes": 10, "sha256": model_hash}
    ]}], require_semantic=semantic)
    return effective_version(snapshot, devices={"detector": "cpu", "semantic": "disabled"})


def test_effective_version_is_stable_when_worker_binds_it_to_job(tmp_path):
    config = PipelineConfig(check_id="0123456789ab", rules_dir=tmp_path)
    version = _version(config)
    assert _version(config.model_copy(update={"pipeline_version": version})) == version
    assert _version(config.model_copy(update={"check_id": "abcdef012345"})) == version


def test_model_sampling_and_semantic_changes_invalidate_cache(tmp_path):
    config = PipelineConfig(check_id="0123456789ab", rules_dir=tmp_path)
    original = _version(config)
    assert _version(config, semantic=True) != original
    assert _version(config, model_hash="b" * 64) != original
    assert _version(config.model_copy(update={"analysis_fps": 1})) != original
    assert _version(config.model_copy(update={"diagnostic_only": True})) != original
    assert _version(config.model_copy(update={"experimental_analysis": True})) != original


def test_rule_content_changes_invalidate_cache_even_with_same_size(tmp_path):
    rule = tmp_path / "rule.yaml"
    rule.write_text("threshold: 1", encoding="utf-8")
    config = PipelineConfig(check_id="0123456789ab", rules_dir=tmp_path)
    original = _version(config)
    rule.write_text("threshold: 2", encoding="utf-8")
    assert _version(config) != original


def test_publication_modes_share_quality_binding_but_not_cached_results(tmp_path):
    config = PipelineConfig(check_id="0123456789ab", rules_dir=tmp_path)
    devices = {"detector": "cpu", "semantic": "cuda", "semantic_dtype": "bf16"}
    snapshots = [configuration_snapshot(config.model_copy(update=flags), [], require_semantic=True)
                 for flags in ({}, {"experimental_analysis": True}, {"diagnostic_only": True})]
    assert len({validation_binding(snapshot, devices=devices) for snapshot in snapshots}) == 1
    assert len({effective_version(snapshot, devices=devices) for snapshot in snapshots}) == 3


def test_threshold_and_generation_options_are_part_of_cache_identity(tmp_path, monkeypatch):
    config = PipelineConfig(check_id="0123456789ab", rules_dir=tmp_path)
    monkeypatch.setenv("NAVIDU_DETECTION_THRESHOLD", "0.45")
    monkeypatch.setenv("NAVIDU_SEMANTIC_MAX_NEW_TOKENS", "512")
    original = _version(config)
    monkeypatch.setenv("NAVIDU_DETECTION_THRESHOLD", "0.6")
    assert _version(config) != original
    monkeypatch.setenv("NAVIDU_DETECTION_THRESHOLD", "0.45")
    monkeypatch.setenv("NAVIDU_SEMANTIC_MAX_NEW_TOKENS", "256")
    assert _version(config) != original


def test_requested_precision_invalidates_cache_and_quality_binding(tmp_path, monkeypatch):
    config = PipelineConfig(check_id="0123456789ab", rules_dir=tmp_path)
    monkeypatch.setenv("NAVIDU_SEMANTIC_DTYPE", "auto")
    initial = configuration_snapshot(config, [], require_semantic=True)
    monkeypatch.setenv("NAVIDU_SEMANTIC_DTYPE", "bf16")
    changed = configuration_snapshot(config, [], require_semantic=True)
    devices = {"detector": "cpu", "semantic": "cuda", "semantic_dtype": "bf16"}
    assert effective_version(initial, devices=devices) != effective_version(changed, devices=devices)
    assert validation_binding(initial, devices=devices) != validation_binding(changed, devices=devices)


def test_effective_precision_is_part_of_loaded_cache_and_quality_identity():
    snapshot = {"base_version": "test", "options": {"NAVIDU_SEMANTIC_DTYPE": "auto"}}
    fp16 = {"detector": "cpu", "semantic": "cuda", "semantic_dtype": "fp16"}
    bf16 = {**fp16, "semantic_dtype": "bf16"}
    assert effective_version(snapshot, devices=fp16) != effective_version(snapshot, devices=bf16)
    assert validation_binding(snapshot, devices=fp16) != validation_binding(snapshot, devices=bf16)


def test_opt_in_adapter_and_task_metadata_require_their_own_quality_profile(tmp_path, monkeypatch):
    config = PipelineConfig(check_id="0123456789ab", rules_dir=tmp_path)
    monkeypatch.delenv("NAVIDU_PROPOSAL_ADAPTER", raising=False)
    base = configuration_snapshot(config, [], require_semantic=True)
    monkeypatch.setenv("NAVIDU_PROPOSAL_ADAPTER", "trained")
    selected = configuration_snapshot(config, [], require_semantic=True)
    assert validation_binding(base) != validation_binding(selected)
    entry = {"id": "trained", "files": [{"path": "adapter.safetensors", "size_bytes": 4, "sha256": "a" * 64}],
             "base_model": {"id": "base", "sha256": "b" * 64}, "task": {"fps": 2.}}
    first = configuration_snapshot(config, [entry], require_semantic=True)
    changed = configuration_snapshot(config, [{**entry, "task": {"fps": 1.}}], require_semantic=True)
    assert validation_binding(first) != validation_binding(changed)
