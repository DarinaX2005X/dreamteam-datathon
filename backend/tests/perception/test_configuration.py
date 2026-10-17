"""Cache identity follows the meaning of a loaded pipeline, not a fixed semver."""

from navidu.contracts.pipeline import PipelineConfig
from navidu.perception.configuration import configuration_snapshot, effective_version


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


def test_rule_content_changes_invalidate_cache_even_with_same_size(tmp_path):
    rule = tmp_path / "rule.yaml"
    rule.write_text("threshold: 1", encoding="utf-8")
    config = PipelineConfig(check_id="0123456789ab", rules_dir=tmp_path)
    original = _version(config)
    rule.write_text("threshold: 2", encoding="utf-8")
    assert _version(config) != original


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
