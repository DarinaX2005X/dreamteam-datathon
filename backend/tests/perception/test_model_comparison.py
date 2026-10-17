import pytest
from navidu.perception.qwen import QwenTemporalVerifier
from navidu.perception.temporal import prompt_profile, visual_request


def test_model_comparison_uses_local_indices_and_does_not_mutate_source_pixels():
    np = pytest.importorskip("numpy")
    pixels = np.zeros((80, 100, 3), dtype=np.uint8)
    frames = [(901.0, pixels), (902.0, pixels), (903.0, pixels)]
    tracks = [{"p1": (0.1, 0.1, 0.6, 0.8)}] * 3
    images, content = visual_request(frames, tracks)
    text = "".join(item.get("text", "") for item in content)
    assert len(images) == sum(item["type"] == "image" for item in content) == 3
    assert all(image.shape == pixels.shape and np.any(image) for image in images)
    assert not np.any(pixels)
    assert "Frame 0" in text and "Frame 2" in text and "0 through 2" in text
    assert "901" not in text and "903" not in text


def test_prompt_variants_keep_frames_and_schema_but_have_distinct_provenance():
    np = pytest.importorskip("numpy")
    frames = [(0., np.zeros((50, 50, 3), dtype=np.uint8)),
              (1., np.zeros((50, 50, 3), dtype=np.uint8))]
    old_images, old_content = visual_request(frames, [{}, {}], "v5")
    new_images, new_content = visual_request(frames, [{}, {}], "v6-evidence")
    assert all(np.array_equal(a, b) for a, b in zip(old_images, new_images))
    assert old_content[:-1] == new_content[:-1]
    assert prompt_profile("v5")[0] != prompt_profile("v6-evidence")[0]
    for field in ("events", "roles", "same_cash_batches", "reflections", "start_frame"):
        assert field in old_content[-1]["text"] and field in new_content[-1]["text"]
    with pytest.raises(RuntimeError, match="INVALID_CONFIG"):
        prompt_profile("typo")


def test_qwen_rejects_unbounded_image_budget_before_loading_weights(monkeypatch):
    monkeypatch.setenv("NAVIDU_QWEN_MAX_PIXELS", "99999999")
    with pytest.raises(RuntimeError, match="pixel budget"):
        QwenTemporalVerifier(None, {}, "cuda")
