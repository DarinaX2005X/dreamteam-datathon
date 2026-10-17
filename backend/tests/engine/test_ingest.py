import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest
from navidu.ingest import MediaProbeError, frame_at, iter_frames, probe_video
from PIL import Image

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or importlib.util.find_spec("av") is None
    or importlib.util.find_spec("numpy") is None,
    reason="ffmpeg, PyAV and NumPy required",
)


def _vfr_video(tmp_path: Path) -> Path:
    Image.new("RGB", (80, 40), "red").save(tmp_path / "a.png")
    Image.new("RGB", (80, 40), "blue").save(tmp_path / "b.png")
    (tmp_path / "concat.txt").write_text(
        "file 'a.png'\nduration 0.2\nfile 'b.png'\nduration 0.9\nfile 'a.png'\n",
        encoding="utf-8",
    )
    output = tmp_path / "vfr.mov"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i",
        str(tmp_path / "concat.txt"), "-vsync", "vfr", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", str(output),
    ], check=True)
    return output


def test_vfr_sampling_uses_actual_pts_and_frame_at_returns_actual_time(tmp_path):
    path = _vfr_video(tmp_path)
    frames = list(iter_frames(path, analysis_fps=10))
    timestamps = [item.timestamp_sec for item in frames]
    assert timestamps == sorted(timestamps)
    assert any(right - left > 0.5 for left, right in zip(timestamps, timestamps[1:]))
    selected = frame_at(path, 0.5)
    assert selected.timestamp_sec >= 0.5
    assert selected.timestamp_sec in timestamps


def test_rotation_metadata_is_applied_before_inference(tmp_path):
    source = _vfr_video(tmp_path)
    rotated = tmp_path / "rotated.mov"
    subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(source),
        "-c", "copy", "-metadata:s:v:0", "rotate=90", str(rotated),
    ], check=True)
    probe = probe_video(rotated)
    assert probe.video.rotation == 90
    decoded = next(iter_frames(rotated, analysis_fps=1, probe=probe))
    assert decoded.rgb.shape[:2] == (80, 40)
    assert (probe.video.width, probe.video.height) == (40, 80)


def test_malformed_media_has_recoverable_code(tmp_path):
    broken = tmp_path / "broken.mov"
    broken.write_bytes(b"not a video")
    with pytest.raises(MediaProbeError) as caught:
        probe_video(broken)
    assert caught.value.code == "UNSUPPORTED_MEDIA"
