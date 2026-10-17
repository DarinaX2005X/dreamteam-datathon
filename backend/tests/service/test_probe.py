import shutil
import subprocess
from dataclasses import replace

import pytest
from navidu.service.exceptions import ServiceError
from navidu.service.uploads import probe_video


def test_ffprobe_rejects_disguised_text(settings, tmp_path):
    if not shutil.which(settings.ffprobe):
        pytest.skip("ffprobe is not installed")
    path = tmp_path / "looks-like-video.mov"
    path.write_text("This is not a video.")
    with pytest.raises(ServiceError) as error:
        probe_video(path, settings)
    assert error.value.code == "UNSUPPORTED_MEDIA"


def test_ffprobe_real_container_and_duration_limit(settings, tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which(settings.ffprobe):
        pytest.skip("ffmpeg is not installed")
    path = tmp_path / "generated-test.mov"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=c=black:s=64x36:r=25:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(path)], check=True, timeout=30)
    meta = probe_video(path, settings)
    assert meta.duration_sec == 1 and meta.width == 64 and meta.height == 36
    with pytest.raises(ServiceError) as error:
        probe_video(path, replace(settings, max_duration_sec=0.5))
    assert error.value.code == "VIDEO_TOO_LONG"
