import json
import subprocess
import sys

from navidu.ingest import probe_video


def test_invalid_camera_metadata_bytes_do_not_abort_numeric_probe(monkeypatch, tmp_path):
    path = tmp_path / "source.mov"
    path.write_bytes(b"fixture")
    original = subprocess.run
    metadata = json.dumps({"streams": [{"codec_type": "video", "duration": "4",
        "width": 100, "height": 80, "avg_frame_rate": "25/1"}]}).encode()

    def run(command, **kwargs):
        script = "import os; os.write(1, " + repr(metadata) + "); os.write(2, b'bad tag: \\xff\\xfe')"
        return original([sys.executable, "-c", script], **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    result = probe_video(path)
    assert result.video.duration_sec == 4 and result.video.width == 100
