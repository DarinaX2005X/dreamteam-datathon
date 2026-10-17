"""Exercise real decoding/evidence with typed test perception, not model quality."""

import base64
import importlib.util
import io
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from navidu.contracts.api_v1 import JobDone
from navidu.contracts.domain import Observation, Track, TrackSample
from navidu.contracts.pipeline import RuntimeStatus
from navidu.pipeline.interfaces import PerceptionArtifacts
from navidu.pipeline.run import RealPipeline
from navidu.service.app import create_app
from navidu.service.worker import Worker
from PIL import Image

from .conftest import upload

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or importlib.util.find_spec("av") is None
    or importlib.util.find_spec("numpy") is None,
    reason="ffmpeg, PyAV and NumPy required for real evidence integration",
)


class TypedPerception:
    def __init__(self):
        self.calls = 0

    def prepare(self, config=None):
        pass

    def runtime_status(self, config=None):
        return RuntimeStatus(model_status="ready", device="cpu", pipeline_version="integration-v1",
                             analysis_ready=True, analysis_mode="validated")

    def perceive(self, frames, *, config, emit, should_stop):
        self.calls += 1
        timestamps = [timestamp for timestamp, rgb in frames]
        assert timestamps and not should_stop()
        people = [
            Track(track_id=identifier, kind="person", first_sec=0, last_sec=last,
                  samples=[TrackSample(t=t, bbox=box, score=0.9) for t in timestamps if t <= last])
            for identifier, last, box in [
                ("p1", timestamps[-1], (0.1, 0.2, 0.35, 0.9)),
                ("p2", 0.5, (0.65, 0.2, 0.9, 0.9)),
            ]
        ]
        desk = Track(track_id="o1", kind="object", object_class="cash_desk",
                     first_sec=0, last_sec=timestamps[-1], samples=[
                         TrackSample(t=t, bbox=(0.25, 0.55, 0.75, 0.9), score=0.85)
                         for t in timestamps
                     ])
        return PerceptionArtifacts(
            tracks=people + [desk],
            observations=[Observation(obs_id="ob-1", kind="personal_item_identified",
                start_sec=1.0, end_sec=1.2, subjects=["p1"], score=None,
                provenance="temporal_model", model_id="typed-integration-test")],
            model_versions={"test_perception": "typed-input-only"},
            capabilities={"personal_item_semantics": True},
            semantic_windows_completed=1,
        )


class PreparedPipeline:
    def __init__(self):
        self.runtime = TypedPerception()
        self.engine = RealPipeline(self.runtime)

    def prepare(self):
        self.runtime.prepare()

    def runtime_status(self):
        return self.runtime.runtime_status()

    def analyze(self, *args, **kwargs):
        return self.engine.analyze(*args, **kwargs)

    def reevaluate(self, *args, **kwargs):
        return self.engine.reevaluate(*args, **kwargs)


def assert_jpeg(raw):
    with Image.open(io.BytesIO(raw)) as image:
        assert image.format == "JPEG" and image.size == (96, 64)
        image.verify()


def test_real_engine_media_publication_cache_and_review(settings, tmp_path):
    source = tmp_path / "source.mov"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=s=96x64:d=2:r=10", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(source)], check=True)
    settings = replace(settings, rules_dir=Path(__file__).parents[3] / "rules",
                       analysis_fps=4, lease_seconds=30)
    pipeline = PreparedPipeline()
    app = create_app(settings, pipeline=pipeline)
    worker = Worker(settings, pipeline=pipeline)
    assert worker.prepare()
    with TestClient(app) as client:
        original = upload(client, api=True, payload=source.read_bytes()).json()["job_id"]
        assert worker.run_once()
        headers = {"X-API-Key": "test-secret-key"}
        jury = JobDone.model_validate(client.get(f"/api/v1/jobs/{original}", headers=headers).json())
        assert jury.verdict == "violation" and jury.violations
        for violation in jury.violations:
            for frame in violation.evidence_frames:
                assert_jpeg(base64.b64decode(frame.image_base64))
        detail = client.get(f"/api/ui/checks/{original}").json()["check"]
        assert any(finding["rule_id"] == "V-PERSONAL-ITEMS" for finding in detail["finding_set"]["findings"])
        frame_id = detail["finding_set"]["findings"][0]["evidence"][0]["frame_id"]
        for overlay in [0, 1]:
            response = client.get(f"/api/ui/checks/{original}/frames/{frame_id}.jpg?revision=1&overlay={overlay}")
            assert response.status_code == 200
            assert_jpeg(response.content)

        duplicate = upload(client, payload=source.read_bytes()).json()["check"]["id"]
        assert worker.run_once() and pipeline.runtime.calls == 1
        duplicate_detail = client.get(f"/api/ui/checks/{duplicate}").json()["check"]
        assert duplicate_detail["analysis_origin"] == "cache"
        directory = settings.checks_dir / duplicate
        manifest = json.loads((directory / "engine_manifest.json").read_text())
        assert Path(manifest["source_path"]).parent == directory
        original_row = app.state.store.get(original)
        (settings.checks_dir / original / original_row["video_file"]).unlink()
        before = {str(path.relative_to(directory)): path.read_bytes()
                  for path in directory.rglob("*") if path.is_file()}
        automatic_jury = client.get(f"/api/v1/jobs/{duplicate}", headers=headers).json()
        request = {"base_revision": 1, "role_overrides": [{"track_id": "p2", "role": "collector"}],
                   "zone_overrides": [], "dry_run": True}
        preview = client.post(f"/api/ui/checks/{duplicate}/reevaluate", json=request)
        assert preview.status_code == 200 and preview.json()["preview_frames"]
        for image in preview.json()["preview_frames"].values():
            assert_jpeg(base64.b64decode(image["image_base64"]))
            assert_jpeg(base64.b64decode(image["overlay_base64"]))
        assert {str(path.relative_to(directory)): path.read_bytes()
                for path in directory.rglob("*") if path.is_file()} == before

        request["dry_run"] = False
        committed = client.post(f"/api/ui/checks/{duplicate}/reevaluate", json=request)
        assert committed.status_code == 200 and committed.json()["finding_set"]["revision"] == 2
        reviewed = client.get(f"/api/ui/checks/{duplicate}").json()["check"]
        assert next(track for track in reviewed["tracks"] if track["track_id"] == "p2")["role"] == {
            "value": "collector", "source": "reviewer",
        }
        for finding in committed.json()["finding_set"]["findings"]:
            for frame in finding["evidence"]:
                media = client.get(f"/api/ui/checks/{duplicate}/frames/{frame['frame_id']}.jpg?revision=2")
                assert media.status_code == 200
                assert_jpeg(media.content)
        assert pipeline.runtime.calls == 1
        assert client.get(f"/api/v1/jobs/{duplicate}", headers=headers).json() == automatic_jury
        assert all((directory / name).read_bytes() == raw for name, raw in before.items())
