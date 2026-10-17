import importlib.util
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image

from navidu.contracts.domain import (
    Decision,
    Observation,
    RoleOverride,
    Track,
    TrackSample,
)
from navidu.contracts.pipeline import PipelineConfig, RuntimeStatus
from navidu.pipeline.interfaces import PerceptionArtifacts
from navidu.pipeline.run import PipelineCancelled, RealPipeline

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or importlib.util.find_spec("av") is None
    or importlib.util.find_spec("numpy") is None,
    reason="ffmpeg, PyAV and NumPy required",
)


class FakeRuntime:
    def __init__(self):
        self.calls = 0

    def prepare(self, config):
        pass

    def runtime_status(self, config):
        return RuntimeStatus(model_status="ready", device="cpu", pipeline_version="test")

    def perceive(self, frames, *, config, emit, should_stop):
        self.calls += 1
        timestamps = [timestamp for timestamp, _ in frames]
        assert timestamps
        def person(track_id, first, last, box):
            return Track(
                track_id=track_id, kind="person", first_sec=first, last_sec=last,
                samples=[TrackSample(t=t, bbox=box, score=0.9)
                         for t in timestamps if first <= t <= last],
            )
        tracks = [
            person("p1", 0, timestamps[-1], (0.1, 0.2, 0.35, 0.9)),
            person("p2", 0, 0.4, (0.65, 0.2, 0.9, 0.9)),
            Track(track_id="o1", kind="object", object_class="cash_desk",
                  first_sec=0, last_sec=timestamps[-1],
                  samples=[TrackSample(t=t, bbox=(0.25, 0.55, 0.75, 0.9), score=0.85)
                           for t in timestamps]),
        ]
        observations = [
            Observation(obs_id="ob-1", kind="cash_handover", start_sec=0.1, end_sec=0.2,
                        subjects=["p1", "p2", "o1"], actor_id="p2", recipient_id="p1",
                        score=None, provenance="temporal_model", model_id="semantic-test"),
            Observation(obs_id="ob-2", kind="counting_motion", start_sec=0.6, end_sec=1.0,
                        subjects=["p1", "o1"], score=None,
                        provenance="temporal_model", model_id="semantic-test"),
            Observation(obs_id="ob-3", kind="personal_item_identified", start_sec=1.1,
                        end_sec=1.3, subjects=["p1"], score=None,
                        provenance="temporal_model", model_id="semantic-test"),
        ]
        return PerceptionArtifacts(tracks=tracks, observations=observations,
            model_versions={"fake": "test"}, capabilities={
                "handover_direction": True, "counting_action": True,
                "continuous_client_visibility": True,
                "continuous_cashier_visibility": True,
                "continuous_other_positions_visibility": True,
                "personal_item_semantics": True,
            })


def _video(path):
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=black:s=96x64:d=2:r=10", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(path)], check=True)


def test_analyze_and_reevaluate_reuse_perception_and_keep_surviving_decision(tmp_path):
    video = tmp_path / "source.mov"
    _video(video)
    runtime = FakeRuntime()
    pipeline = RealPipeline(runtime)
    # Use the repository rules path without relying on the test process cwd layout.
    config = PipelineConfig(check_id="0123456789ab", analysis_fps=4,
                            rules_dir=Path(__file__).parents[3] / "rules")
    artifacts = pipeline.analyze(video, work_dir=tmp_path / "work", config=config,
                                 emit=lambda event: None, should_stop=lambda: False)
    by_rule = {finding.rule_id: finding for finding in artifacts.finding_set.findings}
    assert {"V-COUNT-ABSENT", "V-PERSONAL-ITEMS"} <= set(by_rule)
    assert by_rule["V-COUNT-ABSENT"].timecode_sec == \
        by_rule["V-COUNT-ABSENT"].evidence[0].timestamp_sec
    base = artifacts.finding_set.model_copy(deep=True)
    personal = next(item for item in base.findings if item.rule_id == "V-PERSONAL-ITEMS")
    personal.decision = Decision(value="confirmed", decided_at=datetime.now(timezone.utc))
    reevaluated = pipeline.reevaluate(artifacts,
        role_overrides=[RoleOverride(track_id="p2", role="collector")],
        zone_overrides=[], base=base)
    assert runtime.calls == 1
    assert not any(item.rule_id == "V-COUNT-ABSENT" for item in reevaluated.findings)
    survived = next(item for item in reevaluated.findings if item.rule_id == "V-PERSONAL-ITEMS")
    assert survived.finding_id == personal.finding_id
    assert survived.decision == personal.decision
    assert artifacts.finding_set.revision == 1
    assert reevaluated.revision == 2
    assert (artifacts.work_dir / "finding_set.json").is_file()
    assert (artifacts.work_dir / "finding_set.rev2.json").is_file()
    evidence_id = by_rule["V-COUNT-ABSENT"].evidence[0].frame_id
    for variant in ("original", "overlay"):
        path = artifacts.work_dir / "evidence" / variant / f"{evidence_id}.jpg"
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            assert max(image.size) <= 1280


def test_cancel_before_start_publishes_no_artifacts(tmp_path):
    video = tmp_path / "source.mov"
    _video(video)
    work = tmp_path / "work"
    with pytest.raises(PipelineCancelled):
        RealPipeline(FakeRuntime()).analyze(video, work_dir=work,
            config=PipelineConfig(check_id="0123456789ab"),
            emit=lambda event: None, should_stop=lambda: True)
    assert not work.exists()
