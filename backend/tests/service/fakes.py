"""Test-only pipeline. No environment flag can load it in the production service."""

import base64
import json
import time
from pathlib import Path

from navidu.contracts.domain import AnalysisRun, VideoMeta
from navidu.contracts.pipeline import AnalysisArtifacts, PipelineEvent
from navidu.contracts.ui_api import CheckDetailResponse
from PIL import Image, ImageDraw

FIXTURES = Path(__file__).resolve().parents[3] / "contracts" / "fixtures"


class FakePipeline:
    def __init__(self, *, delay=0, fail=False, version="test-v1", analysis_ready=True,
                 analysis_mode="validated"):
        self.calls = 0
        self.delay = delay
        self.fail = fail
        self.version = version
        self.analysis_ready = analysis_ready
        self.analysis_mode = analysis_mode

    def prepare(self):
        pass

    def runtime_status(self):
        return {"model_status": "ready", "device": "test", "limitations": ["Synthetic test pipeline"],
                "pipeline_version": self.version, "analysis_ready": self.analysis_ready,
                "analysis_mode": self.analysis_mode}

    def analyze(self, video_path, *, work_dir, config, emit, should_stop):
        self.calls += 1
        detail = CheckDetailResponse.model_validate_json((FIXTURES / "ui_check_done.json").read_text()).check
        emit(PipelineEvent(phase="decode", processed_until_sec=0))
        if self.fail:
            raise RuntimeError("deliberate test failure")
        for phase in ["perception", "facts", "rules", "evidence"]:
            if should_stop():
                raise InterruptedError("stop requested")
            if self.delay:
                time.sleep(self.delay)
            emit(PipelineEvent(phase=phase, processed_until_sec=125.04,
                               observations_delta=detail.observations if phase == "facts" else []))
        frames = work_dir / "frames"
        (frames / "original").mkdir(parents=True, exist_ok=True)
        (frames / "overlay").mkdir(exist_ok=True)
        (work_dir / "video.json").write_text(detail.video.model_dump_json(), encoding="utf-8")
        (work_dir / "engine_manifest.json").write_text(json.dumps({
            "source_path": str(video_path.resolve()), "rules_dir": str(config.rules_dir.resolve()),
            "capabilities": {}, "complete": True,
        }), encoding="utf-8")
        source = json.loads((FIXTURES / "v1_done_violation.json").read_text())
        for finding, violation in zip(detail.finding_set.findings, source["violations"]):
            for frame in finding.evidence:
                raw = base64.b64decode(violation["evidence_frames"][0]["image_base64"])
                (frames / "original" / f"{frame.frame_id}.jpg").write_bytes(raw)
                (frames / "overlay" / f"{frame.frame_id}.jpg").write_bytes(raw)
        return AnalysisArtifacts(
            work_dir=work_dir, video=detail.video, layout=detail.layout, tracks=detail.tracks,
            observations=detail.observations, coverage=detail.coverage, finding_set=detail.finding_set,
            run=AnalysisRun(check_id=config.check_id, pipeline_version=config.pipeline_version,
                            analysis_fps=config.analysis_fps, model_versions={"test": "fake"},
                            analysis_ready=self.analysis_ready, analysis_mode=self.analysis_mode),
        )

    def reevaluate(self, artifacts, *, role_overrides, zone_overrides, base):
        assert (artifacts.work_dir / ".reevaluation-staging").is_file()
        updated = base.model_copy(deep=True)
        if any(role.track_id == "p2" and role.role == "collector" for role in role_overrides):
            updated.findings = [finding for finding in updated.findings if finding.rule_id != "V-OUTSIDER"]
        updated.role_overrides = role_overrides
        updated.layout_revision = artifacts.layout.revision + bool(zone_overrides)
        for kind in ["original", "overlay"]:
            (artifacts.work_dir / "frames" / kind).mkdir(parents=True, exist_ok=True)
        for finding in updated.findings:
            for frame in finding.evidence:
                source = artifacts.work_dir.parent / "frames" / "original" / f"{frame.frame_id}.jpg"
                (artifacts.work_dir / "frames" / "original" / source.name).write_bytes(source.read_bytes())
                with Image.open(source) as image:
                    image = image.convert("RGB")
                    ImageDraw.Draw(image).line((0, 0, image.width, image.height), fill="red", width=2)
                    image.save(artifacts.work_dir / "frames" / "overlay" / source.name, "JPEG")
        return updated


def fake_probe(path, settings):
    from navidu.service.exceptions import ServiceError
    if path.read_bytes().startswith(b"not-video"):
        raise ServiceError(415, "UNSUPPORTED_MEDIA", "Файл не является видео.")
    return VideoMeta(duration_sec=125.04, fps_nominal=29.97, width=1920, height=1080)
