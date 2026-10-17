"""Composable local analysis and deterministic review-time reevaluation."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from navidu.contracts.domain import (
    AnalysisRun,
    FindingSet,
    RoleOverride,
    SceneLayout,
    TrackRole,
    Zone,
)
from navidu.contracts.pipeline import (
    AnalysisArtifacts,
    PipelineConfig,
    PipelineEvent,
    RuntimeStatus,
)
from navidu.evidence import materialize_findings
from navidu.facts import derive_coverage, derive_observations
from navidu.ingest import iter_frames, probe_video
from navidu.rules import evaluate, load_catalog
from navidu.scene import assign_roles, infer_layout

from .interfaces import PerceptionArtifacts, PerceptionRuntime


class PipelineCancelled(RuntimeError):
    pass


def _runtime() -> PerceptionRuntime | None:
    try:
        from navidu.perception.runtime import get_runtime

        return get_runtime()
    except (ImportError, AttributeError):
        return None


def prepare(config: PipelineConfig) -> None:
    runtime = _runtime()
    if runtime is None:
        return
    runtime.prepare(config)


def runtime_status(config: PipelineConfig) -> RuntimeStatus:
    runtime = _runtime()
    if runtime is None:
        return RuntimeStatus(model_status="missing", device=config.device,
            limitations=["Локальный адаптер восприятия не установлен."],
            pipeline_version=config.pipeline_version)
    value = runtime.runtime_status(config)
    if isinstance(value, RuntimeStatus):
        return value
    return RuntimeStatus.model_validate(value)


def _write_model(path: Path, value) -> None:
    path.write_text(value.model_dump_json(indent=2), encoding="utf-8")


def _write_artifacts(
    directory: Path,
    artifacts: AnalysisArtifacts,
    perception: PerceptionArtifacts,
    capabilities: dict[str, bool],
    source_path: Path,
    rules_dir: Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_model(directory / "video.json", artifacts.video)
    _write_model(directory / "layout.json", artifacts.layout)
    (directory / "tracks.jsonl").write_text("\n".join(
        item.model_dump_json() for item in artifacts.tracks
    ) + ("\n" if artifacts.tracks else ""), encoding="utf-8")
    (directory / "observations.jsonl").write_text("\n".join(
        item.model_dump_json() for item in artifacts.observations
    ) + ("\n" if artifacts.observations else ""), encoding="utf-8")
    (directory / "coverage.jsonl").write_text("\n".join(
        item.model_dump_json() for item in artifacts.coverage
    ) + ("\n" if artifacts.coverage else ""), encoding="utf-8")
    _write_model(directory / "finding_set.json", artifacts.finding_set)
    _write_model(directory / "run.json", artifacts.run)
    (directory / "keypoints.jsonl").write_text("\n".join(
        json.dumps(asdict(item), ensure_ascii=False) for item in perception.keypoints
    ) + ("\n" if perception.keypoints else ""), encoding="utf-8")
    (directory / "raw_detections.jsonl").write_text("\n".join(
        json.dumps(asdict(item), ensure_ascii=False) for item in perception.raw_detections
    ) + ("\n" if perception.raw_detections else ""), encoding="utf-8")
    (directory / "semantic_signals.jsonl").write_text("\n".join(
        json.dumps(asdict(item), ensure_ascii=False) for item in perception.semantic_signals
    ) + ("\n" if perception.semantic_signals else ""), encoding="utf-8")
    (directory / "engine_manifest.json").write_text(json.dumps({
        "source_path": str(source_path.resolve()),
        "capabilities": capabilities,
        "rules_dir": str(rules_dir.resolve()),
        "complete": True,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


class RealPipeline:
    def __init__(self, runtime: PerceptionRuntime | None = None):
        self.runtime = runtime

    def analyze(
        self,
        video_path: Path,
        *,
        work_dir: Path,
        config: PipelineConfig,
        emit: Callable[[PipelineEvent], None],
        should_stop: Callable[[], bool],
    ) -> AnalysisArtifacts:
        runtime = self.runtime or _runtime()
        if runtime is None:
            raise RuntimeError("PERCEPTION_RUNTIME_MISSING")
        if should_stop():
            raise PipelineCancelled("Анализ отменён до запуска.")
        timings: dict[str, float] = {}
        phase_start = time.perf_counter()
        probe = probe_video(video_path, max_duration_sec=config.max_duration_sec)
        timings["decode"] = time.perf_counter() - phase_start
        emit(PipelineEvent(phase="decode", processed_until_sec=0,
                           message="Метаданные записи прочитаны."))
        runtime.prepare(config)
        timestamps: list[float] = []

        def frames():
            for decoded in iter_frames(video_path, analysis_fps=config.analysis_fps,
                                       probe=probe, should_stop=should_stop):
                timestamps.append(decoded.timestamp_sec)
                yield decoded.timestamp_sec, decoded.rgb

        phase_start = time.perf_counter()
        perception = runtime.perceive(frames(), config=config, emit=emit, should_stop=should_stop)
        timings["perception"] = time.perf_counter() - phase_start
        if should_stop():
            raise PipelineCancelled("Анализ отменён; частичные результаты не опубликованы.")
        phase_start = time.perf_counter()
        layout = infer_layout(perception.tracks)
        roles = assign_roles(perception.tracks, layout)
        tracks = [track.model_copy(update={
            "role": TrackRole(value=roles.get(track.track_id, "unknown"))
        }) if track.kind == "person" else track.model_copy(deep=True) for track in perception.tracks]
        used_ids = [int(item.obs_id.split("-")[1]) for item in perception.observations]
        geometry = derive_observations(tracks, layout, roles,
                                       start_index=max(used_ids, default=0) + 1)
        observations = sorted(perception.observations + geometry,
                              key=lambda item: (item.start_sec, item.obs_id))
        coverage = list(perception.coverage)
        coverage.extend(derive_coverage(timestamps, expected_period_sec=1 / config.analysis_fps,
                                        duration_sec=probe.video.duration_sec))
        timings["facts"] = time.perf_counter() - phase_start
        emit(PipelineEvent(phase="facts", processed_until_sec=probe.video.duration_sec,
                           observations_delta=observations))
        phase_start = time.perf_counter()
        catalog = load_catalog(config.rules_dir / "catalog")
        pending, statuses = evaluate(observations, coverage, roles, catalog,
            perception.capabilities, {zone.zone_id: zone.kind for zone in layout.zones})
        timings["rules"] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        findings = materialize_findings(video_path, pending, observations, tracks,
            work_dir=work_dir, probe=probe, overlay=config.evidence_overlay)
        timings["evidence"] = time.perf_counter() - phase_start
        finding_set = FindingSet(
            revision=1, basis="auto", created_at=datetime.now(timezone.utc),
            layout_revision=layout.revision, findings=findings, rules=statuses,
            verdict="violation" if findings else "no_violation",
        )
        run = AnalysisRun(check_id=config.check_id, pipeline_version=config.pipeline_version,
                          analysis_fps=config.analysis_fps,
                          model_versions=perception.model_versions,
                          phase_timings_sec=timings)
        artifacts = AnalysisArtifacts(work_dir=work_dir, video=probe.video, layout=layout,
            tracks=tracks, observations=observations, coverage=coverage,
            finding_set=finding_set, run=run)
        _write_artifacts(work_dir, artifacts, perception, perception.capabilities, video_path,
                         config.rules_dir)
        emit(PipelineEvent(phase="evidence", processed_until_sec=probe.video.duration_sec,
                           message="Артефакты анализа сохранены."))
        return artifacts

    def reevaluate(
        self,
        artifacts: AnalysisArtifacts,
        *,
        role_overrides: list[RoleOverride],
        zone_overrides: list[Zone],
        base: FindingSet,
    ) -> FindingSet:
        manifest_path = artifacts.work_dir / "engine_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        capabilities = dict(manifest.get("capabilities", {}))
        replacements = {zone.zone_id: zone for zone in zone_overrides}
        zones = [replacements.pop(zone.zone_id, zone) for zone in artifacts.layout.zones]
        zones.extend(replacements.values())
        layout = SceneLayout(revision=artifacts.layout.revision + 1, zones=zones,
                             camera_note=artifacts.layout.camera_note)
        roles = assign_roles(artifacts.tracks, layout)
        roles.update({item.track_id: item.role for item in role_overrides})
        retained = [item for item in artifacts.observations if item.provenance != "geometry"]
        used_ids = [int(item.obs_id.split("-")[1]) for item in retained]
        geometry = derive_observations(artifacts.tracks, layout, roles,
                                       start_index=max(used_ids, default=0) + 1)
        observations = sorted(retained + geometry, key=lambda item: (item.start_sec, item.obs_id))
        catalog = load_catalog(Path(manifest.get("rules_dir", "rules")) / "catalog")
        pending, statuses = evaluate(observations, artifacts.coverage, roles, catalog,
            capabilities, {zone.zone_id: zone.kind for zone in layout.zones})
        old = {finding.finding_id: finding for finding in base.findings}
        source = Path(manifest["source_path"])
        probe = probe_video(source)
        generated = materialize_findings(source, pending, observations, artifacts.tracks,
            work_dir=artifacts.work_dir, probe=probe, overlay=True)
        findings = []
        for finding in generated:
            previous = old.get(finding.finding_id)
            if previous and previous.decision is not None:
                finding.decision = previous.decision
            findings.append(finding)
        result = FindingSet(
            revision=base.revision + 1, basis="reviewer", created_at=datetime.now(timezone.utc),
            layout_revision=layout.revision, role_overrides=role_overrides,
            verdict="violation" if any(
                finding.decision is None or finding.decision.value != "rejected" for finding in findings
            ) else "no_violation",
            findings=findings, rules=statuses,
        )
        # Revision files are append-only; automatic revision 1 remains untouched.
        _write_model(artifacts.work_dir / f"finding_set.rev{result.revision}.json", result)
        return result


_DEFAULT_PIPELINE = RealPipeline()


def analyze(video_path: Path, *, work_dir: Path, config: PipelineConfig,
            emit: Callable[[PipelineEvent], None], should_stop: Callable[[], bool]) -> AnalysisArtifacts:
    return _DEFAULT_PIPELINE.analyze(video_path, work_dir=work_dir, config=config,
                                     emit=emit, should_stop=should_stop)


def reevaluate(artifacts: AnalysisArtifacts, *, role_overrides: list[RoleOverride],
               zone_overrides: list[Zone], base: FindingSet) -> FindingSet:
    return _DEFAULT_PIPELINE.reevaluate(artifacts, role_overrides=role_overrides,
                                        zone_overrides=zone_overrides, base=base)
