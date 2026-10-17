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


class AnalysisNotValidated(RuntimeError):
    code = "ANALYSIS_NOT_PERFORMED"
    retryable = False


class AnalysisIncomplete(RuntimeError):
    code = "ANALYSIS_INCOMPLETE"
    retryable = False

    def __init__(self, rule_statuses, observation_count: int):
        super().__init__(
            "ANALYSIS_INCOMPLETE: ни одно включённое правило не удалось проверить."
        )
        self.rule_statuses = rule_statuses
        self.observation_count = observation_count


def _require_checkable_result(pending, statuses, catalog, *, observation_count: int) -> None:
    enabled = {rule.rule_id for rule in catalog if rule.enabled}
    relevant = [status for status in statuses if status.rule_id in enabled]
    if not pending and relevant and all(status.status == "not_checkable" for status in relevant):
        raise AnalysisIncomplete(relevant, observation_count)


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
    (directory / "inference_metrics.json").write_text(json.dumps({
        "schema_version": 1,
        "semantic_windows_completed": perception.semantic_windows_completed,
        **perception.diagnostics,
    }, ensure_ascii=True, allow_nan=False, indent=2), encoding="utf-8")
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
        # Validate deterministic configuration before decoding or model inference.
        # The already parsed catalog is reused below, so a bad rule cannot waste a
        # full inference attempt or be retried as though it were a transient fault.
        catalog = load_catalog(config.rules_dir / "catalog")
        timings: dict[str, float] = {}
        phase_start = time.perf_counter()
        probe = probe_video(video_path, max_duration_sec=config.max_duration_sec)
        timings["decode"] = time.perf_counter() - phase_start
        emit(PipelineEvent(phase="decode", processed_until_sec=0,
                           message="Метаданные записи прочитаны."))
        runtime.prepare(config)
        prepared_status = runtime.runtime_status(config)
        if prepared_status.model_status != "ready":
            raise RuntimeError(
                f"PERCEPTION_RUNTIME_NOT_READY: {prepared_status.model_status}"
            )
        allowed_mode = "experimental" if config.experimental_analysis else "validated"
        if not config.diagnostic_only and (
            not prepared_status.analysis_ready or prepared_status.analysis_mode != allowed_mode
        ):
            raise AnalysisNotValidated(
                "ANALYSIS_NOT_PERFORMED: нет измеренных возможностей анализа."
            )
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
        if not config.diagnostic_only and perception.semantic_windows_completed < 1:
            raise AnalysisNotValidated(
                "ANALYSIS_NOT_PERFORMED: временной анализ этой записи не выполнен."
            )
        phase_start = time.perf_counter()
        layout = infer_layout(perception.tracks)
        roles = assign_roles(perception.tracks, layout)
        tracks = [track.model_copy(update={
            "role": TrackRole(value=roles.get(track.track_id, "unknown"))
        }) if track.kind == "person" else track.model_copy(deep=True) for track in perception.tracks]
        used_ids = [int(item.obs_id.split("-")[1]) for item in perception.observations]
        fact_gap = max(0.75, 1.5 / config.analysis_fps)
        geometry = derive_observations(tracks, layout, roles, max_gap_sec=fact_gap,
                                       start_index=max(used_ids, default=0) + 1)
        capabilities = dict(perception.capabilities)
        capabilities["zone_departure"] = bool(
            layout.zones and any(track.kind == "person" and any(
                right.t - left.t <= fact_gap
                for left, right in zip(track.samples, track.samples[1:])
            ) for track in tracks)
        )
        observations = sorted(perception.observations + geometry,
                              key=lambda item: (item.start_sec, item.obs_id))
        coverage = list(perception.coverage)
        coverage.extend(derive_coverage(timestamps, expected_period_sec=1 / config.analysis_fps,
                                        duration_sec=probe.video.duration_sec))
        timings["facts"] = time.perf_counter() - phase_start
        emit(PipelineEvent(phase="facts", processed_until_sec=probe.video.duration_sec,
                           observations_delta=observations))
        phase_start = time.perf_counter()
        emit(PipelineEvent(phase="rules", processed_until_sec=probe.video.duration_sec,
                           message="Проверяются условия правил."))
        pending, statuses = evaluate(observations, coverage, roles, catalog,
            capabilities, {zone.zone_id: zone.kind for zone in layout.zones},
            duration_sec=probe.video.duration_sec)
        if not config.diagnostic_only:
            _require_checkable_result(
                pending, statuses, catalog, observation_count=len(observations)
            )
        timings["rules"] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        findings = materialize_findings(video_path, pending, observations, tracks,
            work_dir=work_dir, probe=probe, overlay=config.evidence_overlay,
            evidence_dir_name="frames")
        timings["evidence"] = time.perf_counter() - phase_start
        finding_set = FindingSet(
            revision=1, basis="auto", created_at=datetime.now(timezone.utc),
            layout_revision=layout.revision, findings=findings, rules=statuses,
            verdict="violation" if findings else "no_violation",
        )
        run = AnalysisRun(check_id=config.check_id,
                          pipeline_version=prepared_status.pipeline_version,
                          analysis_fps=config.analysis_fps,
                          analysis_ready=(prepared_status.analysis_ready and
                                          not config.diagnostic_only),
                          analysis_mode=("diagnostic" if config.diagnostic_only else
                                         prepared_status.analysis_mode),
                          model_versions=perception.model_versions,
                          phase_timings_sec=timings)
        artifacts = AnalysisArtifacts(work_dir=work_dir, video=probe.video, layout=layout,
            tracks=tracks, observations=observations, coverage=coverage,
            finding_set=finding_set, run=run)
        _write_artifacts(work_dir, artifacts, perception, capabilities, video_path,
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
        marker = artifacts.work_dir / ".reevaluation-staging"
        if not marker.is_file():
            raise RuntimeError(
                "REEVALUATION_STAGING_REQUIRED: durable automatic artifacts are immutable"
            )
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
        reviewer_ids = {item.track_id for item in role_overrides}
        updated_tracks = [track.model_copy(update={
            "role": TrackRole(value=roles.get(track.track_id, "unknown"),
                              source="reviewer" if track.track_id in reviewer_ids else "auto")
        }) if track.kind == "person" else track.model_copy(deep=True)
            for track in artifacts.tracks]
        retained = [item for item in artifacts.observations if item.provenance != "geometry"]
        used_ids = [int(item.obs_id.split("-")[1]) for item in retained]
        fact_gap = max(0.75, 1.5 / artifacts.run.analysis_fps)
        geometry = derive_observations(updated_tracks, layout, roles, max_gap_sec=fact_gap,
                                       start_index=max(used_ids, default=0) + 1)
        capabilities["zone_departure"] = bool(
            layout.zones and any(track.kind == "person" and any(
                right.t - left.t <= fact_gap
                for left, right in zip(track.samples, track.samples[1:])
            ) for track in updated_tracks)
        )
        observations = sorted(retained + geometry, key=lambda item: (item.start_sec, item.obs_id))
        catalog = load_catalog(Path(manifest.get("rules_dir", "rules")) / "catalog")
        pending, statuses = evaluate(observations, artifacts.coverage, roles, catalog,
            capabilities, {zone.zone_id: zone.kind for zone in layout.zones},
            duration_sec=artifacts.video.duration_sec)
        old = {finding.finding_id: finding for finding in base.findings}
        source = Path(manifest["source_path"])
        probe = probe_video(source)
        generated = materialize_findings(source, pending, observations, updated_tracks,
            work_dir=artifacts.work_dir, probe=probe, overlay=True,
            evidence_dir_name="frames")
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
        # Service owns CAS publication from this isolated directory. Dry runs
        # delete it; the durable automatic revision is never touched here.
        _write_model(artifacts.work_dir / "finding_set.json", result)
        # The service passes an isolated deep copy and persists this reviewed
        # state only after its compare-and-swap succeeds.
        artifacts.layout = layout
        artifacts.tracks = updated_tracks
        artifacts.observations = observations
        artifacts.finding_set = result
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
