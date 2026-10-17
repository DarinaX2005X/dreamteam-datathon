import base64
import json
import re
from pathlib import Path

from navidu.contracts.api_v1 import (
    JobDone,
    JobFailed,
    JobProcessing,
    JuryEvidenceFrame,
    JuryViolation,
)
from navidu.contracts.domain import VideoMeta, verdict_of
from navidu.contracts.errors import AnalysisError, JobError
from navidu.contracts.pipeline import AnalysisArtifacts
from navidu.contracts.ui_api import CheckDetail, CheckProgress, CheckSummary

from .evidence import artifact_frame_path
from .exceptions import ServiceError
from .readiness import NOT_PERFORMED_MESSAGE, published_analysis_valid, usable_analysis_result
from .store import Store, utc


def check_directory(store: Store, check_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{12}", check_id):
        raise ServiceError(404, "NOT_FOUND", "Запись не найдена.")
    store.get(check_id)
    base = store.settings.checks_dir.resolve()
    target = (base / check_id).resolve()
    if target.parent != base:
        raise ServiceError(404, "NOT_FOUND", "Запись не найдена.")
    return target


def evidence_path(store: Store, check_id: str, frame_id: str, overlay: bool,
                  revision: int | None = None) -> Path:
    if not re.fullmatch(r"fr-[0-9]+", frame_id):
        raise ServiceError(404, "NOT_FOUND", "Кадр не найден.")
    directory = check_directory(store, check_id)
    finding_set = store.finding_set(check_id, revision=revision)
    if finding_set is None or not any(frame.frame_id == frame_id
            for finding in finding_set.findings for frame in finding.evidence):
        raise ServiceError(404, "NOT_FOUND", "Кадр не найден.")
    media_revision = store.media_revision(check_id, finding_set.revision)
    root = directory if media_revision is None else directory / "revisions" / str(media_revision)
    if not root.resolve().is_relative_to(directory):
        raise ServiceError(404, "NOT_FOUND", "Кадр не найден.")
    path = artifact_frame_path(root, frame_id, overlay)
    if path is None:
        raise ServiceError(404, "NOT_FOUND", "Кадр не найден.")
    return path


def summary(store: Store, row) -> CheckSummary:
    status = row["status"]
    if status == "done" and not usable_analysis_result(row):
        status = "failed"
    finding_set = store.finding_set(row["id"]) if status == "done" else None
    meta = VideoMeta.model_validate_json(row["video_meta_json"])
    return CheckSummary(
        id=row["id"], file_name=row["file_name"], status=status, created_at=utc(row["created_at"]),
        duration_sec=meta.duration_sec, verdict=verdict_of(finding_set) if finding_set else None,
        findings_count=len(finding_set.findings) if finding_set else None,
        is_example=bool(row["is_example"]),
    )


def detail(store: Store, row) -> CheckDetail:
    base = summary(store, row)
    events = store.events(row["id"])
    last = events[-1] if events else None
    observations = {}
    for event in events:
        observations.update({item.obs_id: item for item in event.observations_delta})
    artifacts = AnalysisArtifacts.model_validate_json(row["artifacts_json"]) if row["artifacts_json"] else None
    status = base.status
    phase = status if status != "processing" else (last.phase if last else "decode")
    finding_set = store.finding_set(row["id"]) if status == "done" else None
    automatic = store.finding_set(row["id"], revision=1) if status == "done" else None
    directory = (store.settings.checks_dir / row["id"]).resolve()
    video = (directory / (row["video_file"] or "expired")).resolve()
    video_available = bool(row["video_file"] and video.parent == directory and video.is_file())
    expires = None if row["is_example"] or row["finished_at"] is None else utc(
        row["finished_at"] + store.settings.retain_video_hours * 3600
    )
    return CheckDetail(
        **base.model_dump(), video=VideoMeta.model_validate_json(row["video_meta_json"]),
        analysis_origin="cache" if row["cached_from_check_id"] else "local",
        analyzed_at=utc(row["analyzed_at"]) if row["analyzed_at"] is not None else None,
        automatic_verdict=verdict_of(automatic, use_decisions=False) if automatic else None,
        video_available=video_available, video_expires_at=expires,
        progress=CheckProgress(
            phase=phase, processed_until_sec=(
                artifacts.video.duration_sec if status == "done" and artifacts else
                next((event.processed_until_sec for event in reversed(events)
                      if event.processed_until_sec is not None), None)
            ), queue_position=store.queue_position(row["id"]) if status == "queued" else None,
        ), layout=artifacts.layout if artifacts else None,
        tracks=artifacts.tracks if artifacts else [],
        observations=artifacts.observations if artifacts else list(observations.values()),
        coverage=artifacts.coverage if artifacts else [], finding_set=finding_set,
        error=AnalysisError.model_validate_json(row["error_json"]) if row["error_json"] else (
            AnalysisError(code="ANALYSIS_NOT_PERFORMED", message=NOT_PERFORMED_MESSAGE,
                          failed_phase="perception") if row["status"] == "done" and status == "failed" else None
        ),
    )


def jury_response(store: Store, row):
    check_id = row["id"]
    if row["status"] in {"queued", "processing"}:
        return JobProcessing(job_id=check_id)
    if row["status"] == "cancelled":
        return JobFailed(job_id=check_id, error=JobError(code="CANCELLED", message="Проверка отменена пользователем."))
    if row["status"] == "failed":
        value = json.loads(row["error_json"]) if row["error_json"] else {}
        return JobFailed(job_id=check_id, error=JobError(
            code=value.get("code", "ANALYSIS_FAILED"), message=value.get("message", "Анализ не завершён.")
        ))
    api_result_allowed = published_analysis_valid(row) or (
        store.settings.allow_experimental_api and usable_analysis_result(row)
    )
    if not api_result_allowed:
        return JobFailed(job_id=check_id, error=JobError(
            code="ANALYSIS_NOT_PERFORMED", message=NOT_PERFORMED_MESSAGE,
        ))
    finding_set = store.finding_set(check_id, revision=1)
    if finding_set is None:
        raise ServiceError(500, "INTERNAL", "Не удалось прочитать результат. Повторите запрос.")
    violations = []
    for finding in sorted(finding_set.findings, key=lambda item: item.timecode_sec):
        evidence = []
        for frame in finding.evidence:
            path = evidence_path(store, check_id, frame.frame_id, store.settings.evidence_overlay, revision=1)
            evidence.append(JuryEvidenceFrame(
                timestamp_sec=frame.timestamp_sec, image_base64=base64.b64encode(path.read_bytes()).decode("ascii")
            ))
        violations.append(JuryViolation(type=finding.type_label, timecode_sec=finding.timecode_sec,
                                        description=finding.description, evidence_frames=evidence))
    return JobDone(job_id=check_id, verdict=verdict_of(finding_set, use_decisions=False), violations=violations)
