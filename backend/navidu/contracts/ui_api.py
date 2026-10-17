"""UI contract. Null values distinguish unavailable data from a measured zero."""

from datetime import datetime
from typing import Literal

from pydantic import Field

from .domain import (
    AnalysisMode,
    Coverage,
    Decision,
    DecisionValue,
    FindingSet,
    Observation,
    RoleOverride,
    SceneLayout,
    Track,
    Verdict,
    VideoMeta,
    Zone,
)
from .errors import AnalysisError
from .pipeline import PIPELINE_PHASES, PipelinePhase
from .units import CheckId, ContractModel, FindingId, Seconds

CheckStatus = Literal["queued", "processing", "done", "failed", "cancelled"]


class CheckSummary(ContractModel):
    id: CheckId
    file_name: str
    status: CheckStatus
    created_at: datetime
    duration_sec: Seconds | None = None
    verdict: Verdict | None = None
    findings_count: int | None = Field(default=None, ge=0)
    is_example: bool = False


class CheckProgress(ContractModel):
    phase: Literal[
        "queued", "decode", "perception", "facts", "rules", "evidence", "done", "failed", "cancelled"
    ]
    phases: list[PipelinePhase] = Field(default_factory=lambda: list(PIPELINE_PHASES))
    processed_until_sec: Seconds | None = None
    queue_position: int | None = Field(default=None, ge=1)


class CheckDetail(CheckSummary):
    analysis_origin: Literal["local", "cache"] = "local"
    analyzed_at: datetime | None = None
    automatic_verdict: Verdict | None = None
    video_available: bool = True
    video_expires_at: datetime | None = None
    video: VideoMeta | None = None
    progress: CheckProgress
    layout: SceneLayout | None = None
    tracks: list[Track] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    coverage: list[Coverage] = Field(default_factory=list)
    finding_set: FindingSet | None = None
    error: AnalysisError | None = None


class CheckSummaryResponse(ContractModel):
    check: CheckSummary


class CheckDetailResponse(ContractModel):
    check: CheckDetail


class DecisionRequest(ContractModel):
    decision: DecisionValue | None
    note: str = Field(default="", max_length=2000)
    base_revision: int = Field(ge=1)


class DecisionResponse(ContractModel):
    decision: Decision | None
    revision: int = Field(ge=1)


class ReevaluateRequest(ContractModel):
    base_revision: int = Field(ge=1)
    role_overrides: list[RoleOverride] = Field(default_factory=list)
    zone_overrides: list[Zone] = Field(default_factory=list)
    dry_run: bool = False


class FindingDiff(ContractModel):
    added: list[FindingId]
    removed: list[FindingId]
    changed: list[FindingId]
    verdict_changed: bool


class PreviewFrame(ContractModel):
    image_base64: str
    overlay_base64: str | None


class ReevaluateResponse(ContractModel):
    finding_set: FindingSet
    diff: FindingDiff
    preview_frames: dict[str, PreviewFrame] = Field(default_factory=dict)


class QueueCounts(ContractModel):
    api: int = Field(ge=0)
    ui: int = Field(ge=0)


class HealthResponse(ContractModel):
    ready: bool
    analysis_ready: bool = False
    analysis_mode: AnalysisMode = "unvalidated"
    queue: QueueCounts
    pipeline_version: str
    model_status: Literal["loading", "ready", "missing", "error"]
    limitations: list[str] = Field(default_factory=list)
    device: str
    example_check_id: CheckId | None = None


class ExampleResponse(ContractModel):
    check_id: CheckId
