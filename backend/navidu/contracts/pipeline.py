"""The stable boundary between durable jobs and local inference."""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field

from .domain import (
    AnalysisMode,
    AnalysisRun,
    Coverage,
    FindingSet,
    Observation,
    RoleOverride,
    SceneLayout,
    Track,
    VideoMeta,
    Zone,
)
from .units import CheckId, ContractModel, Seconds, new_check_id

PipelinePhase = Literal["decode", "perception", "facts", "rules", "evidence"]
PIPELINE_PHASES: list[PipelinePhase] = ["decode", "perception", "facts", "rules", "evidence"]


class PipelineConfig(ContractModel):
    check_id: CheckId
    pipeline_version: str = "0.1.0"
    models_dir: Path = Field(default_factory=lambda: Path("models"))
    rules_dir: Path = Field(default_factory=lambda: Path("rules"))
    device: Literal["cpu", "cuda", "auto"] = "auto"
    analysis_fps: float = Field(default=3.0, gt=0, le=60)
    evidence_overlay: bool = True
    max_duration_sec: float = Field(default=600, gt=0)
    diagnostic_only: bool = False

    @classmethod
    def from_env(cls, *, check_id: str | None = None) -> "PipelineConfig":
        """Standalone entry point; the durable service always passes its own check ID."""
        return cls(
            check_id=check_id or os.getenv("NAVIDU_CHECK_ID") or new_check_id(),
            pipeline_version=os.getenv("NAVIDU_PIPELINE_VERSION", "0.1.0"),
            models_dir=Path(os.getenv("NAVIDU_MODELS_DIR", "models")).resolve(),
            rules_dir=Path(os.getenv("NAVIDU_RULES_DIR", "rules")).resolve(),
            device=os.getenv("NAVIDU_DEVICE", "auto"),
            analysis_fps=float(os.getenv("NAVIDU_ANALYSIS_FPS", "3")),
            evidence_overlay=os.getenv("NAVIDU_EVIDENCE_OVERLAY", "1").lower() in {"1", "true", "yes"},
            max_duration_sec=float(os.getenv("NAVIDU_MAX_DURATION_SEC", "600")),
        )


class PipelineEvent(ContractModel):
    phase: PipelinePhase
    processed_until_sec: Seconds | None = None
    observations_delta: list[Observation] = Field(default_factory=list)
    message: str | None = None


class RuntimeStatus(ContractModel):
    model_status: Literal["loading", "ready", "missing", "error"]
    device: str
    limitations: list[str] = Field(default_factory=list)
    pipeline_version: str
    analysis_ready: bool = False
    analysis_mode: AnalysisMode = "unvalidated"


class AnalysisArtifacts(ContractModel):
    work_dir: Path
    video: VideoMeta
    layout: SceneLayout
    tracks: list[Track]
    observations: list[Observation]
    coverage: list[Coverage]
    finding_set: FindingSet
    run: AnalysisRun


class Pipeline(Protocol):
    def analyze(
        self,
        video_path: Path,
        *,
        work_dir: Path,
        config: PipelineConfig,
        emit: Callable[[PipelineEvent], None],
        should_stop: Callable[[], bool],
    ) -> AnalysisArtifacts: ...

    def reevaluate(
        self,
        artifacts: AnalysisArtifacts,
        *,
        role_overrides: list[RoleOverride],
        zone_overrides: list[Zone],
        base: FindingSet,
    ) -> FindingSet: ...
