"""Internal boundary between timestamp decoding and the warm perception runtime.

These DTOs are intentionally independent from ndarray/Pydantic serialization. The
pipeline converts retained values to C-DATA before writing durable artifacts.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from navidu.contracts.domain import BBox, Coverage, Observation, Track
from navidu.contracts.pipeline import PipelineConfig, PipelineEvent


@dataclass(frozen=True)
class FrameSample:
    timestamp_sec: float
    rgb: Any


@dataclass(frozen=True)
class KeypointSample:
    track_id: str
    timestamp_sec: float
    points: tuple[tuple[float, float, float | None], ...]
    model_id: str


@dataclass(frozen=True)
class RawDetection:
    timestamp_sec: float
    class_name: str
    bbox: BBox
    score: float | None
    model_id: str
    track_id: str | None = None


@dataclass(frozen=True)
class SemanticSignal:
    kind: str
    start_sec: float
    end_sec: float
    subjects: tuple[str, ...] = ()
    boxes: tuple[BBox, ...] = ()
    model_id: str = ""
    prompt_version: str | None = None
    confidence: float | None = None


@dataclass
class PerceptionArtifacts:
    tracks: list[Track] = field(default_factory=list)
    raw_detections: list[RawDetection] = field(default_factory=list)
    keypoints: list[KeypointSample] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    semantic_signals: list[SemanticSignal] = field(default_factory=list)
    coverage: list[Coverage] = field(default_factory=list)
    model_versions: dict[str, str] = field(default_factory=dict)
    capabilities: dict[str, bool] = field(default_factory=dict)


class PerceptionRuntime(Protocol):
    def prepare(self, config: PipelineConfig) -> None: ...

    def perceive(
        self,
        frames: Iterable[tuple[float, Any]],
        *,
        config: PipelineConfig,
        emit: Callable[[PipelineEvent], None],
        should_stop: Callable[[], bool],
    ) -> PerceptionArtifacts: ...

    def runtime_status(self, config: PipelineConfig) -> object: ...
