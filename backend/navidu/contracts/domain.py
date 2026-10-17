"""Authoritative domain types shared by inference, service and user interface."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, model_validator

from .units import (
    CheckId,
    ContractModel,
    Coordinate,
    FindingId,
    FrameId,
    ObservationId,
    Point,
    Score,
    Seconds,
    TrackId,
)

Role = Literal["cashier", "client", "collector", "controller", "staff", "outsider", "unknown"]
ZoneKind = Literal[
    "cashier_post", "client_window", "cash_desk", "storage", "service_door", "transfer_tray",
    "count_room",
]
ObservationKind = Literal[
    "person_in_zone", "cash_in_hands", "cash_on_desk", "cash_handover", "counting_motion",
    "phone_visible", "bag_present", "bag_opened", "storage_open", "visibility_low",
    "cash_sorting", "cash_packing", "bag_integrity_damage", "personal_item_identified",
    "camera_device_visible", "person_left_zone", "extra_person_present", "identity_document_visible",
]
Verdict = Literal["violation", "no_violation"]
DecisionValue = Literal["confirmed", "rejected", "not_visible"]
CoverageScope = Literal["cashier", "client", "desk", "storage", "whole"]
CoverageReason = Literal["out_of_frame", "occluded", "low_light", "motion_blur"]
Source = Literal["auto", "reviewer"]
AnalysisMode = Literal["validated", "experimental", "unvalidated", "diagnostic"]
BBox = tuple[Coordinate, Coordinate, Coordinate, Coordinate]


class TimeWindow(ContractModel):
    start_sec: Seconds
    end_sec: Seconds

    @model_validator(mode="after")
    def ordered(self):
        if self.end_sec < self.start_sec:
            raise ValueError("Конец интервала раньше начала.")
        return self


class TrackRole(ContractModel):
    value: Role
    source: Source = "auto"


class TrackSample(ContractModel):
    t: Seconds
    bbox: BBox
    score: Score | None = None

    @model_validator(mode="after")
    def ordered_box(self):
        if self.bbox[2] < self.bbox[0] or self.bbox[3] < self.bbox[1]:
            raise ValueError("Рамка имеет отрицательную ширину или высоту.")
        return self


class Track(ContractModel):
    track_id: TrackId
    kind: Literal["person", "object"]
    object_class: str | None = None
    first_sec: Seconds
    last_sec: Seconds
    role: TrackRole | None = None
    samples: list[TrackSample] = Field(default_factory=list)

    @model_validator(mode="after")
    def ordered(self):
        if self.last_sec < self.first_sec:
            raise ValueError("Конец трека раньше начала.")
        if any(s.t < self.first_sec or s.t > self.last_sec for s in self.samples):
            raise ValueError("Координата трека находится вне его времени.")
        return self


class Zone(ContractModel):
    zone_id: str = Field(pattern=r"^z-[a-z_]+-[0-9]+$")
    kind: ZoneKind
    polygon: list[Point] = Field(min_length=3, max_length=32)
    source: Source = "auto"


class SceneLayout(ContractModel):
    revision: int = Field(ge=1)
    zones: list[Zone] = Field(default_factory=list)
    camera_note: str | None = None


class Observation(TimeWindow):
    obs_id: ObservationId
    kind: ObservationKind
    subjects: list[TrackId] = Field(default_factory=list)
    zones: list[str] = Field(default_factory=list)
    score: Score | None = None
    frame_ids: list[FrameId] = Field(default_factory=list)
    provenance: Literal["detector", "temporal_model", "geometry", "reviewer"] | None = None
    model_id: str | None = None
    # Handover direction is explicit. The order of subjects carries no direction.
    actor_id: TrackId | None = None
    recipient_id: TrackId | None = None
    relation_id: str | None = Field(default=None, max_length=128,
                                    pattern=r"^[A-Za-z0-9][A-Za-z0-9:._-]{0,127}$")


class Coverage(TimeWindow):
    scope: CoverageScope
    reason: CoverageReason


class OverlayBox(ContractModel):
    bbox: BBox
    label: str
    tone: Literal["observation", "finding"]

    @model_validator(mode="after")
    def ordered_box(self):
        if self.bbox[2] < self.bbox[0] or self.bbox[3] < self.bbox[1]:
            raise ValueError("Рамка имеет отрицательную ширину или высоту.")
        return self


class EvidenceFrame(ContractModel):
    frame_id: FrameId
    timestamp_sec: Seconds
    overlay_boxes: list[OverlayBox] = Field(default_factory=list)


class Decision(ContractModel):
    value: DecisionValue | None
    note: str = Field(default="", max_length=2000)
    decided_at: datetime


class Finding(ContractModel):
    finding_id: FindingId
    rule_id: str
    type_label: str
    timecode_sec: Seconds
    window: TimeWindow
    basis: list[ObservationId]
    what_seen: str
    why_may_violate: str
    norm_refs: list[str] = Field(min_length=1)
    description: str
    evidence: list[EvidenceFrame] = Field(min_length=1, max_length=3)
    strength: Literal["strong", "moderate"]
    decision: Decision | None = None

    @model_validator(mode="after")
    def evidence_time(self):
        if self.timecode_sec != self.evidence[0].timestamp_sec:
            raise ValueError("Таймкод должен совпадать с первым кадром-доказательством.")
        if not self.window.start_sec <= self.timecode_sec <= self.window.end_sec:
            raise ValueError("Основной кадр должен находиться внутри окна эпизода.")
        return self


class NotCheckableWindow(TimeWindow):
    reason: str


class RuleStatus(ContractModel):
    rule_id: str
    status: Literal["violation", "no_violation", "not_checkable", "partly_checkable"]
    not_checkable_windows: list[NotCheckableWindow] = Field(default_factory=list)


class RoleOverride(ContractModel):
    track_id: TrackId
    role: Role


class FindingSet(ContractModel):
    revision: int = Field(ge=1)
    basis: Source
    created_at: datetime
    layout_revision: int = Field(ge=1)
    role_overrides: list[RoleOverride] = Field(default_factory=list)
    verdict: Verdict
    findings: list[Finding] = Field(default_factory=list)
    rules: list[RuleStatus] = Field(default_factory=list)


def verdict_of(finding_set: FindingSet, use_decisions: bool = True) -> Verdict:
    """A lack of visibility never becomes a clean bill; only rejection removes a finding."""
    return "violation" if any(
        not use_decisions or f.decision is None or f.decision.value != "rejected"
        for f in finding_set.findings
    ) else "no_violation"


class VideoMeta(ContractModel):
    duration_sec: Seconds
    fps_nominal: float = Field(gt=0, allow_inf_nan=False)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    rotation: Literal[0, 90, 180, 270] = 0


class AnalysisRun(ContractModel):
    check_id: CheckId
    pipeline_version: str
    analysis_fps: float = Field(gt=0, allow_inf_nan=False)
    analysis_ready: bool = False
    analysis_mode: AnalysisMode = "unvalidated"
    model_versions: dict[str, str] = Field(default_factory=dict)
    phase_timings_sec: dict[str, Annotated[float, Field(ge=0, allow_inf_nan=False)]] = Field(
        default_factory=dict
    )
