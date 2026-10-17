"""Shared source-video time, coordinates and stable identifiers."""

import hashlib
import math
import secrets
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field


def _seconds(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("Время должно быть конечным числом.")
    return round(value, 3)


Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False), AfterValidator(_seconds)]
Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Coordinate = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Point = tuple[Coordinate, Coordinate]
CheckId = Annotated[str, Field(pattern=r"^[0-9a-f]{12}$")]
TrackId = Annotated[str, Field(pattern=r"^[po][0-9]+$")]
FrameId = Annotated[str, Field(pattern=r"^fr-[0-9]+$")]
FindingId = Annotated[str, Field(pattern=r"^fd-[0-9a-f]{10}$")]
ObservationId = Annotated[str, Field(pattern=r"^ob-[0-9]+$")]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def new_check_id() -> str:
    return secrets.token_hex(6)


def frame_id(timestamp_sec: float) -> str:
    return f"fr-{round(timestamp_sec * 1000)}"


def finding_id(rule_id: str, start_sec: float, end_sec: float) -> str:
    """C-UNITS: window rounding, not evidence choice, determines identity."""
    key = f"{rule_id}|{round(start_sec, 1)}|{round(end_sec, 1)}"
    return "fd-" + hashlib.sha1(key.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
