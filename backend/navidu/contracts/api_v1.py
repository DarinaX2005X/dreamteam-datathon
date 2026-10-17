"""The jury's exact successful response: do not add UI or provenance fields here."""

import base64
import binascii
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from .domain import Verdict
from .errors import JobError
from .units import CheckId, ContractModel, Seconds


class JuryEvidenceFrame(ContractModel):
    timestamp_sec: Seconds
    image_base64: str

    @field_validator("image_base64")
    @classmethod
    def valid_jpeg(cls, value: str) -> str:
        try:
            raw = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Ожидался JPEG в base64 без data-префикса.") from exc
        if not raw.startswith(b"\xff\xd8") or not raw.endswith(b"\xff\xd9"):
            raise ValueError("Кадр не является JPEG.")
        return value


class JuryViolation(ContractModel):
    type: str
    timecode_sec: Seconds
    description: str
    evidence_frames: list[JuryEvidenceFrame] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def evidence_time(self):
        if self.timecode_sec != self.evidence_frames[0].timestamp_sec:
            raise ValueError("Таймкод должен совпадать с первым кадром-доказательством.")
        return self


class JobProcessing(ContractModel):
    job_id: CheckId
    status: Literal["processing"] = "processing"


class JobDone(ContractModel):
    job_id: CheckId
    status: Literal["done"] = "done"
    verdict: Verdict
    violations: list[JuryViolation]

    @model_validator(mode="after")
    def verdict_matches(self):
        if (self.verdict == "violation") != bool(self.violations):
            raise ValueError("Вердикт не соответствует списку эпизодов.")
        if self.violations != sorted(self.violations, key=lambda v: v.timecode_sec):
            raise ValueError("Эпизоды должны идти по времени.")
        return self


class JobFailed(ContractModel):
    job_id: CheckId
    status: Literal["failed"] = "failed"
    error: JobError


JobResponse = Annotated[JobProcessing | JobDone | JobFailed, Field(discriminator="status")]
