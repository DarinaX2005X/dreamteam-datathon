"""API failures, deliberately separate from a model's findings."""

from pydantic import Field

from .units import ContractModel, Seconds


class FieldIssue(ContractModel):
    path: str
    message: str


class ErrorInfo(ContractModel):
    code: str
    message: str
    request_id: str
    field_issues: list[FieldIssue] | None = None
    current_revision: int | None = Field(default=None, ge=1)


class ErrorResponse(ContractModel):
    error: ErrorInfo


class JobError(ContractModel):
    code: str
    message: str


class AnalysisError(JobError):
    failed_phase: str
    at_sec: Seconds | None = None
