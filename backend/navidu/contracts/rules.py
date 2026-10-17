"""Normative sources and bounded, deterministic rule descriptions."""

from datetime import date
from typing import Any, Literal

from pydantic import Field, model_validator

from .domain import CoverageScope
from .units import ContractModel


class SourceDocument(ContractModel):
    title: str
    act: str
    registry_code: str
    url: str
    edition_checked: date
    checked_at: date | None = None
    source_url: str | None = None
    source_version: str | None = None


class SourceClause(ContractModel):
    clause_id: str
    document: SourceDocument
    locator: str
    quote: str | None = None
    verified_verbatim: bool = False
    visual_test: str
    not_visual: str


class SourceClauseView(SourceClause):
    """Public source view: unverified quotations never reach user-facing APIs."""

    @model_validator(mode="after")
    def hide_unverified_quote(self):
        if not self.verified_verbatim and self.quote is not None:
            object.__setattr__(self, "quote", None)
        return self

    @classmethod
    def from_source(cls, source: SourceClause) -> "SourceClauseView":
        value = source.model_dump()
        if not source.verified_verbatim:
            value["quote"] = None
        return cls.model_validate(value)


class RuleDefinition(ContractModel):
    rule_id: str
    type_label: str
    norm_refs: list[str] = Field(min_length=1)
    requires_visible: list[CoverageScope]
    requires_capabilities: list[str] = Field(default_factory=list)
    anchor: dict[str, Any]
    condition: dict[str, Any]
    timecode_policy: Literal["max_evidence", "onset", "obligation_due"]
    what_seen: str
    why_may_violate: str
    not_claimed: str
    enabled: bool = True
    strength: Literal["strong", "moderate"] = "moderate"


class RuleView(ContractModel):
    rule_id: str
    type_label: str
    norm_refs: list[str]
    requires_visible: list[CoverageScope]
    visible_sign: str
    why_may_violate: str
    not_claimed: str
    enabled: bool
    sources: list[SourceClauseView] = Field(default_factory=list)


class RulesResponse(ContractModel):
    rules: list[RuleView]
    not_checkable: list[SourceClauseView]
