"""Descriptions made only from the observations that triggered a rule."""

from __future__ import annotations

from navidu.contracts.domain import Observation

from .engine import PendingFinding


def _clock(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:04.1f}".replace(".", ",")


def describe(pending: PendingFinding, observations: dict[str, Observation]) -> tuple[str, str, str]:
    basis = [observations[item] for item in pending.basis if item in observations]
    # The support window may include preceding actions and merged evidence.
    # Only the rule's anchor is the reported onset of the possible violation.
    factual = f"{pending.rule.what_seen} Начало эпизода — {_clock(pending.timecode_sec)}."
    if basis:
        kinds = ", ".join(dict.fromkeys(item.kind for item in basis))
        factual += f" Основание: {kinds}."
    description = f"{factual} {pending.rule.why_may_violate} {pending.rule.not_claimed}"
    return factual, pending.rule.why_may_violate, description
