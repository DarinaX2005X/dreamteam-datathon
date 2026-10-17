"""Deterministic temporal rule interpreter with explicit visibility gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from navidu.contracts.domain import (
    Coverage,
    NotCheckableWindow,
    Observation,
    Role,
    RuleStatus,
    TimeWindow,
)
from navidu.contracts.units import finding_id

from .catalog import RuleSpec


@dataclass(frozen=True)
class PendingFinding:
    finding_id: str
    rule: RuleSpec
    window: TimeWindow
    timecode_sec: float
    basis: tuple[str, ...]


def _overlaps(left_start: float, left_end: float, right_start: float, right_end: float) -> bool:
    return left_start <= right_end and right_start <= left_end


def _role_match(observation: Observation, role: str | None, roles: dict[str, Role]) -> bool:
    return role is None or any(roles.get(subject) == role for subject in observation.subjects)


def _matches(observation: Observation, predicate: dict[str, Any], roles: dict[str, Role],
             zone_kinds: dict[str, str]) -> bool:
    if observation.kind != predicate.get("kind"):
        return False
    if not _role_match(observation, predicate.get("role"), roles):
        return False
    if predicate.get("actor_role") and roles.get(observation.actor_id) != predicate["actor_role"]:
        return False
    if predicate.get("recipient_role") and roles.get(observation.recipient_id) != predicate["recipient_role"]:
        return False
    if predicate.get("zone_kind") and not any(
        zone_kinds.get(zone_id) == predicate["zone_kind"] for zone_id in observation.zones
    ):
        return False
    return True


def _select(expr: dict[str, Any], observations: list[Observation], roles: dict[str, Role],
            zone_kinds: dict[str, str]):
    op, argument = next(iter(expr.items()))
    if op == "observation":
        return [item for item in observations if _matches(item, argument, roles, zone_kinds)]
    if op == "handover":
        return [item for item in observations if _matches(
            item, {"kind": "cash_handover", **argument}, roles, zone_kinds
        )]
    if op == "any":
        found = []
        for child in argument:
            found.extend(_select(child, observations, roles, zone_kinds))
        return list({item.obs_id: item for item in found}.values())
    return []


def _eval(
    expr: dict[str, Any],
    anchor: Observation,
    observations: list[Observation],
    roles: dict[str, Role],
    zone_kinds: dict[str, str],
) -> tuple[bool, list[Observation]]:
    op, argument = next(iter(expr.items()))
    if op in {"observation", "handover"}:
        selected = _select(expr, observations, roles, zone_kinds)
        matches = [item for item in selected if _overlaps(
            item.start_sec, item.end_sec, anchor.start_sec, anchor.end_sec
        )]
        return bool(matches), matches[:1]
    if op == "all":
        basis: list[Observation] = []
        for child in argument:
            result, used = _eval(child, anchor, observations, roles, zone_kinds)
            if not result:
                return False, []
            basis.extend(used)
        return True, basis
    if op == "any":
        for child in argument:
            result, used = _eval(child, anchor, observations, roles, zone_kinds)
            if result:
                return True, used
        return False, []
    if op == "absent":
        child = argument["predicate"]
        selected = _select(child, observations, roles, zone_kinds)
        margin = float(argument.get("within_sec", 0))
        present = any(_overlaps(item.start_sec, item.end_sec,
                                max(0, anchor.start_sec - margin), anchor.end_sec + margin)
                      for item in selected)
        return not present, []
    if op in {"before", "after"}:
        first = _select(argument["first"], observations, roles, zone_kinds)
        second = _select(argument["then"], observations, roles, zone_kinds)
        limit = float(argument.get("within_sec", 30))
        pairs = []
        for left in first:
            for right in second:
                gap = right.start_sec - left.end_sec
                overlap_ok = argument.get("allow_overlap", False) and (
                    left.start_sec <= right.end_sec and 0 <= right.end_sec - left.end_sec <= limit
                )
                if 0 <= gap <= limit or overlap_ok:
                    pairs.append((left, right))
        if op == "after":
            pairs = [(right, left) for left, right in pairs]
        touching = [pair for pair in pairs if any(
            item.obs_id == anchor.obs_id for item in pair
        )]
        return (bool(touching), list(touching[0]) if touching else [])
    raise ValueError(f"Unsupported operator: {op}")


def _covered(window: TimeWindow, required: tuple[str, ...], coverage: list[Coverage]):
    return [item for item in coverage if (item.scope in required or item.scope == "whole") and
            _overlaps(item.start_sec, item.end_sec, window.start_sec, window.end_sec)]


def _merge(findings: list[PendingFinding], gap_sec: float = 3.0) -> list[PendingFinding]:
    merged: list[PendingFinding] = []
    for item in sorted(findings, key=lambda value: (value.rule.rule_id, value.window.start_sec)):
        if merged and merged[-1].rule.rule_id == item.rule.rule_id and \
                item.window.start_sec - merged[-1].window.end_sec <= gap_sec:
            prior = merged[-1]
            window = TimeWindow(start_sec=prior.window.start_sec,
                                end_sec=max(prior.window.end_sec, item.window.end_sec))
            basis = tuple(dict.fromkeys(prior.basis + item.basis))
            merged[-1] = PendingFinding(
                finding_id=finding_id(item.rule.rule_id, window.start_sec, window.end_sec),
                rule=item.rule, window=window, timecode_sec=prior.timecode_sec, basis=basis,
            )
        else:
            merged.append(item)
    return merged


def evaluate(
    observations: list[Observation],
    coverage: list[Coverage],
    roles: dict[str, Role],
    catalog: list[RuleSpec],
    capabilities: dict[str, bool],
    zone_kinds: dict[str, str] | None = None,
) -> tuple[list[PendingFinding], list[RuleStatus]]:
    zone_kinds = zone_kinds or {}
    findings: list[PendingFinding] = []
    statuses: list[RuleStatus] = []
    for rule in catalog:
        if not rule.enabled:
            statuses.append(RuleStatus(rule_id=rule.rule_id, status="not_checkable",
                not_checkable_windows=[NotCheckableWindow(
                    start_sec=0, end_sec=0, reason=rule.not_claimed
                )]))
            continue
        missing = [name for name in rule.requires_capabilities if not capabilities.get(name, False)]
        anchors = _select(rule.anchor, observations, roles, zone_kinds)
        unavailable: list[NotCheckableWindow] = []
        if missing:
            unavailable.append(NotCheckableWindow(start_sec=0, end_sec=0,
                reason="Нет подтверждённой возможности восприятия: " + ", ".join(missing)))
        for anchor in anchors:
            window = TimeWindow(start_sec=anchor.start_sec, end_sec=anchor.end_sec)
            gaps = _covered(window, rule.requires_visible, coverage)
            if gaps:
                unavailable.extend(NotCheckableWindow(start_sec=item.start_sec,
                    end_sec=item.end_sec, reason=f"Область {item.scope} не видна: {item.reason}.")
                    for item in gaps)
                continue
            if missing:
                continue
            matched, used = _eval(rule.condition, anchor, observations, roles, zone_kinds)
            if matched:
                basis = tuple(dict.fromkeys([anchor.obs_id] + [item.obs_id for item in used]))
                sec = anchor.start_sec if rule.timecode_policy in {"onset", "obligation_due"} \
                    else round((anchor.start_sec + anchor.end_sec) / 2, 3)
                findings.append(PendingFinding(
                    finding_id=finding_id(rule.rule_id, window.start_sec, window.end_sec),
                    rule=rule, window=window, timecode_sec=sec, basis=basis,
                ))
        has = any(item.rule.rule_id == rule.rule_id for item in findings)
        status = "violation" if has else ("not_checkable" if unavailable else "no_violation")
        if has and unavailable:
            status = "partly_checkable"
        statuses.append(RuleStatus(rule_id=rule.rule_id, status=status,
                                   not_checkable_windows=unavailable))
    return _merge(findings), statuses
