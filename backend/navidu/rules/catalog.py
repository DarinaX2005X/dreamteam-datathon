"""Strict loader for the bounded rule DSL."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_OPS = {"observation", "handover", "all", "any", "absent", "before", "after", "sequence"}


class CatalogConfigurationError(RuntimeError):
    """A deterministic catalog defect that cannot succeed on a job retry."""

    code = "RULES_INVALID"
    retryable = False


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    type_label: str
    norm_refs: tuple[str, ...]
    requires_visible: tuple[str, ...]
    requires_capabilities: tuple[str, ...]
    anchor: dict[str, Any]
    condition: dict[str, Any]
    timecode_policy: str
    what_seen: str
    why_may_violate: str
    not_claimed: str
    strength: str = "moderate"
    enabled: bool = True


def _validate_expression(value: object) -> None:
    if not isinstance(value, dict) or len(value) != 1:
        raise ValueError("Условие правила должно содержать ровно один оператор.")
    op, argument = next(iter(value.items()))
    if op not in _OPS:
        raise ValueError(f"Неизвестный оператор правила: {op}")
    if op in {"all", "any"}:
        if not isinstance(argument, list) or not argument:
            raise ValueError(f"Оператор {op} ожидает непустой список.")
        for child in argument:
            _validate_expression(child)
    elif op == "sequence":
        if not isinstance(argument, dict) or not isinstance(argument.get("steps"), list):
            raise ValueError("Оператор sequence ожидает список steps.")
        for child in argument["steps"]:
            _validate_expression(child)
        if not isinstance(argument.get("relations"), list):
            raise ValueError("Оператор sequence ожидает список relations.")
    elif op in {"before", "after"}:
        if not isinstance(argument, dict) or set(argument) - {
            "first", "then", "within_sec", "allow_overlap", "shared"
        }:
            raise ValueError(f"Некорректные аргументы оператора {op}.")
        _validate_expression(argument.get("first"))
        _validate_expression(argument.get("then"))
    elif not isinstance(argument, dict):
        raise ValueError(f"Оператор {op} ожидает объект.")


def load_catalog(path: Path) -> list[RuleSpec]:
    try:
        return _load_catalog(path)
    except CatalogConfigurationError:
        raise
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise CatalogConfigurationError(f"RULES_INVALID: {error}") from error


def _load_catalog(path: Path) -> list[RuleSpec]:
    rules: list[RuleSpec] = []
    for file in sorted(path.glob("*.yaml")):
        payload = json.loads(file.read_text(encoding="utf-8"))
        _validate_expression(payload["anchor"])
        _validate_expression(payload["condition"])
        rules.append(RuleSpec(
            rule_id=payload["rule_id"], type_label=payload["type_label"],
            norm_refs=tuple(payload["norm_refs"]),
            requires_visible=tuple(payload.get("requires_visible", [])),
            requires_capabilities=tuple(payload.get("requires_capabilities", [])),
            anchor=payload["anchor"], condition=payload["condition"],
            timecode_policy=payload["timecode_policy"], what_seen=payload["what_seen"],
            why_may_violate=payload["why_may_violate"], not_claimed=payload["not_claimed"],
            strength=payload.get("strength", "moderate"), enabled=payload.get("enabled", True),
        ))
    ids = [rule.rule_id for rule in rules]
    if len(ids) != len(set(ids)):
        raise ValueError("Каталог содержит повторяющийся rule_id.")
    sources_dir = path.parent / "sources"
    source_text = "\n".join(
        file.read_text(encoding="utf-8") for file in sorted(sources_dir.glob("*.yaml"))
    )
    known_refs = set(re.findall(r"(?m)^\s*-?\s*clause_id:\s*([^\s#]+)", source_text))
    missing = sorted({ref for rule in rules for ref in rule.norm_refs if ref not in known_refs})
    if missing:
        raise ValueError("Каталог ссылается на неизвестные нормы: " + ", ".join(missing))
    return rules
