from pathlib import Path

import yaml

from navidu.contracts.rules import (
    RuleDefinition,
    RulesResponse,
    RuleView,
    SourceClause,
    SourceClauseView,
)


def records(path: Path):
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        # Top-level named collections are also convenient for a single catalog file.
        for name in ["clauses", "rules", "sources", "not_checkable"]:
            if isinstance(value.get(name), list):
                return value[name]
        return [value]
    raise ValueError(f"Unsupported YAML collection: {path.name}")


def read_catalog(rules_dir: Path) -> RulesResponse:
    sources = {}
    not_checkable = []
    for path in sorted((rules_dir / "sources").glob("*.yaml")):
        for value in records(path):
            clause = SourceClause.model_validate(value)
            source = SourceClauseView.from_source(clause)
            sources[clause.clause_id] = source
            if path.stem == "not-visual":
                not_checkable.append(source)
    rules = []
    for path in sorted((rules_dir / "catalog").glob("*.yaml")):
        for value in records(path):
            definition = RuleDefinition.model_validate(value)
            referenced = [sources[ref] for ref in definition.norm_refs if ref in sources]
            # Unknown clauses cannot silently become active rules.
            enabled = definition.enabled and len(referenced) == len(definition.norm_refs)
            rules.append(RuleView(
                rule_id=definition.rule_id, type_label=definition.type_label,
                norm_refs=definition.norm_refs, requires_visible=definition.requires_visible,
                visible_sign=definition.what_seen, why_may_violate=definition.why_may_violate,
                not_claimed=definition.not_claimed, enabled=enabled, sources=referenced,
            ))
    return RulesResponse(rules=rules, not_checkable=not_checkable)
