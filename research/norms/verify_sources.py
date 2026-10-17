"""Verify each published quotation against its exact official-source node.

Run from the repository root; no network is used. --schema additionally requires
the project's SourceClause model and validates the full public source contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = Path(__file__).resolve().parent


def normalise_whitespace(value: str) -> str:
    # Preserve punctuation and letters; only PDF/API line wrapping is ignored.
    return " ".join(value.split())


def verify(schema: bool) -> int:
    clause_model = None
    if schema:
        sys.path.insert(0, str(ROOT / "backend"))
        from navidu.contracts.rules import SourceClause

        clause_model = SourceClause
    mapping = json.loads((RESEARCH / "quote-map.json").read_text(encoding="utf-8"))
    snapshots = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in (RESEARCH / "primary").glob("*.json")
    }
    seen = set()
    count = 0
    for path in sorted((ROOT / "rules" / "sources").glob("*.yaml")):
        clauses = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(clauses, list):
            raise ValueError(f"{path.name}: expected a list")
        for clause in clauses:
            clause_id = clause["clause_id"]
            if clause_id in seen:
                raise ValueError(f"Duplicate clause ID: {clause_id}")
            seen.add(clause_id)
            if clause_model:
                clause_model.model_validate(clause)
            if not clause["verified_verbatim"]:
                if clause.get("quote"):
                    raise ValueError(f"{clause_id}: unverified quote must be absent")
                continue
            snapshot_name, node_id = mapping[clause_id]
            snapshot = snapshots[snapshot_name]
            node = next(n for n in snapshot["content"] if n.get("id") == node_id)
            quote = clause.get("quote")
            if not quote or normalise_whitespace(quote) not in normalise_whitespace(node.get("text", "")):
                raise ValueError(f"{clause_id}: quote differs from official node {node_id}")
            if clause["document"]["source_url"] != snapshot["source_url"]:
                raise ValueError(f"{clause_id}: wrong provenance URL")
            if str(clause["document"]["checked_at"]) != snapshot["checked_at"]:
                raise ValueError(f"{clause_id}: check date differs from snapshot")
            if not clause["visual_test"].strip() or not clause["not_visual"].strip():
                raise ValueError(f"{clause_id}: missing interpretation boundary")
            count += 1
    if set(mapping) != seen:
        raise ValueError(f"Quote map and sources differ: {set(mapping) ^ seen}")
    not_visual = yaml.safe_load((ROOT / "rules" / "sources" / "not-visual.yaml").read_text(encoding="utf-8"))
    if len(not_visual) < 5:
        raise ValueError("At least five nonvisual aspects are required")
    print(f"Verified {count} quotations; {len(not_visual)} nonvisual clauses; "
          f"unique IDs and provenance valid; schema={'validated' if schema else 'not requested'}.")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", action="store_true")
    args = parser.parse_args()
    verify(args.schema)
