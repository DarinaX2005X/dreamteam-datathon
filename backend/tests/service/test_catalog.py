from pathlib import Path

from navidu.service.catalog import read_catalog


def test_project_catalog_loads_and_sources_are_safe():
    root = Path(__file__).resolve().parents[3]
    result = read_catalog(root / "rules")
    assert result.rules
    assert result.not_checkable
    for rule in result.rules:
        assert rule.norm_refs
        if rule.enabled:
            assert set(rule.norm_refs) == {source.clause_id for source in rule.sources}
        for source in rule.sources:
            assert source.verified_verbatim or source.quote is None
