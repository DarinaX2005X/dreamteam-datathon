from pathlib import Path

from navidu.contracts.domain import Coverage, Observation
from navidu.rules import evaluate, load_catalog

CATALOG = load_catalog(Path(__file__).parents[3] / "rules" / "catalog")


def obs(index, kind, start, end, *, subjects=(), actor=None, recipient=None, zones=()):
    return Observation(obs_id=f"ob-{index}", kind=kind, start_sec=start, end_sec=end,
                       subjects=list(subjects), actor_id=actor, recipient_id=recipient,
                       zones=list(zones), provenance="temporal_model", model_id="test-model")


def test_count_absent_requires_receipt_sequence_and_verified_visibility():
    observations = [
        obs(1, "cash_handover", 1, 2, subjects=("p1", "o1", "p2"),
            actor="p2", recipient="p1"),
        obs(2, "person_in_zone", 1, 2, subjects=("p2",), zones=("z-client_window-1",)),
        obs(3, "counting_motion", 4, 8, subjects=("p1", "o1")),
    ]
    roles = {"p1": "cashier", "p2": "client", "o1": "unknown"}
    capabilities = {
        "handover_direction": True, "counting_action": True,
        "continuous_client_visibility": True,
    }
    findings, statuses = evaluate(observations, [], roles, CATALOG, capabilities,
                                  {"z-client_window-1": "client_window"})
    assert [item.rule.rule_id for item in findings] == ["V-COUNT-ABSENT"]
    assert next(item for item in statuses if item.rule_id == "V-COUNT-ABSENT").status == "violation"

    findings, _ = evaluate(observations[1:], [], roles, CATALOG, capabilities,
                           {"z-client_window-1": "client_window"})
    assert all(item.rule.rule_id != "V-COUNT-ABSENT" for item in findings)


def test_missing_visibility_never_becomes_absence_conclusion():
    observations = [
        obs(1, "cash_handover", 1, 2, actor="p2", recipient="p1"),
        obs(2, "counting_motion", 4, 8, subjects=("p1",)),
    ]
    roles = {"p1": "cashier", "p2": "client"}
    findings, statuses = evaluate(observations, [], roles, CATALOG, {
        "handover_direction": True, "counting_action": True,
        "continuous_client_visibility": False,
    })
    assert not any(item.rule.rule_id == "V-COUNT-ABSENT" for item in findings)
    status = next(item for item in statuses if item.rule_id == "V-COUNT-ABSENT")
    assert status.status == "not_checkable"
    assert "continuous_client_visibility" in status.not_checkable_windows[0].reason


def test_coverage_gap_suppresses_candidate():
    observations = [
        obs(1, "cash_handover", 1, 2, actor="p2", recipient="p1"),
        obs(2, "counting_motion", 4, 8, subjects=("p1",)),
    ]
    roles = {"p1": "cashier", "p2": "client"}
    findings, statuses = evaluate(observations, [Coverage(
        start_sec=5, end_sec=7, scope="client", reason="occluded"
    )], roles, CATALOG, {
        "handover_direction": True, "counting_action": True,
        "continuous_client_visibility": True,
    })
    assert not any(item.rule.rule_id == "V-COUNT-ABSENT" for item in findings)
    assert next(item for item in statuses if item.rule_id == "V-COUNT-ABSENT").status == "not_checkable"


def test_bag_opening_requires_visible_prior_damage():
    opened = [obs(1, "bag_opened", 10, 12, subjects=("p1", "o1"))]
    roles = {"p1": "cashier", "o1": "unknown"}
    caps = {"bag_damage": True, "bag_opening": True, "continuous_collector_visibility": True}
    findings, _ = evaluate(opened, [], roles, CATALOG, caps)
    assert not any(item.rule.rule_id == "V-BAG-OPENED-ALONE" for item in findings)
    damaged = [obs(2, "bag_integrity_damage", 8, 9, subjects=("o1",)), *opened]
    findings, _ = evaluate(damaged, [], roles, CATALOG, caps)
    assert any(item.rule.rule_id == "V-BAG-OPENED-ALONE" for item in findings)


def test_p42_excludes_client_receipt_and_unknown_is_not_outsider():
    observations = [
        obs(1, "cash_handover", 1, 3, actor="p2", recipient="p1"),
        obs(2, "counting_motion", 1.5, 2.5, subjects=("p1",)),
    ]
    roles = {"p1": "cashier", "p2": "client", "p3": "unknown"}
    caps = {"counting_action": True, "handover_direction": True,
            "continuous_other_positions_visibility": True}
    findings, _ = evaluate(observations, [], roles, CATALOG, caps)
    assert not any(item.rule.rule_id == "V-COUNT-NO-CONTROLLER" for item in findings)
    assert not any(item.rule.rule_id == "V-OUTSIDER" for item in findings)
