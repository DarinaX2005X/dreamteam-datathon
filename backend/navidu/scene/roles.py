"""Appearance-free role assignment from dwell in accepted scene zones."""

from __future__ import annotations

from navidu.contracts.domain import Role, SceneLayout, Track


def _inside(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    previous = len(polygon) - 1
    for current, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[previous]
        crosses = (yi > y) != (yj > y)
        if crosses and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        previous = current
    return inside


def assign_roles(tracks: list[Track], layout: SceneLayout) -> dict[str, Role]:
    zones = {zone.kind: zone for zone in layout.zones}
    result: dict[str, Role] = {}
    dwell: dict[str, dict[str, int]] = {}
    for track in tracks:
        if track.kind != "person":
            continue
        counts: dict[str, int] = {}
        for kind, zone in zones.items():
            counts[kind] = sum(
                _inside((s.bbox[0] + s.bbox[2]) / 2, s.bbox[3], zone.polygon)
                for s in track.samples
            )
        dwell[track.track_id] = counts
        result[track.track_id] = track.role.value if track.role is not None else "unknown"
    if "cashier_post" in zones and dwell:
        cashier, count = max(dwell.items(), key=lambda item: item[1].get("cashier_post", 0))
        if count.get("cashier_post", 0) > 0 and result[cashier] == "unknown":
            result[cashier] = "cashier"
    if "client_window" in zones:
        for track_id, counts in dwell.items():
            if result[track_id] == "unknown" and counts.get("client_window", 0) > 0:
                result[track_id] = "client"
    # Unknown people are deliberately never promoted to outsider or staff.
    return result
