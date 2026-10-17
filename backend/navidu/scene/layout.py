"""Conservative scene geometry inferred only from observed tracks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from navidu.contracts.domain import SceneLayout, Track, Zone

_OBJECT_ZONE = {
    "cash_desk": "cash_desk",
    "counter": "cash_desk",
    "dining table": "cash_desk",
    "transfer_tray": "transfer_tray",
    "storage": "storage",
    "safe": "storage",
    "service_door": "service_door",
}


def _envelope(track: Track, margin: float = 0.02) -> list[tuple[float, float]] | None:
    if not track.samples:
        return None
    x1 = max(0.0, min(sample.bbox[0] for sample in track.samples) - margin)
    y1 = max(0.0, min(sample.bbox[1] for sample in track.samples) - margin)
    x2 = min(1.0, max(sample.bbox[2] for sample in track.samples) + margin)
    y2 = min(1.0, max(sample.bbox[3] for sample in track.samples) + margin)
    if x2 - x1 < 0.01 or y2 - y1 < 0.01:
        return None
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def infer_layout(tracks: Iterable[Track]) -> SceneLayout:
    """Build zones around real stable objects; never synthesize camera coordinates."""
    material = list(tracks)
    grouped: dict[str, list[Track]] = defaultdict(list)
    for track in material:
        if track.kind == "object" and track.object_class in _OBJECT_ZONE:
            grouped[_OBJECT_ZONE[track.object_class]].append(track)
    zones: list[Zone] = []
    for kind in sorted(grouped):
        candidates = sorted(grouped[kind], key=lambda item: item.last_sec - item.first_sec,
                            reverse=True)
        polygon = _envelope(candidates[0]) if candidates else None
        if polygon:
            zones.append(Zone(zone_id=f"z-{kind}-1", kind=kind, polygon=polygon))
    # A persistent person adjacent to a detected desk is a geometric anchor for a
    # post. The polygon follows that real trajectory rather than a camera template.
    desk = next((zone for zone in zones if zone.kind == "cash_desk"), None)
    people = [track for track in material if track.kind == "person" and track.samples]
    if people and (desk or len(people) >= 2):
        persistent = max(people, key=lambda item: item.last_sec - item.first_sec)
        visitors = [item for item in people if item.track_id != persistent.track_id]
        persistent_duration = persistent.last_sec - persistent.first_sec
        visitor_duration = max((item.last_sec - item.first_sec for item in visitors), default=0)
        stable_relation = desk is not None or persistent_duration >= max(1.0, visitor_duration * 1.5)
        post = _envelope(persistent, margin=0.04) if stable_relation else None
        if post:
            zones.append(Zone(zone_id="z-cashier_post-1", kind="cashier_post", polygon=post))
        if visitors:
            visitor = max(visitors, key=lambda item: len(item.samples))
            window = _envelope(visitor, margin=0.04) if stable_relation else None
            if window:
                zones.append(Zone(zone_id="z-client_window-1", kind="client_window",
                                  polygon=window))
    note = None
    if not zones:
        note = "Ориентиры сцены не распознаны; зоны не были придуманы автоматически."
    elif not any(zone.kind in {"cash_desk", "transfer_tray"} for zone in zones):
        note = (
            "Пост и сторона посетителя оценены по устойчивым относительным траекториям; "
            "рабочая поверхность не распознана."
        )
    return SceneLayout(revision=1, zones=zones, camera_note=note)
