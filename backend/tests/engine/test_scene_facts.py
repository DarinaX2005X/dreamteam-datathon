from navidu.contracts.domain import Track, TrackRole, TrackSample
from navidu.facts import derive_coverage, derive_observations
from navidu.scene import assign_roles, infer_layout


def _track(track_id, kind, box, *, object_class=None, start=0, end=5):
    return Track(track_id=track_id, kind=kind, object_class=object_class,
                 first_sec=start, last_sec=end, samples=[
                     TrackSample(t=t, bbox=box, score=0.8) for t in (start, (start + end) / 2, end)
                 ])


def _mirror(track):
    return track.model_copy(update={"samples": [sample.model_copy(update={
        "bbox": (1 - sample.bbox[2], sample.bbox[1], 1 - sample.bbox[0], sample.bbox[3])
    }) for sample in track.samples]})


def test_layout_uses_observed_objects_and_roles_survive_mirror():
    tracks = [
        _track("o1", "object", (0.25, 0.5, 0.75, 0.9), object_class="cash_desk"),
        _track("p1", "person", (0.05, 0.2, 0.3, 0.95), end=10),
        _track("p2", "person", (0.7, 0.2, 0.95, 0.95), start=2, end=4),
    ]
    layout = infer_layout(tracks)
    roles = assign_roles(tracks, layout)
    mirrored = [_mirror(track) for track in tracks]
    mirrored_roles = assign_roles(mirrored, infer_layout(mirrored))
    assert roles["p1"] == mirrored_roles["p1"] == "cashier"
    assert roles["p2"] == mirrored_roles["p2"] == "client"
    assert {zone.kind for zone in layout.zones} >= {"cash_desk", "cashier_post", "client_window"}


def test_no_object_anchor_means_no_invented_zones_or_roles():
    person = _track("p1", "person", (0.2, 0.2, 0.4, 0.9))
    layout = infer_layout([person])
    assert layout.zones == []
    assert layout.camera_note
    assert assign_roles([person], layout)["p1"] == "unknown"
    assert derive_observations([person], layout, {"p1": "unknown"}) == []


def test_stable_relative_person_tracks_can_anchor_roles_without_fixed_coordinates():
    cashier = _track("p1", "person", (0.6, 0.2, 0.85, 0.95), end=12)
    visitor = _track("p2", "person", (0.1, 0.2, 0.35, 0.95), start=3, end=5)
    layout = infer_layout([cashier, visitor])
    assert {zone.kind for zone in layout.zones} == {"cashier_post", "client_window"}
    assert "относительным траекториям" in layout.camera_note
    assert assign_roles([cashier, visitor], layout) == {"p1": "cashier", "p2": "client"}


def test_cashier_post_uses_dwell_cluster_and_excludes_excursion():
    cashier = Track(track_id="p1", kind="person", first_sec=0, last_sec=5, samples=[
        TrackSample(t=t, bbox=box, score=0.9) for t, box in [
            (0, (0.05, 0.2, 0.25, 0.9)), (1, (0.06, 0.2, 0.26, 0.9)),
            (2, (0.05, 0.2, 0.25, 0.9)), (3, (0.07, 0.2, 0.27, 0.9)),
            (4, (0.7, 0.2, 0.9, 0.9)),
        ]
    ])
    visitor = _track("p2", "person", (0.65, 0.2, 0.9, 0.9), start=1, end=2)
    layout = infer_layout([cashier, visitor])
    post = next(zone for zone in layout.zones if zone.kind == "cashier_post")
    assert max(point[0] for point in post.polygon) < 0.5
    roles = assign_roles([cashier, visitor], layout)
    observations = derive_observations([cashier, visitor], layout, roles, max_gap_sec=1.5)
    cashier_post = [item for item in observations if item.subjects == ["p1"] and
                    post.zone_id in item.zones]
    assert cashier_post[0].end_sec == 3
    departure = [item for item in observations if item.kind == "person_left_zone" and
                 item.subjects == ["p1"] and post.zone_id in item.zones]
    assert departure[0].start_sec == 3
    assert departure[0].end_sec == 4


def test_client_window_dwell_cluster_preserves_visibly_observed_exit():
    cashier = _track("p1", "person", (0.65, 0.2, 0.9, 0.9), end=8)
    client = Track(track_id="p2", kind="person", first_sec=1, last_sec=5, samples=[
        TrackSample(t=t, bbox=box, score=0.9) for t, box in [
            (1, (0.08, 0.2, 0.28, 0.9)), (2, (0.09, 0.2, 0.29, 0.9)),
            (3, (0.08, 0.2, 0.28, 0.9)), (4, (0.48, 0.2, 0.68, 0.9)),
            (5, (0.7, 0.2, 0.9, 0.9)),
        ]
    ])
    layout = infer_layout([cashier, client])
    window = next(zone for zone in layout.zones if zone.kind == "client_window")
    assert max(point[0] for point in window.polygon) < 0.5
    observations = derive_observations([cashier, client], layout,
                                       {"p1": "cashier", "p2": "client"}, max_gap_sec=1.5)
    departures = [item for item in observations if item.kind == "person_left_zone"
                  and item.subjects == ["p2"] and item.zones == [window.zone_id]]
    assert [(item.start_sec, item.end_sec) for item in departures] == [(3, 4)]


def test_client_window_requires_repeated_observed_dwell_samples():
    cashier = _track("p1", "person", (0.65, 0.2, 0.9, 0.9), end=8)
    one_point = Track(track_id="p2", kind="person", first_sec=2, last_sec=2, samples=[
        TrackSample(t=2, bbox=(0.1, 0.2, 0.3, 0.9), score=0.9),
    ])
    missing = Track(track_id="p3", kind="person", first_sec=2, last_sec=2, samples=[])
    for client in (one_point, missing):
        layout = infer_layout([cashier, client])
        assert not any(zone.kind == "client_window" for zone in layout.zones)
        observations = derive_observations([cashier, client], layout,
                                           {"p1": "cashier", client.track_id: "client"})
        assert not any(item.kind == "person_left_zone" and
                       item.subjects == [client.track_id] for item in observations)


def test_one_fps_samples_form_observation_when_cadence_is_configured():
    person = _track("p1", "person", (0.1, 0.2, 0.3, 0.9), end=3)
    visitor = _track("p2", "person", (0.7, 0.2, 0.9, 0.9), start=1, end=2)
    layout = infer_layout([person, visitor])
    roles = assign_roles([person, visitor], layout)
    observations = derive_observations([person, visitor], layout, roles, max_gap_sec=1.5)
    assert any(item.subjects == ["p1"] and item.end_sec > item.start_sec
               for item in observations)


def test_grounded_semantic_roles_anchor_layout_without_reassigning_by_duration():
    cashier = _track("p1", "person", (0.6, 0.2, 0.85, 0.95), end=4)
    cashier.role = TrackRole(value="cashier")
    client = _track("p2", "person", (0.1, 0.2, 0.35, 0.95), end=10)
    client.role = TrackRole(value="client")
    layout = infer_layout([cashier, client])
    roles = assign_roles([cashier, client], layout)
    assert roles == {"p1": "cashier", "p2": "client"}
    post = next(zone for zone in layout.zones if zone.kind == "cashier_post")
    assert min(point[0] for point in post.polygon) > 0.5


def test_decoder_gaps_create_coverage_instead_of_absence_facts():
    coverage = derive_coverage([0.0, 0.25, 2.0, 2.25], expected_period_sec=0.25,
                               duration_sec=3)
    assert any(item.start_sec == 0.25 and item.end_sec == 2 for item in coverage)
    assert all(item.scope == "whole" for item in coverage)
