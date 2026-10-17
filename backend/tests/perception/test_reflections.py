from navidu.contracts.domain import Observation, Track, TrackSample
from navidu.perception.reflections import apply_reflection_filters, validate_reflections
from navidu.perception.temporal import TemporalVerifier
from navidu.pipeline.interfaces import PerceptionArtifacts, RawDetection, SemanticSignal


def _frames():
    return [{"p1": (0.05, 0.1, 0.25, 0.9), "p2": (0.65, 0.1, 0.85, 0.9)}] * 4


def test_reflection_claim_requires_known_distinct_pair_in_three_simultaneous_frames():
    times = [0.0, 1.0, 2.0, 3.0]
    valid = {"reflections": [{"reflected_track": "p2", "source_track": "p1",
                              "frames": [0, 1, 2]}]}
    assert validate_reflections(valid, _frames(), times) == [{
        "reflected_track": "p2", "source_track": "p1", "start_sec": 0.0, "end_sec": 2.0,
    }]
    for claim in [
        {"reflected_track": "p2", "source_track": "p1", "frames": [0, 1]},
        {"reflected_track": "p2", "source_track": "p1", "frames": [0, 1, 3]},
        {"reflected_track": "p9", "source_track": "p1", "frames": [0, 1, 2]},
        {"reflected_track": "p1", "source_track": "p1", "frames": [0, 1, 2]},
    ]:
        assert validate_reflections({"reflections": [claim]}, _frames(), times) == []


def test_overlapping_boxes_do_not_support_reflection_claim():
    frames = [{"p1": (0.1, 0.1, 0.5, 0.9), "p2": (0.11, 0.1, 0.51, 0.9)}] * 3
    claim = {"reflections": [{"reflected_track": "p2", "source_track": "p1",
                              "frames": [0, 1, 2]}]}
    assert validate_reflections(claim, frames, [0.0, 1.0, 2.0]) == []


def test_filter_is_interval_bounded_and_keeps_physical_source():
    def track(track_id):
        return Track(track_id=track_id, kind="person", first_sec=0, last_sec=3, samples=[
            TrackSample(t=t, bbox=(0.1, 0.1, 0.2, 0.8), score=0.9) for t in range(4)
        ])

    artifacts = PerceptionArtifacts(tracks=[track("p1"), track("p2")], raw_detections=[
        RawDetection(timestamp_sec=t, class_name="person", bbox=(0.1, 0.1, 0.2, 0.8),
                     score=.9, model_id="det", track_id="p2") for t in range(4)
    ], observations=[
        Observation(obs_id="ob-1", kind="phone_visible", start_sec=1, end_sec=2,
                    subjects=["p2"], provenance="temporal_model", model_id="sem"),
        Observation(obs_id="ob-2", kind="phone_visible", start_sec=3, end_sec=3,
                    subjects=["p2"], provenance="temporal_model", model_id="sem"),
    ], semantic_signals=[
        SemanticSignal(kind="reflection", start_sec=0, end_sec=2, subjects=("p2", "p1")),
        SemanticSignal(kind="phone_visible", start_sec=1, end_sec=2, subjects=("p2",)),
    ])
    metrics = apply_reflection_filters(artifacts)
    assert [sample.t for sample in artifacts.tracks[0].samples] == [0, 1, 2, 3]
    assert [sample.t for sample in artifacts.tracks[1].samples] == [3]
    assert [item.timestamp_sec for item in artifacts.raw_detections] == [3]
    assert [item.obs_id for item in artifacts.observations] == ["ob-1", "ob-2"]
    assert artifacts.observations[0].subjects == ["p1"]
    assert metrics == {
        "accepted_interval_count": 1, "affected_duration_sec": 2,
        "removed_track_samples": 3, "removed_raw_detections": 3,
        "removed_keypoints": 0, "removed_observations": 0,
        "removed_semantic_signals": 0, "canonicalized_observations": 1,
        "canonicalized_semantic_signals": 1, "rejected_self_handovers": 0,
    }


def test_observation_crossing_interval_is_removed_whole_not_clipped():
    artifacts = PerceptionArtifacts(observations=[
        Observation(obs_id="ob-1", kind="phone_visible", start_sec=0, end_sec=4,
                    subjects=["p2"], provenance="temporal_model", model_id="sem"),
    ], semantic_signals=[
        SemanticSignal(kind="reflection", start_sec=1, end_sec=2, subjects=("p2", "p1")),
    ])
    metrics = apply_reflection_filters(artifacts)
    assert artifacts.observations == []
    assert metrics["removed_observations"] == 1


def test_reflected_counting_maps_to_physical_source_without_changing_evidence():
    observation = Observation(obs_id="ob-1", kind="counting_motion", start_sec=1, end_sec=2,
        subjects=["p2"], actor_id="p2", provenance="temporal_model", model_id="sem",
        frame_ids=["fr-1000", "fr-2000"])
    artifacts = PerceptionArtifacts(observations=[observation], semantic_signals=[
        SemanticSignal(kind="reflection", start_sec=0, end_sec=3, subjects=("p2", "p1")),
    ])
    apply_reflection_filters(artifacts)
    mapped = artifacts.observations[0]
    assert mapped.subjects == ["p1"] and mapped.actor_id == "p1"
    assert mapped.start_sec == 1 and mapped.end_sec == 2
    assert mapped.frame_ids == observation.frame_ids and mapped.provenance == "temporal_model"


def test_reflected_handover_to_third_person_is_preserved_and_remapped():
    artifacts = PerceptionArtifacts(observations=[
        Observation(obs_id="ob-1", kind="cash_handover", start_sec=1, end_sec=2,
                    subjects=["p2", "p3"], actor_id="p2", recipient_id="p3",
                    provenance="temporal_model", model_id="sem"),
    ], semantic_signals=[
        SemanticSignal(kind="reflection", start_sec=0, end_sec=3, subjects=("p2", "p1")),
    ])
    apply_reflection_filters(artifacts)
    assert artifacts.observations[0].subjects == ["p1", "p3"]
    assert artifacts.observations[0].actor_id == "p1"
    assert artifacts.observations[0].recipient_id == "p3"


def test_reflected_handover_collapsing_to_self_is_rejected():
    artifacts = PerceptionArtifacts(observations=[
        Observation(obs_id="ob-1", kind="cash_handover", start_sec=1, end_sec=2,
                    subjects=["p2", "p1"], actor_id="p2", recipient_id="p1",
                    provenance="temporal_model", model_id="sem"),
    ], semantic_signals=[
        SemanticSignal(kind="reflection", start_sec=0, end_sec=3, subjects=("p2", "p1")),
    ])
    metrics = apply_reflection_filters(artifacts)
    assert artifacts.observations == []
    assert metrics["rejected_self_handovers"] == 1


def test_event_outside_reflection_interval_is_unchanged():
    observation = Observation(obs_id="ob-1", kind="counting_motion", start_sec=4, end_sec=5,
        subjects=["p2"], actor_id="p2", provenance="temporal_model", model_id="sem")
    artifacts = PerceptionArtifacts(observations=[observation], semantic_signals=[
        SemanticSignal(kind="reflection", start_sec=0, end_sec=3, subjects=("p2", "p1")),
    ])
    apply_reflection_filters(artifacts)
    assert artifacts.observations == [observation]


def test_ambiguous_cycle_never_removes_either_person():
    tracks = [Track(track_id=track_id, kind="person", first_sec=0, last_sec=2, samples=[
        TrackSample(t=t, bbox=(0.1, 0.1, 0.2, 0.8), score=.9) for t in range(3)
    ]) for track_id in ("p1", "p2")]
    artifacts = PerceptionArtifacts(tracks=tracks, semantic_signals=[
        SemanticSignal(kind="reflection", start_sec=0, end_sec=2, subjects=("p2", "p1")),
        SemanticSignal(kind="reflection", start_sec=0, end_sec=2, subjects=("p1", "p2")),
    ])
    apply_reflection_filters(artifacts)
    assert [len(track.samples) for track in artifacts.tracks] == [3, 3]


def test_same_reflected_track_with_two_sources_is_ambiguous():
    track = Track(track_id="p2", kind="person", first_sec=0, last_sec=2, samples=[
        TrackSample(t=t, bbox=(0.1, 0.1, 0.2, 0.8), score=.9) for t in range(3)
    ])
    artifacts = PerceptionArtifacts(tracks=[track], semantic_signals=[
        SemanticSignal(kind="reflection", start_sec=0, end_sec=2, subjects=("p2", "p1")),
        SemanticSignal(kind="reflection", start_sec=1, end_sec=2, subjects=("p2", "p3")),
    ])
    metrics = apply_reflection_filters(artifacts)
    assert len(artifacts.tracks[0].samples) == 3
    assert metrics["accepted_interval_count"] == 0


def test_temporal_metrics_expose_accepted_reflection_count():
    tracks = [Track(track_id=track_id, kind="person", first_sec=0, last_sec=2, samples=[
        TrackSample(t=t, bbox=box, score=.9) for t in range(3)
    ]) for track_id, box in (("p1", (0.05, .1, .25, .9)), ("p2", (.65, .1, .85, .9)))]
    verifier = TemporalVerifier.__new__(TemporalVerifier)
    verifier.model_id = "semantic-test"
    verifier.capabilities = {}
    verifier.last_metrics = {}
    verifier._infer = lambda frames, frame_tracks, should_stop: {
        "events": [], "roles": [], "same_cash_batches": [],
        "reflections": [{"reflected_track": "p2", "source_track": "p1", "frames": [0, 1, 2]}],
    }
    artifacts = PerceptionArtifacts()
    verifier.observe([(0, None), (1, None), (2, None)], tracks, artifacts)
    assert verifier.last_metrics["accepted_reflection_count"] == 1


def test_matching_overlapping_windows_preserve_action_across_their_union():
    observation = Observation(obs_id="ob-1", kind="counting_motion", start_sec=1, end_sec=5,
        subjects=["p2"], actor_id="p2", provenance="temporal_model", model_id="sem")
    artifacts = PerceptionArtifacts(observations=[observation], semantic_signals=[
        SemanticSignal(kind="reflection", start_sec=0, end_sec=3, subjects=("p2", "p1")),
        SemanticSignal(kind="reflection", start_sec=2, end_sec=6, subjects=("p2", "p1")),
    ])
    metrics = apply_reflection_filters(artifacts)
    assert artifacts.observations[0].actor_id == "p1"
    assert metrics["accepted_interval_count"] == 1
    assert metrics["affected_duration_sec"] == 6


def test_mixed_full_and_partial_reflections_cannot_keep_an_unsupported_participant():
    observation = Observation(obs_id="ob-1", kind="cash_handover", start_sec=1, end_sec=5,
        subjects=["p2", "p4"], actor_id="p2", recipient_id="p4",
        provenance="temporal_model", model_id="sem")
    artifacts = PerceptionArtifacts(observations=[observation], semantic_signals=[
        SemanticSignal(kind="reflection", start_sec=0, end_sec=6, subjects=("p2", "p1")),
        SemanticSignal(kind="reflection", start_sec=4, end_sec=7, subjects=("p4", "p3")),
        SemanticSignal(kind="cash_handover", start_sec=1, end_sec=5, subjects=("p2", "p4")),
    ])
    apply_reflection_filters(artifacts)
    assert artifacts.observations == []
    assert all(item.kind == "reflection" for item in artifacts.semantic_signals)
