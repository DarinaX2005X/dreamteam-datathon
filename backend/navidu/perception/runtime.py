"""Warm, bounded local perception. API status never loads model weights."""

from __future__ import annotations

import os
import threading
from collections import deque
from collections.abc import Callable, Iterable
from typing import Any

from navidu.contracts.domain import Observation, Track, TrackRole, TrackSample
from navidu.contracts.pipeline import PipelineConfig, PipelineEvent, RuntimeStatus
from navidu.contracts.units import frame_id
from navidu.pipeline.interfaces import PerceptionArtifacts, RawDetection

from .assets import ModelAssetsError, missing_files, model_entry, read_manifest, verify_model
from .configuration import configuration_snapshot, effective_version, snapshot_hash
from .validation import SEMANTIC_CAPABILITIES, validated_capabilities, validation_binding

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

_SEMANTIC_ID = "internvl3_5-1b"
_UNAVAILABLE = {
    "cash_detection": False, "handover_direction": False, "counting_action": False,
    "bag_damage": False, "bag_opening": False, "personal_item_semantics": False,
    "continuous_client_visibility": False, "continuous_cashier_visibility": False,
    "continuous_other_positions_visibility": False, "continuous_collector_visibility": False,
    "continuous_desk_visibility": False, "cash_on_desk": False, "keypoints": False,
    "authorization_status": False,
}


def _settings() -> tuple[str, str, bool]:
    return (
        os.getenv("NAVIDU_DETECTOR_MODEL", "rfdetr-nano-onnx"),
        os.getenv("NAVIDU_SEMANTIC_MODEL", _SEMANTIC_ID),
        os.getenv("NAVIDU_REQUIRE_SEMANTIC", "1").lower() not in {"0", "false", "no"},
    )


def _selected_ids(detector_id: str, semantic_id: str, require_semantic: bool) -> list[str]:
    adapter_id = os.getenv("NAVIDU_PROPOSAL_ADAPTER", "").strip()
    if adapter_id and not require_semantic:
        raise ModelAssetsError("INVALID_CONFIG: адаптер предложений требует базовую временную модель.")
    return [detector_id] + ([semantic_id] if require_semantic else []) + ([adapter_id] if adapter_id else [])


class LocalPerceptionRuntime:
    def __init__(self):
        self._lock = threading.RLock()
        self._detector = None
        self._semantic = None
        self._prepared_key = None
        self._versions: dict[str, str] = {}
        self._error: str | None = None
        self._loading = False
        self._pipeline_version: str | None = None
        self._validated_capabilities: dict[str, bool] = {}
        self._validation_limitations: list[str] = []
        self.quality_binding: str | None = None
        self._required_semantic_capabilities: set[str] = set(SEMANTIC_CAPABILITIES)

    @staticmethod
    def _configuration_key(config: PipelineConfig) -> tuple:
        detector_id, semantic_id, require_semantic = _settings()
        manifest = read_manifest(config.models_dir)
        entries = [model_entry(manifest, model_id)
                   for model_id in _selected_ids(detector_id, semantic_id, require_semantic)]
        snapshot = configuration_snapshot(config, entries, require_semantic=require_semantic)
        return str(config.models_dir.resolve()), snapshot_hash(snapshot)

    def runtime_status(self, config: PipelineConfig) -> RuntimeStatus:
        detector_id, semantic_id, require_semantic = _settings()
        limitations = []
        analysis_mode = ("diagnostic" if config.diagnostic_only or not require_semantic else
                         "experimental" if config.experimental_analysis else "unvalidated")
        try:
            manifest = read_manifest(config.models_dir)
            for model_id in _selected_ids(detector_id, semantic_id, require_semantic):
                entry = model_entry(manifest, model_id)
                absent = missing_files(config.models_dir, entry)
                if absent:
                    raise ModelAssetsError("MODEL_MISSING: " + ", ".join(absent))
            current_key = self._configuration_key(config)
        except ModelAssetsError as error:
            return RuntimeStatus(model_status="missing", device=config.device,
                                 limitations=[str(error)], pipeline_version=config.pipeline_version,
                                 analysis_ready=False, analysis_mode=analysis_mode)
        except RuntimeError as error:
            return RuntimeStatus(model_status="error", device=config.device,
                                 limitations=[str(error)], pipeline_version=config.pipeline_version,
                                 analysis_ready=False, analysis_mode=analysis_mode)
        if not require_semantic:
            limitations.append("Диагностический профиль: временные действия не анализируются; соответствующие правила непроверяемы.")
        if self._error:
            return RuntimeStatus(model_status="error", device=config.device,
                                 limitations=[self._error], pipeline_version=config.pipeline_version,
                                 analysis_ready=False, analysis_mode=analysis_mode)
        prepared = (self._prepared_key == current_key
                    and self._detector is not None
                    and (not require_semantic or self._semantic is not None)
                    and self._pipeline_version is not None)
        state = "ready" if prepared and not self._loading else "loading"
        if not prepared and not self._loading:
            limitations.append("Файлы моделей найдены; локальная модель ещё не загружена.")
        device = self._detector.device if self._detector is not None else config.device
        if self._semantic is not None and self._semantic.device != device:
            device = f"{device}+{self._semantic.device}"
        validated = (bool(self._required_semantic_capabilities)
                     and all(self._validated_capabilities.get(name, False)
                             for name in self._required_semantic_capabilities))
        analysis_ready = (state == "ready" and require_semantic and not config.diagnostic_only
                          and (validated or config.experimental_analysis))
        if require_semantic:
            if config.experimental_analysis:
                limitations.append("Исходная модель: качество ещё не оценено.")
            else:
                limitations.extend(self._validation_limitations or (
                    ["Проверка качества анализа действий ещё не завершена."] if not analysis_ready else []))
        return RuntimeStatus(model_status=state, device=device, limitations=limitations,
                             pipeline_version=self._pipeline_version if state == "ready" else config.pipeline_version,
                             analysis_ready=analysis_ready,
                             analysis_mode=("experimental" if config.experimental_analysis and analysis_ready else
                                            "validated" if analysis_ready and validated else analysis_mode))

    def prepare(self, config: PipelineConfig) -> None:
        detector_id, semantic_id, require_semantic = _settings()
        key = self._configuration_key(config)
        with self._lock:
            if (key == self._prepared_key and self._detector is not None
                    and (not require_semantic or self._semantic is not None)):
                return
            self._loading, self._error = True, None
            try:
                manifest = read_manifest(config.models_dir)
                detector_entry = model_entry(manifest, detector_id)
                versions = {detector_id: verify_model(config.models_dir, detector_entry)}
                semantic_entry = model_entry(manifest, semantic_id) if require_semantic else None
                if semantic_entry:
                    versions[semantic_id] = verify_model(config.models_dir, semantic_entry)
                adapter_id = os.getenv("NAVIDU_PROPOSAL_ADAPTER", "").strip()
                adapter_entry = model_entry(manifest, adapter_id) if adapter_id else None
                if adapter_entry:
                    if config.analysis_fps < 2:
                        raise ModelAssetsError("INVALID_CONFIG: для адаптера нужна частота анализа не ниже 2 кадров/с.")
                    versions[adapter_id] = verify_model(config.models_dir, adapter_entry)
                from .detector import RfDetrOnnx

                detector = RfDetrOnnx(config.models_dir, detector_entry, config.device)
                semantic = None
                if semantic_entry:
                    from .temporal import TemporalVerifier

                    semantic = TemporalVerifier(config.models_dir, semantic_entry, config.device)
                    if adapter_entry:
                        from .proposal import ProposalVerifier

                        semantic.proposal = ProposalVerifier(semantic, config.models_dir, adapter_entry,
                                                            semantic_entry, versions[semantic_id])
                snapshot = configuration_snapshot(config,
                    [detector_entry] + ([semantic_entry] if semantic_entry else [])
                    + ([adapter_entry] if adapter_entry else []),
                    require_semantic=require_semantic)
                devices = {"detector": detector.device,
                           "semantic": semantic.device if semantic is not None else "disabled",
                           "semantic_dtype": semantic.dtype_name if semantic is not None else "disabled"}
                capabilities, limitations = validated_capabilities(config.models_dir, snapshot,
                    devices=devices, end_to_end_only=True)
                from navidu.rules.catalog import load_catalog

                required_semantic = {name for rule in load_catalog(config.rules_dir / "catalog")
                    if rule.enabled for name in rule.requires_capabilities
                    if name in SEMANTIC_CAPABILITIES}
                if semantic is not None:
                    # Evaluation must exercise candidate features in order to
                    # measure them. The explicit diagnostic flag is never read
                    # from the environment, and its AnalysisRun cannot be
                    # published as ready by the pipeline or service.
                    semantic.capabilities = (
                        dict.fromkeys(SEMANTIC_CAPABILITIES, True)
                        if config.diagnostic_only or config.experimental_analysis else capabilities
                    )
                self._detector, self._semantic = detector, semantic
                self._versions, self._prepared_key = versions, key
                self._validated_capabilities = capabilities if semantic is not None else {}
                self._required_semantic_capabilities = required_semantic
                self._validation_limitations = limitations if semantic is not None else []
                self.quality_binding = validation_binding(snapshot, devices=devices)
                self._pipeline_version = effective_version(snapshot, devices=devices)
            except Exception as error:
                self._error = str(error)
                raise
            finally:
                self._loading = False

    def perceive(
        self,
        frames: Iterable[tuple[float, Any]],
        *,
        config: PipelineConfig,
        emit: Callable[[PipelineEvent], None],
        should_stop: Callable[[], bool],
    ) -> PerceptionArtifacts:
        import numpy as np
        import supervision as sv

        self.prepare(config)
        with self._lock:
            detector = self._detector
            artifacts = PerceptionArtifacts(model_versions=dict(self._versions), capabilities={
                **_UNAVAILABLE, "person_detection": True, "phone_detection": True, "bag_detection": True,
            })
            # A separate tracker per class prevents a phone or bag inheriting a
            # person's ID. Track state belongs to a video, model state is shared.
            trackers = {}
            tracks: dict[str, Track] = {}
            id_map: dict[tuple[int, int], str] = {}
            next_person, next_object = 1, 1
            last_emit, previous_t = -5.0, -1.0
            semantic_frames = []
            next_semantic_t = 0.0
            last_semantic_end = -1.0
            proposal = getattr(self._semantic, "proposal", None)
            dense_frames = deque(maxlen=48)
            proposal_frames = []
            next_proposal_t, last_proposal_end = 0.0, -1.0
            pending_centers = []
            refined_intervals = []

            def observe_window(window, *, baseline=True):
                nonlocal last_semantic_end
                previous = {item.obs_id: item.model_dump_json() for item in artifacts.observations}
                self._semantic.observe(window, list(tracks.values()), artifacts,
                                       should_stop=should_stop)
                if baseline:
                    last_semantic_end = window[-1][0]
                changed = [item for item in artifacts.observations
                           if previous.get(item.obs_id) != item.model_dump_json()]
                emit(PipelineEvent(phase="perception", processed_until_sec=previous_t,
                    observations_delta=changed,
                    message="Обработан следующий фрагмент записи."))

            def propose_window(window):
                nonlocal last_proposal_end
                candidate = proposal.propose(window, should_stop=should_stop)
                last_proposal_end = window[-1][0]
                if candidate.candidate and candidate.time_sec is not None \
                        and not any(start <= candidate.time_sec <= end for start, end in refined_intervals) \
                        and candidate.time_sec not in pending_centers:
                    pending_centers.append(candidate.time_sec)

            def refine_ready(timestamp, *, final=False):
                from .proposal import refinement_window

                for center in list(pending_centers):
                    if not final and timestamp < center + (proposal.window_frames - 1) / (2 * proposal.sample_fps):
                        continue
                    pending_centers.remove(center)
                    if any(start <= center <= end for start, end in refined_intervals):
                        continue
                    window = refinement_window(list(dense_frames), center, proposal.window_frames)
                    # This is the unchanged base verifier. A learned candidate
                    # itself never becomes an observation, role or capability.
                    observe_window(window, baseline=False)
                    refined_intervals.append((window[0][0], window[-1][0]))

            for timestamp, rgb in frames:
                if should_stop():
                    raise RuntimeError("ANALYSIS_CANCELLED")
                if not np.isfinite(timestamp) or timestamp < 0 or timestamp <= previous_t:
                    raise ValueError("Frame timestamps must be finite and strictly increasing")
                previous_t = timestamp
                boxes, class_ids, scores = detector.predict(rgb)
                dimensions = np.array([rgb.shape[1], rgb.shape[0], rgb.shape[1], rgb.shape[0]])
                # Empty detections still advance each class tracker. Otherwise a
                # person can inherit an old ID after being absent for many frames.
                for class_id in sorted(set(class_ids.tolist()) | set(trackers)):
                    label = detector.labels.get(str(class_id), "unknown")
                    mask = class_ids == class_id
                    if class_id not in trackers:
                        trackers[class_id] = sv.ByteTrack(
                            frame_rate=max(1, round(config.analysis_fps)),
                            # Supervision counts updates after birth. One update
                            # confirms a late-arriving object on its second frame.
                            minimum_consecutive_frames=1,
                            # ByteTrack scales this 30 fps value by frame_rate.
                            lost_track_buffer=60,
                        )
                    tracker = trackers[class_id]
                    source_indices = np.flatnonzero(mask)
                    detections = sv.Detections(xyxy=boxes[mask] * dimensions,
                                              confidence=scores[mask], class_id=class_ids[mask],
                                              data={"source_index": source_indices})
                    active = tracker.update_with_detections(detections)
                    assigned_ids = {}
                    for box, confidence, source_id, source_index in zip(
                            active.xyxy, active.confidence, active.tracker_id,
                            active.data.get("source_index", [])):
                        key = (class_id, int(source_id))
                        if key not in id_map:
                            if class_id == 1:
                                id_map[key], next_person = f"p{next_person}", next_person + 1
                            else:
                                id_map[key], next_object = f"o{next_object}", next_object + 1
                        track_id = id_map[key]
                        assigned_ids[int(source_index)] = track_id
                        time_sec = round(timestamp, 3)
                        kind = "person" if class_id == 1 else "object"
                        canonical_label = {"cell phone": "phone", "handbag": "bag", "backpack": "bag", "suitcase": "bag"}.get(label, label)
                        if track_id not in tracks:
                            tracks[track_id] = Track(track_id=track_id, kind=kind,
                                object_class=canonical_label if kind == "object" else None,
                                first_sec=time_sec, last_sec=time_sec,
                                role=TrackRole(value="unknown") if kind == "person" else None)
                        track = tracks[track_id]
                        track.last_sec = time_sec
                        track.samples.append(TrackSample(t=time_sec, bbox=tuple(float(x) for x in (box / dimensions).clip(0, 1)), score=float(confidence)))
                    for source_index, box, confidence in zip(source_indices, boxes[mask], scores[mask]):
                        artifacts.raw_detections.append(RawDetection(
                            timestamp_sec=round(timestamp, 3), class_name=label,
                            bbox=tuple(float(x) for x in box), score=float(confidence),
                            model_id=detector.model_id, track_id=assigned_ids.get(int(source_index)),
                        ))
                # Keep the ordinary base-model sweep even when an optional
                # learned proposer requests denser context around a candidate.
                if self._semantic is not None and timestamp + 1e-7 >= next_semantic_t:
                    semantic_frames.append((round(timestamp, 3), rgb.copy()))
                    while next_semantic_t <= timestamp + 1e-7:
                        next_semantic_t += 1.0 / self._semantic.sample_fps
                    if len(semantic_frames) >= self._semantic.window_frames:
                        observe_window(semantic_frames)
                        semantic_frames = semantic_frames[-2:]
                if proposal is not None and timestamp + 1e-7 >= next_proposal_t:
                    dense_frame = (round(timestamp, 3), rgb.copy())
                    dense_frames.append(dense_frame)
                    proposal_frames.append(dense_frame)
                    while next_proposal_t <= timestamp + 1e-7:
                        next_proposal_t += 1.0 / proposal.sample_fps
                    if len(proposal_frames) >= proposal.window_frames:
                        propose_window(proposal_frames)
                        proposal_frames = proposal_frames[-(proposal.window_frames - proposal.stride_frames):]
                    refine_ready(timestamp)
                if timestamp - last_emit >= 5.0:
                    emit(PipelineEvent(phase="perception", processed_until_sec=timestamp,
                                       message="Анализируются последовательные кадры записи."))
                    last_emit = timestamp
            if (self._semantic is not None and len(semantic_frames) >= 2
                    and semantic_frames[-1][0] > last_semantic_end):
                observe_window(semantic_frames)
            if proposal is not None:
                if len(dense_frames) >= 2 and dense_frames[-1][0] > last_proposal_end:
                    propose_window(list(dense_frames)[-proposal.window_frames:])
                refine_ready(previous_t, final=True)
            if self._semantic is not None:
                from .reflections import apply_reflection_filters
                from .temporal import apply_semantic_roles

                artifacts.tracks = list(tracks.values())
                reflection_metrics = apply_reflection_filters(artifacts)
                artifacts.diagnostics["reflections"] = reflection_metrics
                self._semantic.last_metrics.update({
                    f"reflection_{key}": value for key, value in reflection_metrics.items()
                })
                apply_semantic_roles(artifacts.tracks, artifacts.semantic_signals)
            else:
                artifacts.tracks = list(tracks.values())
            artifacts.observations.extend(_object_observations(artifacts.tracks, detector.model_id,
                                          start_index=len(artifacts.observations) + 1,
                                          max_gap_sec=max(0.5, 1.5 / config.analysis_fps)))
            if previous_t >= 0:
                emit(PipelineEvent(phase="perception", processed_until_sec=previous_t,
                                   observations_delta=artifacts.observations))
            return artifacts


def _object_observations(tracks: list[Track], model_id: str, *, start_index: int,
                         max_gap_sec: float = 0.75) -> list[Observation]:
    output = []
    for track in tracks:
        kind = {"phone": "phone_visible", "bag": "bag_present"}.get(track.object_class)
        if kind is None or len(track.samples) < 2:
            continue
        # Presence only. No camera use, ownership, bag opening, cash handling or
        # temporal violation is inferred from a COCO label.
        runs = [[]]
        for sample in track.samples:
            if runs[-1] and sample.t - runs[-1][-1].t > max_gap_sec:
                runs.append([])
            runs[-1].append(sample)
        for run in runs:
            if len(run) < 2:
                continue
            output.append(Observation(obs_id=f"ob-{start_index + len(output)}", kind=kind,
                start_sec=run[0].t, end_sec=run[-1].t, subjects=[track.track_id],
                score=min(sample.score for sample in run),
                frame_ids=[frame_id(run[0].t), frame_id(run[-1].t)],
                provenance="detector", model_id=model_id))
    return output


_runtime = LocalPerceptionRuntime()


def get_runtime() -> LocalPerceptionRuntime:
    return _runtime


def runtime_status(config: PipelineConfig | None = None) -> RuntimeStatus:
    return _runtime.runtime_status(config or PipelineConfig.from_env())
