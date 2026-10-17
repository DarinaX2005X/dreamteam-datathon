import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from navidu.contracts.domain import Observation
from navidu.contracts.pipeline import PipelineEvent
from navidu.contracts.ui_api import DecisionRequest
from navidu.service.adapter import LocalPipelineAdapter
from navidu.service.exceptions import ServiceError
from navidu.service.queue import JobQueue
from navidu.service.store import Store
from navidu.service.worker import Worker

from .conftest import upload


@pytest.mark.parametrize("mode,ready", [("diagnostic", False), ("unvalidated", False), ("diagnostic", True)])
def test_loaded_assets_without_validated_analysis_cannot_accept_or_acquire(client, app, pipeline, mode, ready):
    queued = upload(client).json()["check"]["id"]
    pipeline.analysis_mode = mode
    pipeline.analysis_ready = ready
    assert app.state.test_worker.run_once() is False
    health = client.get("/api/ui/health").json()
    assert health["ready"] is False and health["analysis_ready"] is False
    assert health["analysis_mode"] == mode
    assert upload(client).status_code == upload(client, api=True).status_code == 503
    assert app.state.store.get(queued)["status"] == "queued"
    assert app.state.store.get(queued)["attempts"] == 0 and pipeline.calls == 0
    assert len(list(app.state.store.settings.checks_dir.iterdir())) == 1


def test_experimental_worker_accepts_ui_and_never_jury_upload_or_result(client, app, pipeline):
    pipeline.analysis_mode = "experimental"
    pipeline.analysis_ready = True
    assert app.state.test_worker.refresh_status()
    assert client.get("/api/ui/health").json()["analysis_mode"] == "experimental"
    assert upload(client, api=True).status_code == 503
    check_id = upload(client).json()["check"]["id"]
    assert app.state.test_worker.run_once()
    row = app.state.store.get(check_id)
    assert row["status"] == "done" and row["analysis_mode"] == "experimental"
    assert client.get(f"/api/ui/checks/{check_id}").json()["check"]["status"] == "done"
    jury = client.get(f"/api/v1/jobs/{check_id}", headers={"X-API-Key": "test-secret-key"}).json()
    assert jury["status"] == "failed" and jury["error"]["code"] == "ANALYSIS_NOT_PERFORMED"


def test_diagnostic_artifacts_cannot_be_published_even_from_ready_worker(client, app, pipeline, monkeypatch):
    original = pipeline.analyze

    def diagnostic_result(*args, **kwargs):
        artifacts = original(*args, **kwargs)
        artifacts.run.analysis_ready = False
        artifacts.run.analysis_mode = "diagnostic"
        return artifacts

    monkeypatch.setattr(pipeline, "analyze", diagnostic_result)
    check_id = upload(client).json()["check"]["id"]
    assert app.state.test_worker.run_once()
    row = app.state.store.get(check_id)
    assert row["status"] == "failed" and row["attempts"] == 1 and row["artifacts_json"] is None
    assert json.loads(row["error_json"])["code"] == "ANALYSIS_NOT_PERFORMED"
    jury = client.get(f"/api/v1/jobs/{check_id}", headers={"X-API-Key": "test-secret-key"}).json()
    assert set(jury) == {"job_id", "status", "error"} and jury["status"] == "failed"
    assert client.get(f"/api/ui/checks/{check_id}").json()["check"]["verdict"] is None


def test_version_change_during_work_cannot_publish_old_analysis(client, app, pipeline, monkeypatch):
    original = pipeline.analyze

    def change_version(*args, **kwargs):
        artifacts = original(*args, **kwargs)
        pipeline.version = "test-v2"
        return artifacts

    monkeypatch.setattr(pipeline, "analyze", change_version)
    check_id = upload(client).json()["check"]["id"]
    assert app.state.test_worker.run_once()
    row = app.state.store.get(check_id)
    assert row["status"] == "failed" and row["attempts"] == 1 and row["artifacts_json"] is None
    assert json.loads(row["error_json"])["code"] == "RUNTIME_RESTART_REQUIRED"


def test_catalog_incompatibility_is_detected_before_loading_models(settings, monkeypatch):
    catalog = settings.rules_dir / "catalog"
    catalog.mkdir(parents=True)
    (catalog / "incompatible.yaml").write_text(json.dumps({
        "anchor": {"observation": {"kind": "counting_motion"}},
        "condition": {"before": {"first": {"observation": {"kind": "cash_handover"}},
            "then": {"observation": {"kind": "counting_motion"}}, "future_unsupported_argument": True}},
    }), encoding="utf-8")
    adapter = LocalPipelineAdapter(settings)

    def forbidden_load():
        raise AssertionError("models must not load for an incompatible catalog")

    monkeypatch.setattr(adapter, "_load", forbidden_load)
    with pytest.raises(ServiceError) as error:
        adapter.prepare()
    assert error.value.code == "RULES_INVALID"
    worker = Worker(settings, pipeline=adapter)
    assert worker.prepare() is False and worker.store.health()["ready"] is False


def test_legacy_diagnostic_is_quarantined_and_retry_preserves_original(client, app, done, pipeline):
    check_id = done["id"]
    store = app.state.store
    finding_id = done["finding_set"]["findings"][0]["finding_id"]
    store.save_decision(check_id, finding_id, DecisionRequest(
        decision="confirmed", note="Keep original review", base_revision=1,
    ))
    row = store.get(check_id)
    revised = {}
    for field in ["artifacts_json", "automatic_artifacts_json"]:
        value = json.loads(row[field])
        value["run"].pop("analysis_ready")
        value["run"].pop("analysis_mode")
        value["run"]["model_versions"] = {"rfdetr-nano-onnx": "test-diagnostic"}
        revised[field] = json.dumps(value)
    with store.connection() as db:
        db.execute("UPDATE checks SET artifacts_json=?,automatic_artifacts_json=? WHERE id=?",
                   (revised["artifacts_json"], revised["automatic_artifacts_json"], check_id))
    directory = store.settings.checks_dir / check_id
    before = {str(path.relative_to(directory)): path.read_bytes()
              for path in directory.rglob("*") if path.is_file()}
    Store(store.settings)
    quarantined = store.get(check_id)
    assert quarantined["status"] == "failed" and quarantined["analysis_mode"] == "diagnostic"
    assert quarantined["automatic_artifacts_json"] == revised["automatic_artifacts_json"]
    assert store.finding_set(check_id).findings[0].decision.note == "Keep original review"
    assert client.get(f"/api/ui/checks/{check_id}").json()["check"]["verdict"] is None
    jury = client.get(f"/api/v1/jobs/{check_id}", headers={"X-API-Key": "test-secret-key"}).json()
    assert jury["status"] == "failed" and jury["error"]["code"] == "ANALYSIS_NOT_PERFORMED"

    pipeline.analysis_ready = False
    assert not app.state.test_worker.refresh_status()
    assert client.post(f"/api/ui/checks/{check_id}/retry").status_code == 503
    pipeline.analysis_ready = True
    assert app.state.test_worker.refresh_status()
    retry = client.post(f"/api/ui/checks/{check_id}/retry")
    assert retry.status_code == 200
    new_id = retry.json()["check"]["id"]
    assert new_id != check_id and app.state.test_worker.run_once()
    assert store.get(new_id)["status"] == "done" and pipeline.calls == 2
    assert store.get(check_id)["status"] == "failed"
    assert store.finding_set(check_id).findings[0].decision.note == "Keep original review"
    assert all((directory / name).read_bytes() == raw for name, raw in before.items())
    Store(store.settings)
    assert store.get(new_id)["status"] == "done"


def test_unvalidated_cache_and_jury_results_are_rejected(client, app, done, pipeline):
    check_id = done["id"]
    row = app.state.store.get(check_id)
    artifact = json.loads(row["automatic_artifacts_json"])
    artifact["run"]["analysis_ready"] = False
    artifact["run"]["analysis_mode"] = "unvalidated"
    with app.state.store.connection() as db:
        db.execute("UPDATE checks SET automatic_artifacts_json=? WHERE id=?", (json.dumps(artifact), check_id))
    jury = client.get(f"/api/v1/jobs/{check_id}", headers={"X-API-Key": "test-secret-key"}).json()
    assert jury["status"] == "failed" and "verdict" not in jury
    duplicate = upload(client).json()["check"]["id"]
    assert app.state.test_worker.run_once() and pipeline.calls == 2
    assert app.state.store.get(duplicate)["cached_from_check_id"] is None


def test_perception_progress_at_end_of_video_is_not_done(client, app, pipeline, monkeypatch):
    arrived, release = threading.Event(), threading.Event()
    original = pipeline.analyze

    def wait_for_actions(video_path, *, work_dir, config, emit, should_stop):
        emit(PipelineEvent(phase="perception", processed_until_sec=125.04))
        arrived.set()
        assert release.wait(10)
        return original(video_path, work_dir=work_dir, config=config, emit=emit, should_stop=should_stop)

    monkeypatch.setattr(pipeline, "analyze", wait_for_actions)
    check_id = upload(client).json()["check"]["id"]
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(app.state.test_worker.run_once)
        try:
            assert arrived.wait(10)
            current = client.get(f"/api/ui/checks/{check_id}").json()["check"]
            assert current["status"] == "processing" and current["verdict"] is None
            assert current["progress"]["phase"] == "perception"
            assert current["progress"]["processed_until_sec"] == current["duration_sec"]
            assert client.get(f"/api/v1/jobs/{check_id}", headers={"X-API-Key": "test-secret-key"}).json() == {
                "job_id": check_id, "status": "processing",
            }
        finally:
            release.set()
        assert future.result(timeout=10)
    assert client.get(f"/api/ui/checks/{check_id}").json()["check"]["status"] == "done"


def test_all_uncheckable_failure_is_nonretryable_and_public(client, app, pipeline, monkeypatch):
    class NoCheckableRules(RuntimeError):
        code = "ANALYSIS_INCOMPLETE"

    def fail_in_rules(video_path, *, work_dir, config, emit, should_stop):
        emit(PipelineEvent(phase="rules", processed_until_sec=125.04))
        raise NoCheckableRules("Private diagnostic detail must not be public")

    monkeypatch.setattr(pipeline, "analyze", fail_in_rules)
    check_id = upload(client).json()["check"]["id"]
    assert app.state.test_worker.run_once()
    row = app.state.store.get(check_id)
    assert row["status"] == "failed" and row["attempts"] == 1 and row["artifacts_json"] is None
    result = client.get(f"/api/ui/checks/{check_id}").json()["check"]
    assert result["verdict"] is None and result["error"]["code"] == "ANALYSIS_INCOMPLETE"
    assert result["error"]["failed_phase"] == "rules"
    assert "Private diagnostic" not in result["error"]["message"]
    assert "автоматически" not in result["error"]["message"]
    jury = client.get(f"/api/v1/jobs/{check_id}", headers={"X-API-Key": "test-secret-key"}).json()
    assert jury["status"] == "failed" and jury["error"]["code"] == "ANALYSIS_INCOMPLETE"
    assert "verdict" not in jury


def test_progress_observation_deltas_upsert_existing_ids(client, app):
    check_id = upload(client).json()["check"]["id"]
    job = JobQueue(app.state.store).acquire()
    first = Observation(obs_id="ob-1", kind="personal_item_identified", start_sec=2, end_sec=3,
                        subjects=["p1"], provenance="temporal_model", model_id="synthetic-test")
    second = first.model_copy(update={"end_sec": 5.25})
    for timestamp, observations in [(3.0, [first]), (5.25, [second]), (5.5, [second])]:
        app.state.store.append_event(check_id, job["lease_token"], PipelineEvent(
            phase="perception", processed_until_sec=timestamp, observations_delta=observations,
        ))
    result = client.get(f"/api/ui/checks/{check_id}").json()["check"]
    assert result["status"] == "processing" and result["verdict"] is None
    assert result["progress"]["processed_until_sec"] == 5.5
    assert len(result["observations"]) == 1 and result["observations"][0]["end_sec"] == 5.25
