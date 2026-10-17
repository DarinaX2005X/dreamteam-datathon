from dataclasses import replace

from fastapi.testclient import TestClient

from navidu.contracts.api_v1 import JobDone
from navidu.contracts.ui_api import CheckDetailResponse, HealthResponse
from navidu.service.app import create_app
from navidu.service.retention import cleanup

from .conftest import upload
from .fakes import fake_probe


def test_health_and_missing_worker(settings):
    with TestClient(create_app(settings, probe=fake_probe)) as client:
        health = HealthResponse.model_validate(client.get("/api/ui/health").json())
        assert not health.ready and health.model_status == "missing"
        response = upload(client)
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "NOT_READY"


def test_api_key_checked_before_reading_upload(client, app):
    for header in [{}, {"X-API-Key": "wrong"}]:
        response = client.post("/api/v1/analyze", headers=header,
                               files={"file": ("private.mov", b"video-content")})
        assert response.status_code == 401
    assert app.state.store.health()["queue"] == {"api": 0, "ui": 0}
    assert list(app.state.store.settings.checks_dir.iterdir()) == []


def test_upload_validation_and_cleanup(client, app):
    assert client.post("/api/ui/checks").status_code == 400
    assert upload(client, payload=b"").status_code == 400
    response = upload(client, payload=b"not-video-but-named-mov")
    assert response.status_code == 415
    assert response.json()["error"]["request_id"].startswith("r-")
    assert response.headers["x-request-id"] == response.json()["error"]["request_id"]
    assert not list(app.state.store.settings.checks_dir.iterdir())


def test_real_file_limit_without_allocating_300mb(settings, pipeline):
    settings = replace(settings, max_upload_mb=0.001)
    app = create_app(settings, pipeline=pipeline, probe=fake_probe)
    app.state.store.worker_state("test", model_status="ready", device="test", limitations=[], pipeline_version="test-v1")
    with TestClient(app) as client:
        assert upload(client, payload=b"x" * 2048).status_code == 413
        assert not list(settings.checks_dir.iterdir())
        assert client.post("/api/ui/checks", content=b"", headers={"Content-Length": "315621376"}).status_code == 413


def test_chunked_upload_limit(settings, pipeline):
    settings = replace(settings, max_upload_mb=0.001)
    app = create_app(settings, pipeline=pipeline, probe=fake_probe)
    app.state.store.worker_state("test", model_status="ready", device="test", limitations=[], pipeline_version="test-v1")

    def chunks():
        yield b'--test\r\nContent-Disposition: form-data; name="file"; filename="video.mov"\r\nContent-Type: video/quicktime\r\n\r\n'
        for _ in range(1100):
            yield b"x" * 1024
        yield b"\r\n--test--\r\n"

    with TestClient(app) as client:
        response = client.post("/api/ui/checks", content=chunks(),
                               headers={"Content-Type": "multipart/form-data; boundary=test"})
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_late_frontend_build_and_asset_boundaries(settings):
    app = create_app(settings, probe=fake_probe)
    with TestClient(app) as client:
        assert client.get("/").status_code == 503
        (settings.web_dist / "assets").mkdir(parents=True)
        (settings.web_dist / "index.html").write_text("<!doctype html><title>Lado</title>")
        (settings.web_dist / "assets" / "app-123.js").write_text("export const name='Lado';")
        (settings.web_dist / "favicon.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
        assert client.get("/app/checks/000000000001").status_code == 200
        asset = client.get("/assets/app-123.js")
        assert asset.status_code == 200 and "immutable" in asset.headers["cache-control"]
        assert client.get("/assets/..%5Cindex.html").status_code == 404
        assert client.get("/favicon.svg").status_code == 200


def test_end_to_end_jury_exact_response(client, app):
    response = upload(client, api=True)
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    headers = {"X-API-Key": "test-secret-key"}
    assert client.get(f"/api/v1/jobs/{job_id}", headers=headers).json()["status"] == "processing"
    assert app.state.test_worker.run_once()
    result = client.get(f"/api/v1/jobs/{job_id}", headers=headers)
    assert result.status_code == 200
    assert set(result.json()) == {"job_id", "status", "verdict", "violations"}
    JobDone.model_validate(result.json())
    CheckDetailResponse.model_validate(client.get(f"/api/ui/checks/{job_id}").json())


def test_video_range_frames_and_path_boundaries(client, done):
    check_id = done["id"]
    response = client.get(f"/api/ui/checks/{check_id}/video", headers={"Range": "bytes=0-4"})
    assert response.status_code == 206 and response.content == b"video"
    assert response.headers["content-range"].startswith("bytes 0-4/")
    frame = done["finding_set"]["findings"][0]["evidence"][0]["frame_id"]
    for overlay in [0, 1]:
        result = client.get(f"/api/ui/checks/{check_id}/frames/{frame}.jpg?overlay={overlay}")
        assert result.status_code == 200 and result.content.startswith(b"\xff\xd8")
    assert client.get(f"/api/ui/checks/{check_id}/frames/..%5C..%5Csecret.jpg").status_code == 404
    assert client.get("/api/ui/checks/000000000000").status_code == 404


def test_decisions_revision_conflict_and_automatic_jury_unchanged(client, done):
    check_id = done["id"]
    findings = done["finding_set"]["findings"]
    path = f"/api/ui/checks/{check_id}/findings/{findings[0]['finding_id']}/decision"
    response = client.put(path, json={"decision": "rejected", "note": "Проверено", "base_revision": 1})
    assert response.status_code == 200 and response.json()["revision"] == 2
    conflict = client.put(path, json={"decision": "confirmed", "note": "Не терять заметку", "base_revision": 1})
    assert conflict.status_code == 409 and conflict.json()["error"]["current_revision"] == 2
    path2 = f"/api/ui/checks/{check_id}/findings/{findings[1]['finding_id']}/decision"
    assert client.put(path2, json={"decision": "rejected", "note": "", "base_revision": 2}).status_code == 200
    current = client.get(f"/api/ui/checks/{check_id}").json()["check"]
    assert current["verdict"] == "no_violation" and current["finding_set"]["revision"] == 3
    assert current["automatic_verdict"] == "violation"
    jury = client.get(f"/api/v1/jobs/{check_id}", headers={"X-API-Key": "test-secret-key"}).json()
    assert jury["verdict"] == "violation" and len(jury["violations"]) == 2
    reset = client.put(path, json={"decision": None, "note": "", "base_revision": 3})
    assert reset.json()["decision"] is None
    assert client.get(f"/api/ui/checks/{check_id}").json()["check"]["verdict"] == "violation"


def test_reevaluate_preview_commit_and_decision_preservation(client, done):
    check_id = done["id"]
    finding = done["finding_set"]["findings"][0]
    client.put(f"/api/ui/checks/{check_id}/findings/{finding['finding_id']}/decision",
               json={"decision": "confirmed", "note": "Сохранить", "base_revision": 1})
    request = {"base_revision": 2, "role_overrides": [{"track_id": "p2", "role": "collector"}],
               "zone_overrides": [], "dry_run": True}
    preview = client.post(f"/api/ui/checks/{check_id}/reevaluate", json=request)
    assert preview.status_code == 200 and len(preview.json()["diff"]["removed"]) == 1
    assert client.get(f"/api/ui/checks/{check_id}").json()["check"]["finding_set"]["revision"] == 2
    request["dry_run"] = False
    saved = client.post(f"/api/ui/checks/{check_id}/reevaluate", json=request)
    assert saved.status_code == 200 and saved.json()["finding_set"]["revision"] == 3
    assert saved.json()["finding_set"]["findings"][0]["decision"]["note"] == "Сохранить"
    assert client.post(f"/api/ui/checks/{check_id}/reevaluate", json=request).status_code == 409


def test_cancel_and_retry_keep_input(client, app):
    check_id = upload(client).json()["check"]["id"]
    cancelled = client.post(f"/api/ui/checks/{check_id}/cancel")
    assert cancelled.json()["check"]["status"] == "cancelled"
    jury = client.get(f"/api/v1/jobs/{check_id}", headers={"X-API-Key": "test-secret-key"}).json()
    assert jury["status"] == "failed" and jury["error"]["code"] == "CANCELLED"
    assert client.post(f"/api/ui/checks/{check_id}/retry").json()["check"]["status"] == "queued"
    assert app.state.test_worker.run_once()
    assert client.get(f"/api/ui/checks/{check_id}").json()["check"]["status"] == "done"
    assert client.post(f"/api/ui/checks/{check_id}/retry").status_code == 409


def test_example_is_read_only_and_retained(client, app, done):
    check_id = done["id"]
    store = app.state.store
    with store.connection() as db:
        db.execute("UPDATE checks SET is_example=1 WHERE id=?", (check_id,))
    store.settings = replace(store.settings, example_check_id=check_id)
    assert client.get("/api/ui/example").json() == {"check_id": check_id}
    finding = done["finding_set"]["findings"][0]
    assert client.put(f"/api/ui/checks/{check_id}/findings/{finding['finding_id']}/decision",
                      json={"decision": "confirmed", "note": "", "base_revision": 1}).status_code == 409
    body = {"base_revision": 1, "role_overrides": [], "zone_overrides": [], "dry_run": True}
    assert client.post(f"/api/ui/checks/{check_id}/reevaluate", json=body).status_code == 200
    body["dry_run"] = False
    assert client.post(f"/api/ui/checks/{check_id}/reevaluate", json=body).status_code == 409
    cleanup(store, now=store.get(check_id)["finished_at"] + 365 * 86400)
    assert store.get(check_id)["status"] == "done"


def test_retention_video_then_results(client, app, done):
    store = app.state.store
    check_id = done["id"]
    finished = store.get(check_id)["finished_at"]
    cleanup(store, now=finished + 25 * 3600)
    assert client.get(f"/api/ui/checks/{check_id}/video").status_code == 410
    frame = done["finding_set"]["findings"][0]["evidence"][0]["frame_id"]
    assert client.get(f"/api/ui/checks/{check_id}/frames/{frame}.jpg").status_code == 200
    cleanup(store, now=finished + 8 * 86400)
    assert client.get(f"/api/ui/checks/{check_id}").status_code == 404


def test_queue_limits_are_enforced(settings, pipeline):
    settings = replace(settings, queue_max=2, ui_max_active_per_ip=1)
    app = create_app(settings, pipeline=pipeline, probe=fake_probe)
    app.state.store.worker_state("test", model_status="ready", device="test", limitations=[], pipeline_version="test-v1")
    with TestClient(app) as client:
        assert upload(client).status_code == 201
        response = upload(client)
        assert response.status_code == 429 and response.headers["retry-after"] == "10"
        assert upload(client, api=True).status_code == 200
        assert upload(client, api=True).status_code == 429


def test_no_production_without_key(settings):
    import pytest
    with pytest.raises(RuntimeError, match="NAVIDU_API_KEYS"):
        create_app(replace(settings, production=True, api_keys=()))


def test_note_before_decision_survives_reload_and_reset(client, done):
    check_id = done["id"]
    finding_id = done["finding_set"]["findings"][0]["finding_id"]
    path = f"/api/ui/checks/{check_id}/findings/{finding_id}/decision"
    response = client.put(path, json={"decision": None, "note": "Нужно уточнить ракурс", "base_revision": 1})
    assert response.status_code == 200
    assert response.json()["decision"]["value"] is None
    result = client.get(f"/api/ui/checks/{check_id}").json()["check"]
    assert result["finding_set"]["findings"][0]["decision"]["note"] == "Нужно уточнить ракурс"
    assert result["verdict"] == "violation"
    assert client.put(path, json={"decision": None, "note": "", "base_revision": 2}).json()["decision"] is None


def test_cache_clones_automatic_revision_without_review_context(client, app, done, pipeline):
    original = done["id"]
    finding_id = done["finding_set"]["findings"][0]["finding_id"]
    client.put(f"/api/ui/checks/{original}/findings/{finding_id}/decision",
               json={"decision": "rejected", "note": "Private reviewer note", "base_revision": 1})
    client.post(f"/api/ui/checks/{original}/reevaluate", json={
        "base_revision": 2, "role_overrides": [{"track_id": "p2", "role": "collector"}],
        "zone_overrides": [], "dry_run": False,
    })
    duplicate = upload(client).json()["check"]["id"]
    assert duplicate != original
    assert app.state.test_worker.run_once()
    assert pipeline.calls == 1
    result = client.get(f"/api/ui/checks/{duplicate}").json()["check"]
    assert result["analysis_origin"] == "cache"
    assert result["analyzed_at"] == done["analyzed_at"]
    assert "cached_from_check_id" not in result
    assert result["finding_set"]["revision"] == 1 and len(result["finding_set"]["findings"]) == 2
    assert all(finding["decision"] is None for finding in result["finding_set"]["findings"])
    assert result["tracks"][1]["role"]["value"] == "unknown"
    artifacts = app.state.store.artifacts(duplicate)
    assert artifacts.run.check_id == duplicate and artifacts.work_dir.name == duplicate
    assert (artifacts.work_dir / "video.json").is_file()
    source = app.state.store.get(original)
    assert source["cached_from_check_id"] is None


def test_cache_requires_matching_pipeline_version(client, app, done, pipeline):
    app.state.store.worker_state("test-new", model_status="ready", device="test", limitations=[], pipeline_version="test-v2")
    duplicate = upload(client).json()["check"]["id"]
    assert app.state.test_worker.run_once()
    assert pipeline.calls == 2
    assert client.get(f"/api/ui/checks/{duplicate}").json()["check"]["analysis_origin"] == "local"
