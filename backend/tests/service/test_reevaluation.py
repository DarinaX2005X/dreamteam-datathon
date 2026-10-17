import base64
import json
import sqlite3

import pytest
from navidu.contracts.ui_api import DecisionRequest, ReevaluateResponse

from .conftest import upload


def snapshot(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


def request_body(revision=1, *, dry_run=True):
    return {"base_revision": revision, "role_overrides": [{"track_id": "p2", "role": "collector"}],
            "zone_overrides": [], "dry_run": dry_run}


def test_preview_is_ephemeral_and_committed_media_is_revision_scoped(client, app, done):
    check_id = done["id"]
    directory = app.state.store.settings.checks_dir / check_id
    automatic_files = snapshot(directory)
    jury_path = f"/api/v1/jobs/{check_id}"
    headers = {"X-API-Key": "test-secret-key"}
    automatic_jury = client.get(jury_path, headers=headers).json()
    response = client.post(f"/api/ui/checks/{check_id}/reevaluate", json=request_body())
    preview = ReevaluateResponse.model_validate(response.json())
    assert response.status_code == 200 and preview.finding_set.revision == 2
    ids = {frame.frame_id for finding in preview.finding_set.findings for frame in finding.evidence}
    assert set(preview.preview_frames) == ids
    for frame_id, image in preview.preview_frames.items():
        assert base64.b64decode(image.image_base64).startswith(b"\xff\xd8")
        assert base64.b64decode(image.overlay_base64).startswith(b"\xff\xd8")
        assert image.image_base64 != image.overlay_base64
    assert snapshot(directory) == automatic_files
    assert app.state.store.finding_set(check_id).revision == 1

    saved = client.post(f"/api/ui/checks/{check_id}/reevaluate", json=request_body(dry_run=False))
    assert saved.status_code == 200 and saved.json()["preview_frames"] == {}
    assert all((directory / name).read_bytes() == value for name, value in automatic_files.items())
    assert client.get(jury_path, headers=headers).json() == automatic_jury
    frame_id = next(iter(ids))
    endpoint = f"/api/ui/checks/{check_id}/frames/{frame_id}.jpg"
    automatic = client.get(endpoint + "?revision=1&overlay=1")
    reviewed = client.get(endpoint + "?revision=2&overlay=1")
    assert automatic.status_code == reviewed.status_code == 200
    assert automatic.content != reviewed.content
    assert client.get(endpoint).content == reviewed.content
    assert client.get(endpoint + "?revision=999").status_code == 404
    assert not list(directory.glob(".reevaluate-*"))
    assert not (directory / "revisions" / "2" / ".reevaluation-staging").exists()
    finding_id = saved.json()["finding_set"]["findings"][0]["finding_id"]
    client.put(f"/api/ui/checks/{check_id}/findings/{finding_id}/decision",
               json={"decision": "confirmed", "note": "", "base_revision": 2})
    assert client.get(endpoint + "?revision=3").content == reviewed.content


def test_failed_engine_preview_cleans_staging_without_writes(client, app, done, pipeline, monkeypatch):
    check_id = done["id"]
    directory = app.state.store.settings.checks_dir / check_id
    before = snapshot(directory)
    original = pipeline.reevaluate

    def fail_after_extract(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("failed after extraction")

    monkeypatch.setattr(pipeline, "reevaluate", fail_after_extract)
    with pytest.raises(RuntimeError, match="failed after extraction"):
        client.post(f"/api/ui/checks/{check_id}/reevaluate", json=request_body(dry_run=False))
    assert snapshot(directory) == before
    assert app.state.store.finding_set(check_id).revision == 1


def test_failed_publication_rolls_back_media_and_database(client, app, done, monkeypatch):
    check_id = done["id"]
    store = app.state.store
    directory = store.settings.checks_dir / check_id
    before = snapshot(directory)

    def fail_insert(*args):
        assert (directory / "revisions" / "2" / "frames").is_dir()
        raise sqlite3.OperationalError("deliberate write failure")

    monkeypatch.setattr(store, "_insert_revision", fail_insert)
    with pytest.raises(sqlite3.OperationalError, match="deliberate write failure"):
        client.post(f"/api/ui/checks/{check_id}/reevaluate", json=request_body(dry_run=False))
    assert snapshot(directory) == before
    assert store.finding_set(check_id).revision == 1
    assert store.media_revision(check_id, 2) is None


def test_stale_computation_cannot_publish_over_concurrent_decision(client, app, done, pipeline, monkeypatch):
    check_id = done["id"]
    store = app.state.store
    directory = store.settings.checks_dir / check_id
    before = snapshot(directory)
    original = pipeline.reevaluate

    def compute_then_change(*args, **kwargs):
        updated = original(*args, **kwargs)
        store.save_decision(check_id, done["finding_set"]["findings"][0]["finding_id"],
                            DecisionRequest(decision="confirmed", note="Concurrent review", base_revision=1))
        return updated

    monkeypatch.setattr(pipeline, "reevaluate", compute_then_change)
    response = client.post(f"/api/ui/checks/{check_id}/reevaluate", json=request_body(dry_run=False))
    assert response.status_code == 409 and response.json()["error"]["current_revision"] == 2
    assert snapshot(directory) == before
    assert store.media_revision(check_id, 2) is None


def test_cached_manifest_rebinds_source_and_rules_before_original_expires(client, app, done):
    original = done["id"]
    duplicate = upload(client).json()["check"]["id"]
    assert app.state.test_worker.run_once()
    store = app.state.store
    directory = store.settings.checks_dir / duplicate
    manifest = json.loads((directory / "engine_manifest.json").read_text())
    assert manifest["source_path"] == str((directory / store.get(duplicate)["video_file"]).resolve())
    assert manifest["rules_dir"] == str(store.settings.rules_dir.resolve())
    source = store.settings.checks_dir / original / store.get(original)["video_file"]
    source.unlink()
    response = client.post(f"/api/ui/checks/{duplicate}/reevaluate", json=request_body())
    assert response.status_code == 200 and response.json()["preview_frames"]
