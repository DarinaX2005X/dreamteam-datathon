"""Synthetic fixtures exercise persistence and access; they do not assess models."""

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from navidu.service.exceptions import ServiceError
from navidu.service.retention import cleanup
from navidu.service.store import Store

from .fakes import fake_probe


@pytest.fixture
def builder(monkeypatch, tmp_path):
    path = Path(__file__).parents[3] / "scripts" / "build_example.py"
    spec = importlib.util.spec_from_file_location("example_builder_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ORGANIZER_INVENTORY", tmp_path / "absent-inventory.csv")
    monkeypatch.setattr(module, "probe_video", fake_probe)
    return module


@pytest.fixture
def own_recording(tmp_path):
    path = tmp_path / "synthetic-team-recording.mp4"
    path.write_bytes(b"explicit synthetic video fixture for queue and permission tests only")
    return path


def test_builder_requires_attestation_and_refuses_organizer_hash(builder, settings, own_recording, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="attestation"):
        builder.enqueue_example(own_recording, settings, own_recording=False)
    inventory = tmp_path / "organizer.csv"
    inventory.write_text("sha256\n" + hashlib.sha256(own_recording.read_bytes()).hexdigest() + "\n")
    monkeypatch.setattr(builder, "ORGANIZER_INVENTORY", inventory)
    with pytest.raises(ValueError, match="organizer source"):
        builder.enqueue_example(own_recording, settings, own_recording=True)
    assert not settings.data_dir.exists()


def test_additional_inventory_extends_the_mandatory_repository_guard(builder, settings, own_recording, tmp_path):
    inventory = tmp_path / "extra-organizer.csv"
    inventory.write_text("sha256\n" + hashlib.sha256(own_recording.read_bytes()).hexdigest() + "\n")
    with pytest.raises(ValueError, match="organizer source"):
        builder.enqueue_example(own_recording, settings, own_recording=True, extra_inventories=(inventory,))
    inventory.write_text("source\nnot-a-hash\n")
    with pytest.raises(ValueError, match="sha256 column"):
        builder.organizer_hashes((inventory,))
    assert not settings.data_dir.exists()


def test_builder_does_not_copy_or_enqueue_without_validated_worker(builder, settings, own_recording):
    store = Store(settings)
    store.worker_state("synthetic-unvalidated", model_status="ready", device="test", limitations=[],
                       pipeline_version="test-v1", analysis_ready=False, analysis_mode="unvalidated")
    with pytest.raises(ServiceError) as error:
        builder.enqueue_example(own_recording, settings, own_recording=True)
    assert error.value.code == "NOT_READY"
    assert list(settings.checks_dir.iterdir()) == []
    assert store.health()["queue"] == {"api": 0, "ui": 0}


def test_queue_built_example_is_persistent_read_only_and_retained(builder, own_recording, client, app):
    store = app.state.store
    check_id = builder.enqueue_example(own_recording, store.settings, own_recording=True)
    row = store.get(check_id)
    assert row["status"] == "queued" and row["is_example"] == 1 and row["attempts"] == 0
    directory = store.settings.checks_dir / check_id
    assert (directory / row["video_file"]).read_bytes() == own_recording.read_bytes()
    provenance = json.loads((directory / "example.json").read_text())
    assert provenance["kind"] == "team-owned-recording" and provenance["ownership_attested"] is True
    assert provenance["source_sha256"] == hashlib.sha256(own_recording.read_bytes()).hexdigest()
    assert client.get("/api/ui/health").json()["example_check_id"] is None
    with pytest.raises(TimeoutError):
        builder.wait_for_example(store, check_id, wait_seconds=0)
    assert app.state.test_worker.run_once()
    builder.wait_for_example(store, check_id, wait_seconds=0)
    store.settings = replace(store.settings, example_check_id=check_id)
    assert client.get("/api/ui/health").json()["example_check_id"] == check_id
    assert client.get("/api/ui/example").json() == {"check_id": check_id}
    detail = client.get(f"/api/ui/checks/{check_id}").json()["check"]
    finding_id = detail["finding_set"]["findings"][0]["finding_id"]
    before = {str(path.relative_to(directory)): path.read_bytes()
              for path in directory.rglob("*") if path.is_file()}
    assert client.put(f"/api/ui/checks/{check_id}/findings/{finding_id}/decision", json={
        "decision": None, "note": "Do not save this synthetic note", "base_revision": 1,
    }).json()["error"]["code"] == "READ_ONLY_EXAMPLE"
    for action in ["retry", "cancel"]:
        response = client.post(f"/api/ui/checks/{check_id}/{action}")
        assert response.status_code == 409 and response.json()["error"]["code"] == "READ_ONLY_EXAMPLE"
    assert client.delete(f"/api/ui/checks/{check_id}").status_code == 405
    body = {"base_revision": 1, "role_overrides": [], "zone_overrides": [], "dry_run": True}
    assert client.post(f"/api/ui/checks/{check_id}/reevaluate", json=body).status_code == 200
    body["dry_run"] = False
    assert client.post(f"/api/ui/checks/{check_id}/reevaluate", json=body).json()["error"]["code"] == "READ_ONLY_EXAMPLE"
    assert {str(path.relative_to(directory)): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()} == before
    assert store.finding_set(check_id).revision == 1
    cleanup(store, now=store.get(check_id)["finished_at"] + 365 * 86400)
    retained = client.get(f"/api/ui/checks/{check_id}").json()["check"]
    assert retained["status"] == "done" and retained["video_available"] is True
    assert retained["video_expires_at"] is None


def test_unvalidated_example_is_not_discoverable_or_printable(builder, client, app, done):
    check_id = done["id"]
    store = app.state.store
    store.settings = replace(store.settings, example_check_id=check_id)
    row = store.get(check_id)
    automatic = json.loads(row["automatic_artifacts_json"])
    automatic["run"]["analysis_ready"] = False
    automatic["run"]["analysis_mode"] = "unvalidated"
    with store.connection() as db:
        db.execute("UPDATE checks SET is_example=1,automatic_artifacts_json=? WHERE id=?",
                   (json.dumps(automatic), check_id))
    assert client.get("/api/ui/health").json()["example_check_id"] is None
    assert client.get("/api/ui/example").status_code == 404
    with pytest.raises(ValueError, match="validated automatic"):
        builder.wait_for_example(store, check_id, wait_seconds=0)


def test_cli_emits_configuration_only_after_validated_completion(builder, own_recording, client, app, monkeypatch, capsys):
    store = app.state.store
    monkeypatch.setattr(builder, "load_dotenv", lambda path: None)
    monkeypatch.setattr(builder, "Settings", SimpleNamespace(from_env=lambda: store.settings))
    enqueue = builder.enqueue_example

    def process_queued(*args, **kwargs):
        check_id = enqueue(*args, **kwargs)
        assert app.state.test_worker.run_once()
        return check_id

    monkeypatch.setattr(builder, "enqueue_example", process_queued)
    assert builder.main(["--video", str(own_recording), "--own-recording"]) == 0
    output = capsys.readouterr()
    assert output.out.startswith("NAVIDU_EXAMPLE_CHECK_ID=")
    check_id = output.out.strip().split("=", 1)[1]
    assert store.get(check_id)["is_example"] and store.get(check_id)["status"] == "done"
    assert store.settings.example_check_id is None
    app.state.test_worker.pipeline.analysis_ready = False
    app.state.test_worker.refresh_status()
    assert builder.main(["--video", str(own_recording), "--own-recording"]) == 1
    assert capsys.readouterr().out == ""
