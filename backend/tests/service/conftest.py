
import pytest
from fastapi.testclient import TestClient
from navidu.service.app import create_app
from navidu.service.config import Settings
from navidu.service.worker import Worker

from .fakes import FakePipeline, fake_probe


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", rules_dir=tmp_path / "rules",
                    web_dist=tmp_path / "web", api_keys=("test-secret-key",),
                    ui_max_active_per_ip=10, lease_seconds=3, heartbeat_seconds=0.1,
                    pipeline_version="test-v1")


@pytest.fixture
def pipeline():
    return FakePipeline()


@pytest.fixture
def app(settings, pipeline):
    result = create_app(settings, pipeline=pipeline, probe=fake_probe)
    worker = Worker(settings, pipeline=pipeline)
    assert worker.prepare()
    result.state.test_worker = worker
    return result


@pytest.fixture
def client(app):
    with TestClient(app) as value:
        yield value


def upload(client, *, api=False, payload=b"video-content", name="test.mov"):
    return client.post("/api/v1/analyze" if api else "/api/ui/checks",
                       files={"file": (name, payload, "video/quicktime")},
                       headers={"X-API-Key": "test-secret-key"} if api else {})


@pytest.fixture
def done(client, app):
    value = upload(client).json()["check"]
    assert app.state.test_worker.run_once()
    return client.get("/api/ui/checks/" + value["id"]).json()["check"]
