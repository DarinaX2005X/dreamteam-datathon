import multiprocessing
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from navidu.contracts.domain import VideoMeta
from navidu.contracts.errors import AnalysisError
from navidu.service.exceptions import LeaseLost
from navidu.service.queue import JobQueue
from navidu.service.store import Store
from navidu.service.worker import Worker, _bounded_stop_check

from .fakes import FakePipeline


def enqueue(store, check_id, priority="ui"):
    path = store.settings.checks_dir / check_id
    path.mkdir()
    (path / "video.mov").write_bytes(b"test-video")
    return store.enqueue(check_id=check_id, file_name="test.mov", sha256=check_id,
                         priority=priority, client_ip_hash="hash-" + check_id,
                         video_file="video.mov", video_meta=VideoMeta(duration_sec=1,
                            fps_nominal=25, width=64, height=36))


def test_queue_api_priority_and_exclusive_lease(settings):
    store = Store(settings)
    enqueue(store, "000000000001")
    enqueue(store, "000000000002", "api")
    queue = JobQueue(store)
    assert queue.acquire()["id"] == "000000000002"
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda _: JobQueue(Store(settings)).acquire(), range(2)))
    assert sum(row is not None for row in rows) == 1
    assert next(row for row in rows if row)["id"] == "000000000001"


def _take_lease_and_wait(settings, sender):
    job = JobQueue(Store(settings)).acquire()
    sender.send(job)
    time.sleep(60)


def test_queue_recovers_killed_process_without_duplicate_publication(settings):
    settings = replace(settings, lease_seconds=3, heartbeat_seconds=0.05)
    store = Store(settings)
    enqueue(store, "000000000001")
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_take_lease_and_wait, args=(settings, sender))
    process.start()
    assert receiver.poll(20), "child worker did not acquire a job"
    previous = receiver.recv()
    process.kill()
    process.join(timeout=10)
    assert not process.is_alive()
    with store.connection() as db:
        db.execute("UPDATE checks SET lease_until=? WHERE id=?", (time.time() - 1, previous["id"]))
    worker = Worker(settings, pipeline=FakePipeline())
    assert worker.run_once()
    assert store.get("000000000001")["status"] == "done"
    assert store.get("000000000001")["attempts"] == 2
    with pytest.raises(LeaseLost):
        JobQueue(store).renew(previous["id"], previous["lease_token"])
    with store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM finding_sets").fetchone()[0] == 1
    receiver.close()
    sender.close()
    process.close()


def test_queue_three_failures_and_partial_events_remain(settings):
    store = Store(settings)
    enqueue(store, "000000000001")
    worker = Worker(settings, pipeline=FakePipeline(fail=True))
    for _ in range(3):
        assert worker.run_once()
    row = store.get("000000000001")
    assert row["status"] == "failed" and row["attempts"] == 3
    assert AnalysisError.model_validate_json(row["error_json"]).code == "RETRIES_EXHAUSTED"
    assert len(store.events(row["id"])) == 1


def test_queue_cancellation_invalidates_active_lease(settings):
    store = Store(settings)
    enqueue(store, "000000000001")
    queue = JobQueue(store)
    job = queue.acquire()
    store.cancel(job["id"])
    assert not queue.owns(job["id"], job["lease_token"])
    queue.release(job["id"], job["lease_token"])
    assert store.get(job["id"])["status"] == "cancelled"


def test_decoded_frames_share_lease_check_and_observe_cancellation_within_half_second(settings):
    store = Store(settings)
    enqueue(store, "000000000001")
    queue = JobQueue(store)
    job = queue.acquire()
    now = [10.0]
    calls = []

    def owns_lease():
        calls.append(now[0])
        return queue.owns(job["id"], job["lease_token"])

    stopped = threading.Event()
    lost = threading.Event()
    should_stop = _bounded_stop_check(stopped, lost, owns_lease, clock=lambda: now[0])
    assert not any(should_stop() for _ in range(1000))
    assert len(calls) == 1
    store.cancel(job["id"])
    now[0] += 0.5
    assert should_stop() and lost.is_set()
    assert len(calls) == 2
    assert should_stop() and len(calls) == 2
    with pytest.raises(LeaseLost):
        queue.renew(job["id"], job["lease_token"])


@pytest.mark.parametrize("signal", ["stop", "lease"])
def test_stop_signals_interrupt_cached_lease_immediately(signal):
    stopped = threading.Event()
    lost = threading.Event()
    checks = []
    should_stop = _bounded_stop_check(stopped, lost, lambda: checks.append(1) or True,
                                     clock=lambda: 10.0)
    assert not should_stop()
    (stopped if signal == "stop" else lost).set()
    assert should_stop()
    assert len(checks) == 1


def test_stale_attempt_cannot_overwrite_retried_result(settings):
    from navidu.contracts.pipeline import PipelineConfig
    store = Store(settings)
    enqueue(store, "000000000001")
    queue = JobQueue(store)
    old = queue.acquire()
    directory = settings.checks_dir / old["id"]
    old_staging = directory / ("attempt-" + old["lease_token"])
    old_staging.mkdir()
    artifact = FakePipeline().analyze(directory / "video.mov", work_dir=old_staging,
        config=PipelineConfig(check_id=old["id"]), emit=lambda _: None, should_stop=lambda: False)
    store.cancel(old["id"])
    store.retry(old["id"])
    worker = Worker(settings, pipeline=FakePipeline())
    assert worker.run_once()
    published_frame = next((directory / "frames" / "original").glob("*.jpg"))
    before = published_frame.read_bytes()
    (old_staging / "frames" / "original" / published_frame.name).write_bytes(b"stale corrupted bytes")
    with pytest.raises(LeaseLost):
        store.complete(old["id"], old["lease_token"], artifact)
    assert published_frame.read_bytes() == before
