import json
import logging
import os
import signal
import threading
import time

from navidu.contracts.errors import AnalysisError
from navidu.contracts.pipeline import AnalysisArtifacts, PipelineConfig

from .adapter import LocalPipelineAdapter
from .config import Settings
from .exceptions import LeaseLost
from .queue import JobQueue
from .retention import cleanup
from .store import Store

logger = logging.getLogger("lado.worker")


class Worker:
    def __init__(self, settings: Settings, pipeline=None):
        settings.validate()
        self.settings = settings
        self.store = Store(settings)
        self.queue = JobQueue(self.store)
        self.pipeline = pipeline or LocalPipelineAdapter(settings)
        self.worker_id = f"worker-{os.getpid()}"
        self.stop_event = threading.Event()
        self._status = {
            "model_status": "loading", "device": settings.device,
            "limitations": [], "pipeline_version": settings.pipeline_version,
        }
        self._last_cleanup = 0.0

    def heartbeat(self):
        self.store.worker_state(self.worker_id, **self._status)

    def prepare(self):
        preparation_done = threading.Event()

        def pulse():
            while not preparation_done.wait(self.settings.heartbeat_seconds):
                self.heartbeat()

        thread = threading.Thread(target=pulse, name="preparation-heartbeat", daemon=True)
        thread.start()
        try:
            self.heartbeat()
            if hasattr(self.pipeline, "prepare"):
                self.pipeline.prepare()
            if hasattr(self.pipeline, "runtime_status"):
                value = self.pipeline.runtime_status()
                if hasattr(value, "model_dump"):
                    value = value.model_dump()
                self._status.update({key: value[key] for key in self._status if key in value})
            else:
                self._status["model_status"] = "ready"
            self.heartbeat()
            return self._status["model_status"] == "ready"
        except (ModuleNotFoundError, FileNotFoundError) as exc:
            self._status["model_status"] = "missing"
            self._status["limitations"] = ["Для локального анализа нужны установленные зависимости и веса модели."]
            self.heartbeat()
            logger.warning(json.dumps({"phase": "prepare", "error_type": type(exc).__name__}))
            return False
        except Exception:
            self._status["model_status"] = "error"
            self._status["limitations"] = ["Не удалось подготовить локальную модель. Подробности в журнале обработчика."]
            self.heartbeat()
            logger.exception("Local model preparation failed")
            return False
        finally:
            preparation_done.set()
            thread.join(timeout=2)

    def run_once(self) -> bool:
        self.heartbeat()
        job = self.queue.acquire()
        if job is None:
            return False
        check_id, token = job["id"], job["lease_token"]
        directory = self.settings.checks_dir / check_id
        staging = directory / ("attempt-" + token)
        staging.mkdir(parents=True, exist_ok=False)
        stop_lease = threading.Event()
        lease_lost = threading.Event()
        phase = "decode"
        processed = None

        def pulse():
            while not stop_lease.wait(min(self.settings.heartbeat_seconds, self.settings.lease_seconds / 3)):
                try:
                    self.queue.renew(check_id, token)
                    self.heartbeat()
                except LeaseLost:
                    lease_lost.set()
                    return

        def should_stop():
            return self.stop_event.is_set() or lease_lost.is_set() or not self.queue.owns(check_id, token)

        def emit(event):
            nonlocal phase, processed
            phase = event.phase
            processed = event.processed_until_sec
            self.store.append_event(check_id, token, event)

        thread = threading.Thread(target=pulse, name="lease-heartbeat", daemon=True)
        thread.start()
        started = time.monotonic()
        try:
            path = directory / (job["video_file"] or "expired")
            if not path.is_file():
                self.queue.fail(check_id, token, AnalysisError(
                    code="VIDEO_EXPIRED", message="Исходная запись больше недоступна.", failed_phase="decode"
                ), retryable=False)
                return True
            config = PipelineConfig(
                check_id=check_id, pipeline_version=job["pipeline_version"],
                models_dir=self.settings.models_dir, rules_dir=self.settings.rules_dir,
                device=self.settings.device, analysis_fps=self.settings.analysis_fps,
                evidence_overlay=self.settings.evidence_overlay,
                max_duration_sec=self.settings.max_duration_sec,
            )
            cached = self.store.cached_artifacts(check_id, token, staging)
            if cached is not None:
                artifacts, source_id, analyzed_at = cached
                self.store.complete(check_id, token, artifacts, cached_from_check_id=source_id,
                                    analyzed_at=analyzed_at)
                logger.info(json.dumps({"check_id": check_id, "phase": "done", "origin": "cache"}))
                return True
            artifacts = self.pipeline.analyze(path, work_dir=staging, config=config,
                                              emit=emit, should_stop=should_stop)
            if should_stop():
                self.queue.release(check_id, token)
                return True
            artifacts = AnalysisArtifacts.model_validate(artifacts)
            self.store.complete(check_id, token, artifacts)
            logger.info(json.dumps({"check_id": check_id, "phase": "done", "duration_sec": round(time.monotonic() - started, 3)}))
        except LeaseLost:
            pass
        except Exception as exc:
            if self.stop_event.is_set():
                self.queue.release(check_id, token)
            elif self.queue.owns(check_id, token):
                self.queue.fail(check_id, token, AnalysisError(
                    code=getattr(exc, "code", "ANALYSIS_FAILED"),
                    message="Обработка прервалась. Сервис попробует повторить её автоматически.",
                    failed_phase=phase, at_sec=processed,
                ))
                logger.exception(json.dumps({"check_id": check_id, "phase": phase}))
        finally:
            stop_lease.set()
            thread.join(timeout=2)
        return True

    def run(self):
        ready = False
        while not self.stop_event.is_set():
            if not ready:
                ready = self.prepare()
                if not ready:
                    self.stop_event.wait(10)
                    continue
            if time.monotonic() - self._last_cleanup > 300:
                cleanup(self.store)
                self._last_cleanup = time.monotonic()
            if not self.run_once():
                self.stop_event.wait(self.settings.worker_poll_seconds)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    worker = Worker(Settings.from_env())
    for event in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(event, lambda *_: worker.stop_event.set())
    worker.run()


if __name__ == "__main__":
    main()
