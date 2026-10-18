"""Single-writer Lado web service on a scale-to-zero Modal L4."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import modal

from deploy.modal_training import image as base_image
from deploy.modal_training import models

ROOT = Path(__file__).resolve().parents[1]

app = modal.App("lado")
data = modal.Volume.from_name("lado-data-v1", version=2, create_if_missing=True)
runtime_secret = modal.Secret.from_name(
    "lado-runtime", required_keys=["NAVIDU_API_KEYS"]
)

image = (
    base_image
    .add_local_file(ROOT / "deploy/modal_training.py", "/workspace/deploy/modal_training.py", copy=True)
    .add_local_dir(ROOT / "rules", "/app/rules", copy=True)
    .add_local_dir(ROOT / "contracts", "/app/contracts", copy=True)
    .add_local_dir(ROOT / "web" / "dist", "/app/web/dist", copy=True)
    .add_local_file(ROOT / "models" / "manifest.json", "/app/models-manifest.json", copy=True)
    .env(
        {
            "NAVIDU_ENV": "production",
            "NAVIDU_HOST": "0.0.0.0",
            "NAVIDU_PORT": "8000",
            "NAVIDU_DATA_DIR": "/app/data",
            "NAVIDU_MODELS_DIR": "/app/models",
            "NAVIDU_RULES_DIR": "/app/rules",
            "NAVIDU_WEB_DIST": "/app/web/dist",
            "NAVIDU_DEVICE": "auto",
            "NAVIDU_WORKERS": "1",
            "NAVIDU_REQUIRE_SEMANTIC": "true",
            "NAVIDU_EXPERIMENTAL_ANALYSIS": "1",
            "NAVIDU_ALLOW_EXPERIMENTAL_API": "1",
            "NAVIDU_DETECTOR_MODEL": "rfdetr-nano-onnx",
            "NAVIDU_SEMANTIC_MODEL": "qwen3-vl-8b",
            "NAVIDU_OBJECT_MODEL": "lado-objects-v7-composite",
            "NAVIDU_CPU_THREADS": "2",
            "NAVIDU_SEMANTIC_WINDOW_FRAMES": "32",
            "NAVIDU_SEMANTIC_FPS": "1",
            "NAVIDU_SEMANTIC_MAX_NEW_TOKENS": "1536",
            "NAVIDU_SEMANTIC_MAX_PATCHES": "4",
            "NAVIDU_SEMANTIC_DTYPE": "bf16",
            "NAVIDU_SEMANTIC_STRUCTURED_OUTPUT": "0",
            "NAVIDU_SEMANTIC_PROMPT_VARIANT": "v10-questions",
            "NAVIDU_QWEN_INPUT_MODE": "video",
            "NAVIDU_QWEN_MAX_PIXELS": "262144",
            "NAVIDU_QWEN_DETAIL_VIEW": "1",
            "NAVIDU_PROPOSAL_ADAPTER": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "USE_TF": "0",
            "USE_FLAX": "0",
        }
    )
)


@app.cls(
    image=image,
    gpu="L4",
    cpu=(2, 2),
    memory=(24576, 24576),
    max_containers=1,
    min_containers=0,
    scaledown_window=90,
    timeout=3600,
    volumes={"/app/data": data, "/app/models": models},
    secrets=[runtime_secret],
)
@modal.concurrent(max_inputs=16, target_inputs=12)
class LadoService:
    processes: list[subprocess.Popen]

    @modal.enter()
    def start_worker(self) -> None:
        from navidu.service.config import Settings
        from navidu.service.readiness import analysis_ready
        from navidu.service.store import Store

        store = Store(Settings.from_env())
        # This deployment has exactly one writer. Old container heartbeats must
        # not make a new cold instance appear ready before its own model loads.
        with store.connection(transaction=True) as db:
            db.execute("DELETE FROM worker_state")
        env = os.environ.copy()
        self.processes = [
            subprocess.Popen(
                ["python", "-m", "navidu.service.worker"], cwd="/app", env=env
            )
        ]
        # A cold POST must reach a prepared worker, rather than immediately
        # failing with NOT_READY while the model is still loading.
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self.processes[0].poll() is not None:
                raise RuntimeError("Analysis worker exited during startup")
            with store.connection() as db:
                state = db.execute("SELECT * FROM worker_state WHERE worker_id=?",
                                   (f"worker-{self.processes[0].pid}",)).fetchone()
            if state is not None and analysis_ready({**dict(state), "analysis_ready": bool(state["analysis_ready"])}):
                return
            time.sleep(1)
        self.processes[0].terminate()
        raise RuntimeError("Analysis worker did not become ready within 180 seconds")

    @modal.asgi_app()
    def web(self):
        from navidu.service.app import create_app
        service = create_app()

        @service.middleware("http")
        async def keep_jobs_alive(request, call_next):
            response = await call_next(request)
            if request.method == "POST" and response.status_code in {200, 201, 202}:
                # Hold a real Modal input while queued work runs, even if the
                # browser closes. max_containers=1 preserves SQLite ownership.
                await LadoService().wait_for_queue.spawn.aio()
            return response

        return service

    @modal.method()
    def wait_for_queue(self) -> None:
        from navidu.service.config import Settings
        from navidu.service.store import Store
        store = Store(Settings.from_env())
        deadline = time.monotonic() + 3500
        while time.monotonic() < deadline:
            with store.connection() as db:
                count = db.execute("SELECT count(*) FROM checks WHERE status IN ('queued','processing')").fetchone()[0]
            if not count:
                return
            time.sleep(3)

    @modal.exit()
    def shutdown(self) -> None:
        for process in getattr(self, "processes", []):
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        for process in getattr(self, "processes", []):
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
        data.commit()
