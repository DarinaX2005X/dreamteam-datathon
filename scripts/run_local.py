#!/usr/bin/env python3
"""Run the Lado API and one warm model worker as owned child processes."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
MIN_AUTO_CUDA_MIB = 8192


def select_device(requested: str, gpu_memory_mib: float | None) -> str:
    if requested in {"cpu", "cuda"}:
        return requested
    return "cuda" if gpu_memory_mib is not None and gpu_memory_mib >= MIN_AUTO_CUDA_MIB else "cpu"


def default_detector(device: str) -> str:
    return "rfdetr-nano-onnx-fp32" if device == "cuda" else "rfdetr-nano-onnx"


def detected_gpu_memory_mib() -> float | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            return max(float(line.strip()) for line in result.stdout.splitlines() if line.strip())
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def load_dotenv(path: Path) -> None:
    """Load literal KEY=VALUE pairs while preserving already-exported environment values."""
    if not path.is_file():
        return
    for number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"{path}:{number}: expected KEY=VALUE")
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip()
        if not ENV_NAME.fullmatch(name):
            raise ValueError(f"{path}:{number}: invalid environment name {name!r}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if "\x00" in value:
            raise ValueError(f"{path}:{number}: NUL is not allowed")
        os.environ.setdefault(name, value)


def wait_until_bound(
    url: str, processes: list[subprocess.Popen], timeout: float
) -> dict:
    deadline = time.monotonic() + timeout
    last_error = "not contacted"
    while time.monotonic() < deadline:
        for process in processes:
            code = process.poll()
            if code is not None:
                raise RuntimeError(
                    f"{process.args!r} exited with code {code}; inspect logs/"
                )
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return json.load(response)
        except (OSError, ValueError, urllib.error.HTTPError) as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise TimeoutError(f"API did not bind within {timeout:g}s: {last_error}")


def main() -> int:
    try:
        load_dotenv(ROOT / ".env")
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"Cannot read .env: {exc}", file=sys.stderr)
        return 2
    device = select_device(os.getenv("NAVIDU_DEVICE", "auto").lower(), detected_gpu_memory_mib())
    os.environ["NAVIDU_DEVICE"] = device
    os.environ.setdefault("NAVIDU_DETECTOR_MODEL", default_detector(device))
    parser = argparse.ArgumentParser(
        description="Start the local API and model worker; Ctrl+C stops only these children."
    )
    parser.add_argument("--host", default=os.getenv("NAVIDU_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("NAVIDU_PORT", "8000"))
    )
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument(
        "--api-python",
        default=os.getenv(
            "NAVIDU_API_PYTHON",
            str(
                BACKEND
                / ".venv"
                / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            ),
        ),
    )
    parser.add_argument("--worker-python", default=os.getenv("NAVIDU_WORKER_PYTHON"))
    parser.add_argument(
        "--no-worker",
        action="store_true",
        help="Diagnostic API-only launch; health will remain not ready.",
    )
    args = parser.parse_args()

    api_python = Path(args.api_python).expanduser().resolve()
    worker_python = (
        Path(args.worker_python).expanduser().resolve()
        if args.worker_python
        else api_python
    )
    if not api_python.is_file():
        parser.error(
            f"API Python is missing: {api_python}. Run the platform setup script first."
        )
    if not args.no_worker and not worker_python.is_file():
        parser.error(
            f"Worker Python is missing: {worker_python}. Set NAVIDU_WORKER_PYTHON or run setup."
        )

    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.setdefault("NAVIDU_DATA_DIR", str(ROOT / "data"))
    env.setdefault("NAVIDU_MODELS_DIR", str(ROOT / "models"))
    env.setdefault("NAVIDU_WEB_DIST", str(ROOT / "web" / "dist"))
    env["NAVIDU_HOST"] = args.host
    env["NAVIDU_PORT"] = str(args.port)
    env.setdefault("PYTHONUNBUFFERED", "1")
    children: list[tuple[subprocess.Popen, object]] = []
    try:
        commands = [("api", api_python, ["-m", "navidu.service"])]
        if not args.no_worker:
            commands.append(("worker", worker_python, ["-m", "navidu.service.worker"]))
        for name, executable, tail in commands:
            stream = (logs / f"{name}.log").open("a", encoding="utf-8", buffering=1)
            process = subprocess.Popen(
                [str(executable), *tail],
                cwd=BACKEND,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                creationflags=CREATE_NO_WINDOW,
            )
            children.append((process, stream))
        health = wait_until_bound(
            f"http://{args.host}:{args.port}/api/ui/health",
            [item[0] for item in children],
            args.timeout,
        )
        state = (
            "model ready"
            if health.get("ready")
            else f"API bound; model {health.get('model_status', 'not ready')}"
        )
        print(f"Lado {state} at http://{args.host}:{args.port}")
        print(f"Logs: {logs}")
        while all(process.poll() is None for process, _ in children):
            time.sleep(0.5)
        return next(
            (process.returncode for process, _ in children if process.returncode), 1
        )
    except KeyboardInterrupt:
        print("Stopping Lado child processes...")
        return 130
    except (RuntimeError, TimeoutError) as exc:
        print(f"Launch failed: {exc}", file=sys.stderr)
        return 1
    finally:
        for process, _ in children:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + 8
        for process, _ in children:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
        for _, stream in children:
            stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
