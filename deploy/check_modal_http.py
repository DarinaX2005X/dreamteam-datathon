"""HTTP-only Modal verification; never opens or drives a browser."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.dataset import file_sha256, load_sources  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("health", "submit", "inspect"))
    parser.add_argument("--url", required=True)
    parser.add_argument("--state", type=Path, default=ROOT / "eval/runs/modal-http.json")
    args = parser.parse_args()
    state = json.loads(args.state.read_text()) if args.state.exists() else {}
    started = time.monotonic()
    with httpx.Client(base_url=args.url.rstrip("/"), timeout=240, follow_redirects=False) as client:
        if args.phase == "health":
            for path in ("/", "/app", "/api/ui/rules"):
                response = client.get(path)
                response.raise_for_status()
                print(json.dumps({"path": path, "status": response.status_code}), flush=True)
            for attempt in range(60):
                response = client.get("/api/ui/health")
                response.raise_for_status()
                health = response.json()
                record = {key: health.get(key) for key in (
                    "ready", "analysis_ready", "model_status", "analysis_mode", "pipeline_version", "device")}
                print(json.dumps({"attempt": attempt, **record}), flush=True)
                if health.get("ready"):
                    state["health"] = record
                    break
                time.sleep(5)
            else:
                raise RuntimeError("Service did not become ready")
        elif args.phase == "submit":
            if "check_id" in state:
                raise RuntimeError("This smoke job was already submitted; inspect it instead")
            sources = load_sources(ROOT / "research/dataset/inventory.csv",
                                   ROOT / "eval/labels.csv", ROOT / "eval/folds.csv")
            # A known clean development source, not the user's reflection upload.
            source = max((item for item in sources if item.fold == 4 and not item.has_violation),
                         key=lambda item: item.duration_sec)
            path = source.video_path(ROOT / "data/dataset")
            if file_sha256(path) != source.sha256:
                raise RuntimeError("Source identity changed")
            with path.open("rb") as video:
                response = client.post("/api/ui/checks", files={"file": ("modal-smoke.mov", video, "video/quicktime")})
            response.raise_for_status()
            result = response.json()["check"]
            state.update(check_id=result["id"], source_duration_sec=source.duration_sec,
                         submitted_at_unix=time.time(), source_sha256=source.sha256)
            print(json.dumps({"status": response.status_code, "check_id": result["id"],
                              "source_duration_sec": source.duration_sec}), flush=True)
        else:
            response = client.get(f"/api/ui/checks/{state['check_id']}")
            response.raise_for_status()
            result = response.json()["check"]
            findings = (result.get("finding_set") or {}).get("findings") or []
            record = {"status": result["status"], "verdict": result.get("verdict"),
                      "error_code": (result.get("error") or {}).get("code"),
                      "finding_count": len(findings), "pipeline_version": result.get("pipeline_version")}
            state.setdefault("inspections", []).append({"at_unix": time.time(), **record})
            print(json.dumps(record), flush=True)
    state[args.phase + "_elapsed_seconds"] = round(time.monotonic() - started, 3)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.state.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
