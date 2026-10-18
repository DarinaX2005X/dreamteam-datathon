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
from scripts.smoke_api import SmokeFailure, validate_done, validate_processing  # noqa: E402


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def jury_check(args, client, state, started) -> None:
    values = dict(line.split("=", 1) for line in args.key_file.read_text().splitlines()
                  if "=" in line and not line.startswith("#"))
    key = values["LADO_API_KEY"].strip()
    headers = {"X-API-Key": key}
    if args.phase == "auth":
        statuses = {}
        for label, supplied in (("missing", {}), ("invalid", {"X-API-Key": "invalid-key"})):
            for method, path in (("POST", "/api/v1/analyze"), ("GET", "/api/v1/jobs/000000000000")):
                response = client.request(method, path, headers=supplied)
                statuses[f"{label}_{method}"] = response.status_code
                if response.status_code != 401:
                    raise RuntimeError("Authentication check failed")
        valid = client.get("/api/v1/jobs/000000000000", headers=headers)
        if valid.status_code != 404:
            raise RuntimeError("Valid jury key was not accepted")
        state["auth"] = {**statuses, "valid_key_unknown_job": valid.status_code}
        print(json.dumps(state["auth"]), flush=True)
        return
    if args.phase == "jury":
        if "job_id" in state:
            raise RuntimeError("Already submitted; use jury-inspect instead")
        sources = load_sources(ROOT / "research/dataset/inventory.csv",
                               ROOT / "eval/labels.csv", ROOT / "eval/folds.csv")
        candidates = [item for item in sources if item.fold != 0
                      and bool(item.has_violation) == (args.source_kind == "positive")
                      and (args.source_sha256 is None or item.sha256 == args.source_sha256)]
        if not candidates:
            raise RuntimeError("No authorized development source matches the selection")
        source = max(candidates, key=lambda item: item.duration_sec)
        path = source.video_path(ROOT / "data/dataset")
        if file_sha256(path) != source.sha256:
            raise RuntimeError("Source identity changed")
        state.update(request_started_at_unix=time.time(), source_duration_sec=source.duration_sec,
                     source_sha256=source.sha256)
        save_state(args.state, state)
        with path.open("rb") as video:
            response = client.post("/api/v1/analyze", headers=headers,
                files={"file": ("jury-smoke.mov", video, "video/quicktime")},
                timeout=min(240, max(1, args.timeout - (time.monotonic() - started))))
        response.raise_for_status()
        try:
            job_id = validate_processing(response.json())
        except SmokeFailure:
            raise RuntimeError("Invalid POST response schema; private payload was not printed") from None
        state.update(job_id=job_id, post_elapsed_seconds=round(time.monotonic() - started, 3),
                     post_analysis_mode=response.headers.get("X-Analysis-Mode"))
        save_state(args.state, state)
        print(json.dumps({"job_id": job_id, "status": "processing",
                          "post_elapsed_seconds": state["post_elapsed_seconds"]}), flush=True)
    while time.monotonic() - started < args.timeout:
        # Inspecting a retained result can wake a cold scale-to-zero container.
        request_limit = 240 if args.phase == "jury-inspect" else 30
        response = client.get(f"/api/v1/jobs/{state['job_id']}", headers=headers,
                              timeout=min(request_limit, max(1, args.timeout - (time.monotonic() - started))))
        response.raise_for_status()
        body = response.json()
        status = body.get("status")
        if status != "processing":
            try:
                verdict, count = validate_done(body)
            except SmokeFailure:
                state["jury_result"] = {"status": status, "error_code": (body.get("error") or {}).get("code")}
                save_state(args.state, state)
                raise RuntimeError("Jury response failed; see numeric state, private payload was not printed") from None
            elapsed = (state.get("jury_result", {}).get("elapsed_seconds")
                       if args.phase == "jury-inspect" else None)
            if elapsed is None:
                elapsed = time.time() - state["request_started_at_unix"]
            state["jury_result"] = {"status": status, "verdict": verdict, "violation_count": count,
                "evidence_frame_count": sum(len(item["evidence_frames"]) for item in body["violations"]),
                "analysis_mode": response.headers.get("X-Analysis-Mode"),
                "schema_and_jpeg_valid": True, "elapsed_seconds": round(elapsed, 3), "within_300_seconds": elapsed <= 300}
            print(json.dumps(state["jury_result"]), flush=True)
            return
        if args.phase == "jury-inspect":
            print(json.dumps({"job_id": state["job_id"], "status": status}), flush=True)
            return
        time.sleep(3)
    state["jury_result"] = {"status": "timeout", "within_300_seconds": False}
    save_state(args.state, state)
    raise RuntimeError("Jury job exceeded the response deadline")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("health", "submit", "inspect", "auth", "jury", "jury-inspect"))
    parser.add_argument("--url", required=True)
    parser.add_argument("--state", type=Path, default=ROOT / "eval/runs/modal-http.json")
    parser.add_argument("--key-file", type=Path, default=ROOT / ".env.jury")
    parser.add_argument("--source-kind", choices=("clean", "positive"), default="clean")
    parser.add_argument("--source-sha256", help="Exact development source SHA; holdout remains excluded")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    state = json.loads(args.state.read_text()) if args.state.exists() else {}
    started = time.monotonic()
    with httpx.Client(base_url=args.url.rstrip("/"), timeout=240, follow_redirects=False) as client:
        if args.phase in {"auth", "jury", "jury-inspect"}:
            jury_check(args, client, state, started)
        elif args.phase == "health":
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
    save_state(args.state, state)


if __name__ == "__main__":
    main()
