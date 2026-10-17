"""Run the actual local pipeline on organizer sources with safe resumable caching.

Only aggregate counts are printed. Raw artifacts, evidence images, exceptions and
per-clip numeric predictions remain under the ignored eval/runs/ directory.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import secrets
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from eval.dataset import (  # noqa: E402
    EvaluationSource,
    dataset_fingerprint,
    file_sha256,
    load_sources,
    select_sources,
)
from eval.metrics import aggregate, markdown_report, validate_prediction  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def private_output_root(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.is_relative_to(ROOT) and not resolved.is_relative_to(ROOT / "eval/runs"):
        raise ValueError("Private evaluation artifacts inside the repository must stay in ignored eval/runs/")
    return resolved


@contextlib.contextmanager
def exclusive_lock(directory: Path) -> Iterator[None]:
    """OS lock automatically releases on exit/crash; persisted PIDs are not trusted."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".evaluation.lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("EVALUATION_BUSY: another local evaluation owns the compute lock") from error
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError("EVALUATION_BUSY: another local evaluation owns the compute lock") from error
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def failure_code(error: BaseException) -> str:
    match = re.match(r"^([A-Z][A-Z_0-9]{2,60}):", str(error))
    return match.group(1) if match else type(error).__name__


class RunStore:
    def __init__(self, root: Path, fingerprint: str):
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise ValueError("Expected a SHA256 fingerprint")
        self.root = root.resolve()
        self.fingerprint = fingerprint
        self.directory = self.root / "profiles" / fingerprint
        self.directory.mkdir(parents=True, exist_ok=True)

    def source_dir(self, source: EvaluationSource) -> Path:
        return self.directory / "sources" / source.sha256

    def new_attempt(self, source: EvaluationSource) -> Path:
        directory = self.source_dir(source) / f"attempt-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(4)}"
        directory.mkdir(parents=True, exist_ok=False)
        return directory

    def cached(self, source: EvaluationSource) -> dict | None:
        pointer = self.source_dir(source) / "complete.json"
        if not pointer.is_file():
            return None
        record = read_json(pointer)
        validate_prediction(record, source, fingerprint=self.fingerprint)
        if record["status"] != "done":
            raise ValueError("Completion pointer is not a completed analysis")
        attempt = (self.source_dir(source) / record["artifact_directory"]).resolve()
        if not attempt.is_relative_to(self.source_dir(source).resolve()):
            raise ValueError("Cached artifacts escape their source directory")
        hashes = record.get("artifact_hashes", {})
        if not {"run.json", "finding_set.json", "engine_manifest.json", "video.json"}.issubset(hashes):
            raise ValueError("Cache has no complete artifact manifest")
        for relative, expected in hashes.items():
            path = (attempt / relative).resolve()
            if not path.is_relative_to(attempt) or not path.is_file() or file_sha256(path) != expected:
                raise ValueError("Cached artifacts are incomplete or changed")
        manifest, run = read_json(attempt / "engine_manifest.json"), read_json(attempt / "run.json")
        if manifest.get("complete") is not True or run.get("pipeline_version") != record.get("pipeline_version"):
            raise ValueError("Cached engine artifacts do not match the effective pipeline")
        return record

    def latest(self, source: EvaluationSource) -> dict | None:
        path = self.source_dir(source) / "latest.json"
        if not path.is_file():
            return None
        value = read_json(path)
        validate_prediction(value, source, fingerprint=self.fingerprint)
        return value

    def save(self, source: EvaluationSource, attempt: Path, record: dict) -> None:
        validate_prediction(record, source, fingerprint=self.fingerprint)
        attempt_root = attempt.resolve()
        if not attempt_root.is_relative_to(self.source_dir(source).resolve()):
            raise ValueError("Attempt is outside the assigned source directory")
        record = {**record, "artifact_directory": attempt.name}
        atomic_json(attempt / "prediction.json", record)
        atomic_json(self.source_dir(source) / "latest.json", record)
        if record["status"] == "done":
            atomic_json(self.source_dir(source) / "complete.json", record)


def freeze_profile(root: Path, profile: dict, dataset_hash: str) -> dict:
    if profile["detector_only"]:
        raise ValueError("A detector-only diagnostic cannot become the final semantic profile")
    path = root / "frozen-profile.json"
    frozen = {
        "schema_version": 1, "fingerprint": profile["fingerprint"],
        "algorithm_fingerprint": profile["algorithm_fingerprint"],
        "pipeline_version": profile["pipeline_version"], "dataset_fingerprint": dataset_hash,
        "frozen_at": utc_now(), "holdout_fold": 0,
    }
    if path.exists():
        previous = read_json(path)
        if any(previous.get(key) != frozen[key] for key in ("fingerprint", "algorithm_fingerprint", "pipeline_version", "dataset_fingerprint", "holdout_fold")):
            raise ValueError("FROZEN_PROFILE_MISMATCH: a different final profile is already frozen")
        return previous
    atomic_json(path, frozen)
    return frozen


def guard_holdout(root: Path, profile: dict, dataset_hash: str, selected: list[EvaluationSource]) -> None:
    if not selected or any(source.fold != 0 for source in selected):
        raise ValueError("Holdout selection must contain the reserved fold 0")
    frozen_path = root / "frozen-profile.json"
    if not frozen_path.is_file():
        raise ValueError("HOLDOUT_NOT_FROZEN: freeze the final semantic profile before using fold 0")
    frozen = read_json(frozen_path)
    if (profile["detector_only"] or frozen.get("fingerprint") != profile["fingerprint"]
            or frozen.get("algorithm_fingerprint") != profile["algorithm_fingerprint"]
            or frozen.get("dataset_fingerprint") != dataset_hash):
        raise ValueError("HOLDOUT_PROFILE_MISMATCH: holdout requires the unchanged frozen profile and labels")
    claim = {
        "schema_version": 1, "fingerprint": profile["fingerprint"],
        "algorithm_fingerprint": profile["algorithm_fingerprint"],
        "pipeline_version": profile["pipeline_version"], "dataset_fingerprint": dataset_hash,
        "source_sha256": sorted(source.sha256 for source in selected),
    }
    path = root / "holdout-session.json"
    if path.exists():
        previous = read_json(path)
        if any(previous.get(key) != value for key, value in claim.items()):
            raise ValueError("HOLDOUT_ALREADY_USED: only resume of the original frozen evaluation is permitted")
        return
    atomic_json(path, {**claim, "started_at": utc_now()})


def require_combined_report_scope(root: Path, profile: dict, dataset_hash: str, sources: list[EvaluationSource]) -> None:
    """Read-only: a combined report cannot claim or initialize reserved holdout."""
    frozen_path, session_path = root / "frozen-profile.json", root / "holdout-session.json"
    if not frozen_path.is_file() or not session_path.is_file():
        raise ValueError("HOLDOUT_NOT_EVALUATED: combined reporting requires an existing frozen holdout session")
    frozen, session = read_json(frozen_path), read_json(session_path)
    expected = {"fingerprint": profile["fingerprint"],
                "algorithm_fingerprint": profile["algorithm_fingerprint"],
                "pipeline_version": profile["pipeline_version"], "dataset_fingerprint": dataset_hash}
    if profile["detector_only"] or any(frozen.get(key) != value or session.get(key) != value
                                       for key, value in expected.items()):
        raise ValueError("HOLDOUT_PROFILE_MISMATCH: combined reporting requires the unchanged frozen profile")
    if session.get("source_sha256") != sorted(source.sha256 for source in sources if source.fold == 0):
        raise ValueError("HOLDOUT_SCOPE_MISMATCH: combined reporting requires the complete reserved fold")


def _artifact_projection(artifacts, source: EvaluationSource, *, fingerprint: str, pipeline_version: str, latency: float, attempt: Path) -> dict:
    if artifacts.run.pipeline_version != pipeline_version:
        raise ValueError("PIPELINE_VERSION_CHANGED: analysis did not use the prepared effective version")
    if artifacts.run.analysis_ready or artifacts.run.analysis_mode != "diagnostic":
        raise ValueError("EVALUATION_MODE_MISMATCH: evaluation artifacts must be nonpublishable diagnostics")
    if abs(artifacts.video.duration_sec - source.duration_sec) > 0.1:
        raise ValueError("SOURCE_DURATION_MISMATCH: decoded source differs from inventory")
    required = ["run.json", "finding_set.json", "engine_manifest.json", "video.json"]
    from PIL import Image

    for finding in artifacts.finding_set.findings:
        for evidence in finding.evidence:
            relative = f"evidence/original/{evidence.frame_id}.jpg"
            path = attempt / relative
            with Image.open(path) as image:
                if image.format != "JPEG":
                    raise ValueError("EVIDENCE_INVALID: original evidence is not JPEG")
                image.verify()
            required.append(relative)
    manifest = read_json(attempt / "engine_manifest.json")
    if manifest.get("complete") is not True:
        raise ValueError("ARTIFACTS_INCOMPLETE: pipeline completion manifest is absent")
    record = {
        "schema_version": 1, "source_sha256": source.sha256, "fingerprint": fingerprint,
        "pipeline_version": pipeline_version, "status": "done", "completed_at": utc_now(),
        "analysis_ready": artifacts.run.analysis_ready, "analysis_mode": artifacts.run.analysis_mode,
        "latency_sec": latency, "verdict": artifacts.finding_set.verdict,
        "findings": [{"timecode_sec": finding.timecode_sec, "rule_id": finding.rule_id} for finding in artifacts.finding_set.findings],
        "rule_statuses": [status.status for status in artifacts.finding_set.rules],
        "phase_timings_sec": artifacts.run.phase_timings_sec,
        "model_versions": artifacts.run.model_versions,
        "artifact_hashes": {relative: file_sha256(attempt / relative) for relative in sorted(set(required))},
    }
    validate_prediction(record, source, fingerprint=fingerprint)
    return record


def _runtime_profile(args, output_root: Path):
    from navidu.contracts.pipeline import PipelineConfig
    from navidu.perception.assets import model_entry, read_manifest
    from navidu.perception.configuration import configuration_snapshot
    from navidu.perception.runtime import get_runtime
    from navidu.perception.validation import SEMANTIC_CAPABILITIES

    os.environ["NAVIDU_REQUIRE_SEMANTIC"] = "0" if args.detector_only else "1"
    # These flags prevent a local model loader from reaching a model hub at inference time.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    config = PipelineConfig.from_env(check_id=secrets.token_hex(6))
    # The explicit evaluation entry point may measure an unvalidated local model.
    # This flag never enables publishing the resulting analysis as a user check.
    updates = {"models_dir": args.models_dir.resolve(), "rules_dir": args.rules_dir.resolve(),
               "diagnostic_only": True}
    if args.device is not None:
        updates["device"] = args.device
    if args.analysis_fps is not None:
        updates["analysis_fps"] = args.analysis_fps
    config = PipelineConfig.model_validate({**config.model_dump(), **updates})
    runtime = get_runtime()
    status = runtime.runtime_status(config)
    if args.command == "preflight":
        print(json.dumps({"event": "preflight", "model_status": status.model_status,
                          "device": status.device, "detector_only": args.detector_only,
                          "analysis_ready": status.analysis_ready, "analysis_mode": status.analysis_mode,
                          "pipeline_version": status.pipeline_version}, ensure_ascii=True), flush=True)
        return None
    if status.model_status in {"missing", "error"}:
        raise RuntimeError(f"MODEL_NOT_READY: local runtime is {status.model_status}; use the local model setup instructions")
    preparation_log = output_root / f"prepare-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(4)}.log"
    started = time.perf_counter()
    with preparation_log.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            runtime.prepare(config)
        except Exception:
            traceback.print_exc(file=log)
            raise
    prepare_sec = time.perf_counter() - started
    status = runtime.runtime_status(config)
    if status.model_status != "ready" or "+runtime." not in status.pipeline_version:
        raise RuntimeError("MODEL_NOT_READY: prepared local runtime has no effective version")
    if not isinstance(runtime.quality_binding, str) or not re.fullmatch(r"[0-9a-f]{64}", runtime.quality_binding):
        raise RuntimeError("ALGORITHM_BINDING_MISSING: prepared runtime has no measured algorithm identity")
    config = config.model_copy(update={"pipeline_version": status.pipeline_version})
    fingerprint = hashlib.sha256(status.pipeline_version.encode("utf-8")).hexdigest()
    manifest = read_manifest(config.models_dir)
    model_ids = [os.getenv("NAVIDU_DETECTOR_MODEL", "rfdetr-nano-onnx")]
    if not args.detector_only:
        model_ids.append(os.getenv("NAVIDU_SEMANTIC_MODEL", "internvl3_5-1b"))
    snapshot = configuration_snapshot(config, [model_entry(manifest, model_id) for model_id in model_ids], require_semantic=not args.detector_only)
    profile = {
        "schema_version": 1, "fingerprint": fingerprint, "pipeline_version": status.pipeline_version,
        "algorithm_fingerprint": runtime.quality_binding,
        "detector_only": args.detector_only, "created_at": utc_now(), "prepare_sec": prepare_sec,
        "diagnostic_only": True,
        "candidate_semantic_capabilities": [] if args.detector_only else sorted(SEMANTIC_CAPABILITIES),
        "runtime": {"model_status": status.model_status, "device": status.device,
                    "analysis_ready": status.analysis_ready, "analysis_mode": status.analysis_mode},
        "package_versions": snapshot["packages"], "configuration": snapshot,
    }
    print(json.dumps({"event": "runtime_ready", "device": status.device, "prepare_sec": round(prepare_sec, 3), "pipeline_version": status.pipeline_version}), flush=True)
    return runtime, config, profile


def write_report(store: RunStore, sources: list[EvaluationSource], records: dict[str, dict], *, profile: dict, split: str, dataset_hash: str, run_stats: dict, output: Path | None = None) -> dict:
    report = {
        "schema_version": 1, "created_at": utc_now(), "split": split,
        "fingerprint": profile["fingerprint"], "pipeline_version": profile["pipeline_version"],
        "algorithm_fingerprint": profile["algorithm_fingerprint"],
        "dataset_fingerprint": dataset_hash, "runtime": profile["runtime"],
        "diagnostic_only": profile.get("diagnostic_only", True),
        "candidate_semantic_capabilities": profile.get("candidate_semantic_capabilities", []),
        "package_versions": profile["package_versions"], "prepare_sec": profile["prepare_sec"],
        "run_stats": run_stats,
        "metrics": aggregate(sources, records, fingerprint=profile["fingerprint"], detector_only=profile["detector_only"]),
    }
    if split == "all":
        report["metrics_by_partition"] = {}
        for partition in ("development", "holdout"):
            partition_sources = select_sources(sources, partition)
            partition_records = {source.sha256: records[source.sha256]
                                 for source in partition_sources if source.sha256 in records}
            report["metrics_by_partition"][partition] = aggregate(
                partition_sources, partition_records, fingerprint=profile["fingerprint"],
                detector_only=profile["detector_only"])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = store.directory / "reports" / f"{split}-{timestamp}-{secrets.token_hex(2)}.json"
    atomic_json(path, report)
    path.with_suffix(".md").write_text(markdown_report(report), encoding="utf-8")
    atomic_json(store.directory / f"latest-{split}-aggregate.json", report)
    if output is not None:
        atomic_json(output.with_suffix(".json"), report)
        output.with_suffix(".md").write_text(markdown_report(report), encoding="utf-8")
    return report


def _load_tables(args):
    tables = [args.inventory.resolve(), args.labels.resolve(), args.folds.resolve()]
    sources = load_sources(*tables)
    if args.split == "all":
        if args.command != "report" or args.video_id:
            raise ValueError("Combined scope is available only for report-only mode without selected video IDs")
        selected = sources
    else:
        selected = select_sources(sources, args.split, args.video_id)
    if args.split == "holdout" and args.video_id:
        raise ValueError("Holdout must evaluate the complete reserved fold, not a selected subset")
    return sources, selected, dataset_fingerprint(tables)


def run(args) -> int:
    output_root = private_output_root(args.output_root)
    _, selected, dataset_hash = _load_tables(args)
    with exclusive_lock(output_root):
        prepared = _runtime_profile(args, output_root)
        if prepared is None:
            return 0
        runtime, config, profile = prepared
        store = RunStore(output_root, profile["fingerprint"])
        profile_path = store.directory / "profile.json"
        if not profile_path.exists():
            atomic_json(profile_path, profile)
        if args.command == "freeze":
            frozen = freeze_profile(output_root, profile, dataset_hash)
            print(json.dumps({"event": "profile_frozen", "fingerprint": frozen["fingerprint"], "holdout_fold": 0}), flush=True)
            return 0
        if args.split == "holdout":
            guard_holdout(output_root, profile, dataset_hash, selected)
        elif (output_root / "holdout-session.json").exists():
            if read_json(output_root / "holdout-session.json").get("fingerprint") != profile["fingerprint"]:
                raise ValueError("HOLDOUT_ALREADY_USED: do not tune a new profile after inspecting final holdout")
        from navidu.pipeline.run import RealPipeline

        pipeline = RealPipeline(runtime)
        stopped = False

        def request_stop(_signum, _frame):
            nonlocal stopped
            stopped = True

        original_handlers = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            original_handlers[signum] = signal.signal(signum, request_stop)
        records = {}
        stats = {"selected": len(selected), "executed": 0, "cache_hits": 0, "invalid_caches": 0, "reused_failures": 0, "interrupted": False}
        try:
            for index, source in enumerate(selected, start=1):
                if stopped:
                    break
                status = runtime.runtime_status(config)
                if status.model_status != "ready" or status.pipeline_version != profile["pipeline_version"]:
                    raise RuntimeError("PIPELINE_CHANGED: evaluation must restart with an unchanged effective profile")
                cached = None
                try:
                    cached = store.cached(source)
                except (OSError, ValueError, KeyError, TypeError):
                    stats["invalid_caches"] += 1
                if cached is not None:
                    records[source.sha256] = cached
                    stats["cache_hits"] += 1
                    continue
                previous = store.latest(source)
                if previous and previous["status"] != "done" and not args.retry_failed:
                    records[source.sha256] = previous
                    stats["reused_failures"] += 1
                    continue
                previous_failures = int(previous.get("previous_failed_attempts", 0)) + int(previous["status"] != "done") if previous else 0
                attempt = store.new_attempt(source)
                source_path = source.video_path(args.dataset_dir)
                phase_path = attempt / "progress.jsonl"

                def emit(event, phase_path=phase_path):
                    # No observations_delta, model text or frames in tool output.
                    with phase_path.open("a", encoding="utf-8") as progress:
                        progress.write(json.dumps({"phase": event.phase, "processed_until_sec": event.processed_until_sec, "at": utc_now()}) + "\n")

                started = time.perf_counter()
                record = None
                try:
                    if file_sha256(source_path) != source.sha256:
                        raise RuntimeError("SOURCE_HASH_MISMATCH: organizer source differs from the inventory")
                    started = time.perf_counter()
                    with (attempt / "runtime.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        artifacts = pipeline.analyze(source_path, work_dir=attempt,
                            config=config.model_copy(update={"check_id": secrets.token_hex(6)}),
                            emit=emit, should_stop=lambda: stopped)
                    elapsed = time.perf_counter() - started
                    status_after = runtime.runtime_status(config)
                    if status_after.model_status != "ready" or status_after.pipeline_version != profile["pipeline_version"]:
                        raise RuntimeError("PIPELINE_CHANGED: analysis changed before completion")
                    record = _artifact_projection(artifacts, source, fingerprint=profile["fingerprint"], pipeline_version=profile["pipeline_version"], latency=elapsed, attempt=attempt)
                except Exception as error:
                    (attempt / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    record = {
                        "schema_version": 1, "source_sha256": source.sha256, "fingerprint": profile["fingerprint"],
                        "pipeline_version": profile["pipeline_version"], "status": "cancelled" if stopped else "failed",
                        "latency_sec": time.perf_counter() - started, "failure_code": failure_code(error),
                        "completed_at": utc_now(),
                    }
                record["previous_failed_attempts"] = previous_failures
                store.save(source, attempt, record)
                records[source.sha256] = record
                stats["executed"] += 1
                print(json.dumps({"event": "source_finished", "index": index, "total": len(selected), "status": record["status"], "elapsed_sec": round(record["latency_sec"], 3)}), flush=True)
        finally:
            stats["interrupted"] = stopped
            for signum, previous_handler in original_handlers.items():
                signal.signal(signum, previous_handler)
            # An interrupted resume must not call already completed, unvisited
            # cache entries "missing" merely because iteration stopped early.
            for source in selected:
                if source.sha256 not in records:
                    try:
                        available = store.cached(source) or store.latest(source)
                    except (OSError, ValueError, KeyError, TypeError):
                        available = None
                    if available is not None:
                        records[source.sha256] = available
            report = write_report(store, selected, records, profile=profile, split=args.split, dataset_hash=dataset_hash, run_stats=stats, output=args.aggregate_report)
            print(json.dumps({"event": "aggregate_saved", "profile": profile["fingerprint"], "counts": report["metrics"]["counts"], "cache_hits": stats["cache_hits"], "quality_metrics_available": not args.detector_only}), flush=True)
        return 130 if stopped else 1 if report["metrics"]["counts"]["failed_unique_sources"] else 0


def report_existing(args) -> int:
    output_root = private_output_root(args.output_root)
    _, selected, dataset_hash = _load_tables(args)
    store = RunStore(output_root, args.profile)
    profile = read_json(store.directory / "profile.json")
    if args.split == "all":
        require_combined_report_scope(output_root, profile, dataset_hash, selected)
    records = {}
    for source in selected:
        record = store.cached(source) or store.latest(source)
        if record is not None:
            records[source.sha256] = record
    result = write_report(store, selected, records, profile=profile, split=args.split, dataset_hash=dataset_hash, run_stats={"executed": 0, "report_only": True}, output=args.aggregate_report)
    print(json.dumps({"event": "aggregate_saved", "profile": args.profile, "counts": result["metrics"]["counts"]}), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "run", "freeze", "report"])
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/dataset")
    parser.add_argument("--inventory", type=Path, default=ROOT / "research/dataset/inventory.csv")
    parser.add_argument("--labels", type=Path, default=ROOT / "eval/labels.csv")
    parser.add_argument("--folds", type=Path, default=ROOT / "eval/folds.csv")
    parser.add_argument("--output-root", type=Path, default=ROOT / "eval/runs")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--rules-dir", type=Path, default=ROOT / "rules")
    parser.add_argument("--split", choices=["development", "holdout", "all"], default="development",
                        help="all is report-only and requires the existing frozen holdout session")
    parser.add_argument("--video-id", action="append", help="Optional development subset; exact Unicode identifier")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--analysis-fps", type=float)
    parser.add_argument("--detector-only", action="store_true", help="Object diagnostic only; disables quality metrics")
    parser.add_argument("--retry-failed", action="store_true", help="Retry only failed attempts under unchanged parameters")
    parser.add_argument("--profile", help="Existing SHA256 profile for report-only mode")
    parser.add_argument("--aggregate-report", type=Path, help="Optional shareable aggregate JSON/Markdown path; never raw artifacts")
    args = parser.parse_args()
    if args.command == "report" and not args.profile:
        parser.error("--profile is required for report-only mode")
    try:
        return report_existing(args) if args.command == "report" else run(args)
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        # Exception messages can contain private source paths or model text.
        print(json.dumps({"event": "evaluation_blocked", "code": failure_code(error), "exception_type": type(error).__name__}), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
