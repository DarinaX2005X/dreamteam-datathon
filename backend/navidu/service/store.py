import hashlib
import json
import secrets
import shutil
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from navidu.contracts.domain import Decision, FindingSet, verdict_of
from navidu.contracts.errors import AnalysisError
from navidu.contracts.pipeline import AnalysisArtifacts, PipelineEvent
from navidu.contracts.ui_api import DecisionRequest, FindingDiff

from .config import Settings
from .exceptions import LeaseLost, ServiceError


def utc(timestamp: float | None = None) -> datetime:
    return datetime.fromtimestamp(time.time() if timestamp is None else timestamp, timezone.utc)


class Store:
    def __init__(self, settings: Settings):
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.checks_dir.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript((Path(__file__).parent / "migrations" / "0001_init.sql").read_text())
            columns = {row[1] for row in db.execute("PRAGMA table_info(checks)")}
            for name, kind in [("automatic_artifacts_json", "TEXT"), ("cached_from_check_id", "TEXT"),
                               ("analyzed_at", "REAL")]:
                if name not in columns:
                    db.execute(f"ALTER TABLE checks ADD COLUMN {name} {kind}")

    @contextmanager
    def connection(self, *, transaction=False):
        db = sqlite3.connect(self.settings.database_path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=30000")
        try:
            if transaction:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if transaction:
                db.commit()
        except BaseException:
            if transaction:
                db.rollback()
            raise
        finally:
            db.close()

    def get(self, check_id: str):
        with self.connection() as db:
            row = db.execute("SELECT * FROM checks WHERE id=?", (check_id,)).fetchone()
        if row is None:
            raise ServiceError(404, "NOT_FOUND", "Запись не найдена.")
        return dict(row)

    def ip_hash(self, address: str) -> str:
        with self.connection(transaction=True) as db:
            db.execute("INSERT OR IGNORE INTO service_metadata(name,value) VALUES('ip_salt',?)", (secrets.token_hex(32),))
            salt = db.execute("SELECT value FROM service_metadata WHERE name='ip_salt'").fetchone()[0]
        return hashlib.sha256((salt + "|" + address).encode()).hexdigest()

    def enqueue(self, *, check_id, file_name, sha256, priority, client_ip_hash,
                video_file, video_meta, pipeline_version=None, is_example=False):
        with self.connection(transaction=True) as db:
            active = db.execute("SELECT COUNT(*) FROM checks WHERE status IN ('queued','processing')").fetchone()[0]
            if active >= self.settings.queue_max:
                raise ServiceError(429, "QUEUE_FULL", "Очередь заполнена. Повторите через несколько секунд.",
                                   headers={"Retry-After": "10"})
            if priority == "ui":
                ip_active = db.execute(
                    "SELECT COUNT(*) FROM checks WHERE status IN ('queued','processing') AND priority='ui' AND client_ip_hash=?",
                    (client_ip_hash,),
                ).fetchone()[0]
                if ip_active >= self.settings.ui_max_active_per_ip:
                    raise ServiceError(429, "QUEUE_FULL", "С этого адреса уже выполняется проверка. Дождитесь результата.",
                                       headers={"Retry-After": "10"})
            db.execute(
                """INSERT INTO checks(id,file_name,sha256,status,priority,client_ip_hash,created_at,
                is_example,pipeline_version,video_file,video_meta_json) VALUES(?,?,?,'queued',?,?,?,?,?,?,?)""",
                (check_id, file_name, sha256, priority, client_ip_hash, time.time(), int(is_example),
                 pipeline_version or self.settings.pipeline_version, video_file, video_meta.model_dump_json()),
            )
        return self.get(check_id)

    def finding_set(self, check_id: str, revision: int | None = None) -> FindingSet | None:
        with self.connection() as db:
            query = "SELECT json FROM finding_sets WHERE check_id=?"
            args = [check_id]
            if revision is not None:
                query += " AND revision=?"
                args.append(revision)
            row = db.execute(query + " ORDER BY revision DESC LIMIT 1", args).fetchone()
        return FindingSet.model_validate_json(row[0]) if row else None

    def events(self, check_id: str) -> list[PipelineEvent]:
        with self.connection() as db:
            rows = db.execute("SELECT payload_json FROM events WHERE check_id=? ORDER BY seq", (check_id,)).fetchall()
        return [PipelineEvent.model_validate_json(row[0]) for row in rows]

    def append_event(self, check_id: str, token: str, event: PipelineEvent):
        with self.connection(transaction=True) as db:
            self._owns_lease(db, check_id, token)
            seq = db.execute("SELECT COALESCE(MAX(seq),0)+1 FROM events WHERE check_id=?", (check_id,)).fetchone()[0]
            db.execute("INSERT INTO events VALUES(?,?,?,?,?)", (
                check_id, seq, event.phase, event.processed_until_sec, event.model_dump_json()
            ))

    def _owns_lease(self, db, check_id: str, token: str):
        row = db.execute("SELECT status,lease_token,lease_until FROM checks WHERE id=?", (check_id,)).fetchone()
        if not row or row["status"] != "processing" or row["lease_token"] != token or row["lease_until"] < time.time():
            raise LeaseLost(check_id)

    def complete(self, check_id: str, token: str, artifacts: AnalysisArtifacts, *,
                 cached_from_check_id=None, analyzed_at=None):
        if artifacts.run.check_id != check_id:
            raise ValueError("Pipeline returned artifacts for a different check.")
        automatic = artifacts.finding_set.model_copy(deep=True)
        automatic.revision = 1
        automatic.basis = "auto"
        for finding in automatic.findings:
            finding.decision = None
        automatic.verdict = verdict_of(automatic, use_decisions=False)
        destination = (self.settings.checks_dir / check_id).resolve()
        staging = artifacts.work_dir.resolve()
        expected = (destination / ("attempt-" + token)).resolve()
        if staging != expected or expected.parent != destination:
            raise ValueError("Pipeline artifacts are outside the worker's attempt directory.")
        artifacts = artifacts.model_copy(update={"finding_set": automatic, "work_dir": destination})
        with self.connection(transaction=True) as db:
            self._owns_lease(db, check_id, token)
            input_name = db.execute("SELECT video_file FROM checks WHERE id=?", (check_id,)).fetchone()[0]
            # Workers write into distinct attempt directories. A cancelled or expired
            # attempt can never overwrite media from a newer successful attempt.
            files = list(staging.rglob("*"))
            for source in files:
                if source.is_symlink() or not source.resolve().is_relative_to(staging):
                    raise ValueError("Pipeline artifacts contain an unsafe filesystem link.")
            for source in files:
                relative = source.relative_to(staging)
                target = destination / relative
                if not target.resolve().is_relative_to(destination):
                    raise ValueError("Pipeline artifacts target escapes the check directory.")
                if relative == Path(input_name or "video.mov") or relative.parts[0].startswith("attempt-"):
                    raise ValueError("Pipeline artifacts collide with reserved input paths.")
                if source.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
            db.execute("INSERT INTO finding_sets VALUES(?,?,?,?)", (
                check_id, 1, automatic.model_dump_json(), time.time()
            ))
            db.execute("""UPDATE checks SET status='done',finished_at=?,lease_until=NULL,
                lease_token=NULL,error_json=NULL,artifacts_json=?,automatic_artifacts_json=?,
                cached_from_check_id=?,analyzed_at=? WHERE id=?""", (
                time.time(), artifacts.model_dump_json(), artifacts.model_dump_json(),
                cached_from_check_id, analyzed_at if analyzed_at is not None else time.time(), check_id
            ))
        if staging.parent == destination and staging.name == "attempt-" + token:
            shutil.rmtree(staging)

    def cached_artifacts(self, check_id: str, token: str, staging: Path):
        """Clone only immutable automatic facts and evidence, never reviewer context."""
        with self.connection(transaction=True) as db:
            self._owns_lease(db, check_id, token)
            current = db.execute("SELECT sha256,pipeline_version FROM checks WHERE id=?", (check_id,)).fetchone()
            row = db.execute("""SELECT * FROM checks WHERE id<>? AND sha256=? AND pipeline_version=?
                AND status='done' AND automatic_artifacts_json IS NOT NULL
                ORDER BY finished_at DESC LIMIT 1""", (check_id, current["sha256"], current["pipeline_version"])).fetchone()
            if row is None:
                return None
            artifacts = AnalysisArtifacts.model_validate_json(row["automatic_artifacts_json"])
            source_dir = (self.settings.checks_dir / row["id"]).resolve()
            if source_dir.parent != self.settings.checks_dir.resolve():
                return None
            # Copy originals named by the automatic revision and its known serializable
            # intermediate files. Reviewed layout/findings are never used as cache input.
            required = [source_dir / "frames" / f"{frame.frame_id}.jpg"
                        for finding in artifacts.finding_set.findings for frame in finding.evidence]
            if any(not path.is_file() or not path.resolve().is_relative_to(source_dir) for path in required):
                return None
            for source in source_dir.iterdir():
                if source.name == row["video_file"] or source.name.startswith(("attempt-", "layout.r", "findings.r")):
                    continue
                if source.is_symlink():
                    continue
                if source.is_dir() and source.name == "frames":
                    target = staging / "frames"
                    target.mkdir(exist_ok=True)
                    for frame in source.glob("*.jpg"):
                        if frame.is_file() and not frame.is_symlink():
                            shutil.copy2(frame, target / frame.name)
                elif source.is_file():
                    shutil.copy2(source, staging / source.name)
            (staging / "layout.r1.json").write_text(artifacts.layout.model_dump_json(), encoding="utf-8")
            (staging / "findings.r1.json").write_text(artifacts.finding_set.model_dump_json(), encoding="utf-8")
            artifacts.work_dir = staging
            artifacts.run.check_id = check_id
            (staging / "run.json").write_text(artifacts.run.model_dump_json(), encoding="utf-8")
            return artifacts, row["id"], row["analyzed_at"] or row["finished_at"]

    def artifacts(self, check_id: str) -> AnalysisArtifacts:
        row = self.get(check_id)
        if not row["artifacts_json"]:
            raise ServiceError(409, "NOT_DONE", "Результат анализа ещё не готов.")
        return AnalysisArtifacts.model_validate_json(row["artifacts_json"])

    def cancel(self, check_id: str):
        with self.connection(transaction=True) as db:
            row = db.execute("SELECT * FROM checks WHERE id=?", (check_id,)).fetchone()
            if not row:
                raise ServiceError(404, "NOT_FOUND", "Запись не найдена.")
            if row["is_example"]:
                raise ServiceError(409, "READ_ONLY_EXAMPLE", "Пример доступен только для просмотра.")
            if row["status"] not in {"queued", "processing", "cancelled"}:
                raise ServiceError(409, "NOT_ACTIVE", "Проверка уже завершена.")
            db.execute("UPDATE checks SET status='cancelled',finished_at=?,lease_until=NULL,lease_token=NULL WHERE id=?", (
                time.time(), check_id
            ))
        return self.get(check_id)

    def retry(self, check_id: str):
        with self.connection(transaction=True) as db:
            row = db.execute("SELECT * FROM checks WHERE id=?", (check_id,)).fetchone()
            if not row:
                raise ServiceError(404, "NOT_FOUND", "Запись не найдена.")
            if row["is_example"]:
                raise ServiceError(409, "READ_ONLY_EXAMPLE", "Пример доступен только для просмотра.")
            if row["status"] not in {"failed", "cancelled"}:
                raise ServiceError(409, "NOT_FAILED", "Повтор доступен для прерванной или неудачной проверки.")
            if not row["video_file"] or not (self.settings.checks_dir / check_id / row["video_file"]).is_file():
                raise ServiceError(410, "VIDEO_EXPIRED", "Срок хранения исходной записи истёк. Загрузите её снова.")
            active = db.execute("SELECT COUNT(*) FROM checks WHERE status IN ('queued','processing')").fetchone()[0]
            if active >= self.settings.queue_max:
                raise ServiceError(429, "QUEUE_FULL", "Очередь заполнена.", headers={"Retry-After": "10"})
            if row["priority"] == "ui":
                ip_active = db.execute("SELECT COUNT(*) FROM checks WHERE status IN ('queued','processing') AND priority='ui' AND client_ip_hash=?", (row["client_ip_hash"],)).fetchone()[0]
                if ip_active >= self.settings.ui_max_active_per_ip:
                    raise ServiceError(429, "QUEUE_FULL", "С этого адреса уже выполняется проверка.", headers={"Retry-After": "10"})
            db.execute("""UPDATE checks SET status='queued',attempts=0,error_json=NULL,started_at=NULL,
                finished_at=NULL,lease_until=NULL,lease_token=NULL WHERE id=?""", (check_id,))
        return self.get(check_id)

    def save_decision(self, check_id: str, finding_id: str, request: DecisionRequest):
        with self.connection(transaction=True) as db:
            current = self._editable_revision(db, check_id, request.base_revision)
            finding = next((f for f in current.findings if f.finding_id == finding_id), None)
            if not finding:
                raise ServiceError(404, "NOT_FOUND", "Эпизод не найден в текущей ревизии.")
            decision = None if request.decision is None and not request.note else Decision(
                value=request.decision, note=request.note, decided_at=utc()
            )
            finding.decision = decision
            current.revision += 1
            current.basis = "reviewer"
            current.created_at = utc()
            current.verdict = verdict_of(current)
            self._insert_revision(db, check_id, current)
            db.execute("INSERT INTO decisions VALUES(?,?,?,?,?,?)", (
                check_id, finding_id, current.revision, request.decision, request.note, time.time()
            ))
        return decision, current.revision

    def _editable_revision(self, db, check_id, base_revision, *, allow_example=False):
        check = db.execute("SELECT is_example,status FROM checks WHERE id=?", (check_id,)).fetchone()
        if not check:
            raise ServiceError(404, "NOT_FOUND", "Запись не найдена.")
        if check["is_example"] and not allow_example:
            raise ServiceError(409, "READ_ONLY_EXAMPLE", "Изменения в примере можно только предпросмотреть.")
        row = db.execute("SELECT json FROM finding_sets WHERE check_id=? ORDER BY revision DESC LIMIT 1", (check_id,)).fetchone()
        if not row or check["status"] != "done":
            raise ServiceError(409, "NOT_DONE", "Дождитесь окончания анализа.")
        current = FindingSet.model_validate_json(row[0])
        if current.revision != base_revision:
            raise ServiceError(409, "STALE_REVISION", "Результат уже изменён. Обновите запись и повторите решение.",
                               current_revision=current.revision)
        return current

    def _insert_revision(self, db, check_id, finding_set):
        db.execute("INSERT INTO finding_sets VALUES(?,?,?,?)", (
            check_id, finding_set.revision, finding_set.model_dump_json(), time.time()
        ))

    def save_reevaluation(self, check_id, base_revision, updated, *, dry_run, artifacts=None):
        with self.connection(transaction=True) as db:
            current = self._editable_revision(db, check_id, base_revision, allow_example=dry_run)
            updated = updated.model_copy(deep=True)
            previous = {finding.finding_id: finding for finding in current.findings}
            for finding in updated.findings:
                if finding.finding_id in previous:
                    finding.decision = previous[finding.finding_id].decision
            updated.revision = current.revision + 1
            updated.basis = "reviewer"
            updated.created_at = utc()
            updated.verdict = verdict_of(updated)
            following = {finding.finding_id: finding for finding in updated.findings}
            diff = FindingDiff(
                added=sorted(following.keys() - previous.keys()),
                removed=sorted(previous.keys() - following.keys()),
                changed=sorted(key for key in previous.keys() & following.keys()
                               if previous[key].model_dump() != following[key].model_dump()),
                verdict_changed=current.verdict != updated.verdict,
            )
            if not dry_run:
                self._insert_revision(db, check_id, updated)
                if artifacts:
                    artifacts = artifacts.model_copy(update={"finding_set": updated})
                    db.execute("UPDATE checks SET artifacts_json=? WHERE id=?", (artifacts.model_dump_json(), check_id))
        return updated, diff

    def worker_state(self, worker_id, *, model_status, device, limitations, pipeline_version):
        with self.connection() as db:
            db.execute("INSERT OR REPLACE INTO worker_state VALUES(?,?,?,?,?,?)", (
                worker_id, time.time(), pipeline_version, model_status, device, json.dumps(limitations)
            ))

    def health(self):
        with self.connection() as db:
            counts = dict(db.execute("SELECT priority,COUNT(*) FROM checks WHERE status IN ('queued','processing') GROUP BY priority").fetchall())
            workers = [dict(row) for row in db.execute("SELECT * FROM worker_state ORDER BY heartbeat DESC").fetchall()]
        active = [worker for worker in workers if worker["heartbeat"] >= time.time() - self.settings.worker_stale_seconds]
        ready = next((worker for worker in active if worker["model_status"] == "ready"), None)
        latest = ready or (active[0] if active else None)
        return {
            "ready": ready is not None,
            "queue": {"api": counts.get("api", 0), "ui": counts.get("ui", 0)},
            "pipeline_version": latest["pipeline_version"] if latest else self.settings.pipeline_version,
            "model_status": latest["model_status"] if latest else "missing",
            "device": latest["device"] if latest else self.settings.device,
            "limitations": json.loads(latest["limitations_json"]) if latest else ["Обработчик локальной модели не запущен."],
            "example_check_id": None,
        }

    def queue_position(self, check_id):
        with self.connection() as db:
            rows = db.execute("SELECT id FROM checks WHERE status='queued' ORDER BY CASE priority WHEN 'api' THEN 0 ELSE 1 END,created_at,id").fetchall()
        return next((index + 1 for index, row in enumerate(rows) if row[0] == check_id), None)

    def fail_without_lease(self, check_id, error: AnalysisError):
        with self.connection(transaction=True) as db:
            db.execute("UPDATE checks SET status='failed',error_json=?,finished_at=? WHERE id=? AND status='queued'", (
                error.model_dump_json(), time.time(), check_id
            ))
