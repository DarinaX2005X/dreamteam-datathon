import secrets
import time

from navidu.contracts.errors import AnalysisError

from .exceptions import LeaseLost
from .store import Store


class JobQueue:
    def __init__(self, store: Store):
        self.store = store
        self.settings = store.settings

    def recover_expired(self, db, now):
        rows = db.execute("SELECT id,attempts FROM checks WHERE status='processing' AND lease_until<?", (now,)).fetchall()
        for row in rows:
            if row["attempts"] >= self.settings.max_attempts:
                error = AnalysisError(code="RETRIES_EXHAUSTED", message="Обработка несколько раз прервалась. Можно повторить проверку.", failed_phase="worker")
                db.execute("UPDATE checks SET status='failed',error_json=?,finished_at=?,lease_token=NULL,lease_until=NULL WHERE id=?", (
                    error.model_dump_json(), now, row["id"]
                ))
            else:
                db.execute("UPDATE checks SET status='queued',lease_token=NULL,lease_until=NULL WHERE id=?", (row["id"],))

    def acquire(self):
        with self.store.connection(transaction=True) as db:
            now = time.time()
            self.recover_expired(db, now)
            row = db.execute("SELECT * FROM checks WHERE status='queued' ORDER BY CASE priority WHEN 'api' THEN 0 ELSE 1 END,created_at,id LIMIT 1").fetchone()
            if not row:
                return None
            token = secrets.token_hex(16)
            db.execute("UPDATE checks SET status='processing',started_at=?,lease_until=?,lease_token=?,attempts=attempts+1,error_json=NULL WHERE id=?", (
                now, now + self.settings.lease_seconds, token, row["id"]
            ))
            db.execute("DELETE FROM events WHERE check_id=?", (row["id"],))
            result = dict(row)
            result.update(status="processing", lease_token=token, attempts=row["attempts"] + 1)
            return result

    def renew(self, check_id, token):
        with self.store.connection(transaction=True) as db:
            self.store._owns_lease(db, check_id, token)
            db.execute("UPDATE checks SET lease_until=? WHERE id=? AND lease_token=?", (
                time.time() + self.settings.lease_seconds, check_id, token
            ))

    def owns(self, check_id, token):
        try:
            with self.store.connection() as db:
                self.store._owns_lease(db, check_id, token)
            return True
        except LeaseLost:
            return False

    def fail(self, check_id, token, error: AnalysisError, *, retryable=True):
        with self.store.connection(transaction=True) as db:
            self.store._owns_lease(db, check_id, token)
            row = db.execute("SELECT attempts FROM checks WHERE id=?", (check_id,)).fetchone()
            retry = retryable and row[0] < self.settings.max_attempts
            if retryable and not retry:
                error = error.model_copy(update={"code": "RETRIES_EXHAUSTED", "message": "Не удалось завершить анализ после трёх попыток. Можно повторить проверку."})
            db.execute("UPDATE checks SET status=?,error_json=?,finished_at=?,lease_until=NULL,lease_token=NULL WHERE id=?", (
                "queued" if retry else "failed", error.model_dump_json(), None if retry else time.time(), check_id
            ))

    def release(self, check_id, token):
        with self.store.connection(transaction=True) as db:
            db.execute("UPDATE checks SET status='queued',lease_until=NULL,lease_token=NULL WHERE id=? AND lease_token=? AND status='processing'", (check_id, token))
