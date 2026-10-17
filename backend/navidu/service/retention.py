import shutil
import time

from .store import Store


def cleanup(store: Store, *, now: float | None = None):
    now = time.time() if now is None else now
    settings = store.settings
    base = settings.checks_dir.resolve()
    with store.connection(transaction=True) as db:
        rows = db.execute("SELECT id,video_file,created_at,finished_at,status FROM checks WHERE is_example=0 AND status NOT IN ('queued','processing')").fetchall()
        for row in rows:
            directory = (base / row["id"]).resolve()
            if directory.parent != base or directory.name != row["id"]:
                continue
            completed = row["finished_at"] or row["created_at"]
            if completed + settings.retain_results_days * 86400 < now:
                if directory.exists():
                    # Resolve the exact target and reject symlinks before recursive removal.
                    if directory.is_symlink() or directory.parent != base:
                        continue
                    shutil.rmtree(directory)
                db.execute("DELETE FROM checks WHERE id=?", (row["id"],))
                db.execute("INSERT INTO retention_log(check_id,resource,removed_at) VALUES(?,?,?)", (row["id"], "results", now))
            elif row["video_file"] and completed + settings.retain_video_hours * 3600 < now:
                video = (directory / row["video_file"]).resolve()
                if video.parent == directory:
                    video.unlink(missing_ok=True)
                    db.execute("UPDATE checks SET video_file=NULL WHERE id=?", (row["id"],))
                    db.execute("INSERT INTO retention_log(check_id,resource,removed_at) VALUES(?,?,?)", (row["id"], "video", now))
