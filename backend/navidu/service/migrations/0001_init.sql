CREATE TABLE IF NOT EXISTS checks (
    id TEXT PRIMARY KEY,
    file_name TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('queued','processing','done','failed','cancelled')),
    priority TEXT NOT NULL CHECK(priority IN ('api','ui')),
    client_ip_hash TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    lease_until REAL,
    lease_token TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    error_json TEXT,
    is_example INTEGER NOT NULL DEFAULT 0,
    pipeline_version TEXT NOT NULL,
    video_file TEXT,
    video_meta_json TEXT NOT NULL,
    artifacts_json TEXT,
    automatic_artifacts_json TEXT,
    cached_from_check_id TEXT,
    analyzed_at REAL
);
CREATE INDEX IF NOT EXISTS checks_queue ON checks(status, priority, created_at);
CREATE INDEX IF NOT EXISTS checks_cache ON checks(sha256, pipeline_version, status);
CREATE TABLE IF NOT EXISTS events (
    check_id TEXT NOT NULL REFERENCES checks(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    phase TEXT NOT NULL,
    processed_until_sec REAL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY(check_id, seq)
);
CREATE TABLE IF NOT EXISTS finding_sets (
    check_id TEXT NOT NULL REFERENCES checks(id) ON DELETE CASCADE,
    revision INTEGER NOT NULL,
    json TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY(check_id, revision)
);
CREATE TABLE IF NOT EXISTS decisions (
    check_id TEXT NOT NULL REFERENCES checks(id) ON DELETE CASCADE,
    finding_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    value TEXT,
    note TEXT NOT NULL,
    decided_at REAL NOT NULL,
    PRIMARY KEY(check_id, finding_id, revision)
);
CREATE TABLE IF NOT EXISTS worker_state (
    worker_id TEXT PRIMARY KEY,
    heartbeat REAL NOT NULL,
    pipeline_version TEXT NOT NULL,
    model_status TEXT NOT NULL,
    device TEXT NOT NULL,
    limitations_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS retention_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    check_id TEXT NOT NULL,
    resource TEXT NOT NULL,
    removed_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS service_metadata (
    name TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
