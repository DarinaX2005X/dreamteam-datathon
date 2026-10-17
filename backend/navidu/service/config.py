import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(int(default))).lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    rules_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "rules")
    web_dist: Path = field(default_factory=lambda: PROJECT_ROOT / "web" / "dist")
    api_keys: tuple[str, ...] = ()
    production: bool = False
    max_upload_mb: float = 300
    max_duration_sec: float = 600
    ui_max_active_per_ip: int = 2
    queue_max: int = 20
    retain_video_hours: float = 24
    retain_results_days: float = 7
    evidence_overlay: bool = True
    example_check_id: str | None = None
    device: str = "auto"
    analysis_fps: float = 3
    pipeline_version: str = "0.1.0"
    experimental_analysis: bool = False
    ffprobe: str = "ffprobe"
    lease_seconds: float = 90
    heartbeat_seconds: float = 30
    worker_poll_seconds: float = 1
    worker_stale_seconds: float = 120
    max_attempts: int = 3

    @property
    def checks_dir(self) -> Path:
        return self.data_dir / "checks"

    @property
    def database_path(self) -> Path:
        return self.data_dir / "navidu.sqlite3"

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @classmethod
    def from_env(cls) -> "Settings":
        result = cls(
            data_dir=Path(os.getenv("NAVIDU_DATA_DIR", str(PROJECT_ROOT / "data"))).resolve(),
            models_dir=Path(os.getenv("NAVIDU_MODELS_DIR", str(PROJECT_ROOT / "models"))).resolve(),
            rules_dir=Path(os.getenv("NAVIDU_RULES_DIR", str(PROJECT_ROOT / "rules"))).resolve(),
            web_dist=Path(os.getenv("NAVIDU_WEB_DIST", str(PROJECT_ROOT / "web" / "dist"))).resolve(),
            api_keys=tuple(k.strip() for k in os.getenv("NAVIDU_API_KEYS", "").split(",") if k.strip()),
            production=os.getenv("NAVIDU_ENV", "development") == "production",
            max_upload_mb=float(os.getenv("NAVIDU_MAX_UPLOAD_MB", "300")),
            max_duration_sec=float(os.getenv("NAVIDU_MAX_DURATION_SEC", "600")),
            ui_max_active_per_ip=int(os.getenv("NAVIDU_UI_MAX_ACTIVE_PER_IP", "2")),
            queue_max=int(os.getenv("NAVIDU_QUEUE_MAX", "20")),
            retain_video_hours=float(os.getenv("NAVIDU_RETAIN_VIDEO_HOURS", "24")),
            retain_results_days=float(os.getenv("NAVIDU_RETAIN_RESULTS_DAYS", "7")),
            evidence_overlay=_bool("NAVIDU_EVIDENCE_OVERLAY", True),
            example_check_id=os.getenv("NAVIDU_EXAMPLE_CHECK_ID") or None,
            device=os.getenv("NAVIDU_DEVICE", "auto"),
            analysis_fps=float(os.getenv("NAVIDU_ANALYSIS_FPS", "3")),
            pipeline_version=os.getenv("NAVIDU_PIPELINE_VERSION", "0.1.0"),
            experimental_analysis=_bool("NAVIDU_EXPERIMENTAL_ANALYSIS", False),
            ffprobe=os.getenv("NAVIDU_FFPROBE", "ffprobe"),
        )
        result.validate()
        return result

    def validate(self):
        if self.production and not self.api_keys:
            raise RuntimeError("NAVIDU_API_KEYS обязателен в production-режиме.")
        if self.device not in {"auto", "cuda", "cpu"}:
            raise RuntimeError("NAVIDU_DEVICE должен быть auto, cuda или cpu.")
        for name in ["max_upload_mb", "max_duration_sec", "queue_max", "ui_max_active_per_ip",
                     "lease_seconds", "heartbeat_seconds", "max_attempts"]:
            if getattr(self, name) <= 0:
                raise RuntimeError(f"Параметр {name} должен быть положительным.")

