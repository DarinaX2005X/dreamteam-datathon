"""Isolate rule previews from immutable automatic evidence and durable revisions."""

import base64
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path

from navidu.contracts.ui_api import PreviewFrame

from .evidence import artifact_frame_path
from .exceptions import ServiceError
from .presenters import check_directory


@contextmanager
def staged_reevaluation(store, check_id, artifacts):
    directory = check_directory(store, check_id)
    row = store.get(check_id)
    source = (directory / (row["video_file"] or "expired")).resolve()
    if not row["video_file"] or source.parent != directory or not source.is_file():
        raise ServiceError(410, "VIDEO_EXPIRED", "Для пересчёта нужна исходная запись. Загрузите её снова.")
    manifest_path = directory / "engine_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ServiceError(409, "NOT_READY", "Сохранённые данные не поддерживают пересчёт правил.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(source_path=str(source), rules_dir=str(store.settings.rules_dir.resolve()))
    with tempfile.TemporaryDirectory(prefix=".reevaluate-", dir=directory) as temporary:
        staging = Path(temporary)
        (staging / ".reevaluation-staging").touch()
        (staging / "engine_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        isolated = artifacts.model_copy(deep=True)
        isolated.work_dir = staging
        yield isolated


def staged_frames(artifacts, finding_set, *, preview):
    result = {}
    frame_ids = {frame.frame_id for finding in finding_set.findings for frame in finding.evidence}
    for frame_id in sorted(frame_ids):
        original = artifact_frame_path(artifacts.work_dir, frame_id, False)
        overlay = artifact_frame_path(artifacts.work_dir, frame_id, True, fallback=False)
        if original is None:
            raise ValueError("Reevaluation did not materialize a referenced evidence frame.")
        if preview:
            result[frame_id] = PreviewFrame(
                image_base64=base64.b64encode(original.read_bytes()).decode("ascii"),
                overlay_base64=base64.b64encode(overlay.read_bytes()).decode("ascii") if overlay else None,
            )
    return result
