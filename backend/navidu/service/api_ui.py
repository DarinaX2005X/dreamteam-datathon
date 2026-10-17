import mimetypes

from fastapi import APIRouter, Query, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from navidu.contracts.domain import RoleOverride, SceneLayout, TrackRole
from navidu.contracts.rules import RulesResponse
from navidu.contracts.ui_api import (
    CheckDetailResponse,
    CheckSummaryResponse,
    DecisionRequest,
    DecisionResponse,
    ExampleResponse,
    HealthResponse,
    ReevaluateRequest,
    ReevaluateResponse,
)

from .api_v1 import accept_upload
from .catalog import read_catalog
from .exceptions import ServiceError
from .presenters import check_directory, detail, evidence_path, summary
from .readiness import PREPARING_MESSAGE, published_analysis_valid
from .reevaluation import staged_frames, staged_reevaluation

router = APIRouter(prefix="/api/ui")


@router.post("/checks", response_model=CheckSummaryResponse, status_code=201)
async def create_check(request: Request, file: UploadFile | None = None):
    row = await accept_upload(request, file, priority="ui")
    return CheckSummaryResponse(check=summary(request.app.state.store, row))


@router.get("/checks/{check_id}", response_model=CheckDetailResponse)
def get_check(check_id: str, request: Request):
    store = request.app.state.store
    return CheckDetailResponse(check=detail(store, store.get(check_id)))


@router.get("/checks/{check_id}/video")
def get_video(check_id: str, request: Request):
    store = request.app.state.store
    directory = check_directory(store, check_id)
    row = store.get(check_id)
    path = (directory / (row["video_file"] or "expired")).resolve()
    if path.parent != directory or not row["video_file"] or not path.is_file():
        raise ServiceError(410, "VIDEO_EXPIRED", "Срок хранения исходной записи истёк.")
    return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "video/mp4",
                        headers={"Cache-Control": "private, max-age=3600"})


@router.get("/checks/{check_id}/frames/{frame_id}.jpg")
def get_frame(check_id: str, frame_id: str, request: Request,
              overlay: int = Query(default=1, ge=0, le=1),
              revision: int | None = Query(default=None, ge=1)):
    path = evidence_path(request.app.state.store, check_id, frame_id, bool(overlay), revision=revision)
    cache = "private, max-age=3600" if revision is not None else "private, no-cache"
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": cache})


@router.put("/checks/{check_id}/findings/{finding_id}/decision", response_model=DecisionResponse)
def put_decision(check_id: str, finding_id: str, body: DecisionRequest, request: Request):
    decision, revision = request.app.state.store.save_decision(check_id, finding_id, body)
    return DecisionResponse(decision=decision, revision=revision)


@router.post("/checks/{check_id}/reevaluate", response_model=ReevaluateResponse)
async def reevaluate(check_id: str, body: ReevaluateRequest, request: Request):
    store = request.app.state.store
    # Check revision and read-only status before potentially expensive recomputation.
    with store.connection() as db:
        base = store._editable_revision(db, check_id, body.base_revision, allow_example=body.dry_run)
    artifacts = store.artifacts(check_id)
    track_ids = {track.track_id for track in artifacts.tracks if track.kind == "person"}
    if any(override.track_id not in track_ids for override in body.role_overrides):
        raise ServiceError(422, "INVALID_INPUT", "Для выбранной роли не найден трек человека.")
    if len({override.track_id for override in body.role_overrides}) != len(body.role_overrides):
        raise ServiceError(422, "INVALID_INPUT", "Роль одного человека указана несколько раз.")
    if len({zone.zone_id for zone in body.zone_overrides}) != len(body.zone_overrides):
        raise ServiceError(422, "INVALID_INPUT", "Зона указана несколько раз.")
    adapter = request.app.state.pipeline
    if adapter is None:
        raise ServiceError(503, "NOT_READY", "Пересчёт правил пока недоступен.")
    roles = {track.track_id: track.role.value for track in artifacts.tracks
             if track.role is not None and track.role.source == "reviewer"}
    roles.update({override.track_id: override.role for override in base.role_overrides})
    roles.update({override.track_id: override.role for override in body.role_overrides})
    effective_roles = [RoleOverride(track_id=track_id, role=role) for track_id, role in roles.items()]
    with staged_reevaluation(store, check_id, artifacts) as staged:
        updated = await run_in_threadpool(
            adapter.reevaluate, staged, role_overrides=effective_roles,
            zone_overrides=body.zone_overrides, base=base,
        )
        zones = {zone.zone_id: zone for zone in staged.layout.zones}
        zones.update({zone.zone_id: zone.model_copy(update={"source": "reviewer"}) for zone in body.zone_overrides})
        staged.layout = SceneLayout(revision=updated.layout_revision, zones=list(zones.values()),
                                    camera_note=staged.layout.camera_note)
        for track in staged.tracks:
            if track.track_id in roles:
                track.role = TrackRole(value=roles[track.track_id], source="reviewer")
        preview = staged_frames(staged, updated, preview=body.dry_run)
        updated, diff = store.save_reevaluation(check_id, body.base_revision, updated,
                                               dry_run=body.dry_run, artifacts=staged)
        return ReevaluateResponse(finding_set=updated, diff=diff, preview_frames=preview)


@router.post("/checks/{check_id}/retry", response_model=CheckSummaryResponse)
def retry(check_id: str, request: Request):
    store = request.app.state.store
    if not store.health()["ready"]:
        raise ServiceError(503, "NOT_READY", PREPARING_MESSAGE)
    return CheckSummaryResponse(check=summary(store, store.retry(check_id)))


@router.post("/checks/{check_id}/cancel", response_model=CheckSummaryResponse)
def cancel(check_id: str, request: Request):
    store = request.app.state.store
    return CheckSummaryResponse(check=summary(store, store.cancel(check_id)))


@router.get("/rules", response_model=RulesResponse)
def rules(request: Request):
    return read_catalog(request.app.state.store.settings.rules_dir)


def example_id(store):
    check_id = store.settings.example_check_id
    if not check_id:
        return None
    try:
        row = store.get(check_id)
        if row["is_example"] and row["status"] == "done" and published_analysis_valid(row):
            return check_id
    except ServiceError:
        pass
    return None


@router.get("/example", response_model=ExampleResponse)
def example(request: Request):
    check_id = example_id(request.app.state.store)
    if check_id is None:
        raise ServiceError(404, "NOT_FOUND", "Готовый пример пока не добавлен.")
    return ExampleResponse(check_id=check_id)


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    store = request.app.state.store
    value = store.health()
    value["example_check_id"] = example_id(store)
    return HealthResponse.model_validate(value)
