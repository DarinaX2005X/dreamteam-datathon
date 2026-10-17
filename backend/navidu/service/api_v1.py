from fastapi import APIRouter, Request, Response, UploadFile

from navidu.contracts.api_v1 import JobProcessing, JobResponse

from .exceptions import ServiceError
from .presenters import jury_response
from .readiness import PREPARING_MESSAGE
from .uploads import persist_upload

router = APIRouter(prefix="/api/v1")


async def accept_upload(request: Request, file: UploadFile | None, *, priority: str):
    store = request.app.state.store
    health = store.health()
    if not health["ready"]:
        raise ServiceError(503, "NOT_READY", PREPARING_MESSAGE)
    api_ready = health["analysis_mode"] == "validated" or (
        store.settings.allow_experimental_api and health["analysis_mode"] == "experimental"
    )
    if priority == "api" and not api_ready:
        raise ServiceError(503, "NOT_READY", PREPARING_MESSAGE)
    check_id, name, path, checksum, meta = await persist_upload(
        file, store.settings, request.app.state.probe
    )
    try:
        row = store.enqueue(
            check_id=check_id, file_name=name, sha256=checksum, priority=priority,
            client_ip_hash=store.ip_hash(request.client.host if request.client else "local") if priority == "ui" else None,
            video_file=path.name, video_meta=meta, pipeline_version=health["pipeline_version"],
        )
        return row
    except BaseException:
        path.unlink(missing_ok=True)
        path.parent.rmdir()
        raise


@router.post("/analyze", response_model=JobProcessing)
async def analyze(request: Request, response: Response, file: UploadFile | None = None):
    row = await accept_upload(request, file, priority="api")
    response.headers["X-Analysis-Mode"] = request.app.state.store.health()["analysis_mode"]
    return JobProcessing(job_id=row["id"])


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, request: Request, response: Response):
    store = request.app.state.store
    row = store.get(job_id)
    response.headers["X-Analysis-Mode"] = row["analysis_mode"] or store.health()["analysis_mode"]
    return jury_response(store, row)
