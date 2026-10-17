from fastapi import APIRouter, Request, UploadFile

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
    if priority == "api" and health["analysis_mode"] != "validated":
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
async def analyze(request: Request, file: UploadFile | None = None):
    row = await accept_upload(request, file, priority="api")
    return JobProcessing(job_id=row["id"])


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, request: Request):
    store = request.app.state.store
    return jury_response(store, store.get(job_id))
