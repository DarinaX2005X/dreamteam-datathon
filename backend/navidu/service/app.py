import json
import logging
import secrets
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException

from navidu.contracts.errors import ErrorInfo, ErrorResponse, FieldIssue

from .adapter import LocalPipelineAdapter
from .api_ui import router as ui_router
from .api_v1 import router as jury_router
from .auth import require_key
from .config import Settings
from .exceptions import ServiceError
from .retention import cleanup
from .store import Store
from .structured_log import log_exception
from .uploads import probe_video

logger = logging.getLogger("lado.http")


def error_response(error: ServiceError, request_id: str):
    value = ErrorResponse(error=ErrorInfo(
        code=error.code, message=error.message, request_id=request_id,
        current_revision=error.current_revision, field_issues=error.field_issues,
    ))
    return JSONResponse(value.model_dump(mode="json", exclude_none=True), status_code=error.status,
                        headers={"X-Request-ID": request_id, **error.headers})


class RequestGuard:
    def __init__(self, app, settings):
        self.app = app
        self.settings = settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request_id = "r-" + secrets.token_hex(6)
        scope.setdefault("state", {})["request_id"] = request_id
        headers = dict(scope.get("headers", []))
        started = time.monotonic()
        response_started = False
        status = 500
        limit = self.settings.max_upload_bytes + 1024 * 1024
        consumed = 0

        async def limited_receive():
            nonlocal consumed
            message = await receive()
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > limit:
                    raise ServiceError(413, "FILE_TOO_LARGE", f"Файл больше {self.settings.max_upload_mb:g} МБ.")
            return message

        async def wrapped_send(message):
            nonlocal response_started, status
            if message["type"] == "http.response.start":
                response_started = True
                status = message["status"]
                message["headers"] = [item for item in message.get("headers", [])
                                      if item[0].lower() not in {b"x-request-id", b"x-content-type-options"}] + [
                    (b"x-request-id", request_id.encode()), (b"x-content-type-options", b"nosniff")
                ]
            await send(message)

        try:
            if scope["path"].startswith("/api/v1/"):
                require_key(headers.get(b"x-api-key", b"").decode("utf-8", errors="replace"), self.settings.api_keys)
            if scope["method"] in {"POST", "PUT", "PATCH"}:
                try:
                    length = int(headers.get(b"content-length", b"0"))
                except ValueError:
                    raise ServiceError(400, "INVALID_INPUT", "Некорректная длина запроса.") from None
                if length > limit:
                    raise ServiceError(413, "FILE_TOO_LARGE", f"Файл больше {self.settings.max_upload_mb:g} МБ.")
            await self.app(scope, limited_receive, wrapped_send)
        except ServiceError as exc:
            if response_started:
                raise
            await error_response(exc, request_id)(scope, receive, wrapped_send)
        finally:
            logger.info(json.dumps({"request_id": request_id, "method": scope["method"],
                                    "path": scope["path"], "status": status,
                                    "duration_ms": round((time.monotonic() - started) * 1000)}))


def create_app(settings: Settings | None = None, *, pipeline=None, probe=probe_video) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.validate()
    store = Store(settings)

    @asynccontextmanager
    async def lifespan(app):
        stopped = threading.Event()

        def maintain_storage():
            while not stopped.is_set():
                try:
                    cleanup(store)
                except Exception as exc:
                    log_exception(logger, exc, code="RETENTION_FAILED", phase="retention")
                stopped.wait(300)

        maintenance = threading.Thread(target=maintain_storage, name="retention", daemon=True)
        maintenance.start()
        try:
            yield
        finally:
            stopped.set()
            maintenance.join(timeout=2)

    app = FastAPI(title="Lado local video analysis", version="0.1.0", lifespan=lifespan,
                  docs_url=None if settings.production else "/docs", redoc_url=None)
    app.state.store = store
    app.state.pipeline = pipeline or LocalPipelineAdapter(settings)
    app.state.probe = probe
    app.add_middleware(RequestGuard, settings=settings)

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        return error_response(exc, request.state.request_id)

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request: Request, exc: RequestValidationError):
        fields = [FieldIssue(path=".".join(str(piece) for piece in issue["loc"]),
                             message="Проверьте значение поля.") for issue in exc.errors()]
        return error_response(ServiceError(422, "INVALID_INPUT", "Некоторые поля заполнены неверно.",
                                            field_issues=fields), request.state.request_id)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        code = "NOT_FOUND" if exc.status_code == 404 else "INVALID_INPUT"
        message = "Адрес не найден." if exc.status_code == 404 else "Запрос не может быть выполнен."
        return error_response(ServiceError(exc.status_code, code, message), request.state.request_id)

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception):
        log_exception(logger, exc, code="INTERNAL", request_id=request.state.request_id)
        return error_response(ServiceError(500, "INTERNAL", "Не удалось выполнить действие. Повторите попытку."),
                              request.state.request_id)

    app.include_router(jury_router)
    app.include_router(ui_router)
    @app.get("/assets/{asset_path:path}")
    def asset(asset_path: str):
        root = (settings.web_dist / "assets").resolve()
        path = (root / asset_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ServiceError(404, "NOT_FOUND", "Файл интерфейса не найден.")
        return FileResponse(path, headers={"Cache-Control": "public, max-age=31536000, immutable"})

    @app.get("/favicon.svg")
    def favicon():
        path = settings.web_dist / "favicon.svg"
        if not path.is_file():
            raise ServiceError(404, "NOT_FOUND", "Значок не найден.")
        return FileResponse(path, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=3600"})

    @app.get("/presentation/{filename}")
    def presentation(filename: str):
        public_files = {
            "lado-pitch.pdf": "application/pdf",
            "lado-pitch.pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        }
        if filename not in public_files:
            raise ServiceError(404, "NOT_FOUND", "Презентация не найдена.")
        path = settings.web_dist / "presentation" / filename
        if not path.is_file():
            raise ServiceError(404, "NOT_FOUND", "Презентация не найдена.")
        return FileResponse(path, media_type=public_files[filename], filename=filename,
            content_disposition_type="inline" if filename.endswith(".pdf") else "attachment",
            headers={"Cache-Control": "public, max-age=0, must-revalidate"})

    @app.get("/")
    @app.get("/app")
    @app.get("/app/{path:path}")
    def frontend(path: str = ""):
        index = settings.web_dist / "index.html"
        if not index.is_file():
            raise ServiceError(503, "NOT_READY", "Интерфейс ещё не собран. Выполните npm run build.")
        return FileResponse(index, media_type="text/html", headers={"Cache-Control": "no-cache"})

    return app
