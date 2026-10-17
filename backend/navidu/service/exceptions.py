from starlette.exceptions import HTTPException


class ServiceError(HTTPException):
    def __init__(self, status: int, code: str, message: str, *, current_revision=None,
                 field_issues=None, headers=None):
        super().__init__(status_code=status, detail=message, headers=headers or {})
        self.status = status
        self.code = code
        self.message = message
        self.current_revision = current_revision
        self.field_issues = field_issues
        self.headers = headers or {}


class LeaseLost(Exception):
    """A worker may no longer publish results for this job."""
