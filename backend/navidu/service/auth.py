import hmac

from .exceptions import ServiceError


def require_key(provided: str | None, configured: tuple[str, ...]):
    candidate = (provided or "").encode("utf-8")
    valid = False
    for key in configured:
        # Evaluate every configured key; never echo either value to a log or response.
        valid |= hmac.compare_digest(candidate, key.encode("utf-8"))
    if not valid:
        raise ServiceError(401, "UNAUTHORIZED", "Нужен действующий ключ X-API-Key.")
