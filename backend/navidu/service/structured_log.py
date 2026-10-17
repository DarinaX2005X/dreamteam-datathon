"""Single-line diagnostics that never serialize exception messages or locals."""

import json
import logging
import math
import traceback
from pathlib import Path


def error_fields(exc: BaseException, *, code: str):
    fields = {
        "code": code,
        "error_type": type(exc).__name__,
        "trace": [{"file": Path(frame.filename).name, "function": frame.name, "line": frame.lineno}
                  for frame in traceback.extract_tb(exc.__traceback__)],
    }
    metrics = getattr(exc, "numeric_metrics", None)
    if isinstance(metrics, dict):
        allowed = {"response_chars", "json_error_pos", "input_tokens", "image_patches",
                   "generated_tokens", "max_new_tokens", "token_limit_reached",
                   "generation_sec", "vision_sec", "peak_cuda_mib"}
        fields["numeric_metrics"] = {
            key: value for key, value in metrics.items() if key in allowed
            and type(value) in (int, float, bool) and math.isfinite(value)
        }
    return fields


def log_exception(logger, exc, *, code, level=logging.ERROR, **fields):
    logger.log(level, json.dumps({**fields, **error_fields(exc, code=code)}, ensure_ascii=False))


class JsonFormatter(logging.Formatter):
    def format(self, record):
        if record.exc_info and record.exc_info[1] is not None:
            value = error_fields(record.exc_info[1], code="UNHANDLED_ERROR")
        else:
            try:
                value = json.loads(record.getMessage())
            except (ValueError, TypeError):
                value = {"message": record.getMessage()}
            if not isinstance(value, dict):
                value = {"message": value}
        return json.dumps({"level": record.levelname.lower(), "logger": record.name, **value},
                          ensure_ascii=False, separators=(",", ":"))


def configure_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
