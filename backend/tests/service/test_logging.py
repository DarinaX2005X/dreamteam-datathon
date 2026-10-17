import json
import logging
import sys

from navidu.service.structured_log import JsonFormatter, log_exception


def test_exception_event_omits_private_message_and_locals(caplog):
    secret = "private-frame-data-and-request-key"
    logger = logging.getLogger("lado.test")
    try:
        raise RuntimeError(secret)
    except RuntimeError as exc:
        log_exception(logger, exc, code="ANALYSIS_FAILED", check_id="000000000001", phase="rules")
    record = caplog.records[-1]
    event = json.loads(record.message)
    assert event["code"] == "ANALYSIS_FAILED" and event["error_type"] == "RuntimeError"
    assert event["check_id"] == "000000000001" and event["phase"] == "rules"
    assert len(event["trace"]) == 1
    frame = event["trace"][0]
    assert frame["file"] == "test_logging.py"
    assert frame["function"] == "test_exception_event_omits_private_message_and_locals"
    assert isinstance(frame["line"], int) and frame["line"] > 0
    assert secret not in record.message and record.exc_info is None
    assert "\n" not in record.message


def test_server_exception_formatter_also_redacts_message_and_context():
    secret = "private-upload-fragment"
    try:
        raise ValueError(secret)
    except ValueError:
        record = logging.LogRecord("uvicorn.error", logging.ERROR, __file__, 32,
                                   "Failed with payload: %s", (secret,), sys.exc_info())
    rendered = JsonFormatter().format(record)
    event = json.loads(rendered)
    assert event["code"] == "UNHANDLED_ERROR" and event["error_type"] == "ValueError"
    assert event["logger"] == "uvicorn.error"
    assert event["trace"][0]["file"] == "test_logging.py"
    assert secret not in rendered and "payload" not in rendered and "\n" not in rendered
