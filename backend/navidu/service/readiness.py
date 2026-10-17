"""Publication requires a validated action analysis, not merely loaded weights."""

import json

PREPARING_MESSAGE = "Подготавливаем проверку действий. Загрузка станет доступна после проверки модели."
NOT_PERFORMED_MESSAGE = "Полная проверка действий не выполнена. Запустите повтор после готовности модели."
INCOMPLETE_MESSAGE = "Для этой записи не удалось проверить условия правил. Итог о наличии нарушений не сформирован."


def analysis_ready(status):
    return (status.get("model_status") == "ready" and status.get("analysis_ready") is True
            and status.get("analysis_mode") in {"validated", "experimental"})


def validated_run(run):
    return run.analysis_ready is True and run.analysis_mode == "validated"


def usable_run(run):
    return run.analysis_ready is True and run.analysis_mode in {"validated", "experimental"}


def usable_analysis_result(row):
    raw = row.get("automatic_artifacts_json") or row.get("artifacts_json")
    if not raw:
        return False
    try:
        run = json.loads(raw).get("run", {})
    except (ValueError, TypeError):
        return False
    return run.get("analysis_ready") is True and run.get("analysis_mode") in {"validated", "experimental"}


def published_analysis_valid(row):
    raw = row.get("automatic_artifacts_json") or row.get("artifacts_json")
    if not raw:
        return False
    try:
        run = json.loads(raw).get("run", {})
    except (ValueError, TypeError):
        return False
    return run.get("analysis_ready") is True and run.get("analysis_mode") == "validated"


def explicit_diagnostic(raw):
    try:
        run = json.loads(raw or "{}").get("run", {})
    except (ValueError, TypeError):
        return False
    if run.get("analysis_mode") == "diagnostic":
        return True
    # The previous local diagnostic worker retained only this detector model ID.
    # Do not infer diagnostic status from an empty model list or an unknown model.
    return ("analysis_mode" not in run
            and set(run.get("model_versions", {})) == {"rfdetr-nano-onnx"})
