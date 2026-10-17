import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import type {
  CheckSummary,
  HealthResponse,
} from "../../../../contracts/ts/types.gen";
import product from "../../../../contracts/product.json";
import { ApiError, api, uploadCheck } from "../../api/client";
import { Button, InlineNotice } from "../../ui";
import { errors } from "../../i18n/ru";
import { readRecentIds, recentStatus } from "./domain";
import { intakeStrings as strings } from "./strings";

const RECENT = "lado.recentChecks";

export function IntakePage() {
  const navigate = useNavigate();
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File>();
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState<number | null>(null);
  const [uploadController, setUploadController] = useState<AbortController>();
  const [recent, setRecent] = useState<CheckSummary[]>([]);
  const [health, setHealth] = useState<HealthResponse>();
  const [healthError, setHealthError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const loadHealth = async () => {
      try {
        const next = await api.health(controller.signal);
        setHealth(next);
        setHealthError("");
        if (next.model_status === "loading")
          timer = setTimeout(loadHealth, 2000);
      } catch (requestError) {
        if ((requestError as Error).name !== "AbortError") {
          setHealthError(
            "Не удалось проверить готовность сервиса. Повторяем запрос.",
          );
          timer = setTimeout(loadHealth, 5000);
        }
      }
    };
    void loadHealth();
    const ids = readRecentIds(localStorage.getItem(RECENT));
    Promise.allSettled(ids.map((id) => api.recent(id, controller.signal))).then(
      (results) => {
        if (controller.signal.aborted) return;
        const available = results.flatMap((result) =>
          result.status === "fulfilled" ? [result.value] : [],
        );
        setRecent(available);
        localStorage.setItem(
          RECENT,
          JSON.stringify(available.map((item) => item.id)),
        );
      },
    );
    return () => {
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, []);

  const pick = (next?: File) => {
    setError("");
    if (!next) return;
    const extension = next.name.split(".").pop()?.toLowerCase();
    if (!["mov", "mp4", "m4v"].includes(extension ?? "")) {
      setError(errors.UNSUPPORTED_MEDIA);
      return;
    }
    if (next.size > 300 * 1024 * 1024) {
      setError(errors.FILE_TOO_LARGE);
      return;
    }
    setFile(next);
  };

  const submit = async () => {
    if (!file || uploadController) return;
    const controller = new AbortController();
    setUploadController(controller);
    setProgress(0);
    try {
      const check = await uploadCheck(
        file,
        (loaded, total) => setProgress(total ? loaded / total : 0),
        controller.signal,
      );
      const ids = [
        check.id,
        ...readRecentIds(localStorage.getItem(RECENT)),
      ].slice(0, 8);
      localStorage.setItem(RECENT, JSON.stringify([...new Set(ids)]));
      navigate(`/app/checks/${check.id}`);
    } catch (requestError) {
      if ((requestError as Error).name !== "AbortError") {
        const apiError = requestError as ApiError;
        setError(errors[apiError.code] ?? apiError.message);
        setFile(undefined);
        if (input.current) {
          input.current.value = "";
          input.current.focus();
        }
      }
    } finally {
      setUploadController(undefined);
      setProgress(null);
    }
  };

  return (
    <div className="page intake-page">
      <header className="page-heading">
        <p className="eyebrow">Рабочее место проверяющего</p>
        <h1>{strings.title}</h1>
        <p>
          Загрузите запись с потолочной камеры. {product.name} отметит
          наблюдаемые эпизоды и свяжет их с нормами.
        </p>
      </header>
      {health && health.model_status !== "ready" && (
        <InlineNotice
          tone={health.model_status === "loading" ? "warning" : "danger"}
        >
          {health.model_status === "loading"
            ? "Модель загружается. Начать проверку можно после её готовности."
            : "Анализ пока недоступен."}{" "}
          {(health.limitations ?? []).join(" ")}
        </InlineNotice>
      )}
      {healthError && <InlineNotice tone="warning">{healthError}</InlineNotice>}
      <section
        className={`dropzone ${dragging ? "is-dragging" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          pick(event.dataTransfer.files[0]);
        }}
      >
        <input
          ref={input}
          type="file"
          accept=".mov,.mp4,.m4v,video/quicktime,video/mp4"
          onChange={(event) => pick(event.target.files?.[0])}
        />
        <div className="dropzone__symbol" aria-hidden="true">
          ↥
        </div>
        <h2>{file ? file.name : "Запись кассы"}</h2>
        <p>
          {file
            ? `${(file.size / 1024 / 1024).toLocaleString("ru", { maximumFractionDigits: 1 })} МБ · готова к передаче`
            : strings.hint}
        </p>
        <div className="action-row">
          <Button variant="quiet" onClick={() => input.current?.click()}>
            {strings.choose}
          </Button>
          <Button
            title={
              health?.model_status !== "ready"
                ? "Дождитесь готовности сервиса"
                : undefined
            }
            onClick={() => void submit()}
            busy={Boolean(uploadController)}
            disabled={!file || health?.model_status !== "ready"}
          >
            {progress == null
              ? "Передать и начать анализ"
              : `Передано ${(progress * 100).toFixed(0)}%`}
          </Button>
        </div>
        {uploadController && (
          <button
            className="text-button"
            onClick={() => uploadController.abort()}
          >
            Отменить передачу
          </button>
        )}
        {error && (
          <p className="field-error" role="alert">
            {error}
          </p>
        )}
      </section>
      <div className="intake-lower">
        <section>
          <div className="section-heading">
            <h2>{strings.recent}</h2>
            <span>{recent.length}</span>
          </div>
          {recent.length ? (
            <ul className="recent-list">
              {recent.map((item) => (
                <li key={item.id}>
                  <Link to={`/app/checks/${item.id}`}>
                    <span>
                      <strong>{item.file_name}</strong>
                      <small>
                        {new Date(item.created_at).toLocaleString("ru")}
                      </small>
                    </span>
                    <span>{recentStatus(item.status, item.verdict)}</span>
                  </Link>
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted">
              Проверки появятся здесь после первой загрузки.
            </p>
          )}
        </section>
        <aside className="norm-callout">
          <p className="eyebrow">Прозрачные правила</p>
          <h2>Проверяем только то, что видно</h2>
          <p>
            Суммы, документы и данные банковской системы по записи определить
            нельзя.
          </p>
          <Link to="/app/norms">Что проверяется →</Link>
          {health?.example_check_id && (
            <Link
              className="example-link"
              to={`/app/checks/${health.example_check_id}`}
            >
              Открыть пример
            </Link>
          )}
        </aside>
      </div>
    </div>
  );
}
