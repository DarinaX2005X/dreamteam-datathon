import type {
  CheckDetail,
  CheckDetailResponse,
  CheckSummary,
  CheckSummaryResponse,
  DecisionRequest,
  DecisionResponse,
  ExampleResponse,
  FindingSet,
  HealthResponse,
  ReevaluateRequest,
  ReevaluateResponse,
  RulesResponse,
} from "../../../contracts/ts/types.gen";

export type UiFindingSet = Omit<
  FindingSet,
  "findings" | "rules" | "role_overrides"
> & {
  findings: NonNullable<FindingSet["findings"]>;
  rules: NonNullable<FindingSet["rules"]>;
  role_overrides: NonNullable<FindingSet["role_overrides"]>;
};

export type UiCheck = Omit<
  CheckDetail,
  "tracks" | "observations" | "coverage" | "finding_set"
> & {
  tracks: NonNullable<CheckDetail["tracks"]>;
  observations: NonNullable<CheckDetail["observations"]>;
  coverage: NonNullable<CheckDetail["coverage"]>;
  finding_set: UiFindingSet | null;
};

export function normalizeFindingSet(value: FindingSet): UiFindingSet {
  return {
    ...value,
    findings: value.findings ?? [],
    rules: value.rules ?? [],
    role_overrides: value.role_overrides ?? [],
  };
}

function normalizeCheck(value: CheckDetail): UiCheck {
  return {
    ...value,
    tracks: value.tracks ?? [],
    observations: value.observations ?? [],
    coverage: value.coverage ?? [],
    finding_set: value.finding_set
      ? normalizeFindingSet(value.finding_set)
      : null,
  };
}

export class ApiError extends Error {
  constructor(
    public code: string,
    message: string,
    public status: number,
    public fieldIssues: unknown[] = [],
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export function parseApiErrorBody(
  body: unknown,
  status: number,
  fallback: string,
): ApiError {
  const root =
    body && typeof body === "object" ? (body as Record<string, unknown>) : {};
  const candidate = root.error ?? root.detail ?? root;
  const detail =
    candidate && typeof candidate === "object"
      ? (candidate as Record<string, unknown>)
      : {};
  return new ApiError(
    typeof detail.code === "string" ? detail.code : "UNKNOWN",
    typeof detail.message === "string" ? detail.message : fallback,
    status,
    Array.isArray(detail.field_issues) ? detail.field_issues : [],
  );
}

async function parseResponseError(response: Response): Promise<ApiError> {
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    body = undefined;
  }
  return parseApiErrorBody(
    body,
    response.status,
    response.statusText || "Ошибка запроса",
  );
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch (error) {
    if ((error as Error).name === "AbortError") throw error;
    throw new ApiError("NETWORK", "Нет связи с сервисом", 0);
  }
  if (!response.ok) throw await parseResponseError(response);
  return response.json() as Promise<T>;
}

export const api = {
  check: async (id: string, signal?: AbortSignal) =>
    normalizeCheck(
      (await request<CheckDetailResponse>(`/api/ui/checks/${id}`, { signal }))
        .check,
    ),
  recent: (id: string, signal?: AbortSignal) =>
    request<CheckDetailResponse>(`/api/ui/checks/${id}`, { signal }).then(
      (value) => value.check,
    ),
  example: (signal?: AbortSignal) =>
    request<ExampleResponse>("/api/ui/example", { signal }),
  health: (signal?: AbortSignal) =>
    request<HealthResponse>("/api/ui/health", { signal }),
  rules: (signal?: AbortSignal) =>
    request<RulesResponse>("/api/ui/rules", { signal }),
  retry: (id: string, signal?: AbortSignal) =>
    request<CheckSummaryResponse>(`/api/ui/checks/${id}/retry`, {
      method: "POST",
      signal,
    }),
  cancel: (id: string, signal?: AbortSignal) =>
    request<CheckSummaryResponse>(`/api/ui/checks/${id}/cancel`, {
      method: "POST",
      signal,
    }),
  decide: (
    id: string,
    finding: string,
    data: DecisionRequest,
    signal?: AbortSignal,
  ) =>
    request<DecisionResponse>(
      `/api/ui/checks/${id}/findings/${finding}/decision`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
        signal,
      },
    ),
  reevaluate: (id: string, data: ReevaluateRequest, signal?: AbortSignal) =>
    request<ReevaluateResponse>(`/api/ui/checks/${id}/reevaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
      signal,
    }),
  video: (id: string) => `/api/ui/checks/${id}/video`,
  frame: (id: string, frame: string, overlay = true) =>
    `/api/ui/checks/${id}/frames/${frame}.jpg?overlay=${overlay ? 1 : 0}`,
};

export function uploadCheck(
  file: File,
  onProgress: (loaded: number, total: number) => void,
  signal: AbortSignal,
) {
  return new Promise<CheckSummary>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const abort = () => xhr.abort();
    xhr.open("POST", "/api/ui/checks");
    xhr.upload.onprogress = (event) => onProgress(event.loaded, event.total);
    xhr.onload = () => {
      signal.removeEventListener("abort", abort);
      let body: unknown;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        reject(
          new ApiError(
            "UNKNOWN",
            "Сервер вернул некорректный ответ",
            xhr.status,
          ),
        );
        return;
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve((body as CheckSummaryResponse).check);
      } else {
        reject(parseApiErrorBody(body, xhr.status, "Ошибка передачи"));
      }
    };
    xhr.onerror = () =>
      reject(new ApiError("NETWORK", "Нет связи с сервисом", 0));
    xhr.onabort = () => reject(new DOMException("Aborted", "AbortError"));
    signal.addEventListener("abort", abort, { once: true });
    const data = new FormData();
    data.append("file", file);
    xhr.send(data);
  });
}

export type PollEvent =
  | { type: "data"; check: UiCheck }
  | {
      type: "error";
      error: ApiError;
      recoverable: boolean;
      nextRetryMs?: number;
    };

export function pollCheck(
  id: string,
  onEvent: (event: PollEvent) => void,
  signal: AbortSignal,
  options: { intervalMs?: number; maxBackoffMs?: number } = {},
) {
  const intervalMs = options.intervalMs ?? 1500;
  const maxBackoffMs = options.maxBackoffMs ?? 12000;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let active = false;
  let stopped = false;
  let failures = 0;

  const schedule = (delay: number) => {
    if (!stopped && !signal.aborted) timer = setTimeout(run, delay);
  };

  const run = async () => {
    if (
      stopped ||
      signal.aborted ||
      active ||
      document.visibilityState === "hidden"
    )
      return;
    active = true;
    try {
      const check = await api.check(id, signal);
      if (stopped || signal.aborted) return;
      failures = 0;
      onEvent({ type: "data", check });
      if (["queued", "processing"].includes(check.status)) schedule(intervalMs);
      else stopped = true;
    } catch (error) {
      if (stopped || signal.aborted || (error as Error).name === "AbortError")
        return;
      const apiError =
        error instanceof ApiError
          ? error
          : new ApiError("UNKNOWN", (error as Error).message, 0);
      const recoverable =
        apiError.status === 0 ||
        apiError.status >= 500 ||
        apiError.status === 429;
      if (!recoverable || apiError.status === 404 || apiError.status === 410)
        stopped = true;
      if (recoverable) failures += 1;
      const nextRetryMs = recoverable
        ? Math.min(intervalMs * 2 ** Math.min(failures, 3), maxBackoffMs)
        : undefined;
      onEvent({ type: "error", error: apiError, recoverable, nextRetryMs });
      if (recoverable && nextRetryMs) schedule(nextRetryMs);
    } finally {
      active = false;
    }
  };

  const onVisibility = () => {
    if (document.visibilityState === "visible" && !stopped && !active) {
      if (timer) clearTimeout(timer);
      void run();
    }
  };
  document.addEventListener("visibilitychange", onVisibility);
  signal.addEventListener(
    "abort",
    () => {
      stopped = true;
      if (timer) clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisibility);
    },
    { once: true },
  );
  void run();
}
