import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import type { DecisionValue, Finding } from "../../../../contracts/ts/types.gen";
import {
  ApiError,
  api,
  pollCheck,
  type UiCheck,
} from "../../api/client";
import {
  Button,
  Chip,
  Dialog,
  Empty,
  FactStrip,
  InlineNotice,
  SegmentedChoice,
  VideoStage,
} from "../../ui";
import { errors, formatTimecode } from "../../i18n/ru";
import {
  decisionPayload,
  evidenceSource,
  type DecisionDraft,
  type PreviewFrames,
} from "./domain";
import { phaseLabels } from "./strings";

type SaveState = "idle" | "saving" | "saved" | "error";

export function CheckPage() {
  const { id = "" } = useParams();
  const [check, setCheck] = useState<UiCheck>();
  const [loadError, setLoadError] = useState("");
  const [pollRetry, setPollRetry] = useState<number>();
  const [time, setTime] = useState(0);
  const [seekToken, setSeekToken] = useState(0);
  const [selected, setSelected] = useState("");
  const [drafts, setDrafts] = useState<Record<string, DecisionDraft>>({});
  const [saveStates, setSaveStates] = useState<Record<string, SaveState>>({});
  const [conflict, setConflict] = useState(false);
  const [diff, setDiff] = useState<string[]>([]);
  const [frame, setFrame] = useState<{ finding: Finding; index: number }>();
  const [frameOverlay, setFrameOverlay] = useState(true);
  const [pollGeneration, setPollGeneration] = useState(0);
  const [previewFrames, setPreviewFrames] = useState<PreviewFrames>({});
  const [previewActive, setPreviewActive] = useState(false);
  const [videoLoadFailed, setVideoLoadFailed] = useState(false);
  const selectedRef = useRef("");
  const pollAbortRef = useRef<AbortController>();
  const persistedCheckRef = useRef<UiCheck>();

  const applyCheck = useCallback((next: UiCheck) => {
    persistedCheckRef.current = next;
    setCheck(next);
    setPreviewFrames({});
    setPreviewActive(false);
    setLoadError("");
    setPollRetry(undefined);
    const findings = next.finding_set?.findings ?? [];
    setSelected((current) => {
      const valid = findings.some((item) => item.finding_id === current);
      const choice = valid ? current : (findings[0]?.finding_id ?? "");
      selectedRef.current = choice;
      if (!current && choice) {
        const first = findings.find((item) => item.finding_id === choice)!;
        setTime(first.timecode_sec);
        setSeekToken((value) => value + 1);
      }
      return choice;
    });
  }, []);

  const refresh = useCallback(async () => {
    const next = await api.check(id);
    applyCheck(next);
    return next;
  }, [applyCheck, id]);

  useEffect(() => {
    const controller = new AbortController();
    pollAbortRef.current = controller;
    pollCheck(
      id,
      (event) => {
        if (event.type === "data") applyCheck(event.check);
        else {
          setLoadError(errors[event.error.code] ?? event.error.message);
          setPollRetry(event.nextRetryMs);
        }
      },
      controller.signal,
    );
    return () => controller.abort();
  }, [applyCheck, id, pollGeneration]);

  useEffect(() => {
    setCheck(undefined);
    setSelected("");
    selectedRef.current = "";
    setVideoLoadFailed(false);
  }, [id]);

  const findingSet = check?.finding_set;
  const duration = check?.video?.duration_sec ?? 1;
  const evidenceRevision = previewActive
    ? (persistedCheckRef.current?.finding_set?.revision ?? 1)
    : (findingSet?.revision ?? 1);
  const videoAvailable = (check?.video_available ?? true) && !videoLoadFailed;
  const reviewedVerdict = useMemo(
    () =>
      findingSet?.findings.some((item) => item.decision?.value !== "rejected")
        ? "violation"
        : "no_violation",
    [findingSet],
  );
  const selectedFinding = findingSet?.findings.find(
    (item) => item.finding_id === selected,
  );

  const seek = (value: number) => {
    setTime(value);
    setSeekToken((token) => token + 1);
  };
  const choose = (finding: Finding) => {
    selectedRef.current = finding.finding_id;
    setSelected(finding.finding_id);
    seek(finding.timecode_sec);
  };

  const handleFindingKeys = (event: React.KeyboardEvent<HTMLElement>) => {
    if (
      event.target instanceof HTMLTextAreaElement ||
      event.target instanceof HTMLInputElement
    )
      return;
    const findings = findingSet?.findings ?? [];
    const index = findings.findIndex((item) => item.finding_id === selected);
    if (
      (event.key === "ArrowDown" || event.key === "ArrowUp") &&
      findings.length
    ) {
      event.preventDefault();
      const offset = event.key === "ArrowDown" ? 1 : -1;
      choose(findings[(index + offset + findings.length) % findings.length]);
    }
    const decision = (
      { "1": "confirmed", "2": "rejected", "3": "not_visible" } as const
    )[event.key as "1" | "2" | "3"];
    if (decision && selectedFinding) {
      event.preventDefault();
      void saveDecision(selectedFinding, { value: decision });
    }
  };

  const draftFor = (finding: Finding): DecisionDraft =>
    drafts[finding.finding_id] ?? {
      value: finding.decision?.value ?? null,
      note: finding.decision?.note ?? "",
    };

  const updateDraft = (finding: Finding, change: Partial<DecisionDraft>) => {
    setDrafts((current) => ({
      ...current,
      [finding.finding_id]: { ...draftFor(finding), ...change },
    }));
  };

  const saveDecision = async (
    finding: Finding,
    override?: Partial<DecisionDraft>,
  ) => {
    if (!findingSet || check?.is_example) return;
    const draft = { ...draftFor(finding), ...override };
    setDrafts((current) => ({ ...current, [finding.finding_id]: draft }));
    setSaveStates((current) => ({
      ...current,
      [finding.finding_id]: "saving",
    }));
    try {
      const response = await api.decide(
        id,
        finding.finding_id,
        decisionPayload(draft, findingSet.revision),
      );
      setCheck(
        (current) =>
          current && {
            ...current,
            finding_set: {
              ...current.finding_set!,
              revision: response.revision,
              findings: current.finding_set!.findings.map((item) =>
                item.finding_id === finding.finding_id
                  ? { ...item, decision: response.decision }
                  : item,
              ),
            },
          },
      );
      setSaveStates((current) => ({
        ...current,
        [finding.finding_id]: "saved",
      }));
      setConflict(false);
    } catch (error) {
      setSaveStates((current) => ({
        ...current,
        [finding.finding_id]: "error",
      }));
      if ((error as ApiError).code === "STALE_REVISION") setConflict(true);
      else setLoadError((error as Error).message);
    }
  };

  const selectAvailableFinding = (findings: Finding[]) => {
    if (findings.some((item) => item.finding_id === selectedRef.current))
      return;
    const first = findings[0];
    selectedRef.current = first?.finding_id ?? "";
    setSelected(selectedRef.current);
    if (first) seek(first.timecode_sec);
  };

  const resetExamplePreview = () => {
    const persisted = persistedCheckRef.current;
    if (!persisted) return;
    setCheck(persisted);
    setPreviewFrames({});
    setPreviewActive(false);
    setDiff([]);
    selectAvailableFinding(persisted.finding_set?.findings ?? []);
  };

  if (!check) {
    return (
      <div className="page">
        <h1>Проверка записи</h1>
        {loadError ? (
          <InlineNotice tone="danger">{loadError}</InlineNotice>
        ) : (
          <p>Получаем состояние проверки…</p>
        )}
      </div>
    );
  }
  const resultReady =
    check.status === "done" &&
    check.progress.phase === "done" &&
    Boolean(check.finding_set);
  if (!resultReady) {
    return (
      <Processing
        check={check}
        restartPolling={() => {
          pollAbortRef.current?.abort();
          setPollGeneration((value) => value + 1);
        }}
        pollError={loadError}
        retryMs={pollRetry}
      />
    );
  }

  return (
    <div className="workspace-page">
      {conflict && (
        <InlineNotice
          tone="warning"
          action={
            <Button variant="quiet" onClick={() => void refresh()}>
              Обновить
            </Button>
          }
        >
          Эту проверку только что изменили в другой вкладке. Черновик заметки
          сохранён.
        </InlineNotice>
      )}
      {loadError && <InlineNotice tone="danger">{loadError}</InlineNotice>}
      {check.video_available === false && (
        <InlineNotice tone="warning">
          Исходная запись удалена по сроку хранения. Кадры-доказательства,
          результат и решения проверяющего сохранены.
        </InlineNotice>
      )}
      <header
        className={`verdict ${reviewedVerdict === "violation" ? "is-alert" : "is-clear"}`}
      >
        <div>
          <p>
            {check.file_name} · {formatTimecode(duration)}
          </p>
          <h1>
            {reviewedVerdict === "violation"
              ? "Обнаружены признаки возможных нарушений"
              : "Нарушение не обнаружено"}
          </h1>
          {findingSet?.basis === "reviewer" && (
            <small>Результат уточнён проверяющим.</small>
          )}
        </div>
        <Link className="button button--quiet" to={`/app/checks/${id}/sheet`}>
          Лист проверки
        </Link>
        {check.is_example && previewActive && (
          <Button variant="quiet" onClick={resetExamplePreview}>
            Сбросить предпросмотр
          </Button>
        )}
      </header>
      {diff.length > 0 && (
        <InlineNotice tone="success">{diff.join(". ")}.</InlineNotice>
      )}
      <div className="workspace">
        <section className="stage-column">
          <VideoStage
            checkId={id}
            duration={duration}
            fps={check.video?.fps_nominal}
            width={check.video?.width}
            height={check.video?.height}
            available={check.video_available ?? true}
            tracks={check.tracks}
            time={time}
            seekToken={seekToken}
            onTime={setTime}
            onAvailabilityChange={(available) => setVideoLoadFailed(!available)}
          />
          {videoAvailable && check.video_expires_at && (
            <p className="coverage-note">
              Исходная запись доступна до{" "}
              {new Date(check.video_expires_at).toLocaleString("ru")}.
            </p>
          )}
          <FactStrip
            duration={duration}
            observations={check.observations}
            coverage={check.coverage}
            findings={findingSet?.findings}
            time={time}
            onSeek={seek}
            onSelect={(findingId) => {
              const item = findingSet?.findings.find(
                (entry) => entry.finding_id === findingId,
              );
              if (item) choose(item);
            }}
          />
          {check.coverage.length > 0 && (
            <p className="coverage-note">
              Заштрихованные участки нельзя полностью проверить: часть сцены не
              видна.
            </p>
          )}
        </section>
        <aside
          className="findings-column"
          aria-label="Эпизоды"
          tabIndex={0}
          onKeyDown={handleFindingKeys}
        >
          {findingSet?.findings.length ? (
            findingSet.findings.map((finding, index) => (
              <FindingCard
                key={finding.finding_id}
                checkId={id}
                revision={evidenceRevision}
                previewFrames={previewFrames}
                finding={finding}
                index={index}
                selected={selected === finding.finding_id}
                readOnly={check.is_example ?? false}
                draft={draftFor(finding)}
                saveState={saveStates[finding.finding_id] ?? "idle"}
                onChoose={() => choose(finding)}
                onDraft={(change) => updateDraft(finding, change)}
                onSave={(change) => void saveDecision(finding, change)}
                onFrame={(frameIndex) => {
                  seek(finding.evidence[frameIndex].timestamp_sec);
                  setFrameOverlay(true);
                  setFrame({ finding, index: frameIndex });
                }}
              />
            ))
          ) : (
            <Empty title="Эпизодов нет">
              Проверенные правила и непроверяемые участки остаются доступны в
              ленте и листе проверки.
            </Empty>
          )}
          <details className="rules-summary">
            <summary>
              Без замечаний:{" "}
              {findingSet?.rules.filter(
                (item) => item.status === "no_violation",
              ).length ?? 0}{" "}
              правил
            </summary>
            <ul>
              {findingSet?.rules.map((item) => (
                <li key={item.rule_id}>
                  <Link to={`/app/norms#${item.rule_id}`}>{item.rule_id}</Link>
                  <span>{ruleStatus(item.status)}</span>
                </li>
              ))}
            </ul>
          </details>
        </aside>
      </div>
      <Dialog
        open={Boolean(frame)}
        title={frame?.finding.type_label ?? "Кадр"}
        onClose={() => setFrame(undefined)}
      >
        {frame && (
          <>
            <div className="modal-body action-row">
              <Button
                variant={frameOverlay ? "primary" : "quiet"}
                onClick={() => setFrameOverlay(true)}
              >
                С разметкой
              </Button>
              <Button
                variant={!frameOverlay ? "primary" : "quiet"}
                onClick={() => setFrameOverlay(false)}
              >
                Без разметки
              </Button>
            </div>
            <img
              className="dialog-image"
              src={evidenceSource(
                id,
                frame.finding.evidence[frame.index].frame_id,
                evidenceRevision,
                previewFrames,
                frameOverlay,
              )}
              alt={
                frameOverlay
                  ? "Кадр-доказательство с разметкой"
                  : "Кадр-доказательство без разметки"
              }
            />
          </>
        )}
      </Dialog>
    </div>
  );
}

function FindingCard({
  checkId,
  revision,
  previewFrames,
  finding,
  index,
  selected,
  readOnly,
  draft,
  saveState,
  onChoose,
  onDraft,
  onSave,
  onFrame,
}: {
  checkId: string;
  revision: number;
  previewFrames: PreviewFrames;
  finding: Finding;
  index: number;
  selected: boolean;
  readOnly: boolean;
  draft: DecisionDraft;
  saveState: SaveState;
  onChoose: () => void;
  onDraft: (change: Partial<DecisionDraft>) => void;
  onSave: (change?: Partial<DecisionDraft>) => void;
  onFrame: (index: number) => void;
}) {
  return (
    <article
      className={`finding-card ${selected ? "is-selected" : ""} ${draft.value === "rejected" ? "is-rejected" : ""}`}
    >
      <button className="finding-card__head" onClick={onChoose}>
        <span className="finding-number">
          {String(index + 1).padStart(2, "0")}
        </span>
        <span>
          <time aria-label={`Начало эпизода ${formatTimecode(finding.timecode_sec)}`}>Начало · {formatTimecode(finding.timecode_sec)}</time>
          <strong>{finding.type_label}</strong>
        </span>
        <Chip tone={finding.strength === "strong" ? "alert" : "neutral"}>
          {finding.strength === "strong" ? "Явный признак" : "Требует внимания"}
        </Chip>
      </button>
      {selected && (
        <div className="finding-body">
          <section>
            <h3>Что видно</h3>
            <p>{finding.what_seen}</p>
          </section>
          <section>
            <h3>Почему это важно</h3>
            <p>{finding.why_may_violate}</p>
            <Link to={`/app/norms#${finding.rule_id}`}>Подробнее о норме</Link>
          </section>
          <div className="evidence-row">
            {finding.evidence.map((item, frameIndex) => (
              <button key={item.frame_id} onClick={() => onFrame(frameIndex)}>
                <img
                  src={evidenceSource(
                    checkId,
                    item.frame_id,
                    revision,
                    previewFrames,
                  )}
                  alt={`Кадр ${formatTimecode(item.timestamp_sec)}`}
                />
                <span>{formatTimecode(item.timestamp_sec)}</span>
              </button>
            ))}
          </div>
          <h3>Решение проверяющего</h3>
          <SegmentedChoice
            value={draft.value}
            disabled={readOnly || saveState === "saving"}
            onChange={(value) => {
              onDraft({ value });
              onSave({ value });
            }}
          />
          <label className="field">
            <span>Заметка</span>
            <textarea
              maxLength={2000}
              readOnly={readOnly}
              value={draft.note}
              onChange={(event) => onDraft({ note: event.target.value })}
              onBlur={() => onSave()}
              onKeyDown={(event) => {
                if (event.ctrlKey && event.key === "Enter") onSave();
              }}
              placeholder="Что важно учесть при разборе"
            />
          </label>
          <div className="save-feedback" role="status">
            {saveState === "saving" ? (
              "Сохраняем…"
            ) : saveState === "saved" ? (
              "Сохранено"
            ) : saveState === "error" ? (
              <button onClick={() => onSave()}>Не сохранено · повторить</button>
            ) : null}
          </div>
          {!readOnly && (draft.value !== null || draft.note) && (
            <button
              className="text-button"
              onClick={() => onSave({ value: null, note: "" })}
            >
              Сбросить решение и заметку
            </button>
          )}
          {readOnly && <small>Пример открыт только для чтения.</small>}
        </div>
      )}
    </article>
  );
}

function Processing({
  check,
  restartPolling,
  pollError,
  retryMs,
}: {
  check: UiCheck;
  restartPolling: () => void;
  pollError: string;
  retryMs?: number;
}) {
  const navigate = useNavigate();
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState("");
  const [now, setNow] = useState(() => Date.now());
  const terminal = ["failed", "cancelled"].includes(check.status);
  const awaitingResult = check.status === "done";
  const phases = ["decode", "perception", "facts", "rules", "evidence"];
  const currentIndex = phases.indexOf(check.progress.phase);
  const createdAt = new Date(check.created_at).getTime();
  const elapsedSeconds = Number.isFinite(createdAt)
    ? Math.max(0, Math.floor((now - createdAt) / 1000))
    : null;
  useEffect(() => {
    if (terminal) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [terminal]);
  const act = async () => {
    setBusy(true);
    setActionError("");
    try {
      if (terminal) {
        const response = await api.retry(check.id);
        if (response.check.id !== check.id) {
          navigate(`/app/checks/${response.check.id}`, { replace: true });
          return;
        }
      } else {
        await api.cancel(check.id);
      }
      restartPolling();
    } catch (error) {
      const apiError = error as ApiError;
      setActionError(errors[apiError.code] ?? apiError.message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="page processing-page">
      <header className="page-heading">
        <p className="eyebrow">{check.file_name}</p>
        <h1>{phaseLabels[check.progress.phase]}</h1>
        <p className="processing-status">
          {!terminal && (
            <span className="processing-spinner" aria-hidden="true" />
          )}
          {check.status === "queued" && check.progress.queue_position
            ? `В очереди: перед этой записью ${check.progress.queue_position}.`
            : check.progress.processed_until_sec != null && check.duration_sec
              ? `Обработано ${formatTimecode(check.progress.processed_until_sec)} из ${formatTimecode(check.duration_sec)}.`
              : "Состояние обновится автоматически."}
        </p>
        <p className="processing-time">
          Прошло:{" "}
          {elapsedSeconds == null ? "—" : formatTimecode(elapsedSeconds)}
          <span>
            Оставшееся время появится, когда его рассчитает анализатор.
          </span>
        </p>
      </header>
      {pollError && (
        <InlineNotice tone="warning">
          {pollError}
          {retryMs
            ? ` Повторим запрос через ${Math.ceil(retryMs / 1000)} с.`
            : " Автоматическое обновление остановлено."}
        </InlineNotice>
      )}
      {check.error && (
        <InlineNotice tone="danger">
          <strong>{errors[check.error.code] ?? check.error.message}</strong>
          <br />
          Этап:{" "}
          {phaseLabels[check.error.failed_phase] ?? check.error.failed_phase}.
          Сохранённые наблюдения доступны ниже.
        </InlineNotice>
      )}
      {actionError && <InlineNotice tone="danger">{actionError}</InlineNotice>}
      <div className="processing-actions">
        <section className="activity-panel" aria-labelledby="activity-title">
          <h2 id="activity-title">Ход анализа</h2>
          <table>
            <thead>
              <tr>
                <th>Этап</th>
                <th>Состояние</th>
              </tr>
            </thead>
            <tbody>
              {phases.map((phase, index) => {
                const state =
                  check.progress.phase === "done" || index < currentIndex
                    ? "Готово"
                    : index === currentIndex
                      ? "Выполняется"
                      : "Ожидает";
                return (
                  <tr
                    key={phase}
                    className={index === currentIndex ? "is-current" : ""}
                  >
                    <td>{phaseLabels[phase]}</td>
                    <td>{state}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>
        {!awaitingResult && (
          <Button
            variant={terminal ? "primary" : "danger"}
            busy={busy}
            onClick={() => void act()}
          >
            {terminal ? "Запустить анализ ещё раз" : "Отменить анализ"}
          </Button>
        )}
      </div>
    </div>
  );
}

function diffMessages(diff: {
  removed: string[];
  added: string[];
  changed: string[];
  verdict_changed: boolean;
}) {
  return [
    ...diff.removed.map((id) => `Эпизод ${id} исчез после уточнения`),
    ...diff.added.map((id) => `Добавлен эпизод ${id}`),
    ...(diff.changed.length
      ? [`Изменено эпизодов: ${diff.changed.length}`]
      : []),
    ...(diff.verdict_changed ? ["Вердикт изменился"] : []),
  ];
}
function ruleStatus(status: string) {
  return status === "not_checkable"
    ? "Нельзя проверить"
    : status === "partly_checkable"
      ? "Проверено частично"
      : status === "no_violation"
        ? "Без замечаний"
        : "Есть эпизод";
}
