import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import type {
  DecisionValue,
  Finding,
  Role,
  Zone,
  ZoneKind,
} from "../../../../contracts/ts/types.gen";
import {
  ApiError,
  api,
  normalizeFindingSet,
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
  ModalSurface,
  PhaseList,
  SegmentedChoice,
  VideoStage,
} from "../../ui";
import { errors, formatTimecode } from "../../i18n/ru";
import {
  decisionPayload,
  mergeRoleOverrides,
  type DecisionDraft,
} from "./domain";
import { phaseLabels, roleLabels } from "./strings";

type SaveState = "idle" | "saving" | "saved" | "error";

export function CheckPage() {
  const { id = "" } = useParams();
  const [check, setCheck] = useState<UiCheck>();
  const [loadError, setLoadError] = useState("");
  const [pollRetry, setPollRetry] = useState<number>();
  const [time, setTime] = useState(0);
  const [seekToken, setSeekToken] = useState(0);
  const [selected, setSelected] = useState("");
  const [saving, setSaving] = useState("");
  const [drafts, setDrafts] = useState<Record<string, DecisionDraft>>({});
  const [saveStates, setSaveStates] = useState<Record<string, SaveState>>({});
  const [conflict, setConflict] = useState(false);
  const [roleTrack, setRoleTrack] = useState("");
  const [role, setRole] = useState<Role>("unknown");
  const [diff, setDiff] = useState<string[]>([]);
  const [frame, setFrame] = useState<{ finding: Finding; index: number }>();
  const [zoneDraft, setZoneDraft] = useState<Zone[] | null>(null);
  const [pollGeneration, setPollGeneration] = useState(0);
  const selectedRef = useRef("");
  const pollAbortRef = useRef<AbortController>();

  const applyCheck = useCallback((next: UiCheck) => {
    setCheck(next);
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
  }, [id]);

  const findingSet = check?.finding_set;
  const duration = check?.video?.duration_sec ?? 1;
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

  const reevaluateRole = async () => {
    if (!findingSet || !roleTrack || !check) return;
    setSaving("role");
    try {
      const roleOverrides = mergeRoleOverrides(findingSet.role_overrides, {
        track_id: roleTrack,
        role,
      });
      const response = await api.reevaluate(id, {
        base_revision: findingSet.revision,
        role_overrides: roleOverrides,
        zone_overrides: check.layout?.zones ?? [],
        dry_run: check.is_example ?? false,
      });
      setCheck(
        (current) =>
          current && {
            ...current,
            tracks: current.tracks.map((track) =>
              track.track_id === roleTrack
                ? { ...track, role: { value: role, source: "reviewer" } }
                : track,
            ),
            finding_set: normalizeFindingSet(response.finding_set),
          },
      );
      setDiff(diffMessages(response.diff));
      setRoleTrack("");
      if (!check.is_example) await refresh();
    } catch (error) {
      if ((error as ApiError).code === "STALE_REVISION") setConflict(true);
      else setLoadError((error as Error).message);
    } finally {
      setSaving("");
    }
  };

  const saveZones = async () => {
    if (!findingSet || !zoneDraft || !check) return;
    setSaving("zones");
    try {
      const response = await api.reevaluate(id, {
        base_revision: findingSet.revision,
        role_overrides: findingSet.role_overrides,
        zone_overrides: zoneDraft,
        dry_run: check.is_example ?? false,
      });
      setCheck(
        (current) =>
          current && {
            ...current,
            layout: current.layout
              ? { ...current.layout, zones: zoneDraft }
              : current.layout,
            finding_set: normalizeFindingSet(response.finding_set),
          },
      );
      setDiff(diffMessages(response.diff));
      setZoneDraft(null);
      if (!check.is_example) await refresh();
    } catch (error) {
      if ((error as ApiError).code === "STALE_REVISION") setConflict(true);
      else setLoadError((error as Error).message);
    } finally {
      setSaving("");
    }
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
  if (["queued", "processing", "failed", "cancelled"].includes(check.status)) {
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
      <header
        className={`verdict ${reviewedVerdict === "violation" ? "is-alert" : "is-clear"}`}
      >
        <div>
          <p>
            {check.file_name} · {formatTimecode(duration)} ·{" "}
            {check.analysis_origin === "cache"
              ? "повторно использован локальный результат"
              : "локальный анализ"}
          </p>
          <h1>
            {reviewedVerdict === "violation"
              ? "Обнаружены признаки возможных нарушений"
              : "Нарушение не обнаружено"}
          </h1>
          <small>
            Текущая ревизия {findingSet?.revision}; основание:{" "}
            {findingSet?.basis === "reviewer"
              ? "уточнения проверяющего"
              : "автоматический анализ"}
            .
            {check.automatic_verdict &&
              ` Автоматический вердикт: ${check.automatic_verdict === "violation" ? "обнаружены признаки" : "нарушение не обнаружено"}.`}
          </small>
        </div>
        <Link className="button button--quiet" to={`/app/checks/${id}/sheet`}>
          Лист проверки
        </Link>
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
            tracks={check.tracks}
            time={time}
            seekToken={seekToken}
            onTime={setTime}
            onTrack={(trackId) => {
              setRoleTrack(trackId);
              setRole(
                check.tracks.find((item) => item.track_id === trackId)?.role
                  ?.value ?? "unknown",
              );
            }}
          />
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
          <section className="people">
            <h2>Люди в кадре</h2>
            {check.tracks
              .filter((item) => item.kind === "person")
              .map((item) => (
                <button
                  key={item.track_id}
                  onClick={() => {
                    setRoleTrack(item.track_id);
                    setRole(item.role?.value ?? "unknown");
                  }}
                >
                  {item.track_id}
                  <span>{roleLabels[item.role?.value ?? "unknown"]}</span>
                </button>
              ))}
          </section>
          {(check.layout?.zones ?? []).length > 0 && (
            <Button
              variant="quiet"
              onClick={() =>
                setZoneDraft(cloneZones(check.layout?.zones ?? []))
              }
            >
              Уточнить зоны сцены
            </Button>
          )}
        </aside>
      </div>
      <ModalSurface
        open={Boolean(roleTrack)}
        title="Кто это на записи?"
        onClose={() => setRoleTrack("")}
      >
        <div className="modal-body">
          <div className="radio-list">
            {Object.entries(roleLabels).map(([value, label]) => (
              <label key={value}>
                <input
                  type="radio"
                  name="role"
                  value={value}
                  checked={role === value}
                  onChange={() => setRole(value as Role)}
                />
                <span>{label}</span>
              </label>
            ))}
          </div>
          {check.is_example && <p>В примере изменения не сохраняются.</p>}
          <div className="action-row">
            <Button variant="quiet" onClick={() => setRoleTrack("")}>
              Отмена
            </Button>
            <Button
              busy={saving === "role"}
              onClick={() => void reevaluateRole()}
            >
              Пересчитать по нормам
            </Button>
          </div>
        </div>
      </ModalSurface>
      <ModalSurface
        open={zoneDraft !== null}
        title="Зоны сцены"
        onClose={() => setZoneDraft(null)}
        className="zone-dialog"
      >
        {zoneDraft && <ZoneEditor zones={zoneDraft} onChange={setZoneDraft} />}
        <div className="modal-body">
          <p>
            Координаты заданы относительно кадра от 0 до 1. Изменённая область
            показана на схеме.
          </p>
          <div className="action-row">
            <Button variant="quiet" onClick={() => setZoneDraft(null)}>
              Отмена
            </Button>
            <Button busy={saving === "zones"} onClick={() => void saveZones()}>
              Пересчитать по нормам
            </Button>
          </div>
        </div>
      </ModalSurface>
      <Dialog
        open={Boolean(frame)}
        title={frame?.finding.type_label ?? "Кадр"}
        onClose={() => setFrame(undefined)}
      >
        {frame && (
          <img
            className="dialog-image"
            src={api.frame(id, frame.finding.evidence[frame.index].frame_id)}
            alt="Кадр-доказательство с разметкой"
          />
        )}
      </Dialog>
    </div>
  );
}

function FindingCard({
  checkId,
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
          <time>{formatTimecode(finding.timecode_sec)}</time>
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
                  src={api.frame(checkId, item.frame_id)}
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

function ZoneEditor({
  zones,
  onChange,
}: {
  zones: Zone[];
  onChange: (zones: Zone[]) => void;
}) {
  const [selected, setSelected] = useState(0);
  const zone = zones[selected];
  const setPoint = (pointIndex: number, coordinate: 0 | 1, value: number) => {
    const bounded = Math.max(
      0,
      Math.min(1, Number.isFinite(value) ? value : 0),
    );
    onChange(
      zones.map((item, zoneIndex) =>
        zoneIndex !== selected
          ? item
          : {
              ...item,
              polygon: item.polygon.map((point, index) =>
                index !== pointIndex
                  ? point
                  : coordinate === 0
                    ? [bounded, point[1]]
                    : [point[0], bounded],
              ),
            },
      ),
    );
  };
  return (
    <div className="zone-editor">
      <div className="zone-preview" data-surface="stage">
        <svg viewBox="0 0 1 1" preserveAspectRatio="none">
          {zones.map((item, index) => (
            <polygon
              key={item.zone_id}
              className={index === selected ? "is-selected" : ""}
              points={item.polygon.map((point) => point.join(",")).join(" ")}
              onClick={() => setSelected(index)}
            />
          ))}
        </svg>
      </div>
      <div className="zone-controls">
        <label className="field">
          <span>Область</span>
          <select
            value={selected}
            onChange={(event) => setSelected(Number(event.target.value))}
          >
            {zones.map((item, index) => (
              <option value={index} key={item.zone_id}>
                {item.zone_id}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>Назначение</span>
          <select
            value={zone.kind}
            onChange={(event) =>
              onChange(
                zones.map((item, index) =>
                  index === selected
                    ? { ...item, kind: event.target.value as ZoneKind }
                    : item,
                ),
              )
            }
          >
            {zoneKinds.map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <fieldset>
          <legend>Вершины области</legend>
          {zone.polygon.map((point, index) => (
            <div className="coordinate-row" key={index}>
              <span>{index + 1}</span>
              <label>
                X{" "}
                <input
                  aria-label={`Вершина ${index + 1}, X`}
                  type="number"
                  min="0"
                  max="1"
                  step="0.01"
                  value={point[0]}
                  onChange={(event) =>
                    setPoint(index, 0, event.target.valueAsNumber)
                  }
                />
              </label>
              <label>
                Y{" "}
                <input
                  aria-label={`Вершина ${index + 1}, Y`}
                  type="number"
                  min="0"
                  max="1"
                  step="0.01"
                  value={point[1]}
                  onChange={(event) =>
                    setPoint(index, 1, event.target.valueAsNumber)
                  }
                />
              </label>
            </div>
          ))}
        </fieldset>
      </div>
    </div>
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
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState("");
  const terminal = ["failed", "cancelled"].includes(check.status);
  const act = async () => {
    setBusy(true);
    setActionError("");
    try {
      terminal ? await api.retry(check.id) : await api.cancel(check.id);
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
        <p>
          {check.status === "queued" && check.progress.queue_position
            ? `В очереди: перед этой записью ${check.progress.queue_position}.`
            : check.progress.processed_until_sec != null && check.duration_sec
              ? `Обработано ${formatTimecode(check.progress.processed_until_sec)} из ${formatTimecode(check.duration_sec)}.`
              : "Состояние обновится автоматически."}
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
      <div className="processing-grid">
        <PhaseList current={check.progress.phase} />
        <div>
          <FactStrip
            duration={
              check.duration_sec ??
              Math.max(1, check.progress.processed_until_sec ?? 1)
            }
            observations={check.observations}
            coverage={check.coverage}
            onSeek={() => {}}
          />
          <Button
            variant={terminal ? "primary" : "danger"}
            busy={busy}
            onClick={() => void act()}
          >
            {terminal ? "Запустить анализ ещё раз" : "Отменить анализ"}
          </Button>
        </div>
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
function cloneZones(zones: Zone[]): Zone[] {
  return zones.map((zone) => ({
    ...zone,
    polygon: zone.polygon.map((point) => [...point] as [number, number]),
  }));
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
const zoneKinds: [ZoneKind, string][] = [
  ["cashier_post", "Место кассира"],
  ["client_window", "Окно клиента"],
  ["cash_desk", "Кассовый стол"],
  ["storage", "Хранилище"],
  ["service_door", "Служебный вход"],
  ["transfer_tray", "Передаточный лоток"],
  ["count_room", "Комната пересчёта"],
];
