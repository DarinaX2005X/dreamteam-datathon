import React, { useEffect, useRef, useState } from "react";
import type {
  Coverage,
  Finding,
  Observation,
  Track,
  TrackSample,
} from "../../../contracts/ts/types.gen";
import { formatTimecode } from "../i18n/ru";
import { roleLabels } from "../features/review/strings";

export function Button({
  variant = "primary",
  busy = false,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "quiet" | "danger";
  busy?: boolean;
}) {
  return (
    <button
      {...props}
      className={`button button--${variant} ${props.className ?? ""}`}
      disabled={props.disabled || busy}
    >
      {busy ? "Подождите…" : props.children}
    </button>
  );
}

export function InlineNotice({
  tone = "info",
  children,
  action,
}: {
  tone?: "info" | "danger" | "warning" | "success";
  children: React.ReactNode;
  action?: React.ReactNode;
}) {
  return (
    <div
      className={`notice notice--${tone}`}
      role={tone === "danger" ? "alert" : "status"}
    >
      <div>{children}</div>
      {action}
    </div>
  );
}

export function Chip({
  children,
  tone = "neutral",
}: {
  children: React.ReactNode;
  tone?: string;
}) {
  return <span className={`chip chip--${tone}`}>{children}</span>;
}

export function PhaseList({ current }: { current: string }) {
  const phases = [
    ["decode", "Читаем запись"],
    ["perception", "Находим людей и предметы"],
    ["facts", "Восстанавливаем действия"],
    ["rules", "Сверяем с нормами"],
    ["evidence", "Готовим кадры"],
  ];
  const currentIndex = phases.findIndex(([id]) => id === current);
  return (
    <ol className="phases">
      {phases.map(([id, label], index) => (
        <li
          key={id}
          className={
            index < currentIndex || current === "done"
              ? "is-done"
              : index === currentIndex
                ? "is-current"
                : ""
          }
        >
          <span aria-hidden="true">
            {index < currentIndex || current === "done" ? "✓" : index + 1}
          </span>
          {label}
        </li>
      ))}
    </ol>
  );
}

export function SegmentedChoice({
  value,
  onChange,
  disabled,
}: {
  value?: string | null;
  onChange: (value: "confirmed" | "rejected" | "not_visible") => void;
  disabled?: boolean;
}) {
  const choices = [
    ["confirmed", "Подтвердить"],
    ["rejected", "Не нарушение"],
    ["not_visible", "Не видно"],
  ] as const;
  return (
    <div
      className="segments"
      role="radiogroup"
      aria-label="Решение проверяющего"
    >
      {choices.map(([choice, label]) => (
        <button
          type="button"
          role="radio"
          aria-checked={value === choice}
          disabled={disabled}
          key={choice}
          onClick={() => onChange(choice)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

const objectNames: Record<string, string> = {
  cash: "Деньги",
  identity_document: "Удостоверение",
  hand: "Кисть руки",
  money_counter: "Счётчик банкнот",
  phone: "Телефон",
  bag: "Сумка",
};

const factNames: Record<string, string> = {
  person_in_zone: "Человек в зоне",
  person_left_zone: "Человек покинул зону",
  extra_person_present: "Более двух людей одновременно",
  identity_document_visible: "Документ, удостоверяющий личность",
  cash_in_hands: "Деньги в руках",
  cash_on_desk: "Деньги на столе",
  cash_handover: "Передача денег",
  counting_motion: "Пересчёт",
  phone_visible: "Телефон виден",
  bag_present: "Сумка в кадре",
  bag_opened: "Сумка открыта",
  storage_open: "Хранилище открыто",
  visibility_low: "Низкая видимость",
  cash_sorting: "Сортировка денег",
  cash_packing: "Упаковка денег",
  bag_integrity_damage: "Повреждение сумки",
  personal_item_identified: "Личная вещь",
  camera_device_visible: "Камера в кадре",
};

type FactGroup = { key: string; label: string; observations: Observation[] };

export function nearestVisibleSample(
  samples: TrackSample[],
  time: number,
): TrackSample | undefined {
  const ordered = [...samples].sort((a, b) => a.t - b.t);
  const nearest = ordered.reduce<TrackSample | undefined>(
    (best, sample) =>
      !best || Math.abs(sample.t - time) < Math.abs(best.t - time)
        ? sample
        : best,
    undefined,
  );
  if (!nearest) return undefined;
  const intervals = ordered
    .slice(1)
    .map((sample, index) => sample.t - ordered[index].t)
    .filter((interval) => interval > 0 && interval <= 2)
    .sort((a, b) => a - b);
  const typicalInterval = intervals[Math.floor(intervals.length / 2)] ?? 0.1;
  const tolerance = Math.min(0.75, Math.max(0.08, typicalInterval * 0.75));
  return Math.abs(nearest.t - time) <= tolerance ? nearest : undefined;
}

export function groupObservations(observations: Observation[]): FactGroup[] {
  const groups = new Map<string, FactGroup>();
  for (const observation of observations) {
    const subject = observation.subjects?.join(", ") || "сцена";
    const key = `${observation.kind}:${subject}`;
    const group = groups.get(key) ?? {
      key,
      label: `${factNames[observation.kind] ?? observation.kind} · ${subject}`,
      observations: [],
    };
    group.observations.push(observation);
    groups.set(key, group);
  }
  return [...groups.values()];
}

export function FactStrip({
  duration,
  observations,
  coverage,
  findings = [],
  time = 0,
  onSeek,
  onSelect,
}: {
  duration: number;
  observations: Observation[];
  coverage: Coverage[];
  findings?: Finding[];
  time?: number;
  onSeek: (seconds: number) => void;
  onSelect?: (id: string) => void;
}) {
  void duration;
  void time;
  return (
    <section className="fact-strip" aria-label="Наблюдаемые факты">
      <div className="fact-strip__head">
        <h3>Наблюдаемые факты</h3>
      </div>
      <div className="fact-table-wrap">
        <table>
          <thead>
            <tr>
              <th>Факт</th>
              <th>Начало</th>
              <th>Конец</th>
            </tr>
          </thead>
          <tbody>
            {observations.map((item) => (
              <tr key={item.obs_id}>
                <td>{factNames[item.kind] ?? item.kind}</td>
                <td>
                  <button
                    className="time-link"
                    onClick={() => onSeek(item.start_sec)}
                  >
                    {formatTimecode(item.start_sec)}
                  </button>
                </td>
                <td>{formatTimecode(item.end_sec)}</td>
              </tr>
            ))}
            {coverage.map((item, index) => (
              <tr key={`coverage-${index}`} className="is-coverage">
                <td>Нельзя полностью проверить: {item.reason}</td>
                <td>
                  <button
                    className="time-link"
                    onClick={() => onSeek(item.start_sec)}
                  >
                    {formatTimecode(item.start_sec)}
                  </button>
                </td>
                <td>{formatTimecode(item.end_sec)}</td>
              </tr>
            ))}
            {findings.map((item) => (
              <tr key={item.finding_id} className="is-finding">
                <td>
                  <button
                    className="fact-link"
                    onClick={() => onSelect?.(item.finding_id)}
                  >
                    {item.type_label}
                  </button>
                </td>
                <td>
                  <button
                    className="time-link"
                    onClick={() => onSeek(item.timecode_sec)}
                  >
                    {formatTimecode(item.timecode_sec)}
                  </button>
                </td>
                <td>Эпизод</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function VideoStage({
  checkId,
  duration = 1,
  fps = 25,
  width = 16,
  height = 9,
  available = true,
  tracks,
  time,
  seekToken,
  onTime,
  onAvailabilityChange,
}: {
  checkId: string;
  duration?: number;
  fps?: number;
  width?: number;
  height?: number;
  available?: boolean;
  tracks: Track[];
  time: number;
  seekToken: number;
  onTime: (value: number) => void;
  onAvailabilityChange?: (available: boolean) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [overlay, setOverlay] = useState(true);
  const [mediaError, setMediaError] = useState(false);
  const [loadGeneration, setLoadGeneration] = useState(0);
  useEffect(() => {
    if (videoRef.current)
      videoRef.current.currentTime = Math.max(0, Math.min(duration, time));
  }, [duration, seekToken]);
  const active = tracks.filter(
    (track) =>
      track.first_sec <= time &&
      track.last_sec >= time &&
      (track.kind === "person" ||
        Object.hasOwn(objectNames, track.object_class ?? "")),
  );
  const seek = (value: number) => {
    if (videoRef.current) videoRef.current.currentTime = value;
    onTime(value);
  };
  return (
    <div
      className="video-stage"
      data-surface="stage"
      style={{ maxWidth: `min(100%, calc(65vh * ${width} / ${height}))` }}
    >
      <div
        className="video-frame"
        style={{ aspectRatio: `${width}/${height}` }}
      >
        {available && !mediaError ? (
          <video
            key={loadGeneration}
            ref={videoRef}
            src={`/api/ui/checks/${checkId}/video`}
            controls
            onCanPlay={() => onAvailabilityChange?.(true)}
            onError={() => {
              setMediaError(true);
              onAvailabilityChange?.(false);
            }}
            onTimeUpdate={(event) => onTime(event.currentTarget.currentTime)}
          >
            Браузер не поддерживает видео.
          </video>
        ) : (
          <div className="video-unavailable">
            <strong>Исходная запись недоступна</strong>
            <p>Кадры-доказательства и решения проверяющего сохранены.</p>
            {available && (
              <Button
                variant="quiet"
                onClick={() => {
                  setMediaError(false);
                  setLoadGeneration((value) => value + 1);
                }}
              >
                Повторить загрузку записи
              </Button>
            )}
          </div>
        )}
        {overlay && available && !mediaError && (
          <svg
            className="track-overlay"
            viewBox="0 0 1 1"
            preserveAspectRatio="none"
            aria-label="Разметка объектов"
          >
            {active.map((track) => {
              const sample = nearestVisibleSample(track.samples ?? [], time);
              if (!sample) return null;
              const [x1, y1, x2, y2] = sample.bbox;
              return (
                <g key={track.track_id}>
                  <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} />
                  <text x={x1} y={Math.max(0.025, y1 - 0.008)}>
                    {track.kind === "person"
                      ? `${track.role && track.role.value !== "unknown" ? roleLabels[track.role.value] : "Человек"} · ${track.track_id}`
                      : (objectNames[track.object_class ?? ""] ??
                        track.object_class ??
                        `Объект ${track.track_id}`)}
                  </text>
                  <title>{`Объект ${track.track_id}`}</title>
                </g>
              );
            })}
          </svg>
        )}
      </div>
      {available && !mediaError && (
        <div className="video-tools">
          <Button
            variant="quiet"
            onClick={() => seek(Math.max(0, time - 1 / fps))}
          >
            − кадр
          </Button>
          <strong>{formatTimecode(time)}</strong>
          <Button
            variant="quiet"
            onClick={() => seek(Math.min(duration, time + 1 / fps))}
          >
            + кадр
          </Button>
          <label>
            <input
              type="checkbox"
              checked={overlay}
              onChange={(event) => setOverlay(event.target.checked)}
            />{" "}
            Разметка
          </label>
        </div>
      )}
    </div>
  );
}

function focusable(container: HTMLElement) {
  return [
    ...container.querySelectorAll<HTMLElement>(
      'button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    ),
  ];
}

export function ModalSurface({
  open,
  title,
  onClose,
  children,
  className = "",
}: {
  open: boolean;
  title: string;
  onClose: () => void;
  children: React.ReactNode;
  className?: string;
}) {
  const container = useRef<HTMLElement>(null);
  const returnFocus = useRef<HTMLElement | null>(null);
  useEffect(() => {
    if (!open) return;
    returnFocus.current = document.activeElement as HTMLElement;
    requestAnimationFrame(() => focusable(container.current!)[0]?.focus());
    const keydown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
      if (event.key === "Tab" && container.current) {
        const items = focusable(container.current);
        if (!items.length) return;
        const first = items[0];
        const last = items[items.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener("keydown", keydown);
    return () => {
      document.removeEventListener("keydown", keydown);
      returnFocus.current?.focus();
    };
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div
      className="dialog-backdrop"
      onMouseDown={(event) => event.target === event.currentTarget && onClose()}
    >
      <section
        ref={container}
        className={className}
        role="dialog"
        aria-modal="true"
        aria-labelledby="modal-title"
      >
        <header>
          <h2 id="modal-title">{title}</h2>
          <button
            className="icon-button"
            aria-label="Закрыть"
            onClick={onClose}
          >
            ×
          </button>
        </header>
        {children}
      </section>
    </div>
  );
}

export function Dialog(
  props: Omit<React.ComponentProps<typeof ModalSurface>, "className">,
) {
  return <ModalSurface {...props} />;
}

export function Empty({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="empty">
      <span className="empty__mark" aria-hidden="true">
        ⌁
      </span>
      <h2>{title}</h2>
      <p>{children}</p>
    </div>
  );
}
