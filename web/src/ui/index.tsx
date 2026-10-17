import React, { useEffect, useMemo, useRef, useState } from "react";
import type {
  Coverage,
  Finding,
  Observation,
  Track,
} from "../../../contracts/ts/types.gen";
import { formatTimecode } from "../i18n/ru";

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

const factNames: Record<string, string> = {
  person_in_zone: "Человек в зоне",
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
  const [table, setTable] = useState(false);
  const groups = useMemo(() => groupObservations(observations), [observations]);
  const pct = (value: number) =>
    `${Math.min(100, Math.max(0, (value / duration) * 100))}%`;
  const keyboardSeek = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (
      event.key !== "ArrowLeft" &&
      event.key !== "ArrowRight" &&
      event.key !== "Home" &&
      event.key !== "End"
    )
      return;
    event.preventDefault();
    const next =
      event.key === "Home"
        ? 0
        : event.key === "End"
          ? duration
          : time + (event.key === "ArrowLeft" ? -1 : 1);
    onSeek(Math.max(0, Math.min(duration, next)));
  };
  return (
    <section className="fact-strip" aria-label="Лента фактов">
      <div className="fact-strip__head">
        <h3>Наблюдаемые факты</h3>
        <button className="text-button" onClick={() => setTable(!table)}>
          {table ? "Показать лентой" : "Показать таблицей"}
        </button>
      </div>
      {table ? (
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
                <td>{formatTimecode(item.start_sec)}</td>
                <td>{formatTimecode(item.end_sec)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div
          className="lanes"
          style={{ height: Math.max(150, groups.length * 30) }}
          role="slider"
          tabIndex={0}
          aria-label="Положение на записи"
          aria-valuemin={0}
          aria-valuemax={duration}
          aria-valuenow={time}
          onKeyDown={keyboardSeek}
          onClick={(event) => {
            const bounds = event.currentTarget.getBoundingClientRect();
            onSeek(((event.clientX - bounds.left) / bounds.width) * duration);
          }}
        >
          {groups.map((group) => (
            <div className="lane" key={group.key}>
              <span>{group.label}</span>
              {group.observations.map((item) => (
                <i
                  key={item.obs_id}
                  style={{
                    left: pct(item.start_sec),
                    width: pct(item.end_sec - item.start_sec),
                  }}
                />
              ))}
            </div>
          ))}
          {coverage.map((item, index) => (
            <i
              key={index}
              className="coverage"
              title={`Нельзя проверить: ${item.reason}`}
              style={{
                left: pct(item.start_sec),
                width: pct(item.end_sec - item.start_sec),
              }}
            />
          ))}
          {findings.map((item) => (
            <button
              key={item.finding_id}
              className={`finding-dot ${item.decision?.value === "rejected" ? "is-rejected" : ""}`}
              style={{ left: pct(item.timecode_sec) }}
              aria-label={`${item.type_label}, ${formatTimecode(item.timecode_sec)}`}
              onClick={(event) => {
                event.stopPropagation();
                onSelect?.(item.finding_id);
              }}
            />
          ))}
          <i className="playhead" style={{ left: pct(time) }} />
        </div>
      )}
    </section>
  );
}

export function VideoStage({
  checkId,
  duration = 1,
  fps = 25,
  width = 16,
  height = 9,
  tracks,
  time,
  seekToken,
  onTime,
  onTrack,
}: {
  checkId: string;
  duration?: number;
  fps?: number;
  width?: number;
  height?: number;
  tracks: Track[];
  time: number;
  seekToken: number;
  onTime: (value: number) => void;
  onTrack?: (id: string) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [overlay, setOverlay] = useState(true);
  useEffect(() => {
    if (videoRef.current)
      videoRef.current.currentTime = Math.max(0, Math.min(duration, time));
  }, [duration, seekToken]);
  const active = tracks.filter(
    (track) => track.first_sec <= time && track.last_sec >= time,
  );
  const seek = (value: number) => {
    if (videoRef.current) videoRef.current.currentTime = value;
    onTime(value);
  };
  return (
    <div className="video-stage" data-surface="stage">
      <div
        className="video-frame"
        style={{ aspectRatio: `${width}/${height}` }}
      >
        <video
          ref={videoRef}
          src={`/api/ui/checks/${checkId}/video`}
          controls
          onTimeUpdate={(event) => onTime(event.currentTarget.currentTime)}
        >
          Браузер не поддерживает видео.
        </video>
        {overlay && (
          <svg
            className="track-overlay"
            viewBox="0 0 1 1"
            preserveAspectRatio="none"
            aria-label="Разметка объектов"
          >
            {active.map((track) => {
              const sample = [...(track.samples ?? [])].sort(
                (a, b) => Math.abs(a.t - time) - Math.abs(b.t - time),
              )[0];
              if (!sample) return null;
              const [x1, y1, x2, y2] = sample.bbox;
              return (
                <g key={track.track_id}>
                  <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} />
                  <title>{`Объект ${track.track_id}`}</title>
                </g>
              );
            })}
          </svg>
        )}
      </div>
      <div className="video-tools">
        <Button
          variant="quiet"
          onClick={() => {
            const video = videoRef.current;
            if (video) void (video.paused ? video.play() : video.pause());
          }}
        >
          Воспроизвести / пауза
        </Button>
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
      {onTrack && active.some((track) => track.kind === "person") && (
        <div className="track-list" aria-label="Объекты в текущем кадре">
          {active
            .filter((track) => track.kind === "person")
            .map((track) => (
              <button
                key={track.track_id}
                onClick={() => onTrack(track.track_id)}
              >
                Уточнить роль {track.track_id}
              </button>
            ))}
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
