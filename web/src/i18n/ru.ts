export const common = {
  newCheck: "Новая проверка",
  norms: "Нормы",
  sheet: "Лист проверки",
  loading: "Загрузка…",
  retry: "Повторить",
  back: "Назад",
  error: "Не удалось получить данные",
};
export const errors: Record<string, string> = {
  UNSUPPORTED_MEDIA:
    "Формат записи не поддерживается. Выберите .mov, .mp4 или .m4v.",
  FILE_TOO_LARGE: "Файл больше 300 МБ.",
  VIDEO_EXPIRED: "Исходная запись удалена по сроку хранения.",
  STALE_REVISION: "Эту проверку только что изменили в другой вкладке.",
  CANCELLED: "Анализ отменён.",
  MODEL_UNAVAILABLE: "Модель анализа сейчас недоступна.",
};
export function formatTimecode(value: number) {
  const s = Math.max(0, value);
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(1).replace(".", ",");
  return `${String(m).padStart(2, "0")}:${sec.padStart(4, "0")}`;
}
export function formatDuration(value: number | null | undefined) {
  if (value == null) return "—";
  const m = Math.floor(value / 60);
  const s = Math.floor(value % 60);
  return m ? `${m} мин ${s} с` : `${s} с`;
}
