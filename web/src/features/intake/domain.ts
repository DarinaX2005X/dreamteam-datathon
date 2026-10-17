export function readRecentIds(raw: string | null): string[] {
  if (!raw) return [];
  try {
    const value: unknown = JSON.parse(raw);
    return Array.isArray(value)
      ? value.filter((item): item is string => typeof item === "string")
      : [];
  } catch {
    return [];
  }
}

export function recentStatus(status: string, verdict?: string | null) {
  if (status === "done")
    return verdict === "violation" ? "Есть замечания" : "Без замечаний";
  if (status === "processing") return "Анализируется";
  if (status === "cancelled") return "Отменена";
  if (status === "failed") return "Завершилась с ошибкой";
  return "В очереди";
}
