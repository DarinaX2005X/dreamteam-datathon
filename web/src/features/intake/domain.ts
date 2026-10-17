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

export function readRecentStorage(
  storage: Pick<Storage, "getItem"> | undefined,
  fallback: string[] = [],
) {
  try {
    return storage
      ? readRecentIds(storage.getItem("lado.recentChecks"))
      : fallback;
  } catch {
    return fallback;
  }
}

export function writeRecentStorage(
  storage: Pick<Storage, "setItem"> | undefined,
  ids: string[],
) {
  try {
    storage?.setItem("lado.recentChecks", JSON.stringify(ids));
    return Boolean(storage);
  } catch {
    return false;
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

export function shouldDiscardUpload(code: string) {
  return ["UNSUPPORTED_MEDIA", "FILE_TOO_LARGE", "INVALID_VIDEO"].includes(
    code,
  );
}
