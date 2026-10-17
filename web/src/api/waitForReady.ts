import { api, ApiError } from "./client";

function pause(signal: AbortSignal, milliseconds: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const cancel = () => {
      clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    };
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", cancel);
      resolve();
    }, milliseconds);
    if (signal.aborted) cancel();
    else signal.addEventListener("abort", cancel, { once: true });
  });
}

export async function waitForReady(signal: AbortSignal): Promise<void> {
  signal.throwIfAborted();
  const controller = new AbortController();
  const cancel = () => controller.abort();
  signal.addEventListener("abort", cancel, { once: true });
  let expired = false;
  const timer = setTimeout(() => {
    expired = true;
    controller.abort();
  }, 180_000);
  try {
    while (true) {
      controller.signal.throwIfAborted();
      if ((await api.health(controller.signal)).ready) return;
      await pause(controller.signal, 3000);
    }
  } catch (error) {
    if (expired) {
      throw new ApiError(
        "NOT_READY",
        "Сервис ещё запускается. Повторите отправку через минуту.",
        503,
      );
    }
    throw error;
  } finally {
    clearTimeout(timer);
    signal.removeEventListener("abort", cancel);
  }
}
