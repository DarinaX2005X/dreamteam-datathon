import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "../src/api/client";
import { waitForReady } from "../src/api/waitForReady";

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe("service wakeup", () => {
  it("bounds a stalled health request by the wakeup deadline", async () => {
    vi.useFakeTimers();
    vi.spyOn(api, "health").mockImplementation(
      (signal) =>
        new Promise((_, reject) => {
          signal?.addEventListener(
            "abort",
            () => reject(new DOMException("Aborted", "AbortError")),
            { once: true },
          );
        }),
    );
    const pending = waitForReady(new AbortController().signal);
    const assertion = expect(pending).rejects.toMatchObject({
      code: "NOT_READY",
    });
    await vi.advanceTimersByTimeAsync(180_000);
    await assertion;
  });

  it("waits until ready and stops polling", async () => {
    vi.useFakeTimers();
    const health = vi
      .spyOn(api, "health")
      .mockResolvedValueOnce({ ready: false } as Awaited<
        ReturnType<typeof api.health>
      >)
      .mockResolvedValue({ ready: true } as Awaited<
        ReturnType<typeof api.health>
      >);
    const pending = waitForReady(new AbortController().signal);
    await vi.advanceTimersByTimeAsync(3000);
    await pending;
    await vi.advanceTimersByTimeAsync(30000);
    expect(health).toHaveBeenCalledTimes(2);
  });

  it("cancels without further health requests", async () => {
    vi.useFakeTimers();
    const health = vi
      .spyOn(api, "health")
      .mockResolvedValue({ ready: false } as Awaited<
        ReturnType<typeof api.health>
      >);
    const controller = new AbortController();
    const pending = waitForReady(controller.signal);
    const assertion = expect(pending).rejects.toMatchObject({
      name: "AbortError",
    });
    await vi.advanceTimersByTimeAsync(0);
    controller.abort();
    await assertion;
    await vi.advanceTimersByTimeAsync(10000);
    expect(health).toHaveBeenCalledTimes(1);
  });
});
