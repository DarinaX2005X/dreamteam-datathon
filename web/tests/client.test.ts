import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  parseApiErrorBody,
  pollCheck,
  type PollEvent,
} from "../src/api/client";

class FakeDocument extends EventTarget {
  visibilityState: DocumentVisibilityState = "visible";
}

function checkResponse(status: "processing" | "done" = "processing") {
  return new Response(
    JSON.stringify({
      check: {
        id: "check-1",
        file_name: "test.mov",
        status,
        created_at: "2026-10-17T00:00:00Z",
        progress: { phase: status === "done" ? "done" : "perception" },
      },
    }),
    { status: 200, headers: { "Content-Type": "application/json" } },
  );
}

describe("API errors", () => {
  it("reads the shared {error} envelope used by fetch and upload", () => {
    const error = parseApiErrorBody(
      {
        error: {
          code: "FILE_TOO_LARGE",
          message: "Слишком большой",
          field_issues: ["file"],
        },
      },
      413,
      "fallback",
    );
    expect(error).toMatchObject({
      code: "FILE_TOO_LARGE",
      message: "Слишком большой",
      status: 413,
      fieldIssues: ["file"],
    });
  });
});

describe("pollCheck", () => {
  let document: FakeDocument;
  beforeEach(() => {
    vi.useFakeTimers();
    document = new FakeDocument();
    vi.stubGlobal("document", document);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("does not overlap requests and ignores a late response after abort", async () => {
    let resolve!: (response: Response) => void;
    const fetchMock = vi.fn(
      () =>
        new Promise<Response>((done) => {
          resolve = done;
        }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const events: PollEvent[] = [];
    const controller = new AbortController();
    pollCheck("check-1", (event) => events.push(event), controller.signal);
    document.dispatchEvent(new Event("visibilitychange"));
    expect(fetchMock).toHaveBeenCalledTimes(1);
    controller.abort();
    resolve(checkResponse());
    await Promise.resolve();
    await Promise.resolve();
    expect(events).toEqual([]);
  });

  it("stops permanently on 404", async () => {
    const fetchMock = vi.fn(
      async () =>
        new Response(
          JSON.stringify({
            error: { code: "NOT_FOUND", message: "Нет проверки" },
          }),
          { status: 404, headers: { "Content-Type": "application/json" } },
        ),
    );
    vi.stubGlobal("fetch", fetchMock);
    const events: PollEvent[] = [];
    pollCheck(
      "missing",
      (event) => events.push(event),
      new AbortController().signal,
    );
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    document.dispatchEvent(new Event("visibilitychange"));
    await vi.runAllTimersAsync();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(events[0]).toMatchObject({ type: "error", recoverable: false });
  });

  it("backs off and retries recoverable network failures", async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce(checkResponse("done"));
    vi.stubGlobal("fetch", fetchMock);
    const events: PollEvent[] = [];
    pollCheck(
      "check-1",
      (event) => events.push(event),
      new AbortController().signal,
      { intervalMs: 100, maxBackoffMs: 500 },
    );
    await vi.waitFor(() => expect(events).toHaveLength(1));
    expect(events[0]).toMatchObject({
      type: "error",
      recoverable: true,
      nextRetryMs: 200,
    });
    await vi.advanceTimersByTimeAsync(200);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(events.at(-1)?.type).toBe("data");
  });
});
