import { describe, expect, it } from "vitest";
import {
  decisionPayload,
  mergeRoleOverrides,
} from "../src/features/review/domain";
import { readRecentIds, recentStatus } from "../src/features/intake/domain";
import { groupObservations } from "../src/ui";

describe("review mutations", () => {
  it("replaces one role without losing existing overrides", () => {
    expect(
      mergeRoleOverrides(
        [
          { track_id: "p1", role: "cashier" },
          { track_id: "p2", role: "client" },
        ],
        { track_id: "p2", role: "collector" },
      ),
    ).toEqual([
      { track_id: "p1", role: "cashier" },
      { track_id: "p2", role: "collector" },
    ]);
  });
  it("allows a note before a decision and preserves the revision", () => {
    expect(
      decisionPayload({ value: null, note: "Нужна ручная проверка" }, 4),
    ).toEqual({
      decision: null,
      note: "Нужна ручная проверка",
      base_revision: 4,
    });
  });
  it("represents reset as null decision and empty note", () => {
    expect(decisionPayload({ value: null, note: "" }, 5)).toEqual({
      decision: null,
      note: "",
      base_revision: 5,
    });
  });
});

describe("recovery helpers", () => {
  it("recovers from corrupt local storage", () => {
    expect(readRecentIds("{broken")).toEqual([]);
    expect(readRecentIds('["ok", 3, null]')).toEqual(["ok"]);
  });
  it("does not describe failed checks as queued", () => {
    expect(recentStatus("failed")).toBe("Завершилась с ошибкой");
  });
});

describe("fact grouping", () => {
  it("groups repeated observations into one subject lane", () => {
    const observations = [
      {
        obs_id: "o1",
        kind: "cash_in_hands" as const,
        start_sec: 1,
        end_sec: 2,
        subjects: ["p1"],
      },
      {
        obs_id: "o2",
        kind: "cash_in_hands" as const,
        start_sec: 4,
        end_sec: 5,
        subjects: ["p1"],
      },
      {
        obs_id: "o3",
        kind: "cash_in_hands" as const,
        start_sec: 6,
        end_sec: 7,
        subjects: ["p2"],
      },
    ];
    expect(groupObservations(observations)).toHaveLength(2);
    expect(groupObservations(observations)[0].observations).toHaveLength(2);
  });
});
