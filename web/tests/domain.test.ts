import { describe, expect, it } from "vitest";
import {
  decisionPayload,
  buildReevaluateRequest,
  evidenceSource,
  mergeRoleOverrides,
  reevaluationBaseRevision,
} from "../src/features/review/domain";
import {
  readRecentIds,
  readRecentStorage,
  recentStatus,
  shouldDiscardUpload,
  writeRecentStorage,
} from "../src/features/intake/domain";
import { groupObservations, nearestVisibleSample } from "../src/ui";
import {
  ruleIdFromHash,
  ruleMatchesQuery,
} from "../src/features/norms/NormsPage";

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
  it("keeps the persisted base revision across repeated dry runs", () => {
    expect(reevaluationBaseRevision(3, 4, true)).toBe(3);
    expect(reevaluationBaseRevision(3, 4, false)).toBe(4);
  });
  it("preserves accumulated preview overrides while using the persisted base", () => {
    const request = buildReevaluateRequest({
      persistedRevision: 3,
      displayedRevision: 4,
      roleOverrides: [
        { track_id: "p1", role: "cashier" },
        { track_id: "p2", role: "collector" },
      ],
      zoneOverrides: [
        {
          zone_id: "z-cash_desk-1",
          kind: "cash_desk",
          polygon: [
            [0.1, 0.1],
            [0.8, 0.1],
            [0.8, 0.6],
          ],
          source: "reviewer",
        },
      ],
      dryRun: true,
    });
    expect(request.base_revision).toBe(3);
    expect(request.role_overrides).toHaveLength(2);
    expect(request.zone_overrides).toHaveLength(1);
  });
  it("uses preview JPEGs and a revision-pinned durable fallback", () => {
    const previews = {
      frame1: { image_base64: "raw", overlay_base64: "marked" },
      frame2: { image_base64: "raw-only", overlay_base64: null },
    };
    expect(evidenceSource("check", "frame1", 3, previews)).toBe(
      "data:image/jpeg;base64,marked",
    );
    expect(evidenceSource("check", "frame2", 3, previews)).toBe(
      "data:image/jpeg;base64,raw-only",
    );
    expect(evidenceSource("check", "old", 3, previews)).toBe(
      "/api/ui/checks/check/frames/old.jpg?overlay=1&revision=3",
    );
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
  it("uses an in-memory fallback when storage access is denied", () => {
    const denied = {
      getItem: () => {
        throw new DOMException("denied", "SecurityError");
      },
      setItem: () => {
        throw new DOMException("denied", "SecurityError");
      },
    };
    expect(readRecentStorage(denied, ["memory-id"])).toEqual(["memory-id"]);
    expect(writeRecentStorage(denied, ["memory-id"])).toBe(false);
  });
  it("keeps the selected file for retryable upload errors", () => {
    expect(shouldDiscardUpload("NETWORK")).toBe(false);
    expect(shouldDiscardUpload("RATE_LIMITED")).toBe(false);
    expect(shouldDiscardUpload("UNSUPPORTED_MEDIA")).toBe(true);
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

describe("video tracking overlay", () => {
  const samples = [
    {
      t: 1,
      bbox: [0.1, 0.1, 0.3, 0.5] as [number, number, number, number],
      score: 0.9,
    },
    {
      t: 1.5,
      bbox: [0.2, 0.1, 0.4, 0.5] as [number, number, number, number],
      score: 0.9,
    },
    {
      t: 8,
      bbox: [0.6, 0.1, 0.8, 0.5] as [number, number, number, number],
      score: 0.9,
    },
  ];

  it("uses a measured box close to the current frame", () => {
    expect(nearestVisibleSample(samples, 1.2)?.t).toBe(1);
  });

  it("hides a stale box across a long tracking gap", () => {
    expect(nearestVisibleSample(samples, 4)).toBeUndefined();
  });
});

describe("norms reference", () => {
  const rule = {
    rule_id: "V-COUNT-ABSENT",
    type_label: "Пересчёт денег",
    norm_refs: ["clause-1"],
    requires_visible: ["desk" as const],
    visible_sign: "Деньги переданы без пересчёта",
    why_may_violate: "Пересчёт не виден",
    not_claimed: "Не определяет сумму",
    enabled: true,
    sources: [
      {
        clause_id: "clause-1",
        document: {
          title: "Положение о кассовых операциях",
          act: "Положение № 318-П",
          registry_code: "318-П",
          url: "https://example.test/rule",
          edition_checked: "2026-01-01",
        },
        locator: "пункт 5.11",
        visual_test: "Виден пересчёт",
        not_visual: "Сумма не видна",
      },
    ],
  };

  it("finds rules by document and clause", () => {
    expect(ruleMatchesQuery(rule, "318-П")).toBe(true);
    expect(ruleMatchesQuery(rule, "5.11")).toBe(true);
    expect(ruleMatchesQuery(rule, "упаковка сумки")).toBe(false);
  });

  it("decodes a deep-linked rule id", () => {
    expect(ruleIdFromHash("#V-COUNT-%41BSENT")).toBe("V-COUNT-ABSENT");
  });
});
