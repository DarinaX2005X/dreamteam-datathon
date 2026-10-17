import type { Role, RoleOverride } from "../../../../contracts/ts/types.gen";

export function mergeRoleOverrides(
  existing: RoleOverride[],
  next: { track_id: string; role: Role },
): RoleOverride[] {
  const withoutTrack = existing.filter(
    (item) => item.track_id !== next.track_id,
  );
  return [...withoutTrack, next];
}

export type DecisionDraft = {
  value: "confirmed" | "rejected" | "not_visible" | null;
  note: string;
};

export function decisionPayload(draft: DecisionDraft, baseRevision: number) {
  return {
    decision: draft.value,
    note: draft.note,
    base_revision: baseRevision,
  };
}
