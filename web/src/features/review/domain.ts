import type {
  ReevaluateResponse,
  ReevaluateRequest,
  Role,
  RoleOverride,
} from "../../../../contracts/ts/types.gen";

export type PreviewFrames = NonNullable<ReevaluateResponse["preview_frames"]>;

export function evidenceSource(
  checkId: string,
  frameId: string,
  revision: number,
  previewFrames: PreviewFrames,
  overlay = true,
) {
  const preview = previewFrames[frameId];
  if (preview) {
    const base64 = overlay
      ? (preview.overlay_base64 ?? preview.image_base64)
      : preview.image_base64;
    return `data:image/jpeg;base64,${base64}`;
  }
  return `/api/ui/checks/${checkId}/frames/${frameId}.jpg?overlay=${overlay ? 1 : 0}&revision=${revision}`;
}

export function reevaluationBaseRevision(
  persistedRevision: number | undefined,
  displayedRevision: number,
  dryRun: boolean,
) {
  return dryRun ? (persistedRevision ?? displayedRevision) : displayedRevision;
}

export function buildReevaluateRequest({
  persistedRevision,
  displayedRevision,
  roleOverrides,
  zoneOverrides,
  dryRun,
}: {
  persistedRevision?: number;
  displayedRevision: number;
  roleOverrides: ReevaluateRequest["role_overrides"];
  zoneOverrides: ReevaluateRequest["zone_overrides"];
  dryRun: boolean;
}): ReevaluateRequest {
  return {
    base_revision: reevaluationBaseRevision(
      persistedRevision,
      displayedRevision,
      dryRun,
    ),
    role_overrides: roleOverrides,
    zone_overrides: zoneOverrides,
    dry_run: dryRun,
  };
}

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
