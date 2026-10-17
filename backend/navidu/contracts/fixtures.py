"""Synthetic contract fixtures only; never load these from the production pipeline."""

import base64
import io
import json
from datetime import datetime, timezone

from PIL import Image

from .api_v1 import JobDone, JobFailed, JobProcessing, JuryEvidenceFrame, JuryViolation
from .domain import (
    Coverage,
    EvidenceFrame,
    Finding,
    FindingSet,
    Observation,
    RuleStatus,
    SceneLayout,
    TimeWindow,
    Track,
    TrackRole,
    VideoMeta,
    Zone,
)
from .errors import AnalysisError, JobError
from .generate import ROOT
from .rules import RulesResponse, RuleView, SourceClauseView, SourceDocument
from .ui_api import (
    CheckDetail,
    CheckDetailResponse,
    CheckProgress,
    FindingDiff,
    ReevaluateResponse,
)
from .units import finding_id, frame_id


def generate_fixtures():
    now = datetime(2026, 10, 17, 7, 30, tzinfo=timezone.utc)
    check_id = "7c1e2a9b40d3"
    image = Image.new("RGB", (64, 36), "#4A0221")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    observations = [
        Observation(obs_id="ob-1", kind="cash_in_hands", start_sec=41.5, end_sec=47.5,
                    subjects=["p1", "o1"], zones=["z-cash_desk-1"], score=0.87,
                    frame_ids=["fr-44200"]),
        Observation(obs_id="ob-2", kind="person_in_zone", start_sec=90, end_sec=96,
                    subjects=["p2"], zones=["z-cashier_post-1"], score=0.84,
                    frame_ids=["fr-92000"]),
    ]
    findings = []
    for rule, label, start, end, sec, obs, text, norm in [
        ("V-COUNT-ABSENT", "Пересчёт денег без клиента у кассы", 41.5, 47.5, 44.2, "ob-1",
         "Кассир держит деньги, место клиента у окна видно и свободно.", "S231-p20"),
        ("V-OUTSIDER", "Посторонний в рабочей зоне кассира", 90.0, 96.0, 92.0, "ob-2",
         "Человек с неуточнённой ролью находится в рабочей зоне кассира.", "S14-p15"),
    ]:
        findings.append(Finding(
            finding_id=finding_id(rule, start, end), rule_id=rule, type_label=label,
            timecode_sec=sec, window=TimeWindow(start_sec=start, end_sec=end), basis=[obs],
            what_seen=text, why_may_violate="Условие требует проверки человеком.",
            norm_refs=[norm], description=text + " Синтетический пример структуры ответа.",
            evidence=[EvidenceFrame(frame_id=frame_id(sec), timestamp_sec=sec)],
            strength="moderate",
        ))
    finding_set = FindingSet(
        revision=1, basis="auto", created_at=now, layout_revision=1, verdict="violation",
        findings=findings, rules=[
            RuleStatus(rule_id="V-COUNT-ABSENT", status="partly_checkable",
                       not_checkable_windows=[{"start_sec": 72, "end_sec": 80,
                                               "reason": "Стол перекрыт."}]),
            RuleStatus(rule_id="V-OUTSIDER", status="violation"),
        ],
    )
    layout = SceneLayout(revision=1, zones=[
        Zone(zone_id="z-cash_desk-1", kind="cash_desk",
             polygon=[(0.1, 0.4), (0.7, 0.4), (0.7, 0.9), (0.1, 0.9)]),
        Zone(zone_id="z-client_window-1", kind="client_window",
             polygon=[(0.3, 0.1), (0.7, 0.1), (0.7, 0.39), (0.3, 0.39)]),
    ])
    detail = CheckDetail(
        id=check_id, file_name="synthetic-contract-example.mov", status="done", created_at=now,
        duration_sec=125.04, verdict="violation", findings_count=2, is_example=True,
        video=VideoMeta(duration_sec=125.04, fps_nominal=29.97, width=1920, height=1080),
        progress=CheckProgress(phase="done", processed_until_sec=125.04), layout=layout,
        tracks=[
            Track(track_id="p1", kind="person", first_sec=0, last_sec=125,
                  role=TrackRole(value="cashier")),
            Track(track_id="p2", kind="person", first_sec=90, last_sec=96,
                  role=TrackRole(value="unknown")),
        ], observations=observations,
        coverage=[Coverage(start_sec=72, end_sec=80, scope="desk", reason="occluded")],
        finding_set=finding_set,
    )
    jury_done = JobDone(job_id=check_id, verdict="violation", violations=[
        JuryViolation(type=f.type_label, timecode_sec=f.timecode_sec, description=f.description,
                      evidence_frames=[JuryEvidenceFrame(timestamp_sec=f.timecode_sec,
                                                         image_base64=encoded)])
        for f in findings
    ])
    processing = detail.model_copy(deep=True)
    processing.status = "processing"
    processing.verdict = None
    processing.findings_count = None
    processing.finding_set = None
    processing.progress = CheckProgress(phase="perception", processed_until_sec=48)
    processing.observations = observations[:1]
    failed = processing.model_copy(deep=True)
    failed.status = "failed"
    failed.progress.phase = "failed"
    failed.error = AnalysisError(code="VIDEO_DECODE_FAILED", message="Не удалось прочитать кадр.",
                                 failed_phase="perception", at_sec=48)
    reevaluated = finding_set.model_copy(deep=True)
    reevaluated.revision = 2
    reevaluated.basis = "reviewer"
    reevaluated.findings = findings[:1]
    source = SourceClauseView(
        clause_id="S231-p20", document=SourceDocument(
            title="Правила осуществления кассовых операций", act="Постановление № 231",
            registry_code="V1900019680", url="https://adilet.zan.kz/rus/docs/V1900019680",
            edition_checked="2026-10-17", checked_at="2026-10-17",
        ), locator="пункт 20", quote=None, verified_verbatim=False,
        visual_test="Присутствие клиента у кассы.", not_visual="Сумма операции.",
    )
    values = {
        "v1_done_violation": jury_done,
        "v1_done_clean": JobDone(job_id=check_id, verdict="no_violation", violations=[]),
        "v1_processing": JobProcessing(job_id=check_id),
        "v1_failed": JobFailed(job_id=check_id, error=JobError(
            code="VIDEO_DECODE_FAILED", message="Не удалось прочитать кадр."
        )),
        "ui_check_processing": CheckDetailResponse(check=processing),
        "ui_check_done": CheckDetailResponse(check=detail),
        "ui_check_failed": CheckDetailResponse(check=failed),
        "ui_reevaluate_diff": ReevaluateResponse(finding_set=reevaluated, diff=FindingDiff(
            added=[], removed=[findings[1].finding_id], changed=[], verdict_changed=False
        )),
        "ui_rules": RulesResponse(rules=[RuleView(
            rule_id="V-COUNT-ABSENT", type_label=findings[0].type_label, norm_refs=["S231-p20"],
            requires_visible=["cashier", "desk"], visible_sign="Клиента у кассы не видно.",
            why_may_violate="Пример формата; не реальный результат анализа.",
            not_claimed="Сумма и документы.", enabled=True, sources=[source],
        )], not_checkable=[]),
    }
    (ROOT / "fixtures").mkdir(exist_ok=True)
    for name, value in values.items():
        (ROOT / "fixtures" / f"{name}.json").write_text(
            json.dumps(value.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    imports = sorted({type(value).__name__ for value in values.values()})
    (ROOT / "ts").mkdir(exist_ok=True)
    (ROOT / "ts" / "fixtures.gen.ts").write_text(
        "// Synthetic contract examples, never production analysis.\n"
        + "import type { " + ", ".join(imports) + " } from './types.gen';\n\n"
        + "\n\n".join(
            f"export const {name}: {type(value).__name__} = "
            + json.dumps(value.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + ";"
            for name, value in values.items()
        ) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    generate_fixtures()
