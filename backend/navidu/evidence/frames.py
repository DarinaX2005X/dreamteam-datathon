"""Extract real source frames and save original plus deterministic overlay JPEGs."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from navidu.contracts.domain import (
    EvidenceFrame,
    Finding,
    Observation,
    OverlayBox,
    TimeWindow,
    Track,
)
from navidu.contracts.units import frame_id
from navidu.ingest.frames import frame_at
from navidu.ingest.probe import ProbeResult
from navidu.rules.describe import describe
from navidu.rules.engine import PendingFinding


def _nearest_box(track: Track, timestamp_sec: float):
    if not track.samples:
        return None
    nearest = min(track.samples, key=lambda sample: abs(sample.t - timestamp_sec))
    # Sparse semantic object boxes must never be carried onto distant frames.
    return nearest.bbox if abs(nearest.t - timestamp_sec) <= .75 else None


def _resize(image: Image.Image) -> Image.Image:
    if max(image.size) <= 1280:
        return image
    scale = 1280 / max(image.size)
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)


def materialize_findings(
    video_path: Path,
    pending: list[PendingFinding],
    observations: list[Observation],
    tracks: list[Track],
    *,
    work_dir: Path,
    probe: ProbeResult,
    overlay: bool = True,
    evidence_dir_name: str = "evidence",
) -> list[Finding]:
    originals = work_dir / evidence_dir_name / "original"
    overlays = work_dir / evidence_dir_name / "overlay"
    originals.mkdir(parents=True, exist_ok=True)
    overlays.mkdir(parents=True, exist_ok=True)
    obs_by_id = {item.obs_id: item for item in observations}
    tracks_by_id = {item.track_id: item for item in tracks}
    result: list[Finding] = []
    for item in pending:
        decoded = frame_at(video_path, item.timecode_sec, probe=probe)
        actual = round(decoded.timestamp_sec, 3)
        identifier = frame_id(actual)
        image = _resize(Image.fromarray(decoded.rgb, mode="RGB"))
        original_path = originals / f"{identifier}.jpg"
        overlay_path = overlays / f"{identifier}.jpg"
        image.save(original_path, format="JPEG", quality=85, optimize=True)
        boxes: list[OverlayBox] = []
        for obs_id in item.basis:
            observation = obs_by_id.get(obs_id)
            if not observation:
                continue
            for subject in observation.subjects:
                track = tracks_by_id.get(subject)
                bbox = _nearest_box(track, actual) if track else None
                if bbox:
                    boxes.append(OverlayBox(bbox=bbox, label=observation.kind, tone="observation"))
        rendered = image.copy()
        if overlay and boxes:
            draw = ImageDraw.Draw(rendered)
            font = ImageFont.load_default()
            for box in boxes:
                x1, y1, x2, y2 = box.bbox
                pixels = (round(x1 * image.width), round(y1 * image.height),
                          round(x2 * image.width), round(y2 * image.height))
                draw.rectangle(pixels, outline="#61CFFF", width=max(2, image.width // 500))
                draw.text((pixels[0] + 3, pixels[1] + 3), box.label, fill="#61CFFF", font=font)
        rendered.save(overlay_path, format="JPEG", quality=85, optimize=True)
        what_seen, why, description = describe(item, obs_by_id)
        window = TimeWindow(start_sec=min(item.window.start_sec, actual),
                            end_sec=max(item.window.end_sec, actual))
        result.append(Finding(
            finding_id=item.finding_id, rule_id=item.rule.rule_id,
            type_label=item.rule.type_label, timecode_sec=actual, window=window,
            basis=list(item.basis), what_seen=what_seen, why_may_violate=why,
            norm_refs=list(item.rule.norm_refs), description=description,
            evidence=[EvidenceFrame(frame_id=identifier, timestamp_sec=actual,
                                    overlay_boxes=boxes)],
            strength=item.rule.strength,
        ))
    return sorted(result, key=lambda finding: finding.timecode_sec)
