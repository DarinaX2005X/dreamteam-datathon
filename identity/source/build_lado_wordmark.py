"""Build the Lado logo lockups while preserving the existing mark exactly.

Usage:
    $env:OPEN_SANS_TTF = "path/to/OpenSans[wdth,wght].ttf"
    python identity/source/build_lado_wordmark.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont


ROOT = Path(__file__).resolve().parents[1]
LOGO = ROOT / "logo"
PRODUCT = ROOT.parent / "contracts" / "product.json"
NAME = json.loads(PRODUCT.read_text(encoding="utf-8"))["name"]
WEIGHT = 780
WIDTH = 100


def wordmark_path(font_path: Path, font_size: float) -> tuple[str, float, float, float]:
    font = TTFont(font_path)
    if "fvar" in font:
        font = instantiateVariableFont(font, {"wght": WEIGHT, "wdth": WIDTH})
    units = font["head"].unitsPerEm
    scale = font_size / units
    glyphs = font.getGlyphSet()
    cmap = font.getBestCmap()
    metrics = font["hmtx"]
    os2 = font["OS/2"]
    tracking = -0.012 * units
    pen = SVGPathPen(glyphs)
    x = 0.0
    names = [cmap[ord(character)] for character in NAME]
    for index, glyph_name in enumerate(names):
        transformed = TransformPen(pen, (scale, 0, 0, -scale, x * scale, 0))
        glyphs[glyph_name].draw(transformed)
        x += metrics[glyph_name][0] + (tracking if index < len(names) - 1 else 0)
    return (
        pen.getCommands(),
        x * scale,
        getattr(os2, "sxHeight", 1096) * scale,
        abs(os2.sTypoDescender) * scale,
    )


def mark_group(bracket: str, strip: str, gradient_id: str, mono: bool = False) -> str:
    gradient = "" if mono else (
        f'<defs><linearGradient id="{gradient_id}" x1="28" y1="39" x2="44" y2="28" gradientUnits="userSpaceOnUse">'
        '<stop offset="0" stop-color="#4A0221"/><stop offset="0.5" stop-color="#A31551"/>'
        '<stop offset="1" stop-color="#FC9385"/></linearGradient></defs>'
    )
    fragment = strip if mono else f"url(#{gradient_id})"
    return gradient + (
        f'<g fill="none" stroke="{bracket}" stroke-width="6.5" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M11 23V17a6 6 0 0 1 6-6h6"/><path d="M41 11h6a6 6 0 0 1 6 6v6"/>'
        '<path d="M53 41v6a6 6 0 0 1-6 6h-6"/><path d="M23 53h-6a6 6 0 0 1-6-6v-6"/></g>'
        f'<rect x="18" y="31" width="28" height="5" rx="2.5" fill="{strip}"/>'
        f'<rect x="28" y="28" width="16" height="11" rx="3.5" fill="{fragment}"/>'
        f'<path d="M31.6 19.5h8.8a1.2 1.2 0 0 1 .95 1.95l-4.4 5.2a1.2 1.2 0 0 1-1.9 0l-4.4-5.2a1.2 1.2 0 0 1 .95-1.95z" fill="{strip}"/>'
    )


def svg(width: float, height: float, title: str, body: str, view_box: str | None = None) -> str:
    view = view_box or f"0 0 {width:.2f} {height:.2f}"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.2f}" height="{height:.2f}" '
        f'viewBox="{view}" role="img" aria-labelledby="title">'
        f'<title id="title">{title}</title>{body}</svg>\n'
    )


def main() -> int:
    font_path = Path(os.environ.get("OPEN_SANS_TTF", ""))
    if not font_path.is_file():
        raise SystemExit("Set OPEN_SANS_TTF to the official Open Sans variable TTF")
    size = 60.0
    path, word_width, x_height, descender = wordmark_path(font_path, size)
    mark_size = 80.0
    gap = 20.0
    baseline = mark_size / 2 + x_height / 2
    total_width = mark_size + gap + word_width
    total_height = max(mark_size, baseline + descender)

    mark_variants = {
        "lado-mark.svg": mark_group("#A31551", "#4A0221", "ladoMarkLit"),
        "lado-mark-mono.svg": mark_group("#4A0221", "#4A0221", "unused", mono=True),
        "lado-mark-on-stage.svg": '<rect width="64" height="64" rx="16" fill="#2A0B19"/>'
        + mark_group("#F7EDF1", "#CDB6C1", "ladoMarkLitStage"),
    }
    for filename, body in mark_variants.items():
        (LOGO / filename).write_text(svg(64, 64, f"{NAME}, знак", body), encoding="utf-8")

    variants = {
        "lado-logo.svg": (mark_group("#A31551", "#4A0221", "ladoLit"), "#4A0221"),
        "lado-logo-mono.svg": (mark_group("#4A0221", "#4A0221", "unused", mono=True), "#4A0221"),
        "lado-logo-on-stage.svg": (mark_group("#F7EDF1", "#CDB6C1", "ladoLitStage"), "#F7EDF1"),
    }
    for filename, (mark, fill) in variants.items():
        body = (
            f'<g transform="scale({mark_size / 64:g})">{mark}</g>'
            f'<path transform="translate({mark_size + gap:g} {baseline:.2f})" fill="{fill}" d="{path}"/>'
        )
        (LOGO / filename).write_text(svg(total_width, total_height, NAME, body), encoding="utf-8")

    metrics = {
        "name": NAME,
        "font": "Open Sans",
        "wght": WEIGHT,
        "wdth": WIDTH,
        "font_size_units": size,
        "x_height": round(x_height, 2),
        "baseline": round(baseline, 2),
        "wordmark_width": round(word_width, 2),
        "gap": gap,
        "lockup": [round(total_width, 2), round(total_height, 2)],
    }
    (LOGO / "lado-wordmark-metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
