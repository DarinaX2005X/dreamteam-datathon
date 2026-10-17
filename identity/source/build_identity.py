#!/usr/bin/env python3
"""Сборка ассетов айдентики «Навиду» из исходников.

Что делает:
  1. tokens/tokens.json  -> tokens/tokens.css (CSS custom properties, светлая тема и тема «сцены»);
  2. рисует знак и логотипы (logo/*.svg), словесный знак переводится в кривые из Open Sans
     (OFL-1.1, начертание wght=780, wdth=100), поэтому SVG не зависит от установленного шрифта;
  3. проверяет контраст основных пар цветов и печатает отчёт;
  4. рендерит PNG-превью логотипов и композиций (previews/*.png) через resvg (не браузер).

Запуск (шрифт в архив не входит, его путь передаётся явно):
  pip install fonttools resvg-py
  OPEN_SANS_TTF=/path/to/OpenSans[wdth,wght].ttf python3 identity/source/build_identity.py

Открытый вариативный файл: https://github.com/google/fonts/tree/main/ofl/opensans
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # identity/
TOKENS = ROOT / "tokens" / "tokens.json"
LOGO = ROOT / "logo"
PREVIEWS = ROOT / "previews"
COMPOSITIONS = ROOT / "compositions"
ART = ROOT / "art"

WORD = "навиду"
WGHT, WDTH = 780, 100


# ---------------------------------------------------------------- tokens
def resolve(value: str, raw: dict) -> str:
    m = re.fullmatch(r"\{([\w-]+)\.([\w-]+)\}", value)
    if not m:
        return value
    return raw[m.group(1)][m.group(2)]


def build_tokens_css(tokens: dict) -> str:
    raw = tokens["raw"]
    out = ["/* Сгенерировано identity/source/build_identity.py из tokens.json. Не редактировать вручную. */", ":root {"]
    for group, values in raw.items():
        for key, val in values.items():
            out.append(f"  --nv-raw-{group}-{key}: {val};")
    for key, val in tokens["semantic"]["light"].items():
        out.append(f"  --nv-{key}: {resolve(val, raw)};")
    for key, val in tokens["gradient"].items():
        out.append(f"  --nv-gradient-{key}: {val};")
    out.append(f"  --nv-font: {tokens['font']['family']};")
    for key, t in tokens["type"].items():
        out.append(f"  --nv-type-{key}-size: {t['size']}px;")
        out.append(f"  --nv-type-{key}-line: {t['line']}px;")
        out.append(f"  --nv-type-{key}-weight: {t['weight']};")
        out.append(f"  --nv-type-{key}-stretch: {t['stretch']};")
        out.append(f"  --nv-type-{key}-tracking: {t['tracking']};")
    for key, val in tokens["space"].items():
        out.append(f"  --nv-space-{key}: {val}px;")
    for key, val in tokens["radius"].items():
        out.append(f"  --nv-radius-{key}: {val}px;")
    for key, val in tokens["size"].items():
        out.append(f"  --nv-size-{key}: {val if isinstance(val, str) else str(val) + 'px'};")
    for key, val in tokens["elevation"].items():
        out.append(f"  --nv-elevation-{key}: {val};")
    for key, val in tokens["motion"].items():
        out.append(f"  --nv-motion-{key}: {val};")
    out.append("}")
    out.append("")
    out.append("/* Тёмная «сцена»: просмотр видео и лента фактов. Не инверсия, отдельный набор ролей. */")
    out.append("[data-surface=\"stage\"] {")
    for key, val in tokens["semantic"]["stage"].items():
        out.append(f"  --nv-{key}: {resolve(val, raw)};")
    out.append("}")
    out.append("")
    out.append("@media (prefers-reduced-motion: reduce) {")
    out.append("  :root { --nv-motion-press: 0ms; --nv-motion-hover: 0ms; --nv-motion-popover: 0ms; --nv-motion-layout: 0ms; --nv-motion-seek: 0ms; --nv-motion-press-scale: 1; }")
    out.append("}")
    out.append("")
    out.append("@media (prefers-reduced-transparency: reduce) {")
    out.append("  :root { --nv-gradient-lit-window: #FC9385; }")
    out.append("}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- contrast
def _lin(c: float) -> float:
    c /= 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_: str) -> float:
    h = hex_.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def contrast_report(tokens: dict) -> list[tuple[str, str, str, float, float]]:
    raw = tokens["raw"]
    L = {k: resolve(v, raw) for k, v in tokens["semantic"]["light"].items()}
    S = {k: resolve(v, raw) for k, v in tokens["semantic"]["stage"].items()}
    pairs = [
        ("text / surface", L["text"], L["surface"], 7.0),
        ("text / canvas", L["text"], L["canvas"], 7.0),
        ("text-secondary / surface", L["text-secondary"], L["surface"], 4.5),
        ("text-secondary / surface-inset", L["text-secondary"], L["surface-inset"], 4.5),
        ("text-on-action / action", L["text-on-action"], L["action"], 4.5),
        ("text-on-action / action-pressed", L["text-on-action"], L["action-pressed"], 4.5),
        ("action (ссылка) / surface", L["action"], L["surface"], 4.5),
        ("action / selected-bg", L["action"], L["selected-bg"], 4.5),
        ("finding-ink / finding-fill", L["finding-ink"], L["finding-fill"], 4.5),
        ("finding-text / surface", L["finding-text"], L["surface"], 4.5),
        ("observation / surface", L["observation"], L["surface"], 4.5),
        ("focus / surface (не текст)", L["focus"], L["surface"], 3.0),
        ("border-control / surface (не текст)", L["border-control"], L["surface"], 3.0),
        ("success / success-bg", L["success"], L["success-bg"], 4.5),
        ("warning / warning-bg", L["warning"], L["warning-bg"], 4.5),
        ("danger / danger-bg", L["danger"], L["danger-bg"], 4.5),
        ("stage text / stage canvas", S["text"], S["canvas"], 7.0),
        ("stage text-secondary / stage surface", S["text-secondary"], S["surface"], 4.5),
        ("stage observation / stage surface (не текст)", S["observation"], S["surface"], 3.0),
        ("stage finding-fill / stage surface (не текст)", S["finding-fill"], S["surface"], 3.0),
        ("stage focus / stage surface (не текст)", S["focus"], S["surface"], 3.0),
    ]
    return [(name, a, b, round(contrast(a, b), 2), need) for name, a, b, need in pairs]


# ---------------------------------------------------------------- mark
def mark_group(*, bracket: str, strip: str, gid: str, mono: bool = False, stroke: float = 6.5) -> str:
    """Знак: видоискатель (четыре угла кадра), лента записи, подсвеченный фрагмент и метка момента.

    Сетка 64×64. Углы: радиус 6, толщина 6.5 (8 для favicon). Лента: x 18..46, y 31..36.
    Фрагмент: x 28..44, y 28..39, радиус 3.5, градиент рассвета (Dark Magenta → Magenta → Coral).
    Метка: скруглённый треугольник вершиной вниз над фрагментом; та же форма отмечает эпизоды в интерфейсе.
    """
    corners = (
        '<path d="M11 23V17a6 6 0 0 1 6-6h6"/>'
        '<path d="M41 11h6a6 6 0 0 1 6 6v6"/>'
        '<path d="M53 41v6a6 6 0 0 1-6 6h-6"/>'
        '<path d="M23 53h-6a6 6 0 0 1-6-6v-6"/>'
    )
    if mono:
        lit_fill, defs, strip_op = bracket, "", "0.5"
    else:
        lit_fill = f"url(#{gid})"
        strip_op = "1"
        defs = (
            f'<defs><linearGradient id="{gid}" x1="28" y1="39" x2="44" y2="28" gradientUnits="userSpaceOnUse">'
            '<stop offset="0" stop-color="#4A0221"/><stop offset="0.5" stop-color="#A31551"/>'
            '<stop offset="1" stop-color="#FC9385"/></linearGradient></defs>'
        )
    marker = '<path d="M31.6 19.5h8.8a1.2 1.2 0 0 1 .95 1.95l-4.4 5.2a1.2 1.2 0 0 1-1.9 0l-4.4-5.2a1.2 1.2 0 0 1 .95-1.95z"'
    return (
        f"{defs}"
        f'<g fill="none" stroke="{bracket}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">{corners}</g>'
        f'<rect x="18" y="31" width="28" height="5" rx="2.5" fill="{strip}" opacity="{strip_op}"/>'
        f'<rect x="28" y="28" width="16" height="11" rx="3.5" fill="{lit_fill}"/>'
        f'{marker} fill="{strip}"/>'
    )


def svg_doc(width: float, height: float, body: str, title: str, view: str | None = None) -> str:
    vb = view or f"0 0 {width:g} {height:g}"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:g}" height="{height:g}" viewBox="{vb}" role="img" aria-labelledby="t">'
        f'<title id="t">{title}</title>{body}</svg>\n'
    )


# ---------------------------------------------------------------- wordmark
def wordmark_path(font_path: str, font_size: float) -> tuple[str, float, float, float]:
    """Возвращает (d, ширина, x-height, нижний вынос) для слова WORD при font_size."""
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen

    font = TTFont(font_path)
    if "fvar" in font:
        font = instantiateVariableFont(font, {"wght": WGHT, "wdth": WDTH})
    upm = font["head"].unitsPerEm
    scale = font_size / upm
    cmap = font.getBestCmap()
    gs = font.getGlyphSet()
    hmtx = font["hmtx"]
    os2 = font["OS/2"]
    x_height = getattr(os2, "sxHeight", 1096) * scale
    tracking = -0.012 * upm  # плотнее на 1.2%: словесный знак крупный
    pen = SVGPathPen(gs)
    x = 0.0
    names = [cmap[ord(ch)] for ch in WORD]
    for i, name in enumerate(names):
        tp = TransformPen(pen, (scale, 0, 0, -scale, x * scale, 0))
        gs[name].draw(tp)
        adv = hmtx[name][0]
        x += adv + (tracking if i < len(names) - 1 else 0)
    width = x * scale
    descender = abs(os2.sTypoDescender) * scale
    return pen.getCommands(), width, x_height, descender


def build_logos(font_path: str) -> dict[str, str]:
    files: dict[str, str] = {}
    ink, mag, coral = "#4A0221", "#A31551", "#FC9385"
    # знак
    files["navidu-mark.svg"] = svg_doc(64, 64, mark_group(bracket=mag, strip=ink, gid="nvLitMark"), "Навиду, знак")
    files["navidu-mark-mono.svg"] = svg_doc(64, 64, mark_group(bracket=ink, strip=ink, gid="x", mono=True), "Навиду, знак, одноцветный")
    files["navidu-mark-on-stage.svg"] = svg_doc(
        64, 64,
        '<rect width="64" height="64" rx="16" fill="#2A0B19"/>' + mark_group(bracket="#F7EDF1", strip="#CDB6C1", gid="nvLitStage"),
        "Навиду, знак на тёмной сцене",
    )
    # favicon: толще штрих, без градиента на 16px
    files["favicon.svg"] = svg_doc(
        64, 64,
        '<rect width="64" height="64" rx="14" fill="#FBF0F4"/>' + mark_group(bracket=mag, strip=ink, gid="nvLitFav", stroke=8),
        "Навиду",
    )
    # словесный знак и горизонтальная связка
    fs = 60.0
    d, w, xh, desc = wordmark_path(font_path, fs)
    M = 80.0  # знак в связке крупнее строчных: высота знака = 2.5 x-height
    baseline = M / 2 + xh / 2  # середина строчных совпадает с центром знака
    gap = 20
    total_w = M + gap + w
    total_h = max(M, baseline + desc)
    def lockup(mark_body: str, text_fill: str, bg: str = "") -> str:
        return svg_doc(
            round(total_w, 2), round(total_h, 2),
            bg + f'<g transform="scale({M / 64:g})">{mark_body}</g>'
            + f'<path transform="translate({M + gap:g} {baseline:.2f})" fill="{text_fill}" d="{d}"/>',
            "Навиду",
        )
    files["navidu-logo.svg"] = lockup(mark_group(bracket=mag, strip=ink, gid="nvLitLogo"), ink)
    files["navidu-logo-mono.svg"] = lockup(mark_group(bracket=ink, strip=ink, gid="x", mono=True), ink)
    files["navidu-logo-on-stage.svg"] = lockup(
        mark_group(bracket="#F7EDF1", strip="#CDB6C1", gid="nvLitLogoStage"), "#F7EDF1",
        bg=f'<rect x="-16" y="-12" width="{total_w + 32:.2f}" height="{total_h + 24:.2f}" rx="18" fill="#2A0B19"/>',
    )
    # фон для on-stage должен попадать во viewBox
    files["navidu-logo-on-stage.svg"] = files["navidu-logo-on-stage.svg"].replace(
        f'viewBox="0 0 {round(total_w, 2):g} {round(total_h, 2):g}"',
        f'viewBox="-16 -12 {total_w + 32:.2f} {total_h + 24:.2f}"',
    ).replace(f'width="{round(total_w, 2):g}" height="{round(total_h, 2):g}"', f'width="{total_w + 32:.2f}" height="{total_h + 24:.2f}"')
    files["wordmark-metrics.json"] = json.dumps(
        {"font": "Open Sans", "wght": WGHT, "wdth": WDTH, "font_size_units": fs, "x_height": round(xh, 2),
         "baseline": round(baseline, 2), "wordmark_width": round(w, 2), "gap": gap, "lockup": [round(total_w, 2), round(total_h, 2)]},
        ensure_ascii=False, indent=2) + "\n"
    return files


# ---------------------------------------------------------------- previews
def render(svg_path: Path, png_path: Path, width: int | None = None, background: str | None = None, font_path: str | None = None) -> None:
    import resvg_py
    kwargs = {"svg_path": str(svg_path)}
    if width:
        kwargs["width"] = width
    if background:
        kwargs["background"] = background
    if font_path:
        kwargs["font_files"] = [font_path]
    kwargs["resources_dir"] = str(svg_path.parent)
    png = resvg_py.svg_to_bytes(**kwargs)
    png_path.write_bytes(bytes(png))


def main() -> int:
    font_path = os.environ.get("OPEN_SANS_TTF")
    if not font_path or not Path(font_path).is_file():
        print("Укажите OPEN_SANS_TTF=/путь/к/OpenSans[wdth,wght].ttf", file=sys.stderr)
        return 2
    tokens = json.loads(TOKENS.read_text(encoding="utf-8"))
    (ROOT / "tokens" / "tokens.css").write_text(build_tokens_css(tokens), encoding="utf-8")

    report = contrast_report(tokens)
    lines = ["# Контраст основных пар (WCAG 2.x)", "", "| Пара | Цвета | Контраст | Минимум | Итог |", "| --- | --- | --- | --- | --- |"]
    failed = 0
    for name, a, b, ratio, need in report:
        ok = ratio >= need
        failed += 0 if ok else 1
        lines.append(f"| {name} | {a} на {b} | {ratio} | {need} | {'да' if ok else 'НЕТ'} |")
    (ROOT / "tokens" / "contrast.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    LOGO.mkdir(exist_ok=True)
    for name, content in build_logos(font_path).items():
        (LOGO / name).write_text(content, encoding="utf-8")

    PREVIEWS.mkdir(exist_ok=True)
    render(LOGO / "navidu-logo.svg", PREVIEWS / "logo.png", width=960, background="#FCFAFB")
    render(LOGO / "navidu-logo-on-stage.svg", PREVIEWS / "logo-on-stage.png", width=960)
    render(LOGO / "navidu-mark.svg", PREVIEWS / "mark-256.png", width=256, background="#FFFFFF")
    render(LOGO / "favicon.svg", PREVIEWS / "favicon-32.png", width=32)
    render(LOGO / "favicon.svg", PREVIEWS / "favicon-16.png", width=16)
    for svg in sorted(COMPOSITIONS.glob("*.svg")):
        render(svg, PREVIEWS / f"{svg.stem}.png", font_path=font_path)
    for svg in sorted(ART.glob("*.svg")):
        render(svg, PREVIEWS / f"art-{svg.stem}.png", font_path=font_path)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
