#!/usr/bin/env python3
"""Генератор иллюстративной ленты фактов (identity/art/fact-strip*.svg).

Данные ниже являются авторским примером для иллюстраций лендинга и дизайн-образцов, а не результатом анализа.
В продукте лента строится из реальных Observation/Coverage/Finding (контракты C-DATA в PLAN.md).
Запуск: python3 identity/source/build_art.py
"""
from pathlib import Path

ART = Path(__file__).resolve().parents[1] / "art"
DURATION = 125.0
FULL = [
    ("Кассир на месте", [(0, 96), (103, 125)]),
    ("Клиент у окна", [(3, 38), (50, 92), (104, 121)]),
    ("Деньги в руках", [(12, 20), (41.5, 47.5), (58, 64)]),
    ("Деньги на столе", [(21, 26), (66, 70)]),
    ("Обзор стола", None),  # особая дорожка: видимость с непроверяемым участком 72–80 с
]
COMPACT = FULL[:3]
SELECTED, SELECTED_AT = (40.5, 48.0), 44.2
OTHER = (96.0, 103.0)
TICKS = [(0, "00:00"), (30, "00:30"), (60, "01:00"), (90, "01:30"), (120, "02:00")]


def strip(lanes, *, width, label_w, lane_gap, font, bar_h, title, desc):
    x0, x1 = label_w, width - 16
    sx = (x1 - x0) / DURATION
    x = lambda t: x0 + t * sx
    top = 36
    height = top + lane_gap * len(lanes) + 34
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d">',
         f'<title id="t">{title}</title><desc id="d">{desc}</desc>',
         '<defs><pattern id="fsHatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="8" height="8" fill="#3A1225"/><rect width="3" height="8" fill="#8A7580" opacity="0.7"/></pattern>'
         '<linearGradient id="fsLit" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#FC9385" stop-opacity="0.10"/><stop offset="0.5" stop-color="#FC9385" stop-opacity="0.32"/><stop offset="1" stop-color="#FC9385" stop-opacity="0.10"/></linearGradient></defs>',
         f'<rect width="{width}" height="{height}" rx="18" fill="#2A0B19"/>']
    lanes_h = lane_gap * len(lanes)
    a, b = SELECTED
    o.append(f'<rect x="{x(a):.1f}" y="{top-8}" width="{x(b)-x(a):.1f}" height="{lanes_h+8}" rx="8" fill="url(#fsLit)" stroke="#FC9385" stroke-width="2"/>')
    c, d = OTHER
    o.append(f'<rect x="{x(c):.1f}" y="{top-8}" width="{x(d)-x(c):.1f}" height="{lanes_h+8}" rx="8" fill="none" stroke="#FC9385" stroke-width="2" stroke-dasharray="6 5" opacity="0.8"/>')
    o.append(f'<g font-family="Open Sans, sans-serif" font-size="{font}" font-weight="600" font-stretch="87.5%">')
    y = top
    for name, bars in lanes:
        cy = y + lane_gap / 2 - 4
        o.append(f'<text x="16" y="{cy + font * 0.36:.1f}" fill="#CDB6C1">{name}</text>')
        o.append(f'<rect x="{x0}" y="{cy - bar_h/2:.1f}" width="{x1-x0}" height="{bar_h}" rx="{bar_h/2}" fill="#3A1225"/>')
        if bars is None:
            for s, e, op in [(0, 72, 0.35), (80, 125, 0.35)]:
                o.append(f'<rect x="{x(s):.1f}" y="{cy - bar_h/2:.1f}" width="{x(e)-x(s):.1f}" height="{bar_h}" rx="{bar_h/2}" fill="#61CFFF" opacity="{op}"/>')
            o.append(f'<rect x="{x(72):.1f}" y="{cy - bar_h/2:.1f}" width="{x(80)-x(72):.1f}" height="{bar_h}" rx="4" fill="url(#fsHatch)"/>')
        else:
            for s, e in bars:
                o.append(f'<rect x="{x(s):.1f}" y="{cy - bar_h/2:.1f}" width="{x(e)-x(s):.1f}" height="{bar_h}" rx="{bar_h/2}" fill="#61CFFF"/>')
        y += lane_gap
    o.append('</g>')
    def marker(t, filled):
        cx = x(t)
        p = f'M{cx-6.6:.1f} 8h13.2a1.6 1.6 0 0 1 1.3 2.6l-6.6 7.8a1.6 1.6 0 0 1-2.5 0l-6.6-7.8a1.6 1.6 0 0 1 1.3-2.6z'
        return f'<path d="{p}" fill="{"#FC9385" if filled else "none"}" stroke="#FC9385" stroke-width="{0 if filled else 2}"/>'
    o.append(marker(SELECTED_AT, True))
    o.append(marker(sum(OTHER) / 2, False))
    o.append(f'<rect x="{x(SELECTED_AT)-1:.1f}" y="{top-12}" width="2" height="{lanes_h+16}" fill="#FFFFFF"/>')
    o.append(f'<g font-family="Open Sans, sans-serif" font-weight="700" font-size="{font+1}" font-stretch="87.5%"><rect x="{x(SELECTED_AT)+14:.1f}" y="4" width="{(font+1)*4.9:.0f}" height="{font+9}" rx="6" fill="#FC9385"/><text x="{x(SELECTED_AT)+21:.1f}" y="{4+(font+9)/2+(font+1)*0.36:.1f}" fill="#4A0221">00:44,2</text></g>')
    ay = top + lanes_h + 4
    o.append(f'<g font-family="Open Sans, sans-serif" font-size="{font-1}" font-weight="600" fill="#CDB6C1" font-stretch="87.5%">')
    for t, lab in TICKS:
        o.append(f'<rect x="{x(t):.1f}" y="{ay}" width="1" height="6" fill="#8A7580"/><text x="{x(t)+4:.1f}" y="{ay+16}">{lab}</text>')
    o.append('</g></svg>')
    return "\n".join(o) + "\n"


if __name__ == "__main__":
    desc = ("Дорожки наблюдаемых фактов по времени записи 2 мин 5 с. Подсвечен эпизод 00:41–00:48: деньги в руках кассира, "
            "клиента у окна нет. Второй эпизод 01:36–01:43 обведён пунктиром. Авторская иллюстрация.")
    (ART / "fact-strip.svg").write_text(strip(FULL, width=960, label_w=176, lane_gap=36, font=13, bar_h=14,
        title="Иллюстрация: лента наблюдаемых фактов", desc=desc + " Участок 01:12–01:20 штрихован: стол не виден."), encoding="utf-8")
    (ART / "fact-strip-compact.svg").write_text(strip(COMPACT, width=640, label_w=150, lane_gap=34, font=14, bar_h=12,
        title="Иллюстрация: лента фактов, три дорожки", desc=desc), encoding="utf-8")
    print("ok")
