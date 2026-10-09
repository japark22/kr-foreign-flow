"""SVG chart builders for the monitor. Plain strings, no dependencies.

Two hues only, and they are the pair validated for colour-vision separation on
this page's navy surface. A third series would need a third hue, and green
against the warm tone fails deuteranope separation, so anything needing three
categories gets faceted instead.
"""
from __future__ import annotations

A, B = "var(--accent)", "var(--hot)"
FAINT, RULE = "var(--ink-faint)", "var(--rule)"


def _txt(x, y, s, cls="tick", anchor="middle"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'class="{cls}">{s}</text>')


def _nice(lo, hi, n=5):
    """Round tick values that bracket the data."""
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** (len(f"{int(abs(raw))}") - 1) if abs(raw) >= 1 else 10 ** -3
    for m in (1, 2, 2.5, 5, 10, 20, 25, 50):
        step = m * mag
        if step >= raw:
            break
    start = (int(lo / step) - 1) * step
    out, v = [], start
    while v <= hi + step:
        out.append(round(v, 6))
        v += step
    return [v for v in out if lo - step <= v <= hi + step]


def dual_line(series_a, series_b, label_a, label_b, unit="%", h=250, w=760):
    """Two time series on one axis. Used for KOSPI against KOSDAQ."""
    if len(series_a) < 3:
        return ""
    vs = [p["v"] for p in series_a] + [p["v"] for p in series_b]
    lo, hi = min(vs), max(vs)
    pad = (hi - lo) * 0.12 or 0.5
    lo, hi = lo - pad, hi + pad
    L, R, T, Bm = 54, 16, 14, 40
    pw, ph = w - L - R, h - T - Bm
    n = max(len(series_a), len(series_b))
    X = lambda i: L + (i / max(n - 1, 1)) * pw
    Y = lambda v: T + (hi - v) / (hi - lo) * ph

    parts = []
    for t in _nice(lo, hi):
        if not (lo <= t <= hi):
            continue
        parts.append(f'<line x1="{L}" x2="{L+pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" '
                     f'stroke="{RULE}" stroke-width="1"/>')
        parts.append(_txt(L - 9, Y(t) + 4, f"{t:g}{unit}", anchor="end"))
    step = max(1, n // 6)
    for i in range(0, n, step):
        if i < len(series_a):
            parts.append(_txt(X(i), h - 14, series_a[i]["date"][2:7]))
    for s, c in ((series_a, A), (series_b, B)):
        if len(s) < 2:
            continue
        d = " ".join(("M" if i == 0 else "L") + f"{X(i):.1f} {Y(p['v']):.1f}"
                     for i, p in enumerate(s))
        parts.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2" '
                     f'stroke-linejoin="round" stroke-linecap="round"/>')
        last = s[-1]
        parts.append(f'<circle cx="{X(len(s)-1):.1f}" cy="{Y(last["v"]):.1f}" r="4" '
                     f'fill="{c}"/>')
        parts.append(f'<text x="{X(len(s)-1)-8:.1f}" y="{Y(last["v"])-10:.1f}" '
                     f'text-anchor="end" class="dlabel" fill="{c}">'
                     f'{last["v"]:.2f}{unit}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{label_a} versus '
            f'{label_b} over time">' + "".join(parts) + "</svg>")


def signed_bars(points, unit="bn", h=200, w=760):
    """Daily net buying. Sign is the message, so colour encodes it."""
    if not points:
        return ""
    vs = [p["v"] for p in points]
    lo, hi = min(vs + [0]), max(vs + [0])
    pad = max(abs(lo), abs(hi)) * 0.1 or 1
    lo, hi = lo - pad, hi + pad
    L, R, T, Bm = 62, 16, 12, 34
    pw, ph = w - L - R, h - T - Bm
    n = len(points)
    bw = max(1.5, pw / n * 0.72)
    X = lambda i: L + (i + 0.5) / n * pw
    Y = lambda v: T + (hi - v) / (hi - lo) * ph
    parts = []
    for t in _nice(lo, hi, 4):
        if not (lo <= t <= hi):
            continue
        parts.append(f'<line x1="{L}" x2="{L+pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" '
                     f'stroke="{RULE}" stroke-width="1"/>')
        parts.append(_txt(L - 9, Y(t) + 4, f"{t:,.0f}", anchor="end"))
    zero = Y(0)
    parts.append(f'<line x1="{L}" x2="{L+pw}" y1="{zero:.1f}" y2="{zero:.1f}" '
                 f'stroke="{FAINT}" stroke-width="1"/>')
    for i, p in enumerate(points):
        v = p["v"]
        top = Y(v) if v >= 0 else zero
        hh = max(1.0, abs(Y(v) - zero))
        parts.append(f'<rect x="{X(i)-bw/2:.1f}" y="{top:.1f}" width="{bw:.1f}" '
                     f'height="{hh:.1f}" rx="1" fill="{A if v>=0 else B}">'
                     f'<title>{p["date"]}  {v:+,.0f}{unit}</title></rect>')
    step = max(1, n // 6)
    for i in range(0, n, step):
        parts.append(_txt(X(i), h - 10, points[i]["date"][5:]))
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="daily net foreign '
            f'buying, last {n} sessions">' + "".join(parts) + "</svg>")


def histogram(bins, h=190, w=760):
    """Distribution of 20-day ownership changes across the universe."""
    if not bins:
        return ""
    L, R, T, Bm = 52, 16, 12, 40
    pw, ph = w - L - R, h - T - Bm
    n = len(bins)
    mx = max(b["n"] for b in bins) or 1
    bw = pw / n
    parts = []
    for t in _nice(0, mx, 4):
        if t > mx:
            continue
        y = T + (1 - t / mx) * ph
        parts.append(f'<line x1="{L}" x2="{L+pw}" y1="{y:.1f}" y2="{y:.1f}" '
                     f'stroke="{RULE}" stroke-width="1"/>')
        parts.append(_txt(L - 9, y + 4, f"{int(t)}", anchor="end"))
    for i, b in enumerate(bins):
        hh = b["n"] / mx * ph
        x = L + i * bw
        parts.append(f'<rect x="{x+1.2:.1f}" y="{T+ph-hh:.1f}" '
                     f'width="{bw-2.4:.1f}" height="{max(hh,0.6):.1f}" rx="2" '
                     f'fill="{A if b["mid"]>=0 else B}">'
                     f'<title>{b["lo"]:+.2f} to {b["hi"]:+.2f}pp &#8212; '
                     f'{b["n"]} names</title></rect>')
    step = max(1, n // 7)
    for i in range(0, n, step):
        parts.append(_txt(L + (i + 0.5) * bw, h - 16, f'{bins[i]["mid"]:+.1f}'))
    parts.append(_txt(L + pw / 2, h - 2, "20-day change in foreign ownership, pp"))
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="distribution of '
            f'20-day ownership changes">' + "".join(parts) + "</svg>")
