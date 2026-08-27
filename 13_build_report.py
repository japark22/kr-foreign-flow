#!/usr/bin/env python3
"""Step 13: build docs/index.html from the result files. No hand-copied numbers.

Every figure on the page is read from a JSON written by the measurement that
produced it. There is no step where a number is typed in, so the page cannot
drift from the code -- and if a measurement has not been re-run, the page says
so instead of showing a stale figure as if it were current.

    results/validate.json      04_validate.py    persistence, control group,
                                                 horizon IC, costs, rebalance
                                                 profile, stock-vs-flow check
    results/combination.json   12_combination.py liquidity split, orthogonality
    results/level_se.json      11_level_se.py    level-signal standard errors

    python 13_build_report.py            # write docs/index.html
    python 13_build_report.py --check    # report staleness and exit

STALENESS
---------
A result is stale if it is older than the newest raw data file. The page prints
every source with its age, and a stale source is labelled on the page itself
rather than only in the console -- a report that quietly shows last month's
numbers is worse than one that admits it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
SOURCES = {
    "validate.json": "04_validate.py",
    "combination.json": "12_combination.py",
    "level_se.json": "11_level_se.py",
}


# ------------------------------------------------------------------ loading
def newest_data_mtime() -> float:
    newest = 0.0
    for p in (ROOT / "data").rglob("*.parquet"):
        try:
            newest = max(newest, p.stat().st_mtime)
        except OSError:
            continue
    return newest


def load():
    data, meta = {}, []
    newest = newest_data_mtime()
    for fn, producer in SOURCES.items():
        p = RESULTS / fn
        if not p.exists():
            meta.append({"file": fn, "producer": producer, "state": "missing",
                         "when": None, "stale": True})
            continue
        mt = p.stat().st_mtime
        stale = newest > 0 and mt < newest
        data[fn.replace(".json", "")] = json.loads(p.read_text(encoding="utf-8"))
        meta.append({"file": fn, "producer": producer,
                     "state": "stale" if stale else "current",
                     "when": dt.datetime.fromtimestamp(mt).strftime("%Y-%m-%d %H:%M"),
                     "stale": stale})
    return data, meta


# ------------------------------------------------------------- formatting
def f(v, nd=4, sign=True):
    if v is None:
        return "&mdash;"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return html.escape(str(v))
    if v != v:
        return "&mdash;"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def pct(v, nd=2):
    if v is None:
        return "&mdash;"
    try:
        return f"{float(v):+.{nd}%}"
    except (TypeError, ValueError):
        return "&mdash;"


def bar(v, scale, color):
    """A CSS bar. No SVG: the generator stays small and the page stays light."""
    w = max(0.0, min(100.0, abs(float(v)) / scale * 100.0))
    side = "left" if v >= 0 else "right"
    return (f'<span class="barwrap"><span class="barfill" style="width:{w:.1f}%;'
            f'background:var(--{color});margin-{side}:auto"></span></span>')


def rows(html_rows):
    return "\n".join(html_rows)


def section(eyebrow, title, body):
    return (f'<section><div class="sec-head"><span class="eyebrow">{eyebrow}</span>'
            f'<h2>{title}</h2></div>{body}</section>')


def missing(what, producer):
    return (f'<p class="absent">Not measured. Run <span class="mono">{producer}'
            f'</span> to fill in {what}.</p>')


# ------------------------------------------------------------------ blocks
def block_control(v):
    ac = (v or {}).get("autocorrelation", {})
    fk = next((k for k in ac if "외국인" in k), None)
    kk = next((k for k in ac if "기관합계" in k), None)
    if not (fk and kk):
        return missing("the control group", "04_validate.py --horizons 1,5,20"), None
    fm, km = ac[fk]["mean"], ac[kk]["mean"]
    ft, kt = ac[fk]["t"], ac[kk]["t"]
    lags = [str(l) for l in (1, 2, 3, 5, 10, 20) if str(l) in fm and str(l) in km]
    scale = max(max(km.values()), max(fm.values()))
    out = []
    for l in lags:
        ratio = fm[l] / km[l] if km[l] else float("nan")
        out.append(
            f"<tr><td>{l}</td>"
            f'<td class="num">{f(km[l])}</td><td class="num soft">{f(kt[l],1)}</td>'
            f"<td>{bar(km[l], scale, 'hot')}</td>"
            f'<td class="num">{f(fm[l])}</td><td class="num soft">{f(ft[l],1)}</td>'
            f"<td>{bar(fm[l], scale, 'accent')}</td>"
            f'<td class="num strong">{f(ratio,2,sign=False)}</td></tr>')
    ratio1 = fm["1"] / km["1"] if km.get("1") else float("nan")
    tbl = ('<div class="scroll"><table><thead><tr>'
           "<th>lag</th><th>Institutions</th><th>t</th><th></th>"
           "<th>Foreign</th><th>t</th><th></th><th>ratio f/k</th>"
           "</tr></thead><tbody>" + rows(out) + "</tbody></table></div>")
    return tbl, ratio1


def block_horizon(v):
    hic = (v or {}).get("horizon_ic")
    cost = (v or {}).get("costs", {})
    if not hic:
        return missing("horizon returns and costs", "04_validate.py --horizons 1,5,20")
    out = []
    for h in sorted(hic, key=int):
        d, c = hic[h], cost.get(h, {})
        net = c.get("net")
        cls = "bad" if (net is not None and net < 0) else ""
        out.append(
            f"<tr><td>{h}d</td>"
            f'<td class="num">{f(d["mean_ic"],5)}</td>'
            f'<td class="num">{f(d["t"],2)}</td>'
            f'<td class="num soft">{d["days"]:,}</td>'
            f'<td class="num">{pct(c.get("gross"))}</td>'
            f'<td class="num soft">{c.get("annual_turnover",float("nan")):.1f}x</td>'
            f'<td class="num {cls}">{pct(net)}</td></tr>')
    one_way = next(iter(cost.values()), {}).get("cost_one_way")
    note = (f'<p class="note">Net is after {one_way*1e4:.0f}bp one-way, applied to '
            f"the measured turnover of both legs.</p>" if one_way else "")
    return ('<div class="scroll"><table><thead><tr><th>horizon</th><th>mean IC</th>'
            "<th>t</th><th>days</th><th>gross/yr</th><th>turnover</th><th>net/yr</th>"
            "</tr></thead><tbody>" + rows(out) + "</tbody></table></div>" + note)


def block_liquidity(c):
    ls = (c or {}).get("liquidity_split")
    if not ls:
        return missing("the liquidity split", "12_combination.py")
    hs = sorted({r["horizon"] for r in ls})
    unis = ["thin", "middle", "liquid"]
    scale = max(abs(r["ic"]) for r in ls) or 1
    out = []
    for h in hs:
        cells = []
        for u in unis:
            r = next((x for x in ls if x["horizon"] == h and x["universe"] == u), None)
            if not r:
                cells.append('<td class="num">&mdash;</td><td></td>')
                continue
            dead = "" if abs(r["t"]) >= 2 else " zero"
            cells.append(f'<td class="num{dead}">{f(r["ic"],5)}'
                         f'<span class="tsub">t {f(r["t"],2)}</span></td>'
                         f'<td>{bar(r["ic"], scale, "accent" if r["ic"]>=0 else "hot")}</td>')
        out.append(f"<tr><td>{h}d</td>" + "".join(cells) + "</tr>")
    return ('<div class="scroll"><table><thead><tr><th>horizon</th>'
            "<th>Thin third</th><th></th><th>Middle</th><th></th>"
            "<th>Liquid third</th><th></th></tr></thead><tbody>"
            + rows(out) + "</tbody></table></div>"
            '<p class="note">Newey-West t beneath each coefficient. '
            "A cell greyed out is indistinguishable from zero.</p>")


def block_orth(c):
    o = (c or {}).get("orthogonality")
    if not o:
        return missing("orthogonality", "12_combination.py")
    ctrls = ", ".join((c or {}).get("controls", []))
    out = []
    for r in sorted(o, key=lambda x: x["horizon"]):
        k = r.get("kept")
        keep = "&mdash;" if k is None or k != k else f"{k:.0%}"
        out.append(f'<tr><td>{r["horizon"]}d</td>'
                   f'<td class="num">{f(r["ic_raw"],5)}</td>'
                   f'<td class="num soft">{f(r["t_raw"],2)}</td>'
                   f'<td class="num">{f(r["ic_resid"],5)}</td>'
                   f'<td class="num">{f(r["t_resid"],2)}</td>'
                   f'<td class="num">{keep}</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>horizon</th><th>raw IC</th>'
            "<th>t</th><th>residual IC</th><th>t</th><th>kept</th>"
            "</tr></thead><tbody>" + rows(out) + "</tbody></table></div>"
            f'<p class="note">Residualised each day against {html.escape(ctrls)}.</p>')


def block_level(lv):
    sig = (lv or {}).get("signals")
    if not sig:
        return missing("level-signal standard errors", "11_level_se.py")
    out = []
    for r in sig:
        if r.get("t_nw") is None:
            continue
        ne = r.get("n_eff")
        flag = "bad" if (ne is not None and ne == ne and ne < 100) else ""
        out.append(f'<tr><td>{html.escape(r["signal"])}, {r["horizon"]}d</td>'
                   f'<td class="num">{f(r["mean_ic"],5)}</td>'
                   f'<td class="num soft">{f(r["t_naive"],2)}</td>'
                   f'<td class="num">{f(r["t_nw"],2)}</td>'
                   f'<td class="num">{f(r["ac1"],3)}</td>'
                   f'<td class="num {flag}">'
                   f'{"&mdash;" if ne is None or ne != ne else f"{ne:,.0f}"}'
                   f'<span class="tsub">of {r["days"]:,}</span></td>'
                   f'<td class="verdict-cell">{html.escape(r.get("reading",""))}</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>signal</th><th>IC</th>'
            "<th>t as published</th><th>t Newey-West</th><th>lag-1 AC</th>"
            "<th>independent obs.</th><th>reading</th></tr></thead><tbody>"
            + rows(out) + "</tbody></table></div>"
            '<p class="note">A high lag-1 autocorrelation means consecutive daily '
            "coefficients are nearly the same number, so the published t counts the "
            "same evidence many times over.</p>")


def block_rebalance(v):
    prof = (v or {}).get("rebalance_profile")
    if not prof:
        return missing("the rebalance-calendar test", "04_validate.py")
    ks = sorted(prof, key=int)
    peak = max(abs(prof[k] - 1) for k in ks) or 0.01
    out = []
    for k in ks:
        rel = prof[k]
        mark = " &larr; effective date" if int(k) == 0 else ""
        out.append(f'<tr><td class="num">{int(k):+d}</td>'
                   f'<td class="num">{rel:.2f}&times;</td>'
                   f'<td>{bar(rel-1, peak, "hot")}</td>'
                   f'<td class="verdict-cell">{mark}</td></tr>')
    n = (v or {}).get("rebalance_dates")
    return ('<div class="scroll"><table><thead><tr><th>day</th>'
            "<th>vs baseline |flow|</th><th></th><th></th></tr></thead><tbody>"
            + rows(out) + "</tbody></table></div>"
            f'<p class="note">{n} rule-derived effective dates. A calendar that '
            "found real events would show a clear hump at zero.</p>")


def block_svf(v):
    s = (v or {}).get("stock_vs_flow")
    if not s:
        return missing("the stock-versus-flow cross-check", "04_validate.py")
    return ('<div class="flag"><h3>Two views of the same activity agree at '
            f'{s["mean_rank_corr"]:.2f}</h3>'
            f'<p>The signal is a stock measure &mdash; the daily change in foreign '
            f"shares held. The exchange separately publishes foreign net buying, a "
            f"flow measure. Across {s['days']:,} overlapping days and "
            f"{s['tickers']:,} tickers the mean daily rank correlation is "
            f'<span class="mono">{f(s["mean_rank_corr"])}</span>, the tenth '
            f'percentile is <span class="mono">{f(s["p10"])}</span> and the worst '
            f'day is <span class="mono">{f(s["worst"])}</span>.</p>'
            "<p>One of the two is not measuring what we assume. Candidates: "
            "settlement-date against trade-date convention, securities lending and "
            "custody transfers moving holdings without a trade, depositary-receipt "
            "conversions, in-kind index creation, and a known misclassification of "
            "orders routed through domestic brokers. It does not change the verdict "
            "&mdash; there is no edge on either reading &mdash; but it changes what "
            "the numbers mean.</p></div>")


# ------------------------------------------------------------------- page
CSS = """
:root{--ground:#F4F6F9;--surface:#FFFFFF;--surface-2:#EDF0F5;--ink:#0F1520;
--ink-soft:#59647A;--ink-faint:#8A93A5;--rule:#DCE1EA;--rule-soft:#E9EDF3;
--accent:#2F63C4;--hot:#C0392B;--ok:#1F6F4A;--warn:#8A6A16}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--ground:#0D1117;--surface:#151B24;--surface-2:#1B222D;--ink:#E4E9F1;
--ink-soft:#93A0B5;--ink-faint:#6C7889;--rule:#242D3B;--rule-soft:#1E2632;
--accent:#5F94E6;--hot:#DD6E56;--ok:#5BB98B;--warn:#D2A73F}}
:root[data-theme="dark"]{--ground:#0D1117;--surface:#151B24;--surface-2:#1B222D;
--ink:#E4E9F1;--ink-soft:#93A0B5;--ink-faint:#6C7889;--rule:#242D3B;
--rule-soft:#1E2632;--accent:#5F94E6;--hot:#DD6E56;--ok:#5BB98B;--warn:#D2A73F}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-size:16px;line-height:1.6;
font-family:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
-webkit-font-smoothing:antialiased}
.wrap{max-width:1000px;margin:0 auto;padding:0 24px 88px}
h1,h2,h3{font-family:Newsreader,Georgia,serif;margin:0;text-wrap:balance}
h1{font-size:clamp(1.9rem,4.4vw,2.8rem);font-weight:500;line-height:1.14;letter-spacing:-.015em}
h2{font-size:1.6rem;font-weight:500} h3{font-size:1.1rem;font-weight:600}
p{margin:0} a{color:var(--accent)}
.mono{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:.72rem;letter-spacing:.14em;
text-transform:uppercase;color:var(--ink-faint);font-weight:500}
header{border-bottom:1px solid var(--rule);background:var(--surface);margin-bottom:52px}
.head-inner{max-width:1000px;margin:0 auto;padding:52px 24px 40px;display:flex;
flex-direction:column;gap:20px}
.kicker{display:flex;gap:14px;flex-wrap:wrap;align-items:baseline}
.lede{max-width:62ch;color:var(--ink-soft);font-size:1.04rem}
.verdict{padding:18px 20px;border-left:3px solid var(--hot);background:var(--surface-2);
max-width:62ch;border-radius:0 6px 6px 0}
section{margin-bottom:56px}
.sec-head{display:flex;flex-direction:column;gap:8px;margin-bottom:20px}
.scroll{overflow-x:auto;border:1px solid var(--rule);border-radius:8px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{padding:9px 13px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--rule-soft)}
th:first-child,td:first-child{text-align:left}
thead th{font-family:"IBM Plex Mono",monospace;font-size:.68rem;letter-spacing:.09em;
text-transform:uppercase;color:var(--ink-faint);font-weight:600;background:var(--surface-2);
border-bottom:1px solid var(--rule)}
tbody tr:last-child td{border-bottom:none}
td.num{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
td.soft{color:var(--ink-faint)} td.strong{font-weight:600}
td.bad{color:var(--hot)} td.zero{color:var(--ink-faint)}
.tsub{display:block;font-size:.72rem;color:var(--ink-faint);font-weight:400}
.verdict-cell{text-align:left;white-space:normal;font-size:.83rem;color:var(--ink-soft)}
.barwrap{display:block;width:110px;height:8px;background:var(--rule-soft);border-radius:99px;
overflow:hidden}
.barfill{display:block;height:100%;border-radius:99px}
.note{font-size:.84rem;color:var(--ink-soft);margin-top:12px;max-width:66ch}
.absent{font-size:.9rem;color:var(--warn);padding:14px 16px;border:1px dashed var(--rule);
border-radius:8px;background:var(--surface)}
.col{max-width:68ch;color:var(--ink-soft)}
.stack{display:flex;flex-direction:column;gap:18px}
.flag{background:var(--surface);border:1px solid var(--hot);border-radius:8px;
padding:20px 22px;display:flex;flex-direction:column;gap:11px}
.flag h3{color:var(--hot)} .flag p{font-size:.92rem;color:var(--ink-soft);max-width:68ch}
.stale{color:var(--hot);font-weight:600}
footer{border-top:1px solid var(--rule);padding-top:22px;font-size:.84rem;color:var(--ink-faint);
display:flex;flex-direction:column;gap:8px}
footer table{font-size:.82rem;margin-top:8px}
@media (max-width:640px){.wrap{padding:0 18px 60px}.head-inner{padding:38px 18px 30px}}
"""


def build(data, meta) -> str:
    v = data.get("validate", {})
    c = data.get("combination", {})
    lv = data.get("level_se", {})

    ctrl_tbl, ratio1 = block_control(v)
    if ratio1 is None or ratio1 != ratio1:
        verdict = ("The control group has not been run, so the central question "
                   "&mdash; whether the persistence is specific to foreign "
                   "investors &mdash; is still open.")
    elif ratio1 < 0.9:
        verdict = (f"<strong>The premise is refuted.</strong> Domestic institutional "
                   f"flow is more persistent than foreign flow at lag 1 "
                   f"(ratio <span class='mono'>{ratio1:.2f}</span>). Persistence is a "
                   f"property of institutional execution, not of foreign investors, "
                   f"so the foreign disclosure has no special claim on it.")
    elif ratio1 > 1.3:
        verdict = (f"<strong>The premise survives its hardest test.</strong> Foreign "
                   f"flow is more persistent than domestic institutional flow "
                   f"(ratio <span class='mono'>{ratio1:.2f}</span>).")
    else:
        verdict = (f"<strong>The two are close</strong> (ratio "
                   f"<span class='mono'>{ratio1:.2f}</span>): persistence looks "
                   f"generic to large-order execution rather than special to foreign "
                   f"investors.")

    win = v.get("window") or c.get("window") or ["?", "?"]
    obs = v.get("observations")
    kick = [f"Korean equities &middot; {win[0]} to {win[1]}"]
    if obs:
        kick.append(f"{obs:,} ticker-days")
    kick.append("Personal project")

    any_stale = any(m["stale"] for m in meta)
    stale_banner = ""
    if any_stale:
        names = ", ".join(m["file"] for m in meta if m["stale"])
        stale_banner = (f'<p class="absent">Some results are older than the raw data '
                        f"or missing: <span class='mono'>{names}</span>. "
                        f"Re-run those measurements before relying on this page.</p>")

    src_rows = "".join(
        f'<tr><td class="mono">{m["file"]}</td><td class="mono">{m["producer"]}</td>'
        f'<td>{m["when"] or "&mdash;"}</td>'
        f'<td class="{"stale" if m["stale"] else ""}">{m["state"]}</td></tr>'
        for m in meta)

    body = "".join([
        section("The control group",
                "Is the persistence specific to foreign investors?",
                '<div class="stack">'
                '<p class="col">Two hypotheses fit the persistence curve equally '
                "well: that foreign flow is special, or that all large "
                "institutional flow is persistent because big orders get split "
                "over days. The domestic-institution series separates them.</p>"
                + ctrl_tbl + "</div>"),
        section("Return prediction and costs",
                "Does it predict price, and can it be traded?",
                block_horizon(v)),
        section("Where the effect lives",
                "Information coefficient by horizon and liquidity",
                block_liquidity(c)),
        section("Is the information its own?",
                "After residualising against standard factors",
                block_orth(c)),
        section("The level family",
                "What the published t-statistics are actually worth",
                block_level(lv)),
        section("The noise filter",
                "Do the rule-derived rebalance dates find anything?",
                block_rebalance(v)),
        section("Still open", "The two series disagree", block_svf(v)),
    ])

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Korean Foreign-Ownership Flow &mdash; results</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style></head><body>
<header><div class="head-inner">
<div class="kicker">{"".join(f'<span class="eyebrow">{k}</span>' for k in kick)}</div>
<h1>Does daily foreign flow predict returns?</h1>
<p class="lede">Korea publishes foreign ownership per stock every day. The premise
was that foreign investors digest information more slowly and split larger orders,
so their flow is persistent, and that a daily disclosure of it is hard to replicate.
This page is generated from the measurement outputs, so every figure below is the
one the code last produced.</p>
<div class="verdict">{verdict}</div>
{stale_banner}
</div></header>
<div class="wrap">{body}
<footer>
<p>Generated {dt.datetime.now().strftime("%Y-%m-%d %H:%M")} by
<span class="mono">13_build_report.py</span> from the files below. No figure on this
page is entered by hand.</p>
<div class="scroll"><table><thead><tr><th>source</th><th>produced by</th>
<th>last written</th><th>state</th></tr></thead><tbody>{src_rows}</tbody></table></div>
</footer></div></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report source freshness and exit without writing")
    a = ap.parse_args()

    data, meta = load()
    for m in meta:
        mark = "STALE" if m["stale"] else "ok"
        print(f"  {m['file']:<22} {m['state']:<8} {m['when'] or '-':<17} {mark}")
    if a.check:
        return 1 if any(m["stale"] for m in meta) else 0
    if not data:
        print("\nNo result files at all. Run the measurements first.")
        return 1

    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / "index.html"
    out.write_text(build(data, meta), encoding="utf-8")
    print(f"\n  wrote {out.relative_to(ROOT)}  ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
