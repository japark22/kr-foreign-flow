#!/usr/bin/env python3
"""Step 14: the recurring monitor -- what foreign investors have been doing lately.

THIS IS NOT A SIGNAL, AND THE PAGE SAYS SO
------------------------------------------
This project tested whether daily foreign ownership predicts returns and found it
does not: the effect sits in names too thin to trade, it reverses by twenty days,
it dies after costs, and domestic institutional flow turns out to be twice as
persistent. Tidy tables invite the opposite reading, so the caveat is rendered at
the top of the page rather than buried in a footnote.

What it is for is the thing the project set out to build in the first place: a
clean daily record of foreign ownership per issuer, and a way to see it. It also
works as a health check -- if collection stalls or the data goes strange, the
freshness block shows it before anything downstream does.

ENGLISH NAMES
-------------
The exchange store carries Korean names only. data/names_en.csv maps a ticker to
an English name, and each row also carries the Korean name it expects. If the
stored Korean name does not match, the row is DROPPED rather than applied: a
silently mislabelled ticker is worse than a missing label. The file is a partial
seed -- extend it as you need, and anything unmapped simply shows its code.

    python 14_monitor.py                 # compute, write results/monitor.json + docs/monitor.html
    python 14_monitor.py --render-only   # rebuild the page from the existing JSON
    python 14_monitor.py --lookback 750  # longer history for the trend charts
"""
from __future__ import annotations

import argparse
import csv as csvmod
import datetime as dt
import html
import json
import sys
from pathlib import Path

import charts
import tracker

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
NAMES_EN = ROOT / "data" / "names_en.csv"
AUTHOR = "Jungan Park"
TOP_N = 15
WINDOWS = (5, 20)
MIN_ADV = 1e8


# ------------------------------------------------------------------ compute
def load_english(korean: dict) -> tuple[dict, int]:
    """ticker -> English name, keeping only rows whose Korean name still matches."""
    if not NAMES_EN.exists():
        return {}, 0
    out, dropped = {}, 0
    with open(NAMES_EN, newline="", encoding="utf-8") as fh:
        for r in csvmod.DictReader(fh):
            t = (r.get("ticker") or "").strip()
            want = (r.get("korean") or "").strip()
            en = (r.get("english") or "").strip()
            if not t or not en:
                continue
            have = (korean.get(t) or "").strip()
            if want and have and want != have:
                dropped += 1
                continue
            out[t] = en
    return out, dropped


def compute(lookback: int) -> dict:
    import numpy as np
    import pandas as pd
    from krxflow import features, storage

    start = (dt.date.today() - dt.timedelta(days=int(lookback * 1.55))).isoformat()
    print(f"loading from {start} ...")
    p = features.load_panels(start, None)
    pct, shares = p["foreign_pct"], p["foreign_shares"]
    listed, limit, mkt = p["shares_listed"], p["foreign_limit_shares"], p["market"]
    idx = pct.index

    m = storage.read_range("market", start, None,
                           columns=["trade_date", "ticker", "close",
                                    "value_traded", "market_cap"])
    if m.empty:
        sys.exit("no market store for this window")
    m["trade_date"] = pd.to_datetime(m["trade_date"])

    def pv(col):
        return (m.pivot_table(index="trade_date", columns="ticker", values=col,
                              aggfunc="last", observed=True)
                 .sort_index().astype("float64")
                 .reindex(index=idx, columns=pct.columns))

    close, vt, cap = pv("close"), pv("value_traded"), pv("market_cap")
    del m
    adv = vt.rolling(20, min_periods=5).mean()

    korean = {}
    try:
        nm = storage.read_range("investor_flow", start, None,
                                columns=["ticker", "name"])
        if not nm.empty:
            korean = (nm.drop_duplicates("ticker", keep="last")
                        .set_index("ticker")["name"].astype(str).to_dict())
    except Exception as exc:                                    # noqa: BLE001
        print(f"  (no Korean names available: {exc})")
    english, dropped = load_english(korean)
    print(f"  names: {len(korean):,} Korean, {len(english):,} English"
          + (f", {dropped} English rows dropped on a Korean-name mismatch"
             if dropped else ""))

    uni = features.universe_mask(p)
    live = uni.iloc[-1] & (adv.iloc[-1] >= MIN_ADV) & pct.iloc[-1].notna()
    tickers = [t for t in pct.columns if bool(live.get(t, False))]
    last = idx[-1]
    by_mkt = {m_: [t for t in tickers if str(mkt.get(t)) == m_]
              for m_ in ("KOSPI", "KOSDAQ")}

    fresh = {"last_date": str(last.date()),
             "days_old": (dt.date.today() - last.date()).days,
             "rows_last_day": int(pct.loc[last].notna().sum()),
             "window_days": int(len(idx)),
             "first_date": str(idx[0].date())}

    def meta(t):
        return {"ticker": t, "name": korean.get(t, ""), "en": english.get(t, ""),
                "market": str(mkt.get(t, ""))}

    def row(t, w):
        d_pp = float(pct[t].iloc[-1] - pct[t].iloc[-1 - w]) if len(idx) > w else float("nan")
        d_sh = float(shares[t].iloc[-1] - shares[t].iloc[-1 - w]) if len(idx) > w else float("nan")
        px = float(close[t].iloc[-1])
        a = float(adv[t].iloc[-1])
        r = meta(t)
        r.update({"d_pp": d_pp,
                  "adv_days": (d_sh * px / a) if (a and a == a and px == px) else float("nan"),
                  "krw_bn": (d_sh * px / 1e9) if (px == px) else float("nan"),
                  "pct_now": float(pct[t].iloc[-1]),
                  "cap_bn": float(cap[t].iloc[-1]) / 1e9 if t in cap else float("nan")})
        return r

    movers = {}
    for w in WINDOWS:
        rs = [row(t, w) for t in tickers]
        rs = [r for r in rs if r["d_pp"] == r["d_pp"]]
        rs.sort(key=lambda r: r["d_pp"], reverse=True)
        movers[str(w)] = {"up": rs[:TOP_N], "down": rs[-TOP_N:][::-1]}
        print(f"  {w}d movers ranked over {len(rs):,} names")

    agg, trend = {}, {}
    for label, sel in (("ALL", tickers), *by_mkt.items()):
        if not sel:
            continue
        held = (shares[sel] * close[sel]).sum(axis=1, min_count=1)
        tot = (listed[sel] * close[sel]).sum(axis=1, min_count=1)
        s = (held / tot * 100).dropna()
        if s.empty:
            continue
        agg[label] = {"now": float(s.iloc[-1]),
                      "chg_5d": float(s.iloc[-1] - s.iloc[-6]) if len(s) > 5 else None,
                      "chg_20d": float(s.iloc[-1] - s.iloc[-21]) if len(s) > 20 else None,
                      "chg_250d": float(s.iloc[-1] - s.iloc[-251]) if len(s) > 250 else None,
                      "names": len(sel)}
        keep = s.iloc[::max(1, len(s) // 220)]
        trend[label] = [{"date": str(d.date()), "v": float(v)} for d, v in keep.items()]

    daily = {}
    d_sh_all = shares.diff()
    for label, sel in by_mkt.items():
        if not sel:
            continue
        krw = (d_sh_all[sel] * close[sel]).sum(axis=1, min_count=1) / 1e9
        daily[label] = [{"date": str(d.date()), "v": float(v)}
                        for d, v in krw.dropna().tail(60).items()]

    hist = []
    w = 20
    if len(idx) > w:
        vals = [float(pct[t].iloc[-1] - pct[t].iloc[-1 - w]) for t in tickers]
        vals = [v for v in vals if v == v]
        edges = [-9e9] + [round(-1.5 + i * 0.25, 3) for i in range(13)] + [9e9]
        for i in range(len(edges) - 1):
            lo_, hi_ = edges[i], edges[i + 1]
            n = sum(1 for v in vals if lo_ <= v < hi_)
            mid = (max(lo_, -1.75) + min(hi_, 1.75)) / 2
            hist.append({"lo": max(lo_, -1.75), "hi": min(hi_, 1.75),
                         "mid": mid, "n": n})

    binding = []
    for t in tickers:
        lim, ls = limit[t].iloc[-1], listed[t].iloc[-1]
        if not (lim == lim and ls == ls and 0 < lim < ls * 0.999):
            continue
        r = meta(t)
        r.update({"exhaustion": float(shares[t].iloc[-1] / lim * 100),
                  "cap_pct": float(lim / ls * 100),
                  "pct_now": float(pct[t].iloc[-1])})
        binding.append(r)
    binding.sort(key=lambda r: r["exhaustion"], reverse=True)

    conc = {}
    if len(idx) > 20:
        val = []
        for t in tickers:
            d = shares[t].iloc[-1] - shares[t].iloc[-21]
            px = close[t].iloc[-1]
            if d == d and px == px:
                val.append(float(d * px))
        buys = sorted((v for v in val if v > 0), reverse=True)
        if buys:
            tot = sum(buys)
            conc = {"window": 20, "gross_buy_bn": tot / 1e9,
                    "top10_share": sum(buys[:10]) / tot,
                    "top50_share": sum(buys[:50]) / tot,
                    "names_bought": len(buys)}

    ride_res = tracker.ride(pct, close,
                            (uni & (adv >= MIN_ADV)).fillna(False))

    return {"schema": 2, "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "freshness": fresh, "movers": movers, "aggregate": agg, "trend": trend,
            "daily_krw": daily, "hist_20d": hist, "limits": binding[:TOP_N],
            "concentration": conc, "min_adv": MIN_ADV, "universe": len(tickers),
            "english_names": len(english), "ride": ride_res}


# ------------------------------------------------------------------ render
def esc(s):
    return html.escape(str(s or ""))


def num(v, nd=2, sign=False, dash="&mdash;"):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return dash
    return dash if v != v else (f"{v:+,.{nd}f}" if sign else f"{v:,.{nd}f}")


def name_cell(r):
    en = esc(r.get("en"))
    ko = esc(r.get("name"))
    sub = f'<span class="sub2">{en}</span>' if en else ""
    return f'<td class="nm">{ko or "&mdash;"}{sub}</td>'


def mover_table(rows_, title, note):
    body = "".join(
        f'<tr><td class="tk">{esc(r["ticker"])}</td>' + name_cell(r) +
        f'<td class="mk">{esc(r["market"])}</td>'
        f'<td class="num big">{num(r["d_pp"],3,sign=True)}</td>'
        f'<td class="num">{num(r["adv_days"],1,sign=True)}</td>'
        f'<td class="num soft">{num(r["krw_bn"],0,sign=True)}</td>'
        f'<td class="num soft">{num(r["pct_now"],2)}</td></tr>' for r in rows_)
    return (f'<div class="half"><h3>{title}</h3><p class="sub">{note}</p>'
            '<div class="scroll"><table><thead><tr>'
            "<th>code</th><th>name</th><th>mkt</th><th>&Delta;pp</th>"
            "<th>ADV days</th><th>KRW bn</th><th>held %</th>"
            f"</tr></thead><tbody>{body}</tbody></table></div></div>")


def render(d: dict) -> str:
    fr = d.get("freshness", {})
    warn = ""
    if fr.get("days_old", 99) > 4:
        warn = (f'<p class="alarm">Last stored trading day is '
                f'<span class="mono">{esc(fr.get("last_date"))}</span>, '
                f'{fr.get("days_old")} days ago. Collection may have stopped.</p>')

    cards = ""
    for k, a in (d.get("aggregate") or {}).items():
        cards += (f'<div class="card"><span class="eyebrow">{esc(k)}</span>'
                  f'<div class="big-num">{num(a["now"],2)}<span class="unit">%</span></div>'
                  f'<div class="deltas"><span>5d {num(a.get("chg_5d"),3,sign=True)}</span>'
                  f'<span>20d {num(a.get("chg_20d"),3,sign=True)}</span>'
                  f'<span>1y {num(a.get("chg_250d"),2,sign=True)}</span></div>'
                  f'<p class="sub">{a["names"]:,} names, weighted by market value</p></div>')

    tr = d.get("trend") or {}
    trend_chart = charts.dual_line(tr.get("KOSPI", []), tr.get("KOSDAQ", []),
                                   "KOSPI", "KOSDAQ")
    dk = d.get("daily_krw") or {}
    bars_kospi = charts.signed_bars(dk.get("KOSPI", []))
    bars_kosdaq = charts.signed_bars(dk.get("KOSDAQ", []))
    hist_chart = charts.histogram(d.get("hist_20d") or [])

    ride = d.get("ride") or {}
    ride_stats = ride.get("stats") or []
    ride_sec = ""
    if ride_stats:
        rows5 = (ride.get("curve") or {}).get("5") or []
        ga = [{"date": q["date"], "v": q["gross"]} for q in rows5]
        na = [{"date": q["date"], "v": q["net"]} for q in rows5]
        ride_chart = charts.dual_line(ga, na, "before costs", "after costs",
                                      unit="%") if rows5 else ""
        rbody = "".join(
            f'<tr><td class="tk">{s_["h"]}d</td>'
            f'<td class="num">{s_["n"]}</td>'
            f'<td class="num big">{num(s_["gross_bp"],1,sign=True)}</td>'
            f'<td class="num big">{num(s_["net_bp"],1,sign=True)}</td>'
            f'<td class="num">{num(s_["t_gross"],2,sign=True)}</td>'
            f'<td class="num soft">{num(s_["hit"],0)}%</td>'
            f'<td class="num soft">{num(s_["ann_net_pct"],1,sign=True)}%</td></tr>'
            for s_ in ride_stats)
        rleg = ('<div class="legend"><span class="lg"><span class="sw acc"></span>'
                'before costs</span><span class="lg"><span class="sw hot"></span>'
                'after costs</span></div>')
        rfig = (f'<figure><figcaption class="fig-top">Cumulative excess return, '
                f'5-session rebalance</figcaption>{rleg}{ride_chart}'
                '<figcaption>Flat-to-down after costs is the expected shape: the '
                'research page measured no tradable edge here, and this chart '
                're-checks that verdict on every refresh.</figcaption></figure>'
                ) if rows5 else ""
        ride_sec = (
            '<section><div class="sec-head"><span class="eyebrow">Riding the flow'
            '</span><h2>If you had bought what they bought</h2></div>'
            f'<p class="note">Every h sessions: rank the universe by its '
            f'{ride.get("sig_window", 20)}-session change in foreign ownership, '
            f'buy the top {ride.get("top_pct", 10)}% equal-weighted, hold h '
            'sessions, sell. Returns are EXCESS of the equal-weighted universe; '
            f'costs charge {num(ride.get("cost_rt_pct"), 2)}% per round trip '
            'including the 0.20% sales tax. Figures are basis points per round '
            'trip.</p>'
            '<div class="scroll"><table><thead><tr><th>hold</th><th>trades</th>'
            '<th>gross bp</th><th>net bp</th><th>t (NW)</th><th>hit</th>'
            '<th>net /yr</th></tr></thead>'
            f'<tbody>{rbody}</tbody></table></div>' + rfig + "</section>")

    movers = ""
    for w in sorted((d.get("movers") or {}), key=int):
        mv = d["movers"][w]
        movers += (f'<section><div class="sec-head"><span class="eyebrow">'
                   f"Last {w} sessions</span><h2>Largest ownership changes</h2></div>"
                   '<div class="two">'
                   + mover_table(mv.get("up", []), "Accumulated",
                                 "Ranked by change in percentage points held.")
                   + mover_table(mv.get("down", []), "Reduced",
                                 "Same measure, the other end of the ranking.")
                   + "</div></section>")

    lim = d.get("limits") or []
    lim_sec = ""
    if lim:
        body = "".join(
            f'<tr><td class="tk">{esc(r["ticker"])}</td>' + name_cell(r) +
            f'<td class="mk">{esc(r["market"])}</td>'
            f'<td class="num big">{num(r["exhaustion"],1)}<span class="unit">%</span></td>'
            f'<td class="num soft">{num(r["cap_pct"],0)}%</td>'
            f'<td class="num soft">{num(r["pct_now"],2)}%</td></tr>' for r in lim)
        lim_sec = ('<section><div class="sec-head"><span class="eyebrow">Statutory '
                   "limits</span><h2>Closest to the foreign ownership cap</h2></div>"
                   '<div class="scroll"><table><thead><tr><th>code</th><th>name</th>'
                   "<th>mkt</th><th>limit used</th><th>cap</th><th>held %</th>"
                   f"</tr></thead><tbody>{body}</tbody></table></div>"
                   '<p class="note">Only issuers whose foreign ownership is capped by '
                   "statute appear here. Near 100% means foreign buyers cannot add "
                   "without another foreign holder selling.</p></section>")

    c = d.get("concentration") or {}
    conc = ""
    if c:
        conc = (f'<div class="card wide"><span class="eyebrow">Concentration, '
                f'{c["window"]} sessions</span>'
                f'<div class="big-num">{num(c.get("top10_share",0)*100,1)}'
                f'<span class="unit">%</span></div>'
                f'<p class="sub">of all net foreign buying went to the ten largest '
                f'recipients; the top fifty took {num(c.get("top50_share",0)*100,1)}%, '
                f'across {c.get("names_bought",0):,} names bought and '
                f'{num(c.get("gross_buy_bn"),0)}bn KRW gross.</p></div>')

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Foreign Ownership Monitor &mdash; Korea</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style></head><body>
<header><div class="head-inner">
<div class="kicker"><span class="eyebrow">Korea &middot; KOSPI + KOSDAQ</span>
<span class="eyebrow">close of {esc(fr.get("last_date"))}</span>
<span class="eyebrow">{d.get("universe",0):,} names</span>
<a class="eyebrow navlink" href="index.html">Research findings &rarr;</a></div>
<h1>Foreign ownership monitor</h1>
<div class="caveat"><strong>Descriptive, not a signal.</strong> This project tested
whether daily foreign ownership predicts returns and found it does not: the effect
sits in names too thin to trade, reverses by twenty days, dies after costs, and
domestic institutional flow is twice as persistent. Read these tables as context for
what has already happened, never as a reason to buy or sell.
<a href="index.html">The evidence is here.</a></div>
{warn}
</div></header>
<div class="wrap">

<section><div class="sec-head"><span class="eyebrow">Market-wide</span>
<h2>Share of listed value held by foreign investors</h2></div>
<div class="cards">{cards}{conc}</div>
<figure><figcaption class="fig-top">KOSPI against KOSDAQ, {fr.get("first_date")}
to {fr.get("last_date")}</figcaption>
<div class="legend"><span class="lg"><span class="sw acc"></span>KOSPI</span>
<span class="lg"><span class="sw hot"></span>KOSDAQ</span></div>
{trend_chart}
<figcaption>Value-weighted, so the line follows where the money is rather than
the average of small issuers. KOSDAQ sits far below KOSPI throughout: foreign
capital in Korea is concentrated in the large-cap board.</figcaption></figure>
</section>

<section><div class="sec-head"><span class="eyebrow">Daily flow</span>
<h2>Net foreign buying, last 60 sessions</h2></div>
<div class="two">
<figure><figcaption class="fig-top">KOSPI, KRW bn</figcaption>{bars_kospi}</figure>
<figure><figcaption class="fig-top">KOSDAQ, KRW bn</figcaption>{bars_kosdaq}</figure>
</div>
<p class="note">The daily change in shares held, valued at that day's close.
Because it is a change in holdings rather than reported trading, it also moves on
lending, custody transfers and in-kind index creation &mdash; the two series agree
with the exchange's own net-buying figure at only 0.10, which is an open question
in this project.</p></section>

<section><div class="sec-head"><span class="eyebrow">Cross-section</span>
<h2>How wide is the move?</h2></div>
<figure>{hist_chart}
<figcaption>Every name in the universe, binned by its 20-day change in foreign
ownership. A tall centre means the market moved as a whole rather than a handful
of issuers being repositioned; fat tails mean the opposite.</figcaption></figure>
</section>

{ride_sec}
{movers}
{lim_sec}

<footer>
<p>Generated {esc(d.get("generated"))} by <span class="mono">14_monitor.py</span>
from the local store. The source is the exchange's end-of-day file; there is no
intraday feed, so the freshest possible view is the previous close.</p>
<p>Names with under {num(d.get("min_adv",0)/1e8,0)}00m KRW of 20-day average traded
value are excluded &mdash; in thinner names a rank change is noise. &ldquo;ADV
days&rdquo; is the net share change valued at the last close divided by average daily
traded value: how many sessions of normal volume the move represents. English names
come from <span class="mono">data/names_en.csv</span>
({d.get("english_names",0)} mapped); a row whose Korean name no longer matches the
exchange record is dropped rather than applied.</p>
</footer></div></body></html>
"""


CSS = """
:root{
  --ground:#07090D; --surface:#0D1220; --surface-2:#131A2C;
  --ink:#DDE4F0; --ink-soft:#8FA0BC; --ink-faint:#7C8CA8;
  --rule:#1E2941; --rule-soft:#161F33;
  --accent:#5F94E6; --hot:#DD6E56; --ok:#5BB98B; --warn:#D6A94A;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-size:15px;line-height:1.62;
font-family:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:0 24px 80px}
h1,h2,h3{font-family:Newsreader,Georgia,serif;margin:0;text-wrap:balance}
h1{font-size:clamp(1.9rem,4.4vw,2.7rem);font-weight:500;line-height:1.14;letter-spacing:-.015em}
h2{font-size:1.45rem;font-weight:500} h3{font-size:1rem;font-weight:600;margin-bottom:4px}
p{margin:0} a{color:var(--accent)}
.mono,.num,.tk{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;
font-variant-numeric:tabular-nums}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:.7rem;letter-spacing:.14em;
text-transform:uppercase;color:var(--ink-faint);font-weight:500}
a.navlink{text-decoration:none;color:var(--accent)}
header{border-bottom:1px solid var(--rule);background:var(--surface);margin-bottom:44px}
.head-inner{max-width:1180px;margin:0 auto;padding:44px 24px 34px;display:flex;
flex-direction:column;gap:18px}
.kicker{display:flex;gap:16px;flex-wrap:wrap;align-items:baseline}
.caveat{padding:16px 18px;border-left:3px solid var(--warn);background:var(--surface-2);
max-width:72ch;border-radius:0 6px 6px 0;font-size:.92rem;color:var(--ink-soft)}
.caveat strong{color:var(--ink)}
.alarm{padding:14px 16px;border:1px solid var(--hot);border-radius:8px;color:var(--hot);
font-size:.9rem;background:var(--surface)}
section{margin-bottom:52px}
.sec-head{display:flex;flex-direction:column;gap:6px;margin-bottom:18px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:14px;
margin-bottom:20px}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:8px;
padding:16px 18px;display:flex;flex-direction:column;gap:7px}
.card.wide{grid-column:1/-1}
.big-num{font-family:"IBM Plex Mono",monospace;font-size:2rem;font-weight:500;line-height:1;
font-variant-numeric:tabular-nums}
.unit{font-size:.95rem;color:var(--ink-faint);margin-left:2px}
.deltas{display:flex;gap:13px;font-family:"IBM Plex Mono",monospace;font-size:.76rem;
color:var(--ink-soft);flex-wrap:wrap}
.sub{font-size:.8rem;color:var(--ink-faint)}
.sub2{display:block;font-size:.72rem;color:var(--ink-faint);
font-family:"IBM Plex Sans",sans-serif}
figure{margin:0;background:var(--surface);border:1px solid var(--rule);border-radius:8px;
padding:18px 18px 14px}
figcaption{font-size:.83rem;color:var(--ink-soft);margin-top:12px;max-width:74ch}
figcaption.fig-top{margin:0 0 12px;color:var(--ink-faint);
font-family:"IBM Plex Mono",monospace;font-size:.72rem;letter-spacing:.09em;
text-transform:uppercase}
svg{display:block;width:100%;height:auto;overflow:visible}
.tick{font-family:"IBM Plex Mono",monospace;font-size:10px;fill:var(--ink-faint);
font-variant-numeric:tabular-nums}
.dlabel{font-family:"IBM Plex Mono",monospace;font-size:11px;font-weight:500;
font-variant-numeric:tabular-nums}
.legend{display:flex;gap:16px;margin-bottom:12px}
.lg{display:flex;align-items:center;gap:6px;font-size:.8rem;color:var(--ink-soft)}
.sw{width:11px;height:11px;border-radius:2px}
.sw.acc{background:var(--accent)} .sw.hot{background:var(--hot)}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:18px}
.scroll{overflow-x:auto;border:1px solid var(--rule);border-radius:8px;
background:var(--surface);margin-top:10px}
table{border-collapse:collapse;width:100%;font-size:.84rem;font-variant-numeric:tabular-nums}
th,td{padding:7px 11px;text-align:right;white-space:nowrap;
border-bottom:1px solid var(--rule-soft)}
th:first-child,td:first-child,td.nm,th:nth-child(2){text-align:left}
thead th{font-family:"IBM Plex Mono",monospace;font-size:.65rem;letter-spacing:.08em;
text-transform:uppercase;color:var(--ink-faint);font-weight:600;background:#101728;
border-bottom:1px solid var(--rule)}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover td{background:#121A2B}
td.nm{max-width:210px;white-space:normal;line-height:1.3}
td.mk{font-size:.72rem;color:var(--ink-faint)}
td.soft{color:var(--ink-faint)} td.big{font-weight:600}
.note{font-size:.83rem;color:var(--ink-soft);margin-top:12px;max-width:74ch}
::selection{background:#1E3A63;color:var(--ink)}
a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
footer{border-top:1px solid var(--rule);padding-top:22px;font-size:.83rem;
color:var(--ink-faint);display:flex;flex-direction:column;gap:9px;max-width:80ch}
@media (max-width:640px){.wrap{padding:0 16px 56px}.head-inner{padding:34px 16px 26px}
.two{grid-template-columns:1fr}}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=500)
    ap.add_argument("--render-only", action="store_true")
    a = ap.parse_args()
    dest = RESULTS / "monitor.json"
    if a.render_only:
        if not dest.exists():
            sys.exit("no results/monitor.json to render")
        d = json.loads(dest.read_text(encoding="utf-8"))
    else:
        d = compute(a.lookback)
        RESULTS.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(d, indent=2, default=float), encoding="utf-8")
        print(f"  wrote {dest.relative_to(ROOT)}")
    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / "monitor.html"
    out.write_text(render(d), encoding="utf-8")
    print(f"  wrote {out.relative_to(ROOT)}  ({out.stat().st_size:,} bytes)")
    print(f"  last trading day {d['freshness']['last_date']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
