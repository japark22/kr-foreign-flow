#!/usr/bin/env python3
"""Step 16: the flipped use case -- foreign ownership as PRE-EVENT positioning.

    python 16_event_study.py                    # provisional (잠정실적) events
    python 16_event_study.py --kind periodic    # periodic reports instead

HYPOTHESIS UNDER TEST
---------------------
If foreign investors have piled into a stock ahead of its earnings release,
the position is crowded and more likely to UNWIND after the result. So:
abnormal pre-event accumulation -> negative post-event abnormal return, and
the ownership series itself should fall back after the event.

DESIGN, STATED BEFORE THE NUMBERS
---------------------------------
Event day D    = first trading day on/after the DART filing date. Filing TIME
                 is not available, so positions enter at the close of D+1 and
                 every post-event window starts there.
Pre signal z   = the [-20d, -1d] change in foreign ownership, standardised by
                 that stock's own trailing 250d distribution of 20d changes
                 (window shifted so it never overlaps the signal window).
Outcomes       = market-adjusted (equal-weight universe) returns over
                 (D+1 -> D+1+5) and (D+1 -> D+1+20), with the split-guard from
                 tracker.py; and the ownership change (D+1 -> D+21): unwind?
Inference      = events are averaged BY DAY first (announcements cluster in
                 season, so per-event t-stats would be fake), then Newey-West
                 over the day series. House rule from step 5 of the record.
Filters        = universe mask, 20d ADV >= 1e8 KRW, corrections dropped,
                 repeat filings within 10 sessions of the same ticker dropped.
Verdict rule (pre-registered): the unwind claim needs the top-quintile 20d
abnormal return negative with |t| >= 2 AND the top-bottom spread |t| >= 2.
Anything else is "not established".
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import tracker

ROOT = Path(__file__).resolve().parent
EVENTS = ROOT / "data" / "events" / "earnings_events.parquet"
RESULTS = ROOT / "results"
MIN_ADV = 1e8
HORIZONS = (5, 20)


def zscore_panel(pct: pd.DataFrame) -> pd.DataFrame:
    """Per-stock z of the 20d ownership change vs its own trailing history."""
    d20 = pct - pct.shift(20)
    hist = d20.shift(21)                       # exclude the signal window itself
    mu = hist.rolling(250, min_periods=120).mean()
    sd = hist.rolling(250, min_periods=120).std()
    return (d20 - mu) / sd.where(sd > 1e-9)


def by_day_t(df: pd.DataFrame, col: str, lag: int) -> tuple[float, float, int]:
    """(mean, NW t, n_days) after averaging events within each event day."""
    day = df.groupby("D")[col].mean().dropna()
    if len(day) < 8:
        return float("nan"), float("nan"), len(day)
    return float(day.mean()), tracker._nw_t(day.to_numpy(), lag), len(day)


def study(pct, close, vt, uni, events: pd.DataFrame) -> dict:
    idx = pct.index
    adv = vt.rolling(20, min_periods=5).mean()
    mask = (uni & (adv >= MIN_ADV) & pct.notna())
    z = zscore_panel(pct)

    f = {h: tracker.clean_forward(close, h) for h in HORIZONS}
    mkt = {h: f[h].where(mask).mean(axis=1) for h in HORIZONS}
    react = close.shift(-1) / close.shift(1) - 1.0          # D-1 close -> D+1 close
    dpost = pct.shift(-21) - pct.shift(-1)                  # D+1 -> D+21, in pp

    col = {t: j for j, t in enumerate(pct.columns)}
    zV, mV = z.to_numpy(), mask.to_numpy()
    fV = {h: f[h].to_numpy() for h in HORIZONS}
    mktV = {h: mkt[h].to_numpy() for h in HORIZONS}
    rV, dV = react.to_numpy(), dpost.to_numpy()

    ev = events[~events["is_correction"]].copy()
    ev["dt"] = pd.to_datetime(ev["rcept_dt"], format="%Y%m%d")
    ev = ev[ev["ticker"].isin(col)]
    pos = idx.searchsorted(ev["dt"].to_numpy())
    ev["p"] = pos
    ev = ev[(ev["p"] >= 300) & (ev["p"] + 22 < len(idx))]
    ev = ev.sort_values(["ticker", "p"])
    ev = ev[~((ev["ticker"] == ev["ticker"].shift())
              & (ev["p"] - ev["p"].shift() <= 10))]          # repeat filings

    rows = []
    for t_, p_ in zip(ev["ticker"].to_numpy(), ev["p"].to_numpy()):
        j = col[t_]
        if not mV[p_ - 1, j]:
            continue
        zz = zV[p_ - 1, j]
        if not np.isfinite(zz):
            continue
        e = p_ + 1                                           # entry row (close D+1)
        r = {"ticker": t_, "D": idx[p_], "z": zz,
             "react": rV[p_, j], "dflow20": dV[p_, j]}
        for h in HORIZONS:
            raw = fV[h][e, j]
            r[f"abn{h}"] = raw - mktV[h][e] if np.isfinite(raw) else np.nan
        rows.append(r)
    d = pd.DataFrame(rows)
    if len(d) < 500:
        sys.exit(f"only {len(d)} usable events -- check the panels cover 2018+")

    d["q"] = pd.qcut(d["z"], 5, labels=False) + 1
    out = {"n_events": int(len(d)), "quintiles": [], "generated":
           dt.datetime.now().strftime("%Y-%m-%d %H:%M")}

    print(f"\n  usable events: {len(d):,}  "
          f"({d['D'].min().date()} .. {d['D'].max().date()})")
    print("\n  pre-event accumulation quintile -> post-event outcome")
    print("  (abn = market-adjusted, bp; flow = ownership change D+1->D+21, pp)")
    print("  q     n      z-mean   abn5bp  t(NW)   abn20bp  t(NW)   flow20pp  t(NW)")
    for qq in range(1, 6):
        s = d[d["q"] == qq]
        a5, t5, _ = by_day_t(s, "abn5", 3)
        a20, t20, _ = by_day_t(s, "abn20", 5)
        fl, tf, _ = by_day_t(s, "dflow20", 5)
        out["quintiles"].append({"q": qq, "n": int(len(s)),
                                 "z": float(s["z"].mean()),
                                 "abn5_bp": a5 * 1e4, "t5": t5,
                                 "abn20_bp": a20 * 1e4, "t20": t20,
                                 "flow20_pp": fl, "tf": tf})
        print(f"  {qq}  {len(s):>6,}   {s['z'].mean():+6.2f}  "
              f"{a5*1e4:+7.1f}  {t5:+5.2f}   {a20*1e4:+7.1f}  {t20:+5.2f}   "
              f"{fl:+7.3f}  {tf:+5.2f}")

    # headline: crowded (Q5) minus uncrowded (Q1), by day then NW
    print("\n  headline (Q5 - Q1 spread):")
    spread = {}
    for h in HORIZONS:
        top = d[d["q"] == 5].groupby("D")[f"abn{h}"].mean()
        bot = d[d["q"] == 1].groupby("D")[f"abn{h}"].mean()
        both = pd.concat([top.rename("t"), bot.rename("b")], axis=1).dropna()
        sp = (both["t"] - both["b"])
        tt = tracker._nw_t(sp.to_numpy(), 5 if h == 5 else 8)
        spread[str(h)] = {"bp": float(sp.mean() * 1e4), "t": tt,
                          "n_days": int(len(sp))}
        print(f"    {h:>2}d: {sp.mean()*1e4:+7.1f} bp  t {tt:+5.2f}  "
              f"({len(sp)} matched days)")
    out["spread"] = spread

    # interaction: within Q5, did the news land well or badly?
    s5 = d[(d["q"] == 5) & d["react"].notna()]
    print("\n  Q5 (crowded) split by the announcement reaction (D-1 -> D+1):")
    inter = {}
    for name, sub in (("news up", s5[s5["react"] > 0]),
                      ("news down", s5[s5["react"] <= 0])):
        a20, t20, nd = by_day_t(sub, "abn20", 5)
        inter[name] = {"n": int(len(sub)), "abn20_bp": a20 * 1e4, "t": t20}
        print(f"    {name:<10} n {len(sub):>6,}   abn20 {a20*1e4:+7.1f} bp  "
              f"t {t20:+5.2f}")
    out["q5_by_reaction"] = inter

    # does the pre-event position anticipate the news itself?
    # This is the "did they already know" test: rank IC between the pre-event
    # z and the announcement reaction, computed within each event day (so the
    # comparison is always across stocks reporting the same day) then averaged.
    print("\n  does pre-event accumulation anticipate the announcement?")
    rr = d[d["react"].notna()]
    ics = []
    for _, g in rr.groupby("D"):
        if len(g) < 8:
            continue
        c = g["z"].rank().corr(g["react"].rank())
        if np.isfinite(c):
            ics.append(float(c))
    if len(ics) >= 8:
        arr = np.array(ics)
        ic_t = tracker._nw_t(arr, 5)
        out["anticipation"] = {"ic": float(arr.mean()), "t": ic_t,
                               "n_days": len(arr)}
        print(f"    rank IC(z, reaction) = {arr.mean():+.4f}  t {ic_t:+5.2f}  "
              f"({len(arr)} days with 8+ same-day reporters)")
    else:
        out["anticipation"] = None
        print("    too few multi-reporter days to measure")
    print("    reaction by quintile (D-1 close -> D+1 close, bp):")
    for qq in range(1, 6):
        sub = rr[rr["q"] == qq]
        mr, tr, _ = by_day_t(sub, "react", 5)
        for row in out["quintiles"]:
            if row["q"] == qq:
                row["react_bp"] = mr * 1e4
                row["react_t"] = tr
        print(f"      q{qq}  {mr*1e4:+7.1f} bp  t {tr:+5.2f}")

    # pre-registered verdict
    q5 = out["quintiles"][-1]
    ok = (q5["abn20_bp"] < 0 and abs(q5["t20"]) >= 2
          and abs(spread["20"]["t"]) >= 2 and spread["20"]["bp"] < 0)
    out["verdict"] = ("UNWIND SUPPORTED" if ok else "NOT ESTABLISHED")
    print(f"\n  verdict by the pre-registered rule: {out['verdict']}")
    print("  (needs Q5 abn20 < 0 with |t|>=2 AND Q5-Q1 20d spread < 0 with |t|>=2)")
    print("  cost reminder: trading either leg costs ~50bp per round trip;")
    print("  any spread smaller than that is a finding, not a trade.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="provisional",
                    choices=["provisional", "periodic"])
    ap.add_argument("--start", default="2017-01-01")
    a = ap.parse_args()

    if not EVENTS.exists():
        sys.exit("run 15_earnings_events.py first")
    events = pd.read_parquet(EVENTS)
    events = events[events["kind"] == a.kind]
    print(f"events loaded: {len(events):,} ({a.kind})")

    from krxflow import features, storage
    p = features.load_panels(a.start, None)
    pct = p["foreign_pct"]
    uni = features.universe_mask(p)
    m = storage.read_range("market", a.start, None,
                           columns=["trade_date", "ticker", "close",
                                    "value_traded"])
    m["trade_date"] = pd.to_datetime(m["trade_date"])

    def pv(colname):
        return (m.pivot_table(index="trade_date", columns="ticker",
                              values=colname, aggfunc="last", observed=True)
                 .sort_index().astype("float64")
                 .reindex(index=pct.index, columns=pct.columns))

    close, vt = pv("close"), pv("value_traded")
    del m

    out = study(pct, close, vt, uni, events)
    out["kind"] = a.kind
    RESULTS.mkdir(exist_ok=True)
    dest = RESULTS / f"event_study_{a.kind}.json"
    dest.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
