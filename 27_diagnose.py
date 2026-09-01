#!/usr/bin/env python3
"""Step 27: subtract the machinery from step 26's table.

    python 27_diagnose.py
    python 27_diagnose.py --draws 40

WHAT WENT WRONG IN STEP 26
--------------------------
Every rule lost 13 to 24 per cent a year against a size-matched benchmark,
including "buy every beat". Size-matched excess returns average zero inside
each decile every day, so that cannot be right for every basket at once.

A book of randomly chosen names -- true edge exactly zero -- run through the
same code reads about -13%/yr at a twenty-session hold and -3.5%/yr at sixty.
The cause is the calendar. Korean issuers report in four short bursts, so a
twenty-session book is invested about half the year but pays all its costs
inside those weeks. Averaging over invested days and multiplying by 250
charges a full year of trading to a third of a year of exposure, and the
shorter the hold the worse it gets. That is why step 26's table looked worst
exactly where the book was thinnest.

THE FIX
-------
Rather than argue about the size of the bias, measure it. For every rule and
every holding period, build placebo books on the SAME entry dates with the
SAME number of positions, drawn at random from the liquid universe. Their
average is what the machinery produces from nothing. Subtract it.

The per-position column is the one to trust. It is the cumulative size-matched
excess of a single position over its own window: no annualising, no book
weighting, thousands of observations, and it sits at zero on data with no edge.
The debiased annual figures still carry three or four points of noise, so read
them as direction, not as a number.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import tracker

ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "results" / "event_panel.parquet"
RESULTS = ROOT / "results"
MIN_ADV = 1e8
N_BUCKETS = 10
COST_RT = 0.0050
HOLDS = (20, 40, 60)
BASE = "i_flow20"


def matched_excess(ret, cap, ok):
    out = np.full_like(ret, np.nan)
    for i in range(ret.shape[0]):
        m = ok[i] & np.isfinite(ret[i]) & np.isfinite(cap[i])
        if m.sum() < 50:
            continue
        c = cap[i][m]
        edges = np.quantile(c, np.linspace(0, 1, N_BUCKETS + 1)[1:-1])
        b = np.digitize(c, edges)
        r = ret[i][m]
        adj = np.empty_like(r)
        for g in range(N_BUCKETS):
            sel = b == g
            adj[sel] = r[sel] - r[sel].mean() if sel.sum() >= 3 else np.nan
        out[i][np.where(m)[0]] = adj
    return out


def daily_book(tick, entry, exc, n_dates, col, hold):
    held, entries = {}, {}
    for t_, e in zip(tick, entry):
        j = col.get(t_) if isinstance(t_, str) else t_
        if j is None:
            continue
        for k in range(int(e) + 1, min(int(e) + 1 + hold, n_dates)):
            held.setdefault(k, []).append(j)
        entries[int(e) + 1] = entries.get(int(e) + 1, 0) + 1
    ex = np.full(n_dates, np.nan)
    for k, js in held.items():
        r = exc[k, js]
        r = r[np.isfinite(r)]
        if len(r):
            ex[k] = float(r.mean()) - (COST_RT * entries.get(k, 0) / len(js))
    return ex


def ann(ex):
    f = np.isfinite(ex)
    return float(ex[f].mean() * 250 * 100) if f.sum() > 100 else np.nan


def per_position(tick, entry, exc, n_dates, col, hold):
    v = []
    for t_, e in zip(tick, entry):
        j = col.get(t_)
        if j is None:
            continue
        w = exc[int(e) + 1:min(int(e) + 1 + hold, n_dates), j]
        w = w[np.isfinite(w)]
        if len(w) >= hold // 2:
            v.append(float(w.sum()))
    return np.array(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-06-01")
    ap.add_argument("--draws", type=int, default=25)
    a = ap.parse_args()
    if not PANEL.exists():
        sys.exit("run 20_event_panel.py first")

    d = pd.read_parquet(PANEL)
    d = d[(d["kind"] == "provisional") & d["surprise"].notna()
          & d[BASE].notna()].copy()

    from krxflow import features, storage
    p = features.load_panels(a.start, None)
    pct, uni = p["foreign_pct"], features.universe_mask(p)
    m = storage.read_range("market", a.start, None,
                           columns=["trade_date", "ticker", "close",
                                    "value_traded", "market_cap"])
    m["trade_date"] = pd.to_datetime(m["trade_date"])

    def pv(c):
        return (m.pivot_table(index="trade_date", columns="ticker", values=c,
                              aggfunc="last", observed=True)
                  .sort_index().astype("float64")
                  .reindex(index=pct.index, columns=pct.columns))

    close, vt, cap = pv("close"), pv("value_traded"), pv("market_cap")
    del m
    adv = vt.rolling(20, min_periods=5).mean()
    ok = (uni & (adv >= MIN_ADV) & close.notna()).to_numpy()
    r = close.pct_change(fill_method=None).mask(lambda x: x.abs() > 0.5).to_numpy()
    print("building size-matched excess ...")
    exc = matched_excess(r, cap.to_numpy(), ok)
    n_dates = len(pct.index)
    col = {t: j for j, t in enumerate(pct.columns)}
    live = [np.where(ok[i])[0] for i in range(n_dates)]
    allv = exc[np.isfinite(exc)]
    print(f"  sanity: matched excess over every stock-day "
          f"{allv.mean()*1e4:+.3f} bp (zero by construction)")

    d["sur_r"] = d.groupby("D")["surprise"].rank(pct=True)
    d["base_r"] = d.groupby("D")[BASE].rank(pct=True)
    beat, quiet = d["sur_r"] >= 0.6, d["base_r"] <= 0.4
    crowd = d["base_r"] >= 0.6
    rules = {"all beats": d[beat], "quiet beats": d[beat & quiet],
             "quiet only": d[quiet], "crowded beats": d[beat & crowd]}

    rng = np.random.default_rng(11)
    out = {"draws": a.draws, "by_hold": {}}
    for hold in HOLDS:
        print(f"\n  hold {hold} sessions "
              f"(placebo = {a.draws} random books, same dates and counts)")
        print("    rule            observed   placebo  debiased | per position bp")
        cell, ex_of = {}, {}
        for name, picks in rules.items():
            tk = picks["ticker"].to_numpy()
            ent = picks["entry"].to_numpy()
            ex = daily_book(tk, ent, exc, n_dates, col, hold)
            ex_of[name] = ex
            obs = ann(ex)
            pl = []
            for _ in range(a.draws):
                js = [int(rng.choice(live[int(e)])) if len(live[int(e)]) else None
                      for e in ent]
                pl.append(ann(daily_book(js, ent, exc, n_dates, col, hold)))
            pl = float(np.nanmean(pl))
            pp = per_position(tk, ent, exc, n_dates, col, hold)
            cell[name] = {"observed": obs, "placebo": pl, "debiased": obs - pl,
                          "per_position_bp": float(pp.mean() * 1e4),
                          "n_positions": int(len(pp))}
            print(f"    {name:<14} {obs:+8.2f}  {pl:+8.2f}  {obs-pl:+8.2f} | "
                  f"{pp.mean()*1e4:+12.1f}")
        gain_obs = ann(ex_of["quiet beats"] - ex_of["all beats"])
        gd = (cell["quiet beats"]["debiased"] - cell["all beats"]["debiased"])
        pq = cell["quiet beats"]["per_position_bp"]
        pa = cell["all beats"]["per_position_bp"]
        cell["filter_gain"] = {"raw_diff": gain_obs, "debiased": gd,
                               "per_position_bp": pq - pa}
        print(f"    {'filter gain':<14} {gain_obs:+8.2f}  {'':>8}  {gd:+8.2f} | "
              f"{pq-pa:+12.1f}")
        out["by_hold"][str(hold)] = cell

    print("\n  the per-position column is the cleanest number here: cumulative")
    print("  size-matched excess of one position over its own window, with no")
    print("  annualising and no book weighting to distort it.")
    gains = [out["by_hold"][str(h)]["filter_gain"]["per_position_bp"]
             for h in HOLDS]
    print(f"  filter gain per position at {HOLDS}: "
          + ", ".join(f"{g:+.0f}bp" for g in gains))
    out["verdict"] = ("FILTER POSITIVE AT EVERY HOLD" if all(g > 0 for g in gains)
                      else "FILTER NOT CONSISTENTLY POSITIVE")
    print(f"\n  verdict: {out['verdict']}")

    RESULTS.mkdir(exist_ok=True)
    dest = RESULTS / "diagnose.json"
    dest.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
