#!/usr/bin/env python3
"""Step 31: three questions the results themselves raised.

    python 31_mechanism.py

These are not another search. Each one is forced by something already
measured, each predicts a SIGN before it is run, and all three are declared
here as a family of exactly three so the permutation bar at the end is honest.

A. ATTENTION. The Fama-MacBeth coefficient (day-weighted) and the portfolio
   contrast (position-weighted) differ by a factor of five. That gap says the
   effect lives on days with few filers and fades on the crowded days that
   carry most of the positions. If investors are distracted when fifty
   companies report at once, prices absorb news slowly and positioning
   information never gets expressed; on a quiet day attention is focused and
   it does. PREDICTION: the coefficient is MORE NEGATIVE on quiet days.

B. WHY SIXTY SESSIONS. Sixty sessions is a quarter, which is exactly the gap
   to the next report. A crowded position unwinding should take days, not a
   full cycle. The alternative is that institutions which loaded up before one
   result simply buy less before the next, so what looks like a return effect
   is the shadow of a flow cycle. PREDICTION: institutional buying before
   consecutive events is NEGATIVELY autocorrelated at the issuer level, and
   controlling for the next event's buying weakens the return coefficient.

C. AVOIDANCE. Everything so far asked what to buy. Long-only books also need
   a do-not-touch list, and that side should be stronger: a disappointing
   result in a name institutions had been accumulating combines bad news with
   a position that has to come off. PREDICTION: the miss-and-crowded cell is
   the worst of the four.
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
CTRL = ["c_mom20", "c_mom60", "c_size", "c_vol", "c_turn"]
BASE, H = "i_flow20", 60
MIN_N = 25


def rank_std(v):
    if np.nanstd(v) <= 1e-12:          # a constant column has no ranking
        return np.zeros(len(v))
    o = v.argsort()
    r = np.empty(len(v), float)
    r[o] = np.arange(len(v), dtype=float)
    r -= r.mean()
    s = r.std()
    return r / s if s > 1e-12 else r


def skey(ts):
    return ts.year * 4 + (ts.month - 1) // 3


def season_t(vals, seas, lag=1):
    per = np.array([np.nanmean(vals[seas == s]) for s in np.unique(seas)])
    per = per[np.isfinite(per)]
    if len(per) < 8:
        return np.nan, np.nan, len(per)
    return float(per.mean()), tracker._nw_t(per, lag), len(per)


def day_level_diff(coef, quiet_m, busy_m, lag=5):
    """Difference in the daily coefficient between quiet and busy days,
    estimated as a regression on a dummy with a Newey-West standard error."""
    m = quiet_m | busy_m
    y = coef[m]
    x = quiet_m[m].astype(float)
    X = np.column_stack([x, np.ones(len(y))])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    n = len(y)
    XtX_inv = np.linalg.pinv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, min(lag, n - 1) + 1):
        w = 1.0 - L / (lag + 1.0)
        A = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += w * (A + A.T)
    V = XtX_inv @ S @ XtX_inv
    se = float(np.sqrt(max(V[0, 0], 1e-24)))
    return float(b[0]), float(b[0] / se)


def fm(d, xs, inter=None, shuffle=None, y=None):
    """Day-by-day regression; returns (coefficient series, season labels)."""
    y = y or f"abn{H}"
    names = list(xs) + ([inter] if inter else [])
    out, seas = {k: [] for k in names}, []
    for day, g in d.groupby("D"):
        g = g.dropna(subset=[y] + [c for c in xs if c in g.columns])
        if len(g) < MIN_N:
            continue
        cols, n = [], len(g)
        for c in xs:
            v = rank_std(g[c].to_numpy(float))
            if shuffle is not None and c == BASE:
                v = v[shuffle.permutation(n)]
            cols.append(v)
        if inter:
            a, b = inter.split("*")
            va = cols[xs.index(a)] if a in xs else rank_std(g[a].to_numpy(float))
            vb = cols[xs.index(b)] if b in xs else rank_std(g[b].to_numpy(float))
            cols.append(va * vb)
        X = np.column_stack(cols + [np.ones(n)])
        yy = g[y].to_numpy(float)
        try:
            bta, *_ = np.linalg.lstsq(X, yy, rcond=None)
        except np.linalg.LinAlgError:
            continue
        for k, val in zip(names, bta[:len(names)]):
            out[k].append(float(val))
        seas.append(skey(day))
    return {k: np.array(v) for k, v in out.items()}, np.array(seas)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=500)
    a = ap.parse_args()
    if not PANEL.exists():
        sys.exit("run 20_event_panel.py first")
    d = pd.read_parquet(PANEL)
    d = d[(d["kind"] == "provisional") & d[BASE].notna()
          & d["surprise"].notna()].copy()
    d["n_filers"] = d.groupby("D")["ticker"].transform("size").astype(float)
    print(f"provisional events: {len(d):,}   "
          f"filers per day: median {d['n_filers'].median():.0f}, "
          f"range {d['n_filers'].min():.0f}-{d['n_filers'].max():.0f}\n")
    out, tstats = {}, {}

    xs = [BASE] + CTRL
    co, se = fm(d, xs)
    b_mn, b_t, _ = season_t(co[BASE], se)
    nf = d.groupby("D")["n_filers"].first()
    keep = [dd for dd in sorted(d.groupby("D").groups)
            if len(d[d["D"] == dd].dropna(
                subset=[f"abn{H}"] + CTRL + [BASE])) >= MIN_N]
    cnt = np.array([nf[dd] for dd in keep])[:len(co[BASE])]
    lo_c, hi_c = np.quantile(cnt, [0.4, 0.6])
    quiet_m, busy_m = cnt <= lo_c, cnt >= hi_c
    q_mn, q_t, q_n = season_t(co[BASE][quiet_m], se[quiet_m])
    b2_mn, b2_t, b2_n = season_t(co[BASE][busy_m], se[busy_m])
    diff, dt_ = day_level_diff(co[BASE], quiet_m, busy_m)
    print("  A. ATTENTION -- is the effect different on quiet reporting days?")
    print(f"     all days               {b_mn*1e4:+8.1f} bp/SD   t {b_t:+6.2f}")
    print(f"     quiet days (<={lo_c:.0f} filers) {q_mn*1e4:+8.1f} bp/SD"
          f"   t {q_t:+6.2f}  ({q_n} seasons)")
    print(f"     busy days  (>={hi_c:.0f} filers) {b2_mn*1e4:+8.1f} bp/SD"
          f"   t {b2_t:+6.2f}  ({b2_n} seasons)")
    print(f"     quiet minus busy       {diff*1e4:+8.1f} bp/SD   t {dt_:+6.2f}"
          f"   (predicted NEGATIVE)")
    tstats["A"] = dt_
    out["A_attention"] = {"all_bp": b_mn * 1e4, "all_t": b_t,
                          "quiet_bp": q_mn * 1e4, "quiet_t": q_t,
                          "busy_bp": b2_mn * 1e4, "busy_t": b2_t,
                          "diff_bp": diff * 1e4, "diff_t": dt_,
                          "cut_lo": float(lo_c), "cut_hi": float(hi_c)}

    print("\n  B. WHY SIXTY SESSIONS -- is this a flow cycle?")
    d = d.sort_values(["ticker", "D"])
    d["next_flow"] = d.groupby("ticker")[BASE].shift(-1)
    pair = d.dropna(subset=[BASE, "next_flow"])
    rho = float(pair[BASE].corr(pair["next_flow"]))
    per = pair.groupby(pair["D"].map(skey)).apply(
        lambda g: g[BASE].corr(g["next_flow"]) if len(g) > 10 else np.nan)
    per = per.dropna().to_numpy()
    rt = tracker._nw_t(per, 1) if len(per) >= 8 else np.nan
    print("     correlation of institutional buying across consecutive events")
    print(f"     for the same issuer: {rho:+.4f}   season t {rt:+6.2f}"
          f"   (predicted NEGATIVE)")
    co2, se2 = fm(d.dropna(subset=["next_flow"]), [BASE, "next_flow"] + CTRL)
    m3, t3, _ = season_t(co2[BASE], se2)
    m4, t4, _ = season_t(co2["next_flow"], se2)
    print("     with the NEXT event's buying in the same regression:")
    print(f"       {BASE:<12} {m3*1e4:+8.1f} bp/SD  t {t3:+6.2f}"
          f"   (was {b_mn*1e4:+.1f}, t {b_t:+.2f})")
    print(f"       next_flow    {m4*1e4:+8.1f} bp/SD  t {t4:+6.2f}")
    tstats["B"] = rt
    out["B_cycle"] = {"rho": rho, "t": rt, "base_with_next_bp": m3 * 1e4,
                      "base_with_next_t": t3, "next_bp": m4 * 1e4, "next_t": t4}

    print("\n  C. AVOIDANCE -- the four cells, market-adjusted bp per position")
    d["sur_r"] = d.groupby("D")["surprise"].rank(pct=True)
    d["base_r"] = d.groupby("D")[BASE].rank(pct=True)
    cells = {}
    for sl, sm in (("beat", d["sur_r"] >= 0.6), ("miss", d["sur_r"] <= 0.4)):
        for bl, bm in (("quiet", d["base_r"] <= 0.4),
                       ("crowded", d["base_r"] >= 0.6)):
            g = d[sm & bm].dropna(subset=[f"abn{H}"])
            mn, t, ns = season_t(g[f"abn{H}"].to_numpy(float),
                                 g["D"].map(skey).to_numpy())
            cells[f"{sl}/{bl}"] = {"bp": mn * 1e4, "t": t, "n": int(len(g))}
            print(f"     {sl:<5} x {bl:<8} {mn*1e4:+8.1f} bp  t {t:+6.2f}  "
                  f"({len(g):,} events)")
    worst = min(cells, key=lambda k: cells[k]["bp"])
    best = max(cells, key=lambda k: cells[k]["bp"])
    print(f"     worst cell: {worst}   best cell: {best}")
    print("     (predicted worst = miss/crowded)")
    mc = cells["miss/crowded"]["bp"]
    others = np.mean([v["bp"] for k, v in cells.items() if k != "miss/crowded"])
    print(f"     miss/crowded minus the other three: {mc - others:+.1f} bp")
    tstats["C"] = cells["miss/crowded"]["t"]
    out["C_avoidance"] = {"cells": cells, "worst": worst, "best": best,
                          "gap_bp": float(mc - others)}

    print(f"\n  permutation bar for this family of three ({a.perms} draws)")
    rng = np.random.default_rng(31)
    mx = []
    for _ in range(a.perms):
        c, s = fm(d, xs, shuffle=rng)
        cb = c[BASE]
        qm, bm = quiet_m[:len(cb)], busy_m[:len(cb)]
        _, tv = day_level_diff(cb, qm, bm)
        mx.append(abs(tv) if np.isfinite(tv) else 0.0)
    mx = np.array(mx)
    crit = float(np.percentile(mx, 95))
    print(f"     null max |t| p50 {np.percentile(mx,50):.2f}  p95 {crit:.2f}")
    print("     observed |t| by test: "
          + ", ".join(f"{k} {abs(v):.2f}" for k, v in tstats.items()
                      if np.isfinite(v)))
    surv = [k for k, v in tstats.items() if np.isfinite(v) and abs(v) >= crit]
    out["fw_crit95"] = crit
    out["survivors"] = surv
    print(f"     clearing the bar: {', '.join(surv) if surv else 'none'}")

    RESULTS.mkdir(exist_ok=True)
    dest = RESULTS / "mechanism.json"
    dest.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
