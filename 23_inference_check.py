#!/usr/bin/env python3
"""Step 23: is the t-stat we reported the right size, on this data?

    python 23_inference_check.py
    python 23_inference_check.py --kind periodic

WHY
---
Step 22 found two cells that cleared a family-wise bar, and both sit at the
60-session horizon. That is exactly where the inference is weakest: a 60-day
forward window opened on Tuesday and one opened on Thursday cover almost the
same stretch of market, so the two days' coefficients are not independent
draws even though the issuers reporting on them are different. Our
Newey-West correction used a lag of five event days, which was chosen for the
short horizon and never revisited.

Whether that matters is an empirical question about THIS panel, not something
to settle by argument or by a simulation whose structure may not match. So
this measures it directly:

  - the autocorrelation of each coefficient series, which is the thing a HAC
    lag is meant to absorb;
  - the t-stat at a ladder of lags, so a claim that survives only at lag 5 is
    visible as such;
  - a season-level t-stat, where all the days inside one reporting season are
    collapsed to a single observation first. Korean issuers report in four
    short bursts a year, so a season is close to the natural independent unit
    and roughly thirty-four of them is the honest sample size at long
    horizons.

A finding that holds at every lag and at the season level is real inference.
A finding that needs lag 5 is a finding about the lag.
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
BASELINES = ["f_flow20", "f_flow60", "f_hold20", "f_level", "i_flow20"]
HORIZONS = (5, 20, 60)
LAGS = (5, 10, 20, 30)
MIN_N = 25
SEASON_GAP_DAYS = 30


def rank_std(v: np.ndarray) -> np.ndarray:
    o = v.argsort()
    r = np.empty(len(v), dtype=float)
    r[o] = np.arange(len(v), dtype=float)
    r -= r.mean()
    s = r.std()
    return r / s if s > 1e-12 else r


def coef(X, y, k) -> float:
    try:
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan
    return float(b[k])


def series(d: pd.DataFrame, baselines: list[str]):
    """{cell -> daily coefficient series} plus the event dates they belong to."""
    out, dates = {}, []
    need = CTRL + baselines
    for day, g in d.groupby("D"):
        g = g.dropna(subset=need)
        if len(g) < MIN_N:
            continue
        n = len(g)
        sur = rank_std(g["surprise"].to_numpy(dtype=float))
        ctl = np.column_stack([rank_std(g[c].to_numpy(dtype=float)) for c in CTRL])
        ys = {}
        ok = True
        for h in HORIZONS:
            yy = g[f"abn{h}"].to_numpy(dtype=float)
            m = np.isfinite(yy)
            if m.sum() < MIN_N:
                ok = False
                break
            yy = np.where(m, yy, np.nanmedian(yy[m]))
            lo, hi = np.percentile(yy, [1, 99])
            ys[h] = np.clip(yy, lo, hi)
        if not ok:
            continue
        dates.append(day)
        one = np.ones(n)
        beat = sur > np.quantile(sur, 0.6)
        for b in baselines:
            bv = rank_std(g[b].to_numpy(dtype=float))
            Xm = np.column_stack([sur, ctl, bv, one])
            Xi = np.column_stack([sur, ctl, bv, sur * bv, one])
            for h in HORIZONS:
                out.setdefault(f"main|{b}|{h}", []).append(coef(Xm, ys[h], 1 + ctl.shape[1]))
                out.setdefault(f"inter|{b}|{h}", []).append(coef(Xi, ys[h], 2 + ctl.shape[1]))
            if beat.sum() >= 15:
                Xb = np.column_stack([ctl[beat], bv[beat], np.ones(int(beat.sum()))])
                out.setdefault(f"beats|{b}|20", []).append(
                    coef(Xb, ys[20][beat], ctl.shape[1]))
            else:
                out.setdefault(f"beats|{b}|20", []).append(np.nan)
    return {k: np.array(v) for k, v in out.items()}, np.array(dates)


def seasons(dates: np.ndarray) -> np.ndarray:
    s, cur = [], 0
    for i, d in enumerate(dates):
        if i and (d - dates[i - 1]).days > SEASON_GAP_DAYS:
            cur += 1
        s.append(cur)
    return np.array(s)


def ac(x: np.ndarray, k: int) -> float:
    x = x[np.isfinite(x)]
    if len(x) <= k + 2:
        return np.nan
    a, b = x[k:], x[:-k]
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="provisional",
                    choices=["provisional", "periodic"])
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()
    if not PANEL.exists():
        sys.exit("run 20_event_panel.py first")
    d = pd.read_parquet(PANEL)
    d = d[d["kind"] == a.kind].copy()
    have = [b for b in BASELINES if b in d.columns and d[b].notna().mean() > 0.3]
    ser, dates = series(d, have)
    sid = seasons(dates)
    n_seas = len(np.unique(sid))
    print(f"{a.kind}: {len(dates)} event days -> {n_seas} reporting seasons")
    print(f"  (a season is the honest unit once outcome windows overlap)\n")

    rows = []
    for k, v in ser.items():
        fin = v[np.isfinite(v)]
        if len(fin) < 12:
            continue
        per = np.array([np.nanmean(v[sid == s]) for s in np.unique(sid)])
        per = per[np.isfinite(per)]
        r = {"cell": k, "days": int(len(fin)), "ac1": ac(v, 1), "ac2": ac(v, 2),
             "t_season": tracker._nw_t(per, 1) if len(per) >= 8 else np.nan,
             "seasons": int(len(per))}
        for L in LAGS:
            r[f"t{L}"] = tracker._nw_t(v[np.isfinite(v)], L)
        rows.append(r)
    rows.sort(key=lambda r_: -abs(r_["t5"]))

    print("  cell                   AC(1)  AC(2) |   t@5    t@10   t@20   t@30 |"
          "  season t  (n)")
    for r in rows[:a.top]:
        print(f"  {r['cell']:<22} {r['ac1']:+5.2f} {r['ac2']:+5.2f} | "
              f"{r['t5']:+6.2f} {r['t10']:+6.2f} {r['t20']:+6.2f} {r['t30']:+6.2f} |"
              f" {r['t_season']:+8.2f}  ({r['seasons']})")

    print("\n  reading it: a cell whose |t| falls away as the lag grows was")
    print("  borrowing significance from overlapping windows. The season column")
    print("  is the one to quote in a report -- it does not depend on a lag")
    print("  choice at all.")
    surv = [r for r in rows if abs(r["t_season"]) >= 2 and
            all(abs(r[f"t{L}"]) >= 2 for L in LAGS)]
    print(f"\n  cells significant at EVERY lag and at season level: "
          f"{', '.join(r['cell'] for r in surv) if surv else 'none'}")
    print("  (before any correction for having searched 35 cells -- the")
    print("   family-wise bar from step 22 still applies on top of this)")

    RESULTS.mkdir(exist_ok=True)
    dest = RESULTS / f"inference_{a.kind}.json"
    dest.write_text(json.dumps({"kind": a.kind, "days": int(len(dates)),
                                "seasons": n_seas, "cells": rows,
                                "robust": [r["cell"] for r in surv]},
                               indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
