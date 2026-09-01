#!/usr/bin/env python3
"""Step 30: does the headline depend on trimming outliers?

    python 30_reconcile.py

THE DISCREPANCY
---------------
Step 24 priced the institutional finding at +268bp per position with t +3.21,
comparing the bottom and top quintiles of pre-event institutional buying over
sixty sessions. Step 29 measured what should be the same contrast on the same
events and got +42bp with t +0.71.

Two choices differ between them, and both are defensible in isolation:

  winsorising  step 24 clipped each day's outcomes at the 1st and 99th
               percentile before averaging, which is standard practice for
               regression coefficients. Step 29 did not.
  weighting    step 24 formed the spread within each day and then averaged
               days; step 29 pooled all events in a season and differenced the
               group means, so a busy day counts for more.

This runs the contrast all four ways. If the answer barely moves, the two
steps simply differed in bookkeeping. If it collapses when the clipping comes
off, the finding lives in the middle of the distribution and is cancelled by
the tails -- and since a real book cannot clip its own losses, the unclipped
number is the one that describes what would have been earned.

The same question is then put to the regression coefficient itself, because
step 24's -131bp per standard deviation was estimated on clipped outcomes too.
"""
from __future__ import annotations

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
BASE = "i_flow20"
H = 60


def rank_std(v):
    o = v.argsort()
    r = np.empty(len(v), float)
    r[o] = np.arange(len(v), dtype=float)
    r -= r.mean()
    s = r.std()
    return r / s if s > 1e-12 else r


def season_key(ts):
    return ts.year * 4 + (ts.month - 1) // 3


def spread_by_day(d, clip):
    """Step 24's construction: spread inside each day, then season-average."""
    rows = []
    for day, g in d.groupby("D"):
        g = g.dropna(subset=[f"abn{H}", BASE])
        if len(g) < 25:
            continue
        y = g[f"abn{H}"].to_numpy(float)
        if clip:
            y = np.clip(y, *np.percentile(y, [1, 99]))
        x = rank_std(g[BASE].to_numpy(float))
        lo, hi = np.quantile(x, 0.2), np.quantile(x, 0.8)
        a, b = y[x <= lo], y[x >= hi]
        if len(a) >= 5 and len(b) >= 5:
            rows.append((season_key(day), float(a.mean() - b.mean())))
    if len(rows) < 8:
        return np.nan, np.nan, 0
    df = pd.DataFrame(rows, columns=["s", "v"])
    per = df.groupby("s")["v"].mean().to_numpy()
    return float(per.mean()), tracker._nw_t(per, 1), len(per)


def spread_pooled(d, clip):
    """Step 29's construction: pool the season, then difference group means."""
    d = d.dropna(subset=[f"abn{H}", BASE]).copy()
    y = d[f"abn{H}"].to_numpy(float).copy()
    if clip:
        for day, idx in d.groupby("D").indices.items():
            y[idx] = np.clip(y[idx], *np.percentile(y[idx], [1, 99]))
    d["_y"] = y
    d["_r"] = d.groupby("D")[BASE].rank(pct=True)
    d["_s"] = d["D"].map(season_key)
    lo = d[d["_r"] <= 0.2].groupby("_s")["_y"].mean()
    hi = d[d["_r"] >= 0.8].groupby("_s")["_y"].mean()
    j = pd.concat([lo.rename("l"), hi.rename("h")], axis=1).dropna()
    if len(j) < 8:
        return np.nan, np.nan, 0
    v = (j["l"] - j["h"]).to_numpy()
    return float(v.mean()), tracker._nw_t(v, 1), len(v)


def coef_series(d, clip):
    out, seas = [], []
    for day, g in d.groupby("D"):
        g = g.dropna(subset=[f"abn{H}", BASE] + CTRL)
        if len(g) < 25:
            continue
        y = g[f"abn{H}"].to_numpy(float)
        if clip:
            y = np.clip(y, *np.percentile(y, [1, 99]))
        X = np.column_stack([rank_std(g[c].to_numpy(float)) for c in CTRL]
                            + [rank_std(g[BASE].to_numpy(float)),
                               np.ones(len(g))])
        try:
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        out.append(float(b[len(CTRL)]))
        seas.append(season_key(day))
    if len(out) < 8:
        return np.nan, np.nan, 0
    df = pd.DataFrame({"s": seas, "v": out})
    per = df.groupby("s")["v"].mean().to_numpy()
    return float(per.mean()), tracker._nw_t(per, 1), len(per)


def main() -> int:
    if not PANEL.exists():
        sys.exit("run 20_event_panel.py first")
    d = pd.read_parquet(PANEL)
    d = d[(d["kind"] == "provisional") & d[BASE].notna()].copy()
    print(f"provisional events with an institutional reading: {len(d):,}\n")

    print(f"  QUINTILE SPREAD at abn{H}, bottom minus top of institutional buying")
    print("    construction              clipped 1/99      unclipped")
    out = {}
    for name, fn in (("spread per day (step 24)", spread_by_day),
                     ("pooled per season (step 29)", spread_pooled)):
        a, ta, na = fn(d, True)
        b, tb, nb = fn(d, False)
        out[name] = {"clipped": {"bp": a * 1e4, "t": ta, "seasons": na},
                     "unclipped": {"bp": b * 1e4, "t": tb, "seasons": nb}}
        print(f"    {name:<26} {a*1e4:+7.1f} (t{ta:+5.2f})  "
              f"{b*1e4:+7.1f} (t{tb:+5.2f})")

    print(f"\n  REGRESSION COEFFICIENT on {BASE}, abn{H}, with controls")
    print("    (step 24 reported -131bp per SD with t -4.41, on clipped data)")
    ca, tca, _ = coef_series(d, True)
    cb, tcb, _ = coef_series(d, False)
    out["coefficient"] = {"clipped": {"bp": ca * 1e4, "t": tca},
                          "unclipped": {"bp": cb * 1e4, "t": tcb}}
    print(f"    clipped 1/99   {ca*1e4:+8.1f} bp/SD   t {tca:+6.2f}")
    print(f"    unclipped      {cb*1e4:+8.1f} bp/SD   t {tcb:+6.2f}")

    shrink = (abs(cb) < abs(ca) * 0.5) or (abs(tcb) < 2 <= abs(tca))
    out["depends_on_clipping"] = bool(shrink)
    print(f"\n  the headline depends on clipping: {shrink}")
    if shrink:
        print("  then the unclipped figures are the ones to quote. A book cannot")
        print("  winsorise its own losses, and a result that needs the tails")
        print("  removed is a result about the middle of the distribution only.")
    else:
        print("  the choice of clipping is bookkeeping, not the result.")

    RESULTS.mkdir(exist_ok=True)
    dest = RESULTS / "reconcile.json"
    dest.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
