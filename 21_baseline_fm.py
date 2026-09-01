#!/usr/bin/env python3
"""Step 21: does the shareholding baseline add anything a price already says?

    python 21_baseline_fm.py                     # primary test
    python 21_baseline_fm.py --kind periodic     # the wider, staler sample

THE HYPOTHESIS, WRITTEN AS A MODEL
----------------------------------
"If the result is good AND foreign investors have not been buying, that is a
stronger signal." That sentence is an interaction, not a main effect: the
payoff to a good result should DECREASE as the foreign baseline rises. So the
coefficient on surprise x baseline is predicted NEGATIVE.

WHY THE CONTROLS DECIDE THIS
----------------------------
"The market already expected it" has an obvious price-based measure: the stock
already went up. Foreign buying and a pre-event run-up travel together, so a
raw baseline result may be nothing but short-term reversal wearing a costume.
The ladder below exists to separate them:

  M1  surprise only ................ is there any post-earnings drift to modulate?
  M2  + size, run-up, vol, turnover . does that drift survive ordinary controls?
  M3  + baseline .................... does ownership add a main effect?
  M4  + surprise x baseline ......... the hypothesis itself

Only the M4 interaction, measured with the controls present, answers the
question. A result that appears in M3 and dies in M2-with-controls was the
run-up all along.

INFERENCE
---------
Fama-MacBeth: one cross-sectional regression per event day, then Newey-West on
the coefficient series. Earnings cluster into four short seasons a year, so
pooled standard errors would be badly overstated. Every regressor is
rank-standardised within the day, so a coefficient reads as basis points per
one standard deviation of rank, and no single outlier can carry a day.

PRE-REGISTERED, BEFORE ANY NUMBER IS SEEN
-----------------------------------------
PRIMARY test: provisional events, 20-session outcome, baseline f_flow20.
The claim is supported only if the M4 interaction is negative with |t| >= 2.
Everything else printed is robustness and is labelled as such. One primary
test, declared in advance, is the only honest way to run a grid this wide.
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
PRIMARY_BASE, PRIMARY_H, PRIMARY_KIND = "f_flow20", 20, "provisional"


def rank_std(s: pd.Series) -> np.ndarray:
    r = s.rank(method="average").to_numpy(dtype=float)
    r -= r.mean()
    sd = r.std(ddof=0)
    return r / sd if sd > 1e-12 else r


def fama_macbeth(d: pd.DataFrame, y: str, xs: list[str], inter: str | None = None,
                 min_n: int = 25, lag: int = 5) -> dict:
    """One regression per event day; Newey-West on the coefficient series."""
    names = list(xs) + ([inter] if inter else [])
    series = {k: [] for k in names}
    for _, g in d.groupby("D"):
        g = g.dropna(subset=[y] + xs)
        if len(g) < min_n:
            continue
        cols = [rank_std(g[c]) for c in xs]
        if inter:
            a, b = inter.split("*")
            cols.append(rank_std(g[a]) * rank_std(g[b]))
        X = np.column_stack(cols + [np.ones(len(g))])
        yy = g[y].to_numpy(dtype=float)
        lo, hi = np.percentile(yy, [1, 99])
        yy = np.clip(yy, lo, hi)
        try:
            beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
        except np.linalg.LinAlgError:
            continue
        for k, b_ in zip(names, beta[:len(names)]):
            series[k].append(float(b_))
    out = {}
    for k, v in series.items():
        arr = np.array(v)
        out[k] = ({"bp": float(arr.mean() * 1e4),
                   "t": tracker._nw_t(arr, lag), "days": len(arr)}
                  if len(arr) >= 8 else {"bp": np.nan, "t": np.nan,
                                         "days": len(arr)})
    return out


def ladder(d: pd.DataFrame, y: str, base: str) -> dict:
    r = {}
    r["M1"] = fama_macbeth(d, y, ["surprise"])
    r["M2"] = fama_macbeth(d, y, ["surprise"] + CTRL)
    r["M3"] = fama_macbeth(d, y, ["surprise"] + CTRL + [base])
    r["M4"] = fama_macbeth(d, y, ["surprise"] + CTRL + [base],
                           inter=f"surprise*{base}")
    return r


def show(tag: str, r: dict, base: str) -> None:
    f = lambda c, k: (f"{c[k]['bp']:+8.1f} ({c[k]['t']:+5.2f})"
                      if k in c and np.isfinite(c[k]["bp"]) else "       --      ")
    print(f"    {tag}")
    print(f"      M1 surprise            {f(r['M1'], 'surprise')}")
    print(f"      M2 surprise +ctrl      {f(r['M2'], 'surprise')}"
          f"   run-up20 {f(r['M2'], 'c_mom20')}")
    print(f"      M3 {base:<12}      {f(r['M3'], base)}")
    print(f"      M4 surprise x {base:<8} {f(r['M4'], 'surprise*' + base)}"
          f"   <- the hypothesis")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default=PRIMARY_KIND,
                    choices=["provisional", "periodic"])
    a = ap.parse_args()
    if not PANEL.exists():
        sys.exit("run 20_event_panel.py first")
    d = pd.read_parquet(PANEL)
    d = d[d["kind"] == a.kind].copy()
    have = [b for b in BASELINES if b in d.columns and d[b].notna().mean() > 0.3]
    print(f"panel: {len(d):,} {a.kind} events, "
          f"{d['D'].min().date()} .. {d['D'].max().date()}")
    print(f"baselines available: {', '.join(have)}")
    if PRIMARY_BASE not in have:
        print(f"  ! {PRIMARY_BASE} is too sparse; the primary test cannot run")

    out = {"kind": a.kind, "n": int(len(d)), "primary": None, "robust": {}}

    print(f"\n  PRIMARY -- {a.kind}, abn{PRIMARY_H}, baseline {PRIMARY_BASE}")
    print("  coefficients are bp per 1 SD of rank; t is Newey-West across days\n")
    if PRIMARY_BASE in have:
        r = ladder(d, f"abn{PRIMARY_H}", PRIMARY_BASE)
        show(f"abn{PRIMARY_H}", r, PRIMARY_BASE)
        out["primary"] = r
        it = r["M4"][f"surprise*{PRIMARY_BASE}"]
        ok = np.isfinite(it["t"]) and it["bp"] < 0 and it["t"] <= -2
        out["verdict"] = "BASELINE HELPS" if ok else "NOT ESTABLISHED"
        print(f"\n  verdict on the pre-registered primary test: {out['verdict']}")
        print(f"  (needs the interaction negative with t <= -2; it is "
              f"{it['bp']:+.1f} bp, t {it['t']:+.2f}, {it['days']} days)")

    print("\n  ROBUSTNESS -- secondary, not the pre-registered claim")
    for h in (5, 20, 60):
        for b in have:
            if b == PRIMARY_BASE and h == PRIMARY_H:
                continue
            r = ladder(d, f"abn{h}", b)
            k = f"abn{h}|{b}"
            out["robust"][k] = r
            it = r["M4"][f"surprise*{b}"]
            print(f"    abn{h:<3} {b:<9} interaction {it['bp']:+8.1f} bp "
                  f"(t {it['t']:+5.2f})   M3 main {r['M3'][b]['bp']:+8.1f} bp "
                  f"(t {r['M3'][b]['t']:+5.2f})")

    print("\n  BEATS ONLY -- the brief's own framing: among good results, does a")
    print("  lower baseline mean a bigger drift? (coefficient should be NEGATIVE)")
    beats = d[d["surprise"] > d["surprise"].quantile(0.6)]
    out["beats_only"] = {}
    for b in have:
        r = fama_macbeth(beats, f"abn{PRIMARY_H}", CTRL + [b], min_n=15)
        out["beats_only"][b] = r
        print(f"    {b:<9} {r[b]['bp']:+8.1f} bp  t {r[b]['t']:+5.2f}  "
              f"({r[b]['days']} days, n={len(beats):,})")

    RESULTS.mkdir(exist_ok=True)
    dest = RESULTS / f"baseline_fm_{a.kind}.json"
    dest.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
