"""Are the two periods actually different, or just noisy?

The pre-registered out-of-sample block, 2011-2017, comes back at -7.4 bp/SD
with t -0.35 while the original window reads -61.9 at t -2.71. Read as two
separate numbers that looks like a failure to replicate. But two subsample
estimates being on opposite sides of significance says nothing on its own --
the question is whether the difference between them exceeds what sampling
noise produces, and that has to be estimated directly rather than eyeballed.

So: one interaction on the full sample, baseline times a post-2018 indicator,
with the same season-clustered errors. The main coefficient is the effect in
the early block, the interaction is the change, and the interaction carries
the standard error that decides the question.

Two further checks, because a break date chosen after seeing the split is not
evidence: the same interaction swept across every candidate break year, and
the year-by-year coefficient series. If 2018 is special, it should stand out
from its neighbours. If the sweep is flat, there is no break to speak of --
only an imprecise estimate, and the full window is the honest one.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path("results/event_panel.parquet")
OUT = Path("results/regime.json")
BASE, MIN_N = "i_flow20", 5

spec = importlib.util.spec_from_file_location("bfm", "21_baseline_fm.py")
bfm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bfm)
CTRL = list(bfm.CTRL)


def build(sub, extra=None, y="abn60"):
    xs = ["surprise"] + CTRL + [BASE]
    others = [c for c in xs if c != BASE]
    need = [y] + xs + ([extra] if extra else [])
    X, Y, S = [], [], []
    for day, g in sub.groupby("D"):
        g = g.dropna(subset=need)
        if len(g) < MIN_N:
            continue
        b = bfm.rank_std(g[BASE])
        cols = [b]
        if extra:
            m = g[extra].to_numpy(dtype=float)
            cols += [b * m, m]
        cols += [bfm.rank_std(g[c]) for c in others] + [np.ones(len(g))]
        X.append(np.column_stack(cols))
        v = g[y].to_numpy(dtype=float)
        lo, hi = np.percentile(v, [1, 99])
        Y.append(np.clip(v, lo, hi))
        t = pd.Timestamp(day)
        S.append(np.full(len(g), t.year * 4 + (t.month - 1) // 3))
    if not X:
        return None
    return np.vstack(X), np.concatenate(Y), np.concatenate(S)


def fit(p, want=0):
    X, y, sea = p
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    e = y - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(sea):
        m = sea == c
        u = X[m].T @ e[m]
        meat += np.outer(u, u)
    V = XtX_inv @ meat @ XtX_inv
    se = float(np.sqrt(max(V[want, want], 1e-30)))
    return {"bp": float(b[want] * 1e4), "t": float(b[want] / se),
            "se_bp": float(se * 1e4)}


d = pd.read_parquet(PANEL)
d["D"] = pd.to_datetime(d["D"])
p = d[d["kind"] == "provisional"].copy()
print(f"provisional {len(p):,}   {p['D'].min():%Y-%m} .. {p['D'].max():%Y-%m}\n")

R = {}
full = fit(build(p))
print(f"  full window                  {full['bp']:+8.1f} bp/SD"
      f"   se {full['se_bp']:5.1f}   t {full['t']:+5.2f}")
R["full"] = full

p["post"] = (p["D"] >= "2018-01-01").astype(float)
pp = build(p, "post")
early, delta = fit(pp, 0), fit(pp, 1)
print(f"  2011-2017 (main)             {early['bp']:+8.1f} bp/SD"
      f"   se {early['se_bp']:5.1f}   t {early['t']:+5.2f}")
print(f"  change after 2018            {delta['bp']:+8.1f} bp/SD"
      f"   se {delta['se_bp']:5.1f}   t {delta['t']:+5.2f}"
      f"   <- the question")
R["break_2018"] = {"early": early, "change": delta}

print("\n  break-year sweep (a real break stands out from its neighbours)")
R["sweep"] = {}
for yr in range(2014, 2024):
    p["cut"] = (p["D"] >= f"{yr}-01-01").astype(float)
    q = build(p, "cut")
    if q is None:
        continue
    dd = fit(q, 1)
    R["sweep"][yr] = dd
    print(f"    break at {yr}   change {dd['bp']:+8.1f}   t {dd['t']:+5.2f}")

print("\n  coefficient by year (each on its own, so each is noisy)")
R["by_year"] = {}
for yr, g in p.groupby(p["D"].dt.year):
    q = build(g)
    if q is None or len(np.unique(q[2])) < 3:
        continue
    r = fit(q)
    R["by_year"][int(yr)] = r
    bar = "#" * min(30, int(abs(r["bp"]) / 10))
    print(f"    {yr}  {r['bp']:+8.1f}  t {r['t']:+5.2f}  {bar}")

verdict = ("the two periods are not statistically distinguishable -- the full "
           "window is the honest estimate and no regime claim is available"
           if abs(R["break_2018"]["change"]["t"]) < 2.0 else
           "the periods differ by more than sampling noise -- a regime "
           "statement is defensible, but only with an independent reason for "
           "the break date")
print(f"\nverdict: {verdict}")
R["verdict"] = verdict
OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2))
print(f"wrote {OUT}")
