#!/usr/bin/env python3
"""Is it the short-selling suspension, or is it simply after 2020?

The two suspensions both fall after 2020, and the year-by-year table shows
2022 positive as well, which lies in the gap between them when shorting was
normal. Those two explanations cannot be separated by comparing suspended
against everything else, so the sample is cut three ways:

    before 2020        shorting normal, old regime
    gap                after 2020, shorting normal
    suspended          after 2020, shorting suspended

If the gap resembles the first block, the constraint is doing the work. If
it resembles the third, the date is.

The split-sample estimates rest on as few as eleven reporting seasons, which
is too few for cluster asymptotics. The same question asked as one
interaction over the whole sample uses every season, and a wild cluster
bootstrap under the restricted null gives it an honest p-value.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V5 = ROOT / "results" / "event_panel_v5.parquet"
OUT = ROOT / "results" / "ban_vs_era.json"
REPS = 2000
BANS = [("2020-03-16", "2021-05-02"), ("2023-11-06", "2025-03-30")]


def load(name):
    spec = importlib.util.spec_from_file_location(name.split("_")[0],
                                                  ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


th = load("64_threats_flevel")
bfm = load("21_baseline_fm")
HARD = th.CTRL + th.LONG


def design(d, base, extra):
    """Rank inside the day, then stack. `extra` are already event-level flags."""
    xs = ["surprise"] + HARD + [base]
    X, Y, S, T = [], [], [], []
    for day, g in d.groupby("D"):
        g = g.dropna(subset=["abn60"] + xs)
        if len(g) < th.MIN_N:
            continue
        xb = bfm.rank_std(g[base])
        cols = [xb] + [xb * g[e].to_numpy(dtype=float) for e in extra]
        cols += [g[e].to_numpy(dtype=float) for e in extra]
        cols += [bfm.rank_std(g[c]) for c in xs if c != base]
        cols += [np.ones(len(g))]
        X.append(np.column_stack(cols))
        v = g["abn60"].to_numpy(dtype=float)
        lo, hi = np.percentile(v, [1, 99])
        Y.append(np.clip(v, lo, hi))
        ts = pd.Timestamp(day)
        S.append(np.full(len(g), ts.year * 4 + (ts.month - 1) // 3))
        T.append(g["ticker"].to_numpy())
    return (np.vstack(X), np.concatenate(Y), np.concatenate(S),
            np.concatenate(T))


def cluster_fit(X, y, g, want):
    XtX = np.linalg.pinv(X.T @ X)
    b = XtX @ (X.T @ y)
    e = y - X @ b
    u = X * e[:, None]
    df = pd.DataFrame(u)
    df["g"] = g
    S = df.groupby("g").sum().to_numpy()
    meat = S.T @ S
    V = XtX @ meat @ XtX
    se = float(np.sqrt(max(V[want, want], 1e-30)))
    return float(b[want]), float(b[want] / se), b, XtX


def wild_bootstrap(X, y, g, want, reps=REPS, seed=5):
    """Restricted-null wild cluster bootstrap, Rademacher weights."""
    keep = [i for i in range(X.shape[1]) if i != want]
    Xr = X[:, keep]
    br = np.linalg.pinv(Xr.T @ Xr) @ (Xr.T @ y)
    er = y - Xr @ br
    _, t0, _, _ = cluster_fit(X, y, g, want)
    rng = np.random.default_rng(seed)
    groups = pd.Series(g)
    codes = groups.astype("category").cat.codes.to_numpy()
    ng = codes.max() + 1
    ts = []
    for _ in range(reps):
        w = rng.choice([-1.0, 1.0], size=ng)[codes]
        yb = Xr @ br + er * w
        _, tb, _, _ = cluster_fit(X, yb, g, want)
        ts.append(tb)
    ts = np.array(ts)
    p = float((np.abs(ts) >= abs(t0)).mean())
    return t0, p, int(ng)


def main() -> int:
    d = pd.read_parquet(V5)
    d = d[d["kind"] == "periodic"].copy()
    d["fl_minus_i"] = (d.groupby("D")["f_level"].transform(lambda s: s.rank(pct=True))
                       - d.groupby("D")["i_flow20_v2"].transform(lambda s: s.rank(pct=True)))
    d["fl_x_surp"] = (d.groupby("D")["f_level"].transform(lambda s: s.rank(pct=True))
                      * d.groupby("D")["surprise"].transform(lambda s: s.rank(pct=True)))
    ban = pd.Series(False, index=d.index)
    for a_, b_ in BANS:
        ban |= (d["D"] >= a_) & (d["D"] <= b_)
    d["ban"] = ban.astype(float)
    d["era"] = (d["D"] >= "2020-03-16").astype(float)
    d["gap"] = ((d["era"] == 1) & (d["ban"] == 0)).astype(float)
    R = {}

    print("three blocks")
    blocks = {"before 2020-03-16": d[d["era"] == 0],
              "after 2020, shorting normal": d[d["gap"] == 1],
              "shorting suspended": d[d["ban"] == 1]}
    for b in ("fl_minus_i", "fl_x_surp"):
        print(f"  [{b}]")
        for tag, sub in blocks.items():
            p = th.prepare(sub, b, HARD)
            r = th.fit(p) if p else None
            R[f"block_{b}_{tag}"] = r
            if r:
                print(f"    {tag:<30}{r['bp']:+8.1f}   t {r['t_two_way']:+6.2f}"
                      f"   n={r['events']:>6,}  seasons={r['seasons']:>3}")

    print("\none regression, every season, two interactions")
    print("  the question is which interaction carries the effect")
    for b in ("fl_minus_i", "fl_x_surp"):
        X, y, S, T = design(d, b, ["ban", "gap"])
        names = [b, f"{b} x suspended", f"{b} x after-2020-normal"]
        print(f"  [{b}]  n={len(y):,}  seasons={pd.Series(S).nunique()}")
        out = {}
        for k, nm in enumerate(names):
            bp, t, _, _ = cluster_fit(X, y, S, k)
            out[nm] = {"bp": bp * 1e4, "t_season": t}
            print(f"    {nm:<34}{bp * 1e4:+8.1f}   t {t:+6.2f}")
        for k, nm in ((1, names[1]), (2, names[2])):
            t0, p, ng = wild_bootstrap(X, y, S, k)
            out[nm]["wild_p"] = p
            out[nm]["clusters"] = ng
            print(f"    wild cluster bootstrap, {nm:<26} "
                  f"t {t0:+6.2f}   p {p:.4f}   clusters {ng}")
        R[f"interaction_{b}"] = out

    prev = json.loads((ROOT / "results" / "shortban.json").read_text())
    print("\nplacebo line that scrolled past earlier:")
    for b in ("fl_minus_i", "fl_x_surp"):
        k = f"placebo_{b}"
        if k in prev:
            v = prev[k]
            print(f"  {b:<14} after removing suspensions {v['after_removal']['bp']:+7.1f}"
                  f"   random band [{v['p05']:+7.1f},{v['p95']:+7.1f}]"
                  f"   {'OUTSIDE' if v['special'] else 'inside'}")

    OUT.write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
