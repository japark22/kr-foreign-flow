#!/usr/bin/env python3
"""Ask the same question with an estimator that fits the data.

The outcome has a tenth percentile near minus twenty percent and a ninetieth
near plus twenty. Regressing a winsorised raw return on a ranked feature is a
weak way to detect a monotone cross-sectional relationship in a distribution
that shape: a handful of extreme observations carry the coefficient and the
standard error pays for them. Ranking both sides is the standard remedy and
it measures exactly the same monotone relationship.

Three estimator choices, each declared here and each standard:

  rank outcome   the return replaced by its rank inside the day, scaled the
                 same way the feature is. This is a rank information
                 coefficient expressed on the same axis as everything else.
  precision      events where crowding was measured over more trading days
                 carry more information, so they are weighted by how many.
  horizon        the effect accumulated to sixty days; one hundred and twenty
                 is checked in case it is still accumulating.

All of them keep the frozen universe, the frozen controls, the two-way
clusters and the 200-draw placebo. Nothing about the hypothesis changes.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V6 = ROOT / "results" / "event_panel_v6.parquet"
OUT = ROOT / "results" / "efficiency.json"
RUNG, DRAWS = 10, 200
BANS = [("2020-03-16", "2021-05-02"), ("2023-11-06", "2025-03-30")]
FEATURE = "i_flow20_v2"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name.split("_")[0],
                                                  ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


th = load_module("64_threats_flevel")
bfm = load_module("21_baseline_fm")
HARD = th.CTRL + th.LONG


def pieces(d, outcome, rank_y, weight):
    X, Y, W, S, T = [], [], [], [], []
    for day, g in d.groupby("D"):
        g = g.dropna(subset=[outcome, "surprise", FEATURE] + HARD)
        if len(g) < th.MIN_N:
            continue
        C = np.column_stack([bfm.rank_std(g[c]) for c in ["surprise"] + HARD]
                            + [np.ones(len(g))])
        Q, _ = np.linalg.qr(C)
        x = bfm.rank_std(g[FEATURE])
        if rank_y:
            y = bfm.rank_std(g[outcome])
        else:
            y = g[outcome].to_numpy(dtype=float)
            lo, hi = np.percentile(y, [1, 99])
            y = np.clip(y, lo, hi)
        w = (g["inst_days20"].to_numpy(dtype=float) / 20.0) if weight \
            else np.ones(len(g))
        sw = np.sqrt(w)
        X.append((x - Q @ (Q.T @ x)) * sw)
        Y.append((y - Q @ (Q.T @ y)) * sw)
        W.append(w)
        ts = pd.Timestamp(day)
        S.append(np.full(len(g), ts.year * 4 + (ts.month - 1) // 3))
        T.append(g["ticker"].to_numpy())
    return (np.concatenate(X), np.concatenate(Y), np.concatenate(S),
            np.concatenate(T))


def coef(x, y, sea, tic, scale):
    den = float(x @ x)
    b = float(x @ y) / den
    u = x * (y - x * b)
    def meat(k):
        return float((pd.DataFrame({"u": u, "k": k})
                      .groupby("k")["u"].sum() ** 2).sum())
    pair = (pd.Series(sea).astype(str) + "|" + pd.Series(tic).astype(str)).to_numpy()
    m2 = max(meat(sea) + meat(tic) - meat(pair), 1e-30)
    return {"coef": b * scale, "events": int(len(y)),
            "seasons": int(pd.Series(sea).nunique()),
            "t_season": b / (np.sqrt(meat(sea)) / den),
            "t_two_way": b / (np.sqrt(m2) / den)}


def main() -> int:
    d = pd.read_parquet(V6)
    d = d[d["inst_days20"] >= RUNG].copy()
    d = d[~(((d["D"] >= BANS[0][0]) & (d["D"] <= BANS[0][1])) |
            ((d["D"] >= BANS[1][0]) & (d["D"] <= BANS[1][1])))]
    R = {}

    specs = [
        ("frozen spec, raw return",        "abn60",  False, False, 1e4, "bp"),
        ("rank outcome",                   "abn60",  True,  False, 1.0, "sd"),
        ("rank outcome + precision",       "abn60",  True,  True,  1.0, "sd"),
        ("raw return + precision",         "abn60",  False, True,  1e4, "bp"),
        ("rank outcome, 20-day horizon",   "abn20",  True,  False, 1.0, "sd"),
        ("rank outcome, 5-day horizon",    "abn5",   True,  False, 1.0, "sd"),
    ]

    print(f"  {'specification':<34}{'coef':>10}{'t(season)':>12}"
          f"{'t(two-way)':>13}{'events':>9}{'placebo band':>22}")
    for tag, outcome, rank_y, weight, scale, unit in specs:
        if outcome not in d.columns:
            continue
        x, y, S, T = pieces(d, outcome, rank_y, weight)
        r = coef(x, y, S, T, scale)
        rng = np.random.default_rng(613)
        pl = [coef(x[rng.permutation(len(x))], y, S, T, scale)["coef"]
              for _ in range(DRAWS)]
        lo, hi = np.percentile(pl, [5, 95])
        outside = r["coef"] < lo or r["coef"] > hi
        print(f"  {tag:<34}{r['coef']:>8.3f}{unit:>2}{r['t_season']:>12.2f}"
              f"{r['t_two_way']:>13.2f}{r['events']:>9,}"
              f"{f'[{lo:+.2f},{hi:+.2f}]':>20}"
              f"{'  OUT' if outside else '  in'}")
        R[tag] = {**r, "unit": unit, "p05": float(lo), "p95": float(hi),
                  "outside_band": bool(outside)}

        # halves, for whichever specification is being read
        seasons = np.unique(S)
        cut = seasons[len(seasons) // 2]
        hv = {}
        for h, m in (("first", S < cut), ("second", S >= cut)):
            hv[h] = coef(x[m], y[m], S[m], T[m], scale)
        print(f"      halves  first {hv['first']['coef']:+7.3f} "
              f"(t {hv['first']['t_two_way']:+5.2f})   "
              f"second {hv['second']['coef']:+7.3f} "
              f"(t {hv['second']['t_two_way']:+5.2f})")
        R[tag]["halves"] = hv

    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2, default=str))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print("\na rank outcome cannot manufacture a relationship that is not")
    print("there -- it can only stop the fat tails from hiding one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
