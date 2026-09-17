#!/usr/bin/env python3
"""A tail statistic that does not depend on how big the groups are.

Comparing empirical tenth percentiles between a large group and a small one
is biased: the larger sample reaches further into the tail simply by having
more draws. The placebo band in the original design absorbs that bias, so
the published verdict stands, but the headline number does not mean what it
looks like -- a large part of -236 bp is the estimator, not the market.

Two repairs, run side by side.

  A. A proportion instead of a percentile. The probability that the sixty-day
     abnormal return falls below -10% is unbiased at any group size, it is
     already in the published table, and it says something a risk report can
     use directly.

  B. The percentile, but with the groups forced to the same size. Inside each
     season the larger group is sampled down to the smaller one, repeatedly,
     and the contrast averaged. Whatever survives that is not a counting
     artefact.

Both are tested the same way the original was: per reporting season, then a
clustered t, against a 200-draw placebo band.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V6 = ROOT / "results" / "event_panel_v6.parquet"
OUT = ROOT / "results" / "tail_statistic.json"
RUNGS = [1, 10, 14, 18]
DRAWS, SUBS = 200, 40
NS, NC = 5, 3
CROWD = "i_flow20_v2"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name.split("_")[0],
                                                  ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


bfm = load_module("21_baseline_fm")


def groups(sub, crowd_vals):
    sq = sub.groupby("D")["surprise"].rank(pct=True)
    cq = pd.Series(crowd_vals, index=sub.index).groupby(
        sub["D"].to_numpy()).rank(pct=True)
    s = np.clip((sq * NS).astype(int), 0, NS - 1).to_numpy()
    c = np.clip((cq * NC).astype(int), 0, NC - 1).to_numpy()
    top = s == NS - 1
    return top & (c == NC - 1), top & (c == 0)


def season_stat(sub, crowd_vals, mode, rng):
    crowded, quiet = groups(sub, crowd_vals)
    y = sub["abn60"].to_numpy()
    season = sub["season"].to_numpy()
    rows = []
    for ss in np.unique(season):
        m = season == ss
        a, b = y[m & crowded], y[m & quiet]
        if len(a) < 8 or len(b) < 8:
            continue
        if mode == "big_loss":
            rows.append((a < -0.10).mean() - (b < -0.10).mean())
        else:
            k = min(len(a), len(b))
            vals = [np.percentile(rng.choice(a, k, replace=False), 10)
                    - np.percentile(rng.choice(b, k, replace=False), 10)
                    for _ in range(SUBS)]
            rows.append(float(np.mean(vals)))
    if len(rows) < 8:
        return None
    r = np.array(rows, dtype=float)
    return {"diff": float(r.mean()), "t": float(bfm.tracker._nw_t(r, 1)),
            "seasons": int(len(r))}


def main() -> int:
    d = pd.read_parquet(V6)
    d["season"] = d["D"].dt.year * 4 + (d["D"].dt.month - 1) // 3
    R = {"draws": DRAWS, "subsamples": SUBS}

    for mode, unit, scale in (("big_loss", "pp", 100.0),
                              ("p10_equal", "bp", 1e4)):
        print(f"\n{'=' * 82}")
        print("probability of losing more than 10 percent, crowded minus quiet"
              if mode == "big_loss"
              else "tenth percentile, groups forced to equal size")
        print(f"{'=' * 82}")
        for kind in ("provisional", None):
            key = kind or "all"
            base = d if kind is None else d[d["kind"] == kind]
            base = base.dropna(subset=["abn60", "surprise", CROWD, "inst_days20"])
            print(f"  [{key} filings]")
            for rung in RUNGS:
                sub = base[base["inst_days20"] >= rung]
                rng = np.random.default_rng(20260917 + rung)
                real = season_stat(sub, sub[CROWD].to_numpy(), mode, rng)
                if real is None:
                    print(f"    rung {rung:<4} too few comparable seasons")
                    continue
                band = []
                for _ in range(DRAWS):
                    v = season_stat(sub, rng.random(len(sub)), mode, rng)
                    if v:
                        band.append(v["diff"])
                b = np.array(band)
                lo, hi = np.percentile(b, [5, 95])
                outside = real["diff"] < lo or real["diff"] > hi
                print(f"    rung {rung:<4} n={len(sub):>7,}  "
                      f"{real['diff'] * scale:+8.2f} {unit}   "
                      f"t {real['t']:+5.2f}   "
                      f"placebo mean {b.mean() * scale:+7.2f}   "
                      f"[{lo * scale:+.2f}, {hi * scale:+.2f}]   "
                      f"{'OUTSIDE' if outside else 'inside'}")
                R[f"{mode}_{key}_{rung}"] = {
                    "n": int(len(sub)), "diff": real["diff"], "t": real["t"],
                    "seasons": real["seasons"], "placebo_mean": float(b.mean()),
                    "p05": float(lo), "p95": float(hi),
                    "outside_band": bool(outside)}

    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2, default=str))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print("\nfor an unbiased statistic the placebo mean should sit near zero.")
    print("if it does not, the repair did not work and the statistic is still")
    print("measuring the counting, not the market.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
