#!/usr/bin/env python3
"""Does splitting the flow help, and is the split what it claims to be?

The rebuild matched the panel's column at a Pearson correlation of exactly
1.000000, which is the number two series produce when a handful of shared
extreme values dominate both. The reconstructed whole has a standard
deviation of 1192 where its own legs sit near 2, so the first thing here is
a rank correlation, which cannot be bought by outliers. If that is not also
near one, the match was an artefact and nothing below is worth reading.

Then the question the decomposition exists for. Declared before estimating:

  the idiosyncratic leg beats the combined column, because averaging a
  signal with an allocation that carries nothing is what an information
  coefficient of 0.015 looks like;

  the systematic leg is weak, because an allocation wave is not a view;

  crowding bites harder where the variance ratio is high, because a campaign
  ends when the position is filled and that is a reason for the unwind, while
  churn has no such reason.

Everything runs on the frozen specification: the participation universe, the
suspensions excluded, momentum controlled to 500 days, two-way clusters, and
a 200-draw placebo that shuffles within the day.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V7 = ROOT / "results" / "event_panel_v7.parquet"
OUT = ROOT / "results" / "flow_factor_test.json"
RUNG, DRAWS = 10, 200
BANS = [("2020-03-16", "2021-05-02"), ("2023-11-06", "2025-03-30")]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name.split("_")[0],
                                                  ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ef = load_module("78_efficiency")
th = load_module("64_threats_flevel")
bv = load_module("67_ban_vs_era")
bfm = load_module("21_baseline_fm")
HARD = th.CTRL + th.LONG


def frozen_universe():
    d = pd.read_parquet(V7)
    d = d[d["inst_days20"] >= RUNG].copy()
    return d[~(((d["D"] >= BANS[0][0]) & (d["D"] <= BANS[0][1])) |
               ((d["D"] >= BANS[1][0]) & (d["D"] <= BANS[1][1])))]


def run(d, feature, rank_y=False):
    ef.FEATURE = feature
    x, y, S, T = ef.pieces(d, "abn60", rank_y, False)
    r = ef.coef(x, y, S, T, 1.0 if rank_y else 1e4)
    rng = np.random.default_rng(404)
    pl = [ef.coef(x[rng.permutation(len(x))], y, S, T,
                  1.0 if rank_y else 1e4)["coef"] for _ in range(DRAWS)]
    lo, hi = np.percentile(pl, [5, 95])
    r.update({"p05": float(lo), "p95": float(hi),
              "outside_band": bool(r["coef"] < lo or r["coef"] > hi)})
    return r


def show(tag, r, unit):
    print(f"  {tag:<34}{r['coef']:+9.3f}{unit:>3}   t(two-way) {r['t_two_way']:+6.2f}"
          f"   [{r['p05']:+8.2f},{r['p95']:+8.2f}] "
          f"{'OUT' if r['outside_band'] else 'in ':>4}   n={r['events']:>6,}")


def main() -> int:
    d = frozen_universe()
    R = {"rung": RUNG, "draws": DRAWS}

    print("0. is the rebuild real, or bought by outliers?")
    b = d["i_all20"].notna() & d["i_flow20_v2"].notna()
    pear = float(d.loc[b, "i_all20"].corr(d.loc[b, "i_flow20_v2"]))
    a_r = d.loc[b, "i_all20"].rank()
    v_r = d.loc[b, "i_flow20_v2"].rank()
    spear = float(np.corrcoef(a_r, v_r)[0, 1])
    print(f"   cells {int(b.sum()):,}   pearson {pear:.6f}   "
          f"spearman {spear:.6f}")
    print(f"   i_all20 sd {d['i_all20'].std():.1f}   "
          f"p1 {d['i_all20'].quantile(0.01):.2f}   "
          f"p99 {d['i_all20'].quantile(0.99):.2f}   "
          f"max |x| {d['i_all20'].abs().max():.1f}")
    R["rebuild_pearson"], R["rebuild_spearman"] = pear, spear
    if spear < 0.98:
        print("   STOP -- the rank correlation says the rebuild is not the "
              "column it claimed to match")
        OUT.write_text(json.dumps(R, indent=2, ensure_ascii=False))
        return 1
    print("   ok -- the match survives ranking, so it is the same variable")

    print("\n1. main effects, frozen specification, raw return")
    for feat, tag in (("i_flow20_v2", "combined  (the frozen feature)"),
                      ("i_idio20", "idiosyncratic leg"),
                      ("i_sys20", "systematic leg"),
                      ("flow_beta", "flow beta (a characteristic)"),
                      ("flow_vr", "variance ratio (a characteristic)")):
        if feat not in d.columns:
            continue
        r = run(d, feat)
        R[f"raw_{feat}"] = r
        show(tag, r, "bp")

    print("\n2. the same, as rank information coefficients")
    for feat, tag in (("i_flow20_v2", "combined"),
                      ("i_idio20", "idiosyncratic leg"),
                      ("i_sys20", "systematic leg")):
        r = run(d, feat, rank_y=True)
        R[f"ic_{feat}"] = r
        show(tag, r, "IC")

    print("\n3. does crowding bite harder where accumulation trends?")
    sub = d.dropna(subset=["abn60", "surprise", "i_flow20_v2", "flow_vr"] + HARD)
    X, Y, S, T = [], [], [], []
    for day, g in sub.groupby("D"):
        g = g.dropna(subset=["abn60", "surprise", "i_flow20_v2", "flow_vr"] + HARD)
        if len(g) < th.MIN_N:
            continue
        x = bfm.rank_std(g["i_flow20_v2"])
        v = bfm.rank_std(g["flow_vr"])
        cols = [x, x * v, v]
        cols += [bfm.rank_std(g[c]) for c in ["surprise"] + HARD]
        cols += [np.ones(len(g))]
        X.append(np.column_stack(cols))
        yy = g["abn60"].to_numpy(dtype=float)
        lo, hi = np.percentile(yy, [1, 99])
        Y.append(np.clip(yy, lo, hi))
        ts = pd.Timestamp(day)
        S.append(np.full(len(g), ts.year * 4 + (ts.month - 1) // 3))
    X, Y, S = np.vstack(X), np.concatenate(Y), np.concatenate(S)
    names = ["crowding", "crowding x variance ratio", "variance ratio"]
    print(f"   n={len(Y):,}  seasons={pd.Series(S).nunique()}")
    inter = {}
    for k, nm in enumerate(names):
        bp, t, _, _ = bv.cluster_fit(X, Y, S, k)
        inter[nm] = {"bp": bp * 1e4, "t_season": t}
        print(f"   {nm:<30}{bp * 1e4:+9.1f} bp/SD   t {t:+6.2f}")
    t0, p, ng = bv.wild_bootstrap(X, Y, S, 1, reps=2000)
    inter["crowding x variance ratio"]["wild_p"] = p
    print(f"   wild cluster bootstrap on the interaction   "
          f"t {t0:+.2f}   p {p:.4f}   clusters {ng}")
    R["interaction"] = inter

    OUT.write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print("\nthe frozen feature stands unless a leg beats it on every one of")
    print("the five estimator variants, which is the next script and only")
    print("runs if something here is worth taking that far.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
