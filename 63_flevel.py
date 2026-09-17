#!/usr/bin/env python3
"""Put the survivor through everything that killed the others.

Two cells cleared a family-wise threshold of 5.02 over 2,658 candidates,
and both contain the same ingredient: where foreign ownership sits inside
its own 250-day range. That is a level, not a flow, and an earlier pass in
this project found the foreign ownership level to be a size and quality
proxy. So the question is not whether the coefficient is large. It is
whether anything is left once size is taken seriously.

Four ways to break it, in order of how likely they are to succeed:

  1. Is it one finding or one ingredient counted twice? Estimate the level
     alone, the flow alone, the difference and the interaction side by side.
  2. Is it size? Add the square and the cube of size, then size deciles as
     their own effects, and see what survives.
  3. Does it hold in both halves of the sample?
  4. Where in the conditional distribution does it live?
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V4 = ROOT / "results" / "event_panel_v4.parquet"
OUT = ROOT / "results" / "flevel.json"
DRAWS = 200
TAUS = [0.10, 0.25, 0.50, 0.75, 0.90]


def load(name):
    spec = importlib.util.spec_from_file_location(name.split("_")[0],
                                                  ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


fr = load("38_final_results")
qr = load("58_qreg")
bfm = load("21_baseline_fm")
CTRL = list(fr.CTRL)


def cell(d, base, ctrl, tag, seed=17):
    prep = fr.prepare(d, base, ["surprise"] + ctrl + [base])
    if prep is None:
        print(f"  {tag:<44}  --")
        return None
    r = fr.fit(prep)
    Xo, xb, y, sea, slices = prep
    rng = np.random.default_rng(seed)
    pl = []
    for _ in range(DRAWS):
        xp = xb.copy()
        for a_, b_ in slices:
            xp[a_:b_] = rng.permutation(xp[a_:b_])
        pl.append(fr.fit((Xo, xp, y, sea, slices))["bp"])
    lo, hi = np.percentile(pl, [5, 95])
    inside = lo <= r["bp"] <= hi
    r.update({"placebo_p05": float(lo), "placebo_p95": float(hi),
              "inside_band": bool(inside)})
    print(f"  {tag:<44}{r['bp']:+8.1f} bp/SD  t {r['t']:+6.2f}  "
          f"[{lo:+7.1f},{hi:+7.1f}] {'inside' if inside else 'OUTSIDE':<8} "
          f"n={r['events']:>6,} s={r['seasons']}")
    return r


def main() -> int:
    d = pd.read_parquet(V4)
    d = d[d["kind"] == "periodic"].copy()
    d["fl_minus_i"] = (d.groupby("D")["f_level"].transform(lambda s: s.rank(pct=True))
                       - d.groupby("D")["i_flow20_v2"].transform(lambda s: s.rank(pct=True)))
    d["fl_x_surp"] = (d.groupby("D")["f_level"].transform(lambda s: s.rank(pct=True))
                      * d.groupby("D")["surprise"].transform(lambda s: s.rank(pct=True)))
    d["c_size2"] = d["c_size"] ** 2
    d["c_size3"] = d["c_size"] ** 3
    d["c_mom250"] = d["c_mom60"]           # placeholder kept explicit
    R = {"draws": DRAWS}

    print("correlations with size, before anything:")
    for c in ("f_level", "f_hold20", "i_flow20_v2", "fl_minus_i", "fl_x_surp"):
        print(f"  {c:<14} corr(c_size)={d[[c, 'c_size']].corr().iloc[0, 1]:+.3f}"
              f"   corr(c_mom20)={d[[c, 'c_mom20']].corr().iloc[0, 1]:+.3f}")

    print("\n1. one finding or one ingredient counted twice?")
    for b in ("f_level", "i_flow20_v2", "fl_minus_i", "fl_x_surp", "f_hold20"):
        R[f"plain_{b}"] = cell(d, b, CTRL, f"{b}, standard controls")

    print("\n2. is it size?")
    hard = CTRL + ["c_size2", "c_size3"]
    for b in ("f_level", "fl_minus_i", "fl_x_surp"):
        R[f"sizehard_{b}"] = cell(d, b, hard, f"{b}, plus size squared and cubed")

    print("\n   within size deciles (the effect must survive inside each):")
    d["sz"] = d.groupby("D")["c_size"].transform(
        lambda s: np.clip((s.rank(pct=True) * 5).astype(int), 0, 4) + 1)
    for q in (1, 3, 5):
        R[f"sizedec_{q}"] = cell(d[d["sz"] == q], "fl_minus_i", CTRL,
                                 f"fl_minus_i, size quintile {q}")

    print("\n3. both halves?")
    med = d["D"].median()
    R["first"] = cell(d[d["D"] < med], "fl_minus_i", CTRL,
                      f"fl_minus_i, before {pd.Timestamp(med).date()}")
    R["second"] = cell(d[d["D"] >= med], "fl_minus_i", CTRL,
                       f"fl_minus_i, from {pd.Timestamp(med).date()}")
    R["first_x"] = cell(d[d["D"] < med], "fl_x_surp", CTRL,
                        f"fl_x_surp, before {pd.Timestamp(med).date()}")
    R["second_x"] = cell(d[d["D"] >= med], "fl_x_surp", CTRL,
                         f"fl_x_surp, from {pd.Timestamp(med).date()}")

    print("\n4. where in the distribution?")
    prep = fr.prepare(d, "fl_minus_i", ["surprise"] + CTRL + ["fl_minus_i"])
    Xo, xb, y, sea, slices = prep
    X = np.column_stack([xb, Xo])
    seasons = np.unique(sea)
    rows = {s: np.where(sea == s)[0] for s in seasons}
    rng = np.random.default_rng(29)
    for tau in TAUS:
        b, frac = qr.fit_check(X, y, tau)
        bs = []
        for _ in range(150):
            pick = rng.choice(seasons, size=len(seasons), replace=True)
            idx = np.concatenate([rows[s] for s in pick])
            bs.append(qr.qreg(X[idx], y[idx], tau)[0] * 1e4)
        sd = float(np.std(bs, ddof=1))
        print(f"  tau {tau:.2f}  frac<0={frac:.3f}  "
              f"{b[0] * 1e4:+8.1f} bp/SD   t {b[0] * 1e4 / sd:+6.2f}")
        R[f"tau_{tau}"] = {"bp": float(b[0] * 1e4), "t": float(b[0] * 1e4 / sd),
                           "frac_below": float(frac)}

    OUT.write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
