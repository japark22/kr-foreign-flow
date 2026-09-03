"""Does the fatter left tail survive the two obvious attacks.

50_baseline found what the mean tests could not: after the strongest earnings
surprises, names institutions were already crowded into have a tenth
percentile 236 bp worse than uncrowded names, at t -3.24 and outside a
200-draw placebo band, while the mean and the hit rate do not move. Before
that is shown to anyone it has to survive the two things a reader will ask
first.

  volatility   crowded names may simply be more volatile names, in which
               case the tail gap is a stock characteristic wearing a
               positioning costume. The gap must hold inside each
               volatility tercile.
  time         every result today that lived in one half of the sample died
               when the other half arrived. The gap must carry the same sign
               in both halves.

Declared before the run: negative in all three volatility terciles, negative
in both halves. The shape of the lower tail (p5, p10, p25) is reported as
description, not as a test.

    python 51_tail_robust.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("results/decompose_panel.parquet")
OUT = Path("results/tail_robust.json")
CROWD, NS, NC = "i_flow20", 5, 3

spec = importlib.util.spec_from_file_location("bfm", "21_baseline_fm.py")
bfm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bfm)


def season_diff(g, ma, mb, stat, min_n=8):
    rows = []
    v = g["abn60"].to_numpy()
    for s in np.unique(g["season"]):
        m = (g["season"] == s).to_numpy()
        a, b = v[m & ma], v[m & mb]
        if len(a) >= min_n and len(b) >= min_n:
            rows.append(stat(a) - stat(b))
    r = np.array(rows, dtype=float)
    if len(r) < 8:
        return None
    return {"bp": float(r.mean() * 1e4), "t": float(bfm.tracker._nw_t(r, 1)),
            "seasons": int(len(r))}


def line(tag, r):
    if r is None:
        print(f"  {tag:<34}too few comparable seasons")
    else:
        print(f"  {tag:<34}{r['bp']:+8.1f} bp   t {r['t']:+5.2f}"
              f"   seasons {r['seasons']}")


def main() -> int:
    d = pd.read_parquet(CACHE)
    d["D"] = pd.to_datetime(d["D"])
    R = {}
    for kind in ("provisional", None):
        sub = d if kind is None else d[d["kind"] == kind]
        sub = sub.dropna(subset=["abn60", "surprise", CROWD, "c_vol"]).copy()
        sub["season"] = sub["D"].dt.year * 4 + (sub["D"].dt.month - 1) // 3
        g = sub.groupby("D")
        sub["s"] = np.clip((g["surprise"].rank(pct=True) * NS).astype(int), 0, NS - 1)
        sub["c"] = np.clip((g[CROWD].rank(pct=True) * NC).astype(int), 0, NC - 1)
        sub["v"] = np.clip((g["c_vol"].rank(pct=True) * 3).astype(int), 0, 2)

        top = sub[sub["s"] == NS - 1].copy()
        label = kind or "all filings"
        print(f"\n================ {label}: strongest-surprise events "
              f"{len(top):,} ================")
        crowded = (top["c"] == NC - 1).to_numpy()
        quiet = (top["c"] == 0).to_numpy()
        p10 = lambda x: np.percentile(x, 10)
        key = kind or "all"
        R[key] = {}

        print("shape of the lower tail, crowded minus quiet")
        R[key]["shape"] = {}
        for q in (5, 10, 25, 50):
            r = season_diff(top, crowded, quiet, lambda x, q=q: np.percentile(x, q))
            R[key]["shape"][f"p{q}"] = r
            line(f"   p{q}", r)

        print("\n1  inside each volatility tercile  (declared: all negative)")
        R[key]["by_vol"] = {}
        for vi, name in enumerate(("low vol", "mid vol", "high vol")):
            m = (top["v"] == vi).to_numpy()
            r = season_diff(top, crowded & m, quiet & m, p10, min_n=5)
            R[key]["by_vol"][name] = r
            line(f"   {name}", r)

        print("\n2  by half  (declared: both negative)")
        R[key]["by_half"] = {}
        cut = top["D"].quantile(0.5)
        for name, m in (("first half", (top["D"] <= cut).to_numpy()),
                        ("second half", (top["D"] > cut).to_numpy())):
            r = season_diff(top, crowded & m, quiet & m, p10)
            R[key]["by_half"][name] = r
            line(f"   {name}", r)

        print("\n3  monotone across crowding terciles?  (p10, pooled, bp)")
        vals = [float(np.percentile(top[top["c"] == c]["abn60"], 10) * 1e4)
                for c in range(NC)]
        R[key]["p10_by_tercile"] = vals
        mono = vals[0] > vals[1] > vals[2]
        print(f"   quiet {vals[0]:+8.1f}   mid {vals[1]:+8.1f}   "
              f"crowded {vals[2]:+8.1f}   {'monotone' if mono else 'not monotone'}")

        vol_ok = all(r and r["bp"] < 0 for r in R[key]["by_vol"].values())
        half_ok = all(r and r["bp"] < 0 for r in R[key]["by_half"].values())
        R[key]["passes"] = {"volatility": vol_ok, "halves": half_ok}
        print(f"\n   volatility check {'PASS' if vol_ok else 'FAIL'}   "
              f"halves check {'PASS' if half_ok else 'FAIL'}")

    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
