#!/usr/bin/env python3
"""The conditional table, with the chance baseline printed beside every number.

The published table compared empirical tenth percentiles between groups of
different sizes, and a larger group reaches further into the tail simply by
having more draws. Measured here, that counting effect is worth -73 to -117
basis points -- roughly two fifths of the headline. The placebo band in the
original design already absorbed it, so the verdict was sound, but the
magnitude was not what it appeared to be.

This rebuilds the table so the magnitude is readable:

  * the tenth percentile with the groups forced to the same size
  * the probability of a loss worse than ten percent, which is a proportion
    and therefore does not care how big the groups are
  * the mean and the hit rate, unchanged, since neither has the problem

and prints, next to each, what the same statistic returns when crowding is
assigned at random. A number is only worth reading as a size if its chance
baseline sits near zero.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V6 = ROOT / "results" / "event_panel_v6.parquet"
OUT = ROOT / "results" / "corrected_table.json"
RUNG = 10
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


def buckets(sub, vals):
    sq = sub.groupby("D")["surprise"].rank(pct=True)
    cq = pd.Series(vals, index=sub.index).groupby(
        sub["D"].to_numpy()).rank(pct=True)
    return (np.clip((sq * NS).astype(int), 0, NS - 1).to_numpy(),
            np.clip((cq * NC).astype(int), 0, NC - 1).to_numpy())


STATS = {
    "mean": lambda a, b, r: a.mean() - b.mean(),
    "median": lambda a, b, r: np.median(a) - np.median(b),
    "p10 equal size": None,          # handled separately
    "P(loss > 10%)": lambda a, b, r: (a < -0.10).mean() - (b < -0.10).mean(),
    "hit rate": lambda a, b, r: (a > 0).mean() - (b > 0).mean(),
}


def p10_equal(a, b, rng):
    k = min(len(a), len(b))
    return float(np.mean([
        np.percentile(rng.choice(a, k, replace=False), 10)
        - np.percentile(rng.choice(b, k, replace=False), 10)
        for _ in range(SUBS)]))


def season_stat(sub, vals, fn, rng):
    s, c = buckets(sub, vals)
    top = s == NS - 1
    crowded, quiet = top & (c == NC - 1), top & (c == 0)
    y, season = sub["abn60"].to_numpy(), sub["season"].to_numpy()
    rows = []
    for ss in np.unique(season):
        m = season == ss
        a, b = y[m & crowded], y[m & quiet]
        if len(a) >= 8 and len(b) >= 8:
            rows.append(fn(a, b, rng))
    if len(rows) < 8:
        return None
    r = np.array(rows, dtype=float)
    return {"diff": float(r.mean()), "t": float(bfm.tracker._nw_t(r, 1)),
            "seasons": int(len(r))}


def main() -> int:
    d = pd.read_parquet(V6)
    d["season"] = d["D"].dt.year * 4 + (d["D"].dt.month - 1) // 3
    d = d[d["inst_days20"] >= RUNG]
    R = {"rung": RUNG, "draws": DRAWS, "subsamples": SUBS}

    for kind in (None, "provisional"):
        key = kind or "all"
        base = d if kind is None else d[d["kind"] == kind]
        base = base.dropna(subset=["abn60", "surprise", CROWD])
        print(f"\n{'=' * 90}\n{key} filings, institutions active on "
              f"{RUNG}+ of the 20 days, {len(base):,} events\n{'=' * 90}")
        print(f"  {'statistic':<18}{'crowded - quiet':>18}{'t':>8}"
              f"{'chance baseline':>18}{'placebo 5-95%':>24}{'':>10}")

        for name in STATS:
            fn = p10_equal if name == "p10 equal size" else STATS[name]
            if fn is None:
                continue
            rng = np.random.default_rng(777)
            real = season_stat(base, base[CROWD].to_numpy(), fn, rng)
            if real is None:
                continue
            band = [v["diff"] for v in
                    (season_stat(base, rng.random(len(base)), fn, rng)
                     for _ in range(DRAWS)) if v]
            b = np.array(band)
            lo, hi = np.percentile(b, [5, 95])
            outside = real["diff"] < lo or real["diff"] > hi
            pct = name in ("P(loss > 10%)", "hit rate")
            sc, u = (100.0, "pp") if pct else (1e4, "bp")
            print(f"  {name:<18}{real['diff'] * sc:>15.1f} {u}"
                  f"{real['t']:>8.2f}{b.mean() * sc:>15.1f} {u}"
                  f"{f'[{lo * sc:+.1f}, {hi * sc:+.1f}]':>24}"
                  f"{'  OUTSIDE' if outside else '  inside':>10}")
            R[f"{key}_{name}"] = {
                "diff": real["diff"], "t": real["t"], "seasons": real["seasons"],
                "chance_baseline": float(b.mean()),
                "p05": float(lo), "p95": float(hi),
                "outside_band": bool(outside)}

    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2, default=str))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print("\nread the chance baseline first. where it is far from zero the")
    print("statistic is measuring the estimator, and only the band test means")
    print("anything; where it is near zero the difference is a size.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
