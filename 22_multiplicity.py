#!/usr/bin/env python3
"""Step 22: how surprising is our best t, given how many cells we searched?

    python 22_multiplicity.py                 # 200 permutations, provisional
    python 22_multiplicity.py --perms 500

THE PROBLEM THIS SOLVES
-----------------------
Step 21 ran a family of tests: five baselines x three horizons x {main effect,
interaction}, plus a beats-only main effect for each baseline. Thirty-five
numbers. The largest was t = -3.30. Read alone that looks decisive. Read as
the maximum of thirty-five draws it may be nothing at all, and no single-test
p-value can tell the difference.

WHAT THIS DOES INSTEAD
----------------------
Shuffle the baseline columns among the events that filed on the SAME day, so
any real link between positioning and subsequent return is destroyed while
everything else survives untouched: the day structure, the surprise, the
controls, the correlations among the baselines themselves, the number of
events per day, even the fat tails in returns. Then re-run the entire search
and write down the largest |t| it produced. Repeat.

That yields the distribution of "best t found by this exact search when there
is nothing to find". Comparing our observed best against it is the honest
version of a p-value for a study that looked in many places.

The same machinery gives a family-wise critical value: the 95th percentile of
that distribution is the bar a single cell must clear to count as evidence,
rather than the usual 2.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import tracker

ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "results" / "event_panel.parquet"
RESULTS = ROOT / "results"
CTRL = ["c_mom20", "c_mom60", "c_size", "c_vol", "c_turn"]
BASELINES = ["f_flow20", "f_flow60", "f_hold20", "f_level", "i_flow20"]
HORIZONS = (5, 20, 60)
MIN_N = 25


def rank_std(v: np.ndarray) -> np.ndarray:
    order = v.argsort()
    r = np.empty(len(v), dtype=float)
    r[order] = np.arange(len(v), dtype=float)
    r -= r.mean()
    s = r.std()
    return r / s if s > 1e-12 else r


def pack(d: pd.DataFrame, baselines: list[str]) -> list[dict]:
    """One entry per event day, with everything already rank-standardised."""
    days = []
    need = CTRL + baselines
    for _, g in d.groupby("D"):
        g = g.dropna(subset=need)
        if len(g) < MIN_N:
            continue
        entry = {"n": len(g),
                 "sur": rank_std(g["surprise"].to_numpy(dtype=float)),
                 "ctrl": np.column_stack(
                     [rank_std(g[c].to_numpy(dtype=float)) for c in CTRL]),
                 "base": {b: rank_std(g[b].to_numpy(dtype=float))
                          for b in baselines},
                 "y": {}, "beat": None}
        ok = True
        for h in HORIZONS:
            yy = g[f"abn{h}"].to_numpy(dtype=float)
            if not np.isfinite(yy).all():
                m = np.isfinite(yy)
                if m.sum() < MIN_N:
                    ok = False
                    break
                yy = np.where(m, yy, np.nanmedian(yy[m]))
            lo, hi = np.percentile(yy, [1, 99])
            entry["y"][h] = np.clip(yy, lo, hi)
        if not ok:
            continue
        entry["beat"] = entry["sur"] > np.quantile(entry["sur"], 0.6)
        days.append(entry)
    return days


def coef(X: np.ndarray, y: np.ndarray, k: int) -> float:
    try:
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan
    return float(b[k])


def family(days: list[dict], baselines: list[str], shuffle) -> dict:
    """Every test we reported, returned as {name: NW t}."""
    series = {}
    for e in days:
        n = e["n"]
        base = e["base"]
        if shuffle is not None:
            base = {b: v[shuffle.permutation(n)] for b, v in base.items()}
        one = np.ones(n)
        for b in baselines:
            bv = base[b]
            Xm = np.column_stack([e["sur"], e["ctrl"], bv, one])
            Xi = np.column_stack([e["sur"], e["ctrl"], bv, e["sur"] * bv, one])
            for h in HORIZONS:
                y = e["y"][h]
                series.setdefault(f"main|{b}|{h}", []).append(
                    coef(Xm, y, 1 + e["ctrl"].shape[1]))
                series.setdefault(f"inter|{b}|{h}", []).append(
                    coef(Xi, y, 2 + e["ctrl"].shape[1]))
            m = e["beat"]
            if m.sum() >= 15:
                Xb = np.column_stack([e["ctrl"][m], bv[m], np.ones(int(m.sum()))])
                series.setdefault(f"beats|{b}|20", []).append(
                    coef(Xb, e["y"][20][m], e["ctrl"].shape[1]))
    out = {}
    for k, v in series.items():
        arr = np.array([x for x in v if np.isfinite(x)])
        out[k] = tracker._nw_t(arr, 5) if len(arr) >= 8 else np.nan
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="provisional",
                    choices=["provisional", "periodic"])
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260831)
    a = ap.parse_args()
    if not PANEL.exists():
        sys.exit("run 20_event_panel.py first")

    d = pd.read_parquet(PANEL)
    d = d[d["kind"] == a.kind].copy()
    have = [b for b in BASELINES if b in d.columns and d[b].notna().mean() > 0.3]
    days = pack(d, have)
    if len(days) < 20:
        sys.exit(f"only {len(days)} usable event days")
    print(f"{a.kind}: {len(d):,} events, {len(days)} usable days, "
          f"{len(have)} baselines")

    obs = family(days, have, None)
    obs = {k: v for k, v in obs.items() if np.isfinite(v)}
    n_tests = len(obs)
    best_name = max(obs, key=lambda k: abs(obs[k]))
    best_t = obs[best_name]
    print(f"\n  observed family: {n_tests} tests")
    print(f"  largest |t|: {best_name}  t = {best_t:+.2f}")
    print("\n  the six largest, as actually found:")
    for k in sorted(obs, key=lambda k: -abs(obs[k]))[:6]:
        print(f"    {k:<22} t {obs[k]:+6.2f}")

    print(f"\n  running {a.perms} permutations "
          f"(baselines shuffled within each event day) ...")
    rng = np.random.default_rng(a.seed)
    maxes, t0 = [], time.time()
    for i in range(1, a.perms + 1):
        r = family(days, have, rng)
        vals = [abs(v) for v in r.values() if np.isfinite(v)]
        maxes.append(max(vals) if vals else np.nan)
        if i % 25 == 0:
            el = time.time() - t0
            print(f"    {i}/{a.perms}   {el/i:.2f}s each, "
                  f"~{(a.perms-i)*el/i/60:.1f} min left")
    mx = np.array([m for m in maxes if np.isfinite(m)])

    print("\n  distribution of the LARGEST |t| the same search finds when the")
    print("  baselines carry no information:")
    for q in (50, 75, 90, 95, 99):
        print(f"    {q}th percentile   {np.percentile(mx, q):5.2f}")
    p_fw = float((mx >= abs(best_t)).mean())
    crit = float(np.percentile(mx, 95))
    print(f"\n  our best was |t| = {abs(best_t):.2f}")
    print(f"  family-wise p-value: {p_fw:.3f}   "
          f"({int((mx >= abs(best_t)).sum())} of {len(mx)} null searches did "
          f"at least as well)")
    print(f"  family-wise 5% bar for a single cell: |t| >= {crit:.2f} "
          f"(not the usual 2.00)")
    survivors = [k for k in obs if abs(obs[k]) >= crit]
    print(f"\n  cells clearing that bar: "
          f"{', '.join(survivors) if survivors else 'none'}")
    verdict = ("SURVIVES THE SEARCH" if p_fw <= 0.05 else
               "INDISTINGUISHABLE FROM SEARCH LUCK")
    print(f"\n  verdict: {verdict}")

    RESULTS.mkdir(exist_ok=True)
    dest = RESULTS / f"multiplicity_{a.kind}.json"
    dest.write_text(json.dumps(
        {"kind": a.kind, "n_tests": n_tests, "perms": int(len(mx)),
         "observed": obs, "best": {"name": best_name, "t": best_t},
         "fw_p": p_fw, "fw_crit95": crit, "survivors": survivors,
         "null_max_percentiles": {str(q): float(np.percentile(mx, q))
                                  for q in (50, 75, 90, 95, 99)},
         "verdict": verdict}, indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
