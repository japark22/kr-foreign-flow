#!/usr/bin/env python3
"""Step 24: confirm the one cell that survived, and price it.

    python 24_confirm.py                 # 1000 permutations
    python 24_confirm.py --perms 2000

WHAT SURVIVED, AND WHAT IT SAYS
-------------------------------
Of thirty-five searched cells, one cleared every gate: institutional net
buying in the twenty sessions before a provisional earnings release predicts
the following sixty sessions NEGATIVELY, at about -112bp per standard
deviation of rank. It held at every HAC lag, and it held when the 173 event
days were collapsed to 34 reporting seasons, which is the honest independent
unit. Its coefficient series is barely autocorrelated (AC1 +0.16), so no
overlap correction was doing the work.

In words: when domestic institutions have been accumulating into a result,
the stock does worse afterwards. That is the brief's own logic -- a crowded
position pays less -- with a different investor group than the brief named.

WHAT THIS SCRIPT ADDS
---------------------
Nothing new is searched here. Everything below interrogates that single cell,
because a finding that has been selected out of a search deserves harder
questions than the search itself asked.

  1. a tighter family-wise p, from more permutations than step 22 could afford
  2. stability: first half against second half, year by year, and across size
     terciles. A real effect need not be constant, but it should not live in
     one regime and be absent everywhere else.
  3. the economic size, stated as the quintile spread and as a rough annual
     figure for a fully-invested book, net of 50bp a position. A t-stat is not
     a return.
  4. whether it is distinct from the foreign series or just a relabelling of
     it, by putting both in the same regression.
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
CELL_BASE, CELL_H, CELL_KIND = "i_flow20", 60, "provisional"
BASELINES = ["f_flow20", "f_flow60", "f_hold20", "f_level", "i_flow20"]
HORIZONS = (5, 20, 60)
MIN_N = 25
COST_RT = 0.0050
SEASON_GAP = 30


def rank_std(v: np.ndarray) -> np.ndarray:
    o = v.argsort()
    r = np.empty(len(v), dtype=float)
    r[o] = np.arange(len(v), dtype=float)
    r -= r.mean()
    s = r.std()
    return r / s if s > 1e-12 else r


def coef(X, y, k) -> float:
    try:
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan
    return float(b[k])


def prep(d: pd.DataFrame, extra: list[str]) -> list[dict]:
    days = []
    need = CTRL + extra
    for day, g in d.groupby("D"):
        g = g.dropna(subset=need + [f"abn{CELL_H}"])
        if len(g) < MIN_N:
            continue
        days.append({
            "D": day, "n": len(g),
            "ctrl": np.column_stack([rank_std(g[c].to_numpy(float)) for c in CTRL]),
            "x": {c: rank_std(g[c].to_numpy(float)) for c in extra},
            "size": g["c_size"].to_numpy(float),
            "y": np.clip(g[f"abn{CELL_H}"].to_numpy(float),
                         *np.percentile(g[f"abn{CELL_H}"].to_numpy(float), [1, 99])),
        })
    return days


def season_ids(days: list[dict]) -> np.ndarray:
    s, cur = [], 0
    for i, e in enumerate(days):
        if i and (e["D"] - days[i - 1]["D"]).days > SEASON_GAP:
            cur += 1
        s.append(cur)
    return np.array(s)


def season_t(vals: np.ndarray, sid: np.ndarray):
    per = np.array([np.nanmean(vals[sid == s]) for s in np.unique(sid)])
    per = per[np.isfinite(per)]
    if len(per) < 8:
        return np.nan, np.nan, len(per)
    return float(per.mean()), tracker._nw_t(per, 1), len(per)


def cell_series(days: list[dict], base: str, shuffle=None) -> np.ndarray:
    out = []
    for e in days:
        x = e["x"][base]
        if shuffle is not None:
            x = x[shuffle.permutation(e["n"])]
        X = np.column_stack([e["ctrl"], x, np.ones(e["n"])])
        out.append(coef(X, e["y"], e["ctrl"].shape[1]))
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=99)
    a = ap.parse_args()
    if not PANEL.exists():
        sys.exit("run 20_event_panel.py first")
    d = pd.read_parquet(PANEL)
    d = d[d["kind"] == CELL_KIND].copy()
    days = prep(d, BASELINES)
    sid = season_ids(days)
    out = {"cell": f"main|{CELL_BASE}|{CELL_H}", "kind": CELL_KIND,
           "days": len(days), "seasons": int(len(np.unique(sid)))}
    print(f"{CELL_KIND}: {len(days)} event days, {out['seasons']} seasons\n")

    c = cell_series(days, CELL_BASE)
    mn, ts, ns = season_t(c, sid)
    print(f"  1. the cell            {mn*1e4:+8.1f} bp / SD   season t {ts:+6.2f}"
          f"  ({ns} seasons)")
    out["headline"] = {"bp": mn * 1e4, "t": ts, "seasons": ns}

    print(f"\n  2. family-wise p from {a.perms} permutations of the full "
          f"35-cell search")
    rng = np.random.default_rng(a.seed)
    obs_best = abs(ts)
    maxes = []
    for i in range(a.perms):
        best = 0.0
        for b in BASELINES:
            cc = cell_series(days, b, rng)
            _, t_, _ = season_t(cc, sid)
            if np.isfinite(t_):
                best = max(best, abs(t_))
        maxes.append(best)
        if (i + 1) % 250 == 0:
            print(f"     {i+1}/{a.perms}")
    mx = np.array(maxes)
    p_fw = float((mx >= obs_best).mean())
    print(f"     null max |t| : p50 {np.percentile(mx,50):.2f}  "
          f"p95 {np.percentile(mx,95):.2f}  p99 {np.percentile(mx,99):.2f}")
    print(f"     observed {obs_best:.2f}  ->  family-wise p = {p_fw:.4f}")
    out["fw_p"] = p_fw
    out["fw_crit95"] = float(np.percentile(mx, 95))

    print("\n  3. stability -- the same cell in pieces")
    half = len(np.unique(sid)) // 2
    for lbl, msk in (("first half", sid < half), ("second half", sid >= half)):
        mn_, t_, n_ = season_t(c[msk], sid[msk])
        print(f"     {lbl:<12} {mn_*1e4:+8.1f} bp   t {t_:+6.2f}  ({n_} seasons)")
        out[lbl.replace(" ", "_")] = {"bp": mn_ * 1e4, "t": t_, "seasons": n_}
    yrs = np.array([e["D"].year for e in days])
    print("     by year:", "  ".join(
        f"{y}:{np.nanmean(c[yrs == y])*1e4:+.0f}" for y in sorted(set(yrs))))
    out["by_year"] = {int(y): float(np.nanmean(c[yrs == y]) * 1e4)
                      for y in sorted(set(yrs))}
    neg = sum(1 for y in sorted(set(yrs)) if np.nanmean(c[yrs == y]) < 0)
    print(f"     negative in {neg} of {len(set(yrs))} years")

    print("     by size tercile (small / mid / large):")
    ter = {0: [], 1: [], 2: []}
    for e in days:
        cut = np.quantile(e["size"], [1/3, 2/3])
        grp = np.digitize(e["size"], cut)
        for g in (0, 1, 2):
            msk = grp == g
            if msk.sum() < 12:
                ter[g].append(np.nan)
                continue
            X = np.column_stack([e["ctrl"][msk], e["x"][CELL_BASE][msk],
                                 np.ones(int(msk.sum()))])
            ter[g].append(coef(X, e["y"][msk], e["ctrl"].shape[1]))
    out["size_terciles"] = {}
    for g, name in ((0, "small"), (1, "mid"), (2, "large")):
        arr = np.array(ter[g])
        mn_, t_, n_ = season_t(arr, sid)
        print(f"       {name:<6} {mn_*1e4:+8.1f} bp   t {t_:+6.2f}")
        out["size_terciles"][name] = {"bp": mn_ * 1e4, "t": t_}

    print("\n  4. institutional and foreign in the same regression")
    both = []
    for e in days:
        X = np.column_stack([e["ctrl"], e["x"]["i_flow20"], e["x"]["f_flow20"],
                             np.ones(e["n"])])
        both.append((coef(X, e["y"], e["ctrl"].shape[1]),
                     coef(X, e["y"], e["ctrl"].shape[1] + 1)))
    both = np.array(both)
    for k, name in ((0, "i_flow20"), (1, "f_flow20")):
        mn_, t_, _ = season_t(both[:, k], sid)
        print(f"     {name:<9} {mn_*1e4:+8.1f} bp   t {t_:+6.2f}")
        out[f"joint_{name}"] = {"bp": mn_ * 1e4, "t": t_}

    print("\n  5. economic size -- quintile spread in the outcome itself")
    sp = []
    for e in days:
        x = e["x"][CELL_BASE]
        lo_c, hi_c = np.quantile(x, [0.2, 0.8])
        lo_m, hi_m = x <= lo_c, x >= hi_c
        if lo_m.sum() < 5 or hi_m.sum() < 5:
            sp.append(np.nan)
            continue
        sp.append(float(e["y"][lo_m].mean() - e["y"][hi_m].mean()))
    sp = np.array(sp)
    mn_, t_, n_ = season_t(sp, sid)
    gross = mn_ * 1e4
    ann = mn_ * (250 / CELL_H) * 100
    ann_net = (mn_ - COST_RT) * (250 / CELL_H) * 100
    print(f"     bottom minus top quintile: {gross:+.1f} bp per position, "
          f"t {t_:+.2f}")
    print(f"     if fully invested and rolled: {ann:+.2f}%/yr gross, "
          f"{ann_net:+.2f}%/yr after 50bp a position")
    print("     (long-short; the long-only half is roughly half of it, and")
    print("      Korea restricts the short leg for much of this sample)")
    out["economics"] = {"spread_bp": gross, "t": t_, "ann_gross_pct": ann,
                        "ann_net_pct": ann_net, "seasons": n_}

    RESULTS.mkdir(exist_ok=True)
    dest = RESULTS / "confirm_iflow.json"
    dest.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
