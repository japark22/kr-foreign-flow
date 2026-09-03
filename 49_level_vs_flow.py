"""Level or flow -- which half of the disclosure carries anything.

Every test in this project measured a flow: net buying over a window. Today
established why they all failed. A flow over the last twenty days is very
nearly the last twenty days of return -- the top and bottom quintiles differ
by 985 bp going in and by 37 bp coming out. A flow cannot forecast a price
because it is a record of one.

A level is a different object. Ownership standing at the top of its own range
is a state, not a change, and it is not mechanically equal to the recent
return. The distinction sorts the datasets that have been proposed: Korean and
Taiwanese daily trading are flows; Hong Kong participant holdings, short
interest balances and 13F positions are levels. It also sorts this project's
own history -- Korea publishes an ownership LEVEL every day, and every feature
built here converted it into a flow first.

So the test is an ordered one rather than a single cell. Four measures span
the spectrum, and if the level reading is right their informativeness should
rise along it:

    f_flow20   twenty-day net buying intensity      most flow-like
    f_flow60   sixty-day net buying intensity
    f_hold20   twenty-day change in ownership
    f_level    where ownership sits in its own 250-day range   most level-like

Declared before the run: the coefficients should become more negative, and
their t-statistics stronger, moving down that list. A single significant cell
proves little; the ordering is the hypothesis. Failing that ordering closes
the level idea for Korea and moves it to the markets where the data is
actually a level.

    python 49_level_vs_flow.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path("results/event_panel.parquet")
OUT = Path("results/level_vs_flow.json")
MIN_N, PERM = 5, 200
LADDER = ["f_flow20", "f_flow60", "f_hold20", "f_level"]

spec = importlib.util.spec_from_file_location("bfm", "21_baseline_fm.py")
bfm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bfm)
CTRL = list(bfm.CTRL)


def prep(sub, base, y="abn60"):
    xs = ["surprise"] + [c for c in CTRL if c != base] + [base]
    X, Y, S, SL = [], [], [], []
    n = 0
    for day, g in sub.groupby("D"):
        g = g.dropna(subset=[y] + xs)
        if len(g) < MIN_N:
            continue
        cols = [bfm.rank_std(g[base])]
        cols += [bfm.rank_std(g[c]) for c in xs if c != base]
        cols += [np.ones(len(g))]
        X.append(np.column_stack(cols))
        v = g[y].to_numpy(dtype=float)
        lo, hi = np.percentile(v, [1, 99])
        Y.append(np.clip(v, lo, hi))
        d = pd.Timestamp(day)
        S.append(np.full(len(g), d.year * 4 + (d.month - 1) // 3))
        SL.append((n, n + len(g)))
        n += len(g)
    if not X:
        return None
    return np.vstack(X), np.concatenate(Y), np.concatenate(S), SL


def fit(p, rng=None):
    X, y, sea, sl = p
    if rng is not None:
        X = X.copy()
        for a, b in sl:
            X[a:b, 0] = rng.permutation(X[a:b, 0])
    XtX_inv = np.linalg.pinv(X.T @ X)
    b_ = XtX_inv @ (X.T @ y)
    e = y - X @ b_
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(sea):
        m = sea == c
        u = X[m].T @ e[m]
        meat += np.outer(u, u)
    V = XtX_inv @ meat @ XtX_inv
    se = float(np.sqrt(max(V[0, 0], 1e-30)))
    return {"bp": float(b_[0] * 1e4), "t": float(b_[0] / se),
            "events": int(len(y)), "seasons": int(len(np.unique(sea)))}


def main() -> int:
    d = pd.read_parquet(PANEL)
    d["D"] = pd.to_datetime(d["D"])
    prov = d[d["kind"] == "provisional"]
    print(f"provisional {len(prov):,}   {prov['D'].min():%Y-%m}"
          f" .. {prov['D'].max():%Y-%m}\n")

    print("the ladder, most flow-like first (declared: should strengthen)")
    R, preps = {"ladder": {}}, {}
    for name in LADDER:
        p = prep(prov, name)
        if p is None:
            print(f"  {name:<12}unavailable")
            continue
        preps[name] = p
        r = fit(p)
        R["ladder"][name] = r
        print(f"  {name:<12}{r['bp']:+8.1f} bp/SD   t {r['t']:+5.2f}"
              f"   events {r['events']:,}   seasons {r['seasons']}")

    # Is the ordering itself what was predicted? Rank correlation between the
    # position on the ladder and the strength of the estimate.
    names = [n for n in LADDER if n in R["ladder"]]
    ts = np.array([R["ladder"][n]["t"] for n in names])
    pos = np.arange(len(names), dtype=float)
    if len(names) >= 3:
        a = pos - pos.mean()
        b = ts - ts.mean()
        rho = float((a @ b) / np.sqrt((a @ a) * (b @ b) + 1e-30))
        R["ordering_rho"] = rho
        print(f"\n  correlation between ladder position and t: {rho:+.3f}"
              f"   (declared: negative, t falling as the measure becomes a "
              f"level)")

    print(f"\nfamily bar over {len(preps)} measures, {PERM} permutations")
    rng = np.random.default_rng(20260903)
    maxes = np.empty(PERM)
    for i in range(PERM):
        maxes[i] = max(abs(fit(p, rng)["t"]) for p in preps.values())
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{PERM}   running bar "
                  f"{np.quantile(maxes[:i + 1], 0.95):.2f}")
    bar = float(np.quantile(maxes, 0.95))
    R["bar"] = bar
    print(f"\n  bar |t| >= {bar:.2f}")
    winners = []
    for n in names:
        t = R["ladder"][n]["t"]
        R["ladder"][n]["p_fwer"] = float((maxes >= abs(t)).mean())
        if abs(t) >= bar:
            winners.append(n)
        print(f"  {'PASS' if abs(t) >= bar else '    '}  {n:<12}"
              f"t {t:+5.2f}   p {R['ladder'][n]['p_fwer']:.3f}")
    R["winners"] = winners

    ordered = R.get("ordering_rho", 0.0) < -0.5
    verdict = ("the level end of the ladder carries what the flow end does not"
               if winners and ordered else
               "levels behave like flows here -- the distinction does not save "
               "Korea, and belongs in a market whose disclosure is a level")
    print(f"\nverdict: {verdict}")
    R["verdict"] = verdict
    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
