"""The extended sample, against the window it replaces.

The event set now runs from 2011 rather than 2018, which nearly doubles it.
Extending a sample after the result is known is the point at which a research
programme can quietly start choosing its own window, so the rule was fixed
before the data arrived: take the maximum the data allows, and report the old
window and the new one side by side. Both appear below whatever they say.

The 2011-2017 block is the closest thing to an out-of-sample test this project
can run without leaving Korea -- those years were never examined while the
hypothesis was being formed.

Estimator is unchanged: provisional filings, one vote per position, regressors
rank-standardised within the day, returns winsorised 1/99 within the day,
standard errors clustered on the reporting season.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path("results/event_panel.parquet")
OUT = Path("results/extended.json")
MIN_N = 5

spec = importlib.util.spec_from_file_location("bfm", "21_baseline_fm.py")
bfm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bfm)
CTRL = list(bfm.CTRL)


def prepare(sub, base, y="abn60"):
    xs = ["surprise"] + CTRL + [base]
    others = [c for c in xs if c != base]
    X, Y, S = [], [], []
    for day, g in sub.groupby("D"):
        g = g.dropna(subset=[y] + xs)
        if len(g) < MIN_N:
            continue
        cols = [bfm.rank_std(g[base])]
        cols += [bfm.rank_std(g[c]) for c in others] + [np.ones(len(g))]
        X.append(np.column_stack(cols))
        v = g[y].to_numpy(dtype=float)
        lo, hi = np.percentile(v, [1, 99])
        Y.append(np.clip(v, lo, hi))
        t = pd.Timestamp(day)
        S.append(np.full(len(g), t.year * 4 + (t.month - 1) // 3))
    if not X:
        return None
    return np.vstack(X), np.concatenate(Y), np.concatenate(S)


def fit(p):
    X, y, sea = p
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    e = y - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(sea):
        m = sea == c
        u = X[m].T @ e[m]
        meat += np.outer(u, u)
    V = XtX_inv @ meat @ XtX_inv
    se = float(np.sqrt(max(V[0, 0], 1e-30)))
    return {"bp": float(b[0] * 1e4), "t": float(b[0] / se),
            "events": int(len(y)), "seasons": int(len(np.unique(sea)))}


def show(tag, sub, base="i_flow20", y="abn60", store=None):
    p = prepare(sub, base, y)
    if p is None or len(np.unique(p[2])) < 8:
        print(f"  {tag:<28}too few seasons")
        return None
    r = fit(p)
    print(f"  {tag:<28}{r['bp']:+8.1f} bp/SD   t {r['t']:+5.2f}"
          f"   events {r['events']:,}   seasons {r['seasons']}")
    if store is not None:
        store[tag.strip()] = r
    return r


def main() -> int:
    d = pd.read_parquet(PANEL)
    d["D"] = pd.to_datetime(d["D"])
    prov = d[d["kind"] == "provisional"]
    print(f"panel {len(d):,} events   {d['D'].min():%Y-%m} .. {d['D'].max():%Y-%m}")
    print(f"provisional {len(prov):,}   CTRL {CTRL}\n")

    old = prov[prov["D"] >= "2018-01-01"]
    new = prov[prov["D"] < "2018-01-01"]

    R = {"windows": {}}
    print("institutional flow, abn60, by window")
    show("full 2011-2026", prov, store=R["windows"])
    show("original 2018-2026", old, store=R["windows"])
    show("added 2011-2017", new, store=R["windows"])

    print("\nhorizons on the full window")
    R["horizons"] = {}
    for y in ("abn5", "abn20", "abn60"):
        show(f"  {y}", prov, y=y, store=R["horizons"])

    print("\nforeign flow on the full window (the series that does not work)")
    R["foreign"] = {}
    show("f_flow20, abn60", prov, base="f_flow20", store=R["foreign"])

    print("\nby filing kind, full window")
    R["kinds"] = {}
    show("periodic", d[d["kind"] == "periodic"], store=R["kinds"])
    show("all filings", d, store=R["kinds"])

    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
