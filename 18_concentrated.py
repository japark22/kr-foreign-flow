#!/usr/bin/env python3
"""Step 18: the concentrated coat-tail test -- hold a few names alongside them.

    python 18_concentrated.py
    python 18_concentrated.py --signal 60      # rank on a 60-session change

WHAT IS DIFFERENT FROM STEP 11
------------------------------
The tracker bought the top DECILE (~200 names) and sold on a fixed calendar.
Two things that hides:
  - if the information lives only in the handful of most extreme accumulations,
    averaging 200 names buries it;
  - a fixed calendar exit pays for a full round trip every h sessions, which is
    what killed the economics (320 rebalances a year).
This tests the version a person would actually run: buy the top N, keep holding
while the name stays among the most-accumulated, and let go once it drops out.

THE EXIT IS THE POINT. Entry is top N on the signal. A position is kept while
it stays inside the top N x buffer; it leaves only when it falls out of that
wider band. So holds lengthen by themselves when foreign buying persists, and
turnover is paid only when the flow actually rotates.

HONESTY RULES, FIXED BEFORE RUNNING
-----------------------------------
  - Universe is the liquid top third by 20d traded value. The illiquid tail is
    where earlier effects lived and it cannot be traded.
  - Costs: 50bp per round trip (commission, spread, 0.20% sales tax), charged
    on realised turnover, not assumed.
  - EVERY cell of the grid is printed, not the best one. The count of cells
    tried is printed next to the result, because with enough cells one of them
    always looks good.
  - Train 2018-2022 / test 2023 onward. A cell that only works in train is a
    fitted cell.
  - Split-guarded returns: an unadjusted 1:10 split otherwise reads as -90%.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import tracker

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
COST_RT = 0.0050
BAD_DAY = 0.50
LIQ_Q = 0.67                      # keep the top third by traded value
SPLIT = "2023-01-01"

GRID_N = (5, 10, 20, 50)
GRID_R = (5, 20)                  # sessions between rebalance checks
GRID_BUF = (1.0, 2.0)             # exit when out of top N*buffer


def daily_returns(close: pd.DataFrame) -> pd.DataFrame:
    r = close.pct_change(fill_method=None)
    return r.mask(r.abs() > BAD_DAY)


def run_cell(sig: np.ndarray, ret: np.ndarray, ok: np.ndarray,
             n: int, rstep: int, buf: float) -> dict:
    """One (N, rebalance, buffer) cell. Returns daily strategy/benchmark series."""
    n_d, n_t = sig.shape
    held: list[int] = []
    w = np.zeros(n_t)
    strat = np.full(n_d, np.nan)
    bench = np.full(n_d, np.nan)
    costs = np.zeros(n_d)
    holds, turn_events = [], []
    entry_day = {}
    wide = int(round(n * buf))

    for d in range(21, n_d - 1):
        avail = ok[d]
        # today's return earned on yesterday's weights
        if w.sum() > 0:
            r = ret[d]
            m = w > 0
            rr = np.where(np.isfinite(r[m]), r[m], 0.0)
            strat[d] = float((w[m] * rr).sum() / w[m].sum())
        b = avail & np.isfinite(ret[d])
        if b.sum() >= 20:
            bench[d] = float(ret[d][b].mean())

        if (d - 21) % rstep:
            continue

        s = np.where(avail, sig[d], np.nan)
        if np.isfinite(s).sum() < max(50, wide):
            continue
        order = np.argsort(-np.nan_to_num(s, nan=-np.inf))
        top_wide = set(order[:wide].tolist())

        keep = [j for j in held if j in top_wide and avail[j]]
        room = n - len(keep)
        adds = [j for j in order[:n] if j not in keep][:max(room, 0)]
        new = keep + adds
        if not new:
            continue

        w_new = np.zeros(n_t)
        w_new[new] = 1.0 / len(new)
        turn = float(np.abs(w_new - w).sum()) / 2.0
        costs[d] = turn * COST_RT
        turn_events.append(turn)

        for j in held:
            if j not in new and j in entry_day:
                holds.append(d - entry_day.pop(j))
        for j in new:
            entry_day.setdefault(j, d)

        held, w = new, w_new

    return {"strat": strat, "bench": bench, "cost": costs,
            "avg_hold": float(np.mean(holds)) if holds else np.nan,
            "turn_per_reb": float(np.mean(turn_events)) if turn_events else np.nan,
            "n_rebal": len(turn_events)}


def summarise(cell: dict, idx: pd.DatetimeIndex, lo=None, hi=None) -> dict:
    s, b, c = cell["strat"], cell["bench"], cell["cost"]
    m = np.isfinite(s) & np.isfinite(b)
    if lo is not None:
        m &= np.asarray(idx >= lo)
    if hi is not None:
        m &= np.asarray(idx < hi)
    if m.sum() < 250:
        return {"days": int(m.sum())}
    ex = (s - b - c)[m]
    ann = float(ex.mean() * 250 * 100)
    sd = float(ex.std(ddof=1))
    sharpe = float(ex.mean() / sd * np.sqrt(250)) if sd > 1e-12 else np.nan
    return {"days": int(m.sum()), "ann_excess_pct": ann, "sharpe": sharpe,
            "t": tracker._nw_t(ex, 10),
            "cost_pct_yr": float(c[m].sum() / m.sum() * 250 * 100)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", type=int, default=20)
    ap.add_argument("--start", default="2017-06-01")
    a = ap.parse_args()

    from krxflow import features, storage
    p = features.load_panels(a.start, None)
    pct = p["foreign_pct"]
    uni = features.universe_mask(p)
    m = storage.read_range("market", a.start, None,
                           columns=["trade_date", "ticker", "close",
                                    "value_traded"])
    m["trade_date"] = pd.to_datetime(m["trade_date"])

    def pv(c):
        return (m.pivot_table(index="trade_date", columns="ticker", values=c,
                              aggfunc="last", observed=True)
                 .sort_index().astype("float64")
                 .reindex(index=pct.index, columns=pct.columns))

    close, vt = pv("close"), pv("value_traded")
    del m
    adv = vt.rolling(20, min_periods=5).mean()
    cut = adv.quantile(LIQ_Q, axis=1)
    liquid = uni & adv.ge(cut, axis=0) & pct.notna() & close.notna()

    sig = (pct - pct.shift(a.signal)).to_numpy()
    ret = daily_returns(close).to_numpy()
    ok = liquid.to_numpy()
    idx = pct.index
    print(f"panel {idx[0].date()}..{idx[-1].date()}  "
          f"liquid names/day ~{int(ok.sum(axis=1).mean()):,}  "
          f"signal = {a.signal}d ownership change")

    cells, rows = len(GRID_N) * len(GRID_R) * len(GRID_BUF), []
    print(f"\n  grid: {cells} cells, all reported\n")
    print("   N  reb  buf | full: ann%   Sharpe   t  | train ann%  test ann% |"
          " hold  turn/reb  cost%/yr")
    for n in GRID_N:
        for r in GRID_R:
            for bf in GRID_BUF:
                cell = run_cell(sig, ret, ok, n, r, bf)
                full = summarise(cell, idx)
                tr = summarise(cell, idx, hi=pd.Timestamp(SPLIT))
                te = summarise(cell, idx, lo=pd.Timestamp(SPLIT))
                if "ann_excess_pct" not in full:
                    continue
                rows.append({"n": n, "rebal": r, "buffer": bf, "full": full,
                             "train": tr, "test": te,
                             "avg_hold": cell["avg_hold"],
                             "turn_per_reb": cell["turn_per_reb"]})
                print(f"  {n:>3} {r:>4} {bf:>4.1f} | "
                      f"{full['ann_excess_pct']:+8.2f} {full['sharpe']:+7.2f} "
                      f"{full['t']:+5.2f} | "
                      f"{tr.get('ann_excess_pct', float('nan')):+9.2f} "
                      f"{te.get('ann_excess_pct', float('nan')):+9.2f} | "
                      f"{cell['avg_hold']:5.0f} {cell['turn_per_reb']:8.2f} "
                      f"{full['cost_pct_yr']:9.2f}")

    if not rows:
        sys.exit("no cell produced enough days -- widen --start")

    best = max(rows, key=lambda r_: r_["full"]["ann_excess_pct"])
    pos_test = [r_ for r_ in rows
                if r_["test"].get("ann_excess_pct", -9e9) > 0
                and r_["train"].get("ann_excess_pct", -9e9) > 0]
    print(f"\n  best cell by full-sample excess: N={best['n']} reb={best['rebal']} "
          f"buf={best['buffer']}  {best['full']['ann_excess_pct']:+.2f}%/yr  "
          f"t {best['full']['t']:+.2f}")
    print(f"  cells positive in BOTH train and test: {len(pos_test)} of {len(rows)}")
    print(f"  cells tried: {len(rows)}. With this many, the best one is expected")
    print("  to look good even under a true null -- read the train/test columns,")
    print("  not the headline. A t below ~3 here is not evidence.")

    RESULTS.mkdir(exist_ok=True)
    dest = RESULTS / "concentrated.json"
    dest.write_text(json.dumps(
        {"generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
         "signal_days": a.signal, "cost_rt": COST_RT, "split": SPLIT,
         "cells": rows}, indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
