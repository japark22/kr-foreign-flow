#!/usr/bin/env python3
"""Step 26: measure the book against a matched benchmark, not a style bet.

    python 26_matched.py

THE PROBLEM WITH STEP 25
------------------------
Every rule there lost to the equal-weight liquid universe at every holding
period, including "buy every beat" -- while the cross-sectional regression in
step 21 found a real post-earnings drift (+37bp per SD, t +2.18). Both can be
true, and the reason matters: the regression controlled for size, run-up,
volatility and turnover; the portfolio controlled for nothing. Stocks that
beat skew large and high-momentum, the equal-weight benchmark leans small, and
that difference is large enough to swamp the drift. Step 25 was measuring a
style bet.

WHAT CHANGES HERE
-----------------
Each position is compared with the stocks it most resembles rather than with
the market as a whole. Every day, the liquid universe is cut into size deciles
and a position earns the return of its own stock minus the average return of
its decile that day. A book of large caps is then judged against large caps.
This is the portfolio version of the control the regression already had, so
the two finally measure the same thing.

One property to keep in mind while reading the result: matching also removes
any effect that is uniform inside a size bucket. If the institutional signal
disappears here, that does not mean the measurement failed -- it means the
signal was a size effect wearing another name, which is itself the answer.

READ THE WHOLE TABLE, NOT THE BEST CELL
---------------------------------------
Three holding periods are printed together, on purpose. Step 25 was run at 60,
then 20, then 40 sessions, and one of the three produced a filter gain with
t = 2.09. Selecting that one and reporting it would repeat exactly the mistake
step 22 was built to catch. So the verdict below demands the gain be positive
at ALL THREE holds and in BOTH halves of the sample -- a bar that cannot be
cleared by choosing a holding period after the fact.
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
HOLDS = (20, 40, 60)
COST_RT = 0.0050
MIN_ADV = 1e8
SPLIT = "2022-07-01"
BASE = "i_flow20"
N_BUCKETS = 10


def matched_excess(ret: np.ndarray, cap: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Stock return minus the mean return of its own size decile that day."""
    out = np.full_like(ret, np.nan)
    for i in range(ret.shape[0]):
        m = ok[i] & np.isfinite(ret[i]) & np.isfinite(cap[i])
        if m.sum() < 50:
            continue
        c = cap[i][m]
        edges = np.quantile(c, np.linspace(0, 1, N_BUCKETS + 1)[1:-1])
        b = np.digitize(c, edges)
        r = ret[i][m]
        adj = np.empty_like(r)
        for g in range(N_BUCKETS):
            sel = b == g
            if sel.sum() >= 3:
                adj[sel] = r[sel] - r[sel].mean()
            else:
                adj[sel] = np.nan
        out[i][np.where(m)[0]] = adj
    return out


def book(picks: pd.DataFrame, exc: np.ndarray, n_dates: int, col: dict,
         hold: int) -> dict:
    held: dict = {}
    entries: dict = {}
    for t_, e in zip(picks["ticker"], picks["entry"]):
        j = col.get(t_)
        if j is None:
            continue
        for k in range(int(e) + 1, min(int(e) + 1 + hold, n_dates)):
            held.setdefault(k, []).append(j)
        entries[int(e) + 1] = entries.get(int(e) + 1, 0) + 1
    ex = np.full(n_dates, np.nan)
    for k, js in held.items():
        r = exc[k, js]
        r = r[np.isfinite(r)]
        if len(r):
            ex[k] = float(r.mean())
            if entries.get(k):
                ex[k] -= COST_RT * entries[k] / len(js)
    return {"ex": ex, "n": int(len(picks)),
            "avg_book": float(np.mean([len(v) for v in held.values()]))
            if held else 0.0}


def season_of(dates: pd.DatetimeIndex) -> np.ndarray:
    return dates.year.to_numpy() * 4 + (dates.month.to_numpy() - 1) // 3


def stats(ex: np.ndarray, dates: pd.DatetimeIndex, lo=None, hi=None) -> dict:
    m = np.isfinite(ex)
    if lo is not None:
        m &= np.asarray(dates >= lo)
    if hi is not None:
        m &= np.asarray(dates < hi)
    if m.sum() < 120:
        return {"days": int(m.sum())}
    sid = season_of(dates)[m]
    e = ex[m]
    per = np.array([e[sid == s].mean() for s in np.unique(sid)])
    return {"days": int(m.sum()), "ann_pct": float(e.mean() * 250 * 100),
            "season_t": tracker._nw_t(per, 1) if len(per) >= 8 else np.nan,
            "seasons": int(len(per))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-06-01")
    a = ap.parse_args()
    if not PANEL.exists():
        sys.exit("run 20_event_panel.py first")

    d = pd.read_parquet(PANEL)
    d = d[(d["kind"] == "provisional") & d[BASE].notna()
          & d["surprise"].notna()].copy()

    from krxflow import features, storage
    p = features.load_panels(a.start, None)
    pct, uni = p["foreign_pct"], features.universe_mask(p)
    m = storage.read_range("market", a.start, None,
                           columns=["trade_date", "ticker", "close",
                                    "value_traded", "market_cap"])
    m["trade_date"] = pd.to_datetime(m["trade_date"])

    def pv(c):
        return (m.pivot_table(index="trade_date", columns="ticker", values=c,
                              aggfunc="last", observed=True)
                  .sort_index().astype("float64")
                  .reindex(index=pct.index, columns=pct.columns))

    close, vt, cap = pv("close"), pv("value_traded"), pv("market_cap")
    del m
    adv = vt.rolling(20, min_periods=5).mean()
    ok = (uni & (adv >= MIN_ADV) & close.notna()).to_numpy()
    r = close.pct_change(fill_method=None)
    r = r.mask(r.abs() > 0.5).to_numpy()
    print("building size-matched excess returns ...")
    exc = matched_excess(r, cap.to_numpy(), ok)
    dates = pct.index
    col = {t: j for j, t in enumerate(pct.columns)}

    d["sur_r"] = d.groupby("D")["surprise"].rank(pct=True)
    d["base_r"] = d.groupby("D")[BASE].rank(pct=True)
    beat = d["sur_r"] >= 0.6
    quiet = d["base_r"] <= 0.4
    crowd = d["base_r"] >= 0.6
    rules = {"all beats": d[beat], "quiet beats": d[beat & quiet],
             "quiet only": d[quiet], "crowded beats": d[beat & crowd]}

    print(f"\nprovisional events: {len(d):,}   size-decile matched, "
          f"{COST_RT*1e4:.0f}bp a position\n")
    out = {"holds": list(HOLDS), "cost_rt": COST_RT, "by_hold": {}}
    gains = []
    for hold in HOLDS:
        print(f"  hold {hold} sessions")
        print("    rule            book |   ann%   season t |  first half"
              "  second half")
        bks, cell = {}, {}
        for name, picks in rules.items():
            bk = book(picks, exc, len(dates), col, hold)
            bks[name] = bk
            f = stats(bk["ex"], dates)
            h1 = stats(bk["ex"], dates, hi=pd.Timestamp(SPLIT))
            h2 = stats(bk["ex"], dates, lo=pd.Timestamp(SPLIT))
            cell[name] = {"avg_book": bk["avg_book"], "full": f,
                          "first_half": h1, "second_half": h2}
            print(f"    {name:<14} {bk['avg_book']:>5.0f} | "
                  f"{f.get('ann_pct', float('nan')):+7.2f} "
                  f"{f.get('season_t', float('nan')):+9.2f} | "
                  f"{h1.get('ann_pct', float('nan')):+10.2f} "
                  f"{h2.get('ann_pct', float('nan')):+12.2f}")
        diff = bks["quiet beats"]["ex"] - bks["all beats"]["ex"]
        g = stats(diff, dates)
        g1 = stats(diff, dates, hi=pd.Timestamp(SPLIT))
        g2 = stats(diff, dates, lo=pd.Timestamp(SPLIT))
        cell["filter_gain"] = {"full": g, "first_half": g1, "second_half": g2}
        gains.append((hold, g, g1, g2))
        print(f"    {'filter gain':<14} {'':>5} | "
              f"{g.get('ann_pct', float('nan')):+7.2f} "
              f"{g.get('season_t', float('nan')):+9.2f} | "
              f"{g1.get('ann_pct', float('nan')):+10.2f} "
              f"{g2.get('ann_pct', float('nan')):+12.2f}\n")
        out["by_hold"][str(hold)] = cell

    allpos = all(g.get("ann_pct", -9) > 0 for _, g, _, _ in gains)
    halves = all(g1.get("ann_pct", -9) > 0 and g2.get("ann_pct", -9) > 0
                 for _, _, g1, g2 in gains)
    sig = sum(1 for _, g, _, _ in gains if abs(g.get("season_t", 0)) >= 2)
    print(f"  filter gain positive at all {len(HOLDS)} holds: {allpos}")
    print(f"  positive in both halves at all holds: {halves}")
    print(f"  holds where |t| >= 2: {sig} of {len(HOLDS)}")
    out["verdict"] = ("FILTER ADDS VALUE" if (allpos and halves and sig >= 2)
                      else "NOT ESTABLISHED")
    print(f"\n  verdict: {out['verdict']}")
    print("  (a gain that appears at one holding period only is a choice of")
    print("   holding period, not a property of the data)")

    RESULTS.mkdir(exist_ok=True)
    dest = RESULTS / "matched_book.json"
    dest.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
