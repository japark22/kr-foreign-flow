#!/usr/bin/env python3
"""Step 25: turn the institutional finding into a long-only rule, or fail to.

    python 25_longonly.py

WHAT IS BEING BUILT
-------------------
The finding is a coefficient: institutional buying into a provisional result
predicts the next sixty sessions negatively, -126bp per standard deviation
with foreign flow held alongside it and dying there. A coefficient is not a
decision. This turns it into one, in the only form the brief allows -- long
only, because shorting Korean equities was restricted for a large part of this
sample and a rule that needs the short leg is not a rule we can use.

THE RULES TESTED, ALL DECIDED BEFORE RUNNING
--------------------------------------------
  all beats        buy every provisional beat. The base case: whatever drift
                   exists without any ownership information at all.
  quiet beats      buy the beats institutions were NOT accumulating (bottom
                   two quintiles of i_flow20). If the finding is real this
                   should beat "all beats", and the DIFFERENCE between them is
                   the entire value of the data.
  quiet only       buy the bottom quintile of i_flow20 regardless of result,
                   to see whether the earnings condition matters at all.
  crowded beats    buy the beats institutions WERE accumulating. The rule the
                   finding says should be worst. If it is not worst, the
                   finding is not what we think it is.

Each position enters at the D+1 close and is held sixty sessions. The book is
equal weighted across whatever is open, so it is thin at the start of a
reporting season and full in the middle -- which is what actually happens.
Costs are 50bp a position, charged at entry.

Everything is measured against the equal-weight liquid universe, aggregated to
reporting seasons before any t-stat, and split into halves. The number that
decides this is not the headline return; it is "quiet beats" minus "all
beats", because that difference is what the ownership data buys.
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
HOLD = 60          # overridden by --hold
COST_RT = 0.0050
MIN_ADV = 1e8
SPLIT = "2022-07-01"
BASE = "i_flow20"


def book(picks: pd.DataFrame, ret: np.ndarray, bench: np.ndarray,
         n_dates: int, col: dict) -> dict:
    """Equal-weight book: each pick held HOLD sessions from its entry row."""
    held: dict = {}
    entries: dict = {}
    for t_, e in zip(picks["ticker"], picks["entry"]):
        j = col.get(t_)
        if j is None:
            continue
        for k in range(int(e) + 1, min(int(e) + 1 + HOLD, n_dates)):
            held.setdefault(k, []).append(j)
        entries[int(e) + 1] = entries.get(int(e) + 1, 0) + 1
    strat = np.full(n_dates, np.nan)
    cost = np.zeros(n_dates)
    for k, js in held.items():
        r = ret[k, js]
        r = r[np.isfinite(r)]
        if len(r):
            strat[k] = float(r.mean())
            if entries.get(k):
                cost[k] = COST_RT * entries[k] / len(js)
    return {"strat": strat, "cost": cost, "bench": bench,
            "n": int(len(picks)),
            "avg_book": float(np.mean([len(v) for v in held.values()]))
            if held else 0.0}


def season_of(dates: pd.DatetimeIndex, mask: np.ndarray) -> np.ndarray:
    """Label each date with a reporting-season index (quarterly buckets)."""
    y = dates.year.to_numpy()
    q = ((dates.month.to_numpy() - 1) // 3)
    return y * 4 + q


def stats(bk: dict, dates: pd.DatetimeIndex, lo=None, hi=None) -> dict:
    ex = bk["strat"] - bk["bench"] - bk["cost"]
    m = np.isfinite(ex)
    if lo is not None:
        m &= np.asarray(dates >= lo)
    if hi is not None:
        m &= np.asarray(dates < hi)
    if m.sum() < 120:
        return {"days": int(m.sum())}
    sid = season_of(dates, m)[m]
    e = ex[m]
    per_mean = np.array([e[sid == s].mean() for s in np.unique(sid)])
    return {"days": int(m.sum()),
            "ann_pct": float(e.mean() * 250 * 100),
            "season_t": (tracker._nw_t(per_mean, 1)
                         if len(per_mean) >= 8 else np.nan),
            "seasons": int(len(per_mean))}


def main() -> int:
    global HOLD
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-06-01")
    ap.add_argument("--hold", type=int, default=HOLD,
                    help="sessions held. 60 is where the institutional effect "
                         "was measured; 20 is where this panel's own "
                         "post-earnings drift actually sits, and holding past "
                         "it gives that drift back.")
    a = ap.parse_args()
    if not PANEL.exists():
        sys.exit("run 20_event_panel.py first")
    HOLD = a.hold

    d = pd.read_parquet(PANEL)
    d = d[(d["kind"] == "provisional") & d[BASE].notna()
          & d["surprise"].notna()].copy()

    from krxflow import features, storage
    p = features.load_panels(a.start, None)
    pct, uni = p["foreign_pct"], features.universe_mask(p)
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
    mask = uni & (adv >= MIN_ADV) & close.notna()
    r = close.pct_change(fill_method=None)
    r = r.mask(r.abs() > 0.5)
    bench = r.where(mask).mean(axis=1).to_numpy()
    ret = r.to_numpy()
    dates = pct.index
    col = {t: j for j, t in enumerate(pct.columns)}

    # within-day ranks, so a cut means the same thing on a thin day
    d["sur_r"] = d.groupby("D")["surprise"].rank(pct=True)
    d["base_r"] = d.groupby("D")[BASE].rank(pct=True)
    beat = d["sur_r"] >= 0.6
    quiet = d["base_r"] <= 0.4
    crowd = d["base_r"] >= 0.6

    rules = {
        "all beats": d[beat],
        "quiet beats": d[beat & quiet],
        "quiet only": d[quiet],
        "crowded beats": d[beat & crowd],
    }
    print(f"provisional events usable: {len(d):,}   hold {HOLD} sessions, "
          f"{COST_RT*1e4:.0f}bp a position\n")
    print("  rule            picks  book |  full ann%  season t |  "
          "first half  second half")
    out = {"hold": HOLD, "cost_rt": COST_RT, "rules": {}}
    books = {}
    for name, picks in rules.items():
        bk = book(picks, ret, bench, len(dates), col)
        books[name] = bk
        f = stats(bk, dates)
        h1 = stats(bk, dates, hi=pd.Timestamp(SPLIT))
        h2 = stats(bk, dates, lo=pd.Timestamp(SPLIT))
        out["rules"][name] = {"picks": bk["n"], "avg_book": bk["avg_book"],
                              "full": f, "first_half": h1, "second_half": h2}
        print(f"  {name:<14} {bk['n']:>5} {bk['avg_book']:>5.0f} | "
              f"{f.get('ann_pct', float('nan')):+9.2f} "
              f"{f.get('season_t', float('nan')):+8.2f} | "
              f"{h1.get('ann_pct', float('nan')):+10.2f} "
              f"{h2.get('ann_pct', float('nan')):+12.2f}")

    qb, ab = books["quiet beats"], books["all beats"]
    diff = (qb["strat"] - qb["cost"]) - (ab["strat"] - ab["cost"])
    m2 = np.isfinite(diff)
    sid = season_of(dates, m2)[m2]
    per = np.array([diff[m2][sid == s].mean() for s in np.unique(sid)])
    t_ = tracker._nw_t(per, 1) if len(per) >= 8 else np.nan
    add = float(diff[m2].mean() * 250 * 100)
    print(f"\n  what the ownership filter adds over buying every beat:")
    print(f"    {add:+.2f}%/yr   season t {t_:+.2f}  ({len(per)} seasons)")
    print("    this is the whole commercial case for the data. A filter that")
    print("    adds nothing here is not worth collecting, whatever its t-stat")
    print("    in a regression.")
    out["filter_value"] = {"ann_pct": add, "season_t": t_,
                           "seasons": int(len(per))}

    ok = (np.isfinite(t_) and add > 0 and t_ >= 2
          and out["rules"]["quiet beats"]["first_half"].get("ann_pct", -9) > 0
          and out["rules"]["quiet beats"]["second_half"].get("ann_pct", -9) > 0)
    out["verdict"] = "USABLE LONG ONLY" if ok else "NOT USABLE AS A LONG-ONLY RULE"
    print(f"\n  verdict: {out['verdict']}")
    print("  (needs a positive, significant filter gain AND both halves positive)")

    RESULTS.mkdir(exist_ok=True)
    dest = RESULTS / "longonly.json"
    dest.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
