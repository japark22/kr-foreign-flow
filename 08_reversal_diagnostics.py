#!/usr/bin/env python3
"""Step 8: resolve the two contradictions in the sweep result.

    python 08_reversal_diagnostics.py
    python 08_reversal_diagnostics.py --accum 20 --split 20200101

The sweep found that 20-day accumulated foreign flow, normalised by average
daily traded value, has a robust NEGATIVE information coefficient: names with
the heaviest recent foreign accumulation underperform. In-sample t=-12.8,
out-of-sample t=-10.7, sign consistent, magnitude above the 0.01 threshold,
turnover low enough to survive costs.

Two things in that result contradict each other, and both have to be settled
before the finding means anything.

CONTRADICTION 1 — IC says down, the quintile portfolio says up
--------------------------------------------------------------
Rank IC is -0.014 while the top-minus-bottom spread on raw returns is
positive. For a monotone relationship that is impossible.

The likely cause: IC is rank-based and so describes the *typical* name, while
the quintile spread averages *raw* returns and is therefore dominated by a few
extreme winners. If so, the median heavy-accumulation name underperforms while
the bucket mean is dragged positive by outliers — and the positive net return
in the sweep is an artifact of skew, not a tradable edge.

The test: decile returns computed three ways — mean, median, and winsorised.
If mean and median disagree in sign, skew is doing the work.

CONTRADICTION 2 — the sign flips with liquidity
-----------------------------------------------
Full universe gives OOS IC -0.014. Restricted to the most liquid 20% it gives
+0.014. Opposite signs.

Either the effect is unstable noise, or two different mechanisms are being
mixed: information in liquid names that the market absorbs and continues, and
pure price pressure in thin names that reverts. The second would be a real and
useful decomposition; the first would be nothing.

The test: IC by liquidity tercile, in-sample and out-of-sample separately. A
genuine mechanism split shows the same pattern in both periods.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from krxflow import features, storage

W = 88


def rule(title: str = "") -> None:
    print()
    print("=" * W)
    if title:
        print(f"  {title}")
        print("=" * W)


def load_market():
    df = storage.read_range("market", columns=["trade_date", "ticker",
                                               "close", "value_traded"])
    if df.empty:
        return None
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    def pivot(col):
        return (df.pivot_table(index="trade_date", columns="ticker", values=col,
                               aggfunc="last", observed=True)
                .sort_index().astype("float64"))

    return {"close": pivot("close"), "value_traded": pivot("value_traded")}


def winsorise(df: pd.DataFrame, lo: float = 0.01, hi: float = 0.99) -> pd.DataFrame:
    """Clip each row at its own percentiles, so extremes stop dominating means."""
    ql = df.quantile(lo, axis=1)
    qh = df.quantile(hi, axis=1)
    return df.clip(lower=ql, upper=qh, axis=0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--accum", type=int, default=20, help="accumulation window")
    p.add_argument("--horizon", type=int, default=20, help="holding period")
    p.add_argument("--split", default="20200101")
    p.add_argument("--adv-window", type=int, default=20)
    p.add_argument("--deciles", type=int, default=10)
    args = p.parse_args()

    split = pd.Timestamp(args.split)

    rule("Loading")
    panels = features.load_panels()
    market = load_market()
    if market is None:
        print("  No market data stored.")
        return 1

    idx = panels["foreign_shares"].index
    cols = panels["foreign_shares"].columns
    close = market["close"].reindex(index=idx, columns=cols)
    adv = (market["value_traded"].reindex(index=idx, columns=cols)
           .rolling(args.adv_window, min_periods=5).mean())

    # The variant the sweep singled out.
    d_held = panels["foreign_shares"].diff().mask(
        features.corporate_action_mask(panels))
    if args.accum > 1:
        d_held = d_held.rolling(args.accum,
                                min_periods=max(2, args.accum // 2)).sum()
    flow = (d_held * close) / adv.where(adv > 0)
    flow = flow.where(features.universe_mask(panels))

    signal = features.cross_sectional_rank(flow).shift(1)
    fwd = features.forward_returns(close, args.horizon)

    print(f"  construction: {args.accum}d accumulation / ADV, {args.horizon}d hold")
    print(f"  in-sample before {split.date()}, out-of-sample after")
    print(f"  observations: {int(signal.notna().to_numpy().sum()):,}")

    periods = {
        "in-sample": signal.index < split,
        "out-of-sample": signal.index >= split,
    }

    # ------------------------------------------------------------------ 1 --
    rule(f"1. Decile returns — is the relationship monotone?")
    print(f"  Forward {args.horizon}-day return by signal decile. Decile 1 =")
    print("  heaviest foreign SELLING, decile 10 = heaviest BUYING.")
    print()
    print("  If IC is negative and the relationship is monotone, returns should")
    print("  fall from decile 1 to decile 10. If mean and median disagree in")
    print("  sign, the mean is being driven by outliers and the quintile")
    print("  spread in the sweep cannot be trusted.\n")

    q = signal.rank(axis=1, pct=True)
    fwd_w = winsorise(fwd)

    for name, mask in periods.items():
        sig_p = signal.loc[mask]
        q_p = q.loc[mask]
        fwd_p = fwd.loc[mask]
        fwd_wp = fwd_w.loc[mask]
        if sig_p.notna().to_numpy().sum() < 1000:
            continue

        print(f"  {name}")
        print("  decile      mean       median   winsorised   n/day")
        print("  ------   --------   ----------   ----------   -----")
        rows = []
        for d in range(args.deciles):
            lo, hi = d / args.deciles, (d + 1) / args.deciles
            sel = (q_p > lo) & (q_p <= hi) if d else (q_p >= 0) & (q_p <= hi)
            m = fwd_p.where(sel)
            rows.append((d + 1, m.stack().mean(), m.stack().median(),
                         fwd_wp.where(sel).stack().mean(),
                         sel.sum(axis=1).mean()))
        for d, mean, med, wins, n in rows:
            print(f"  {d:>6}   {mean:>+8.3%}   {med:>+10.3%}   {wins:>+10.3%}   "
                  f"{n:>5.0f}")

        top, bot = rows[-1], rows[0]
        print(f"\n    top-bottom, mean       : {top[1] - bot[1]:+.3%}")
        print(f"    top-bottom, median     : {top[2] - bot[2]:+.3%}")
        print(f"    top-bottom, winsorised : {top[3] - bot[3]:+.3%}")

        means = np.array([r[1] for r in rows])
        meds = np.array([r[2] for r in rows])
        if np.sign(top[1] - bot[1]) != np.sign(top[2] - bot[2]):
            print("    -> MEAN AND MEDIAN DISAGREE. The spread is an artifact of")
            print("       skew; the median is the honest description.")
        else:
            print("    -> mean and median agree in sign.")

        # Monotonicity: correlation of decile number with decile return.
        r_mean = np.corrcoef(np.arange(1, len(means) + 1), means)[0, 1]
        r_med = np.corrcoef(np.arange(1, len(meds) + 1), meds)[0, 1]
        print(f"    monotonicity (decile vs return): mean {r_mean:+.2f}, "
              f"median {r_med:+.2f}")
        print()

    # ------------------------------------------------------------------ 2 --
    rule("2. Does the sign really flip with liquidity?")
    print("  IC within liquidity terciles, computed separately in each period.")
    print("  A genuine two-mechanism story (information in liquid names,")
    print("  price pressure in thin ones) shows the SAME pattern in both.")
    print("  Noise does not.\n")

    adv_rank = adv.rank(axis=1, pct=True)
    buckets = {
        "thin      (bottom 33%)": (0.00, 0.33),
        "mid       (33-67%)": (0.33, 0.67),
        "liquid    (top 33%)": (0.67, 1.01),
    }

    print("  liquidity bucket          in-sample IC     t      "
          "out-of-sample IC     t")
    print("  ----------------------   -------------  ------   "
          "-----------------  ------")
    for label, (lo, hi) in buckets.items():
        sel = (adv_rank >= lo) & (adv_rank < hi)
        sig_b = signal.where(sel)
        cells = []
        for name, mask in periods.items():
            s = sig_b.loc[mask]
            f = features.cross_sectional_rank(fwd.loc[mask].where(sel.loc[mask]))
            ic = s.corrwith(f, axis=1).dropna()
            if len(ic) < 60:
                cells.append((np.nan, np.nan))
                continue
            t = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))
            cells.append((ic.mean(), t))
        (ic_i, t_i), (ic_o, t_o) = cells
        flag = ""
        if not np.isnan(ic_i) and not np.isnan(ic_o) and np.sign(ic_i) != np.sign(ic_o):
            flag = "  <- sign flip across periods"
        print(f"  {label:<22}   {ic_i:>+13.5f}  {t_i:>+6.2f}   "
              f"{ic_o:>+17.5f}  {t_o:>+6.2f}{flag}")

    print()
    print("  Read it: if thin names are negative and liquid names positive in")
    print("  BOTH periods, that is two mechanisms and worth building on. If the")
    print("  pattern differs between periods, it is noise and the sweep's")
    print("  liquidity result should be discarded.")

    # ------------------------------------------------------------------ 3 --
    rule("3. Is it stable year by year?")
    print("  A signal that only works in a few years is not a signal. Annual IC")
    print("  for the full universe.\n")
    print("  year      IC        t     days")
    print("  ----   --------   -----   ----")

    fwd_rank_all = features.cross_sectional_rank(fwd)
    ic_all = signal.corrwith(fwd_rank_all, axis=1).dropna()
    by_year = ic_all.groupby(ic_all.index.year)
    pos = neg = 0
    for year, s in by_year:
        if len(s) < 60:
            continue
        t = s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))
        pos += s.mean() > 0
        neg += s.mean() < 0
        print(f"  {year}   {s.mean():>+8.5f}   {t:>+5.1f}   {len(s):>4}")

    print(f"\n  {neg} year(s) negative, {pos} positive.")
    if min(pos, neg) / max(pos + neg, 1) > 0.3:
        print("  The sign is not stable across years. Whatever the pooled t-stat")
        print("  says, this is not one effect — it is a mix, and pooling hides that.")
    else:
        print("  The sign is consistent across most years, which is what a real")
        print("  effect looks like.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
