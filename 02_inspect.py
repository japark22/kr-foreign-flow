#!/usr/bin/env python3
"""Step 3: QA the collected data and take the first look at the hypothesis.

    python 02_inspect.py

Runs entirely offline against what is already on disk. Three things:

  1. Coverage — which trading days are stored, any gaps, ticker counts drifting.
  2. Precision — how badly the float16 column would have hurt us.
  3. Persistence — the autocorrelation curve of daily normalised foreign flow.

(3) is the first real test of the hypothesis: if foreign flow is
persistent, today's flow should correlate with tomorrow's, decaying slowly
over several days rather than dropping to zero at lag 1. This is a raw,
un-denoised read — index rebalancing and ex-dividend effects are still in
here — so treat it as a smell test, not a result.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from krxflow import storage

MAX_LAG = 15


def main() -> int:
    dates = storage.stored_dates("foreign_ownership")
    if not dates:
        print("No data stored yet. Run 01_backfill.py first.")
        return 1

    print("=" * 74)
    print("  Coverage")
    print("=" * 74)
    print(f"  stored trading days : {len(dates):,}")
    print(f"  range               : {dates[0]} -> {dates[-1]}")

    df = storage.read_range("foreign_ownership")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    print(f"  rows                : {len(df):,}")
    print(f"  unique tickers      : {df['ticker'].nunique():,}")

    per_day = df.groupby("trade_date").size()
    print(f"  tickers/day         : min {per_day.min():,}  "
          f"median {int(per_day.median()):,}  max {per_day.max():,}")

    thin = per_day[per_day < per_day.median() * 0.8]
    if len(thin):
        print(f"\n  {len(thin)} day(s) with unusually few tickers — worth a look:")
        for d, n in thin.head(10).items():
            print(f"    {d.date()}  {n:,}")

    # ------------------------------------------------------------------ 2 --
    print()
    print("=" * 74)
    print("  Precision: exact ratio vs pykrx's float16 column")
    print("=" * 74)
    if "foreign_pct_krx_lossy" in df.columns:
        err = (df["foreign_pct"] - df["foreign_pct_krx_lossy"]).abs()
        print(f"  median error  {err.median():.6f} pp")
        print(f"  p99 error     {err.quantile(0.99):.6f} pp")
        print(f"  max error     {err.max():.6f} pp")
        print("  For reference, the median absolute one-day change in foreign")
        print("  ownership is printed below — if the rounding error is the same")
        print("  order of magnitude, the lossy column would have destroyed the signal.")

    # ------------------------------------------------------------------ 3 --
    print()
    print("=" * 74)
    print("  Persistence of daily foreign flow (raw, not yet denoised)")
    print("=" * 74)

    panel = df.pivot_table(index="trade_date", columns="ticker",
                           values="foreign_pct", aggfunc="last").sort_index()
    shares = df.pivot_table(index="trade_date", columns="ticker",
                            values="shares_listed", aggfunc="last").sort_index()
    shares = shares.astype("float64")  # Int64 -> float so diff/mask stay non-nullable

    # Daily change in ownership, in percentage points.
    d_pct = panel.astype("float64").diff()

    # Drop days where shares outstanding moved: that is a corporate action, not
    # a trade. Crude version of the corporate-action scrub in the spec.
    shares_changed = (shares.pct_change().abs() > 1e-9).fillna(False).astype(bool)
    d_pct = d_pct.mask(shares_changed)

    print(f"  median |1-day change| : {d_pct.abs().stack().median():.6f} pp")
    print(f"  days x tickers        : {d_pct.shape[0]:,} x {d_pct.shape[1]:,}")
    print(f"  masked as corp action : {int(shares_changed.to_numpy().sum()):,} cells")

    # Cross-sectionally rank-normalise each day, then measure how much the
    # cross-section persists from day t to day t+lag. Ranking makes the measure
    # robust to outliers and to the level differences between big and small caps.
    ranked = d_pct.rank(axis=1, pct=True) - 0.5

    if len(ranked) < MAX_LAG + 5:
        print(f"\n  Only {len(ranked)} days stored — need at least ~{MAX_LAG + 5} "
              "for the autocorrelation curve. Backfill more history.")
        return 0

    print("\n  lag    corr    -0.2        0        +0.2")
    print("  ---   ------   |----------|----------|")
    half = 10  # characters per 0.2 of correlation
    for lag in range(1, MAX_LAG + 1):
        corr = ranked.corrwith(ranked.shift(-lag), axis=1).mean()
        n = max(-half, min(half, int(round(corr / 0.2 * half))))
        if n >= 0:
            bar = " " * half + "|" + "#" * n
        else:
            bar = " " * (half + n) + "#" * (-n) + "|"
        print(f"  {lag:>3}   {corr:+.4f}   {bar}")

    print("\n  Reading it: a curve that starts clearly positive and decays over")
    print("  several days supports the hypothesis. A spike at lag 1 that dies")
    print("  immediately is more likely settlement/reporting mechanics than")
    print("  genuine multi-day accumulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
