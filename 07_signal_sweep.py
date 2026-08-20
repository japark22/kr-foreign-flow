#!/usr/bin/env python3
"""Step 7: does a better signal construction change the verdict?

    python 07_signal_sweep.py
    python 07_signal_sweep.py --split 20200101 --horizons 1,5,20

WHY THIS EXISTS
---------------
`04_validate.py` tested exactly one construction: a one-day change in holdings,
divided by shares outstanding, across every listed name. It found flow
predicts flow overwhelmingly (t=64) but returns only marginally (t=3.9), and
nothing that survives Korean transaction costs.

That is a result about *that construction*, not about the data. Three things
were left untested, and each is where a flow signal normally lives:

  ACCUMULATION — persistence out to lag 20 means information accumulates. A
    one-day delta throws that away. Cumulative flow over N days is the standard
    construction and should be materially stronger if the effect is real.

  DENOMINATOR — dividing by shares outstanding measures "fraction of the
    company bought". Dividing by average daily traded value measures "how big
    was this relative to what the stock can absorb", which is what actually
    moves price.

  LIQUIDITY FLOOR — 3,600 names, most of them thin. If the effect lives in
    tradable names, the full cross-section dilutes it; if it lives only in
    illiquid names, it was never tradable anyway. Either answer is useful.

THE OVERFITTING PROBLEM, AND WHAT WE DO ABOUT IT
------------------------------------------------
Sweeping constructions and reporting the best one is how false discoveries are
made. With enough variants something always looks good.

So the sample is split. Variants are ranked on the IN-SAMPLE period, and the
number that matters is the OUT-OF-SAMPLE one for the same variant. A variant
that looks strong in-sample and dies out-of-sample was noise. The script prints
both, side by side, and prints how many variants were tried so the multiple-
testing burden is visible rather than hidden.

Nothing here is evidence until the out-of-sample column agrees.
"""
from __future__ import annotations

import argparse
import itertools
import sys

import numpy as np
import pandas as pd

from krxflow import features, storage

W = 96
COST_ONE_WAY = 0.0030  # 20bp Korean transaction tax + ~10bp spread; optimistic


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


def build_variant(panels, market, accum: int, denom: str, liquidity: str,
                  adv: pd.DataFrame):
    """One signal construction -> ranked panel, aligned to the market grid."""
    held = panels["foreign_shares"]
    shares = panels["shares_listed"]

    d_held = held.diff().mask(features.corporate_action_mask(panels))

    # Accumulate the raw share flow BEFORE normalising, so the sum is a real
    # quantity of shares bought over the window rather than a sum of ratios.
    if accum > 1:
        d_held = d_held.rolling(accum, min_periods=max(2, accum // 2)).sum()

    if denom == "shares":
        flow = d_held / shares.where(shares > 0)
    elif denom == "adv":
        # Shares bought, valued at close, relative to what the stock trades.
        close = market["close"].reindex(index=d_held.index, columns=d_held.columns)
        flow = (d_held * close) / adv.where(adv > 0)
    else:
        raise ValueError(denom)

    flow = flow.where(features.universe_mask(panels))

    if liquidity != "all":
        rank = adv.rank(axis=1, pct=True)
        cut = {"adv_top50": 0.50, "adv_top20": 0.80}[liquidity]
        flow = flow.where(rank >= cut)

    return features.cross_sectional_rank(flow)


def evaluate(ranked, close, horizons, lo=None, hi=None):
    """IC and cost-adjusted long-short for one variant over a date window."""
    sig = ranked.shift(1)  # D-1: trade on the following day
    if lo is not None:
        sig = sig.loc[sig.index >= lo]
    if hi is not None:
        sig = sig.loc[sig.index < hi]

    out = {}
    for h in horizons:
        fwd = features.forward_returns(close, h).reindex(index=sig.index,
                                                         columns=sig.columns)
        fwd_rank = features.cross_sectional_rank(fwd)
        ic = sig.corrwith(fwd_rank, axis=1).dropna()
        if len(ic) < 60:
            out[h] = None
            continue

        t = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))

        q = sig.rank(axis=1, pct=True)
        long_leg = q >= 0.8
        ls = (fwd.where(long_leg).mean(axis=1)
              - fwd.where(q <= 0.2).mean(axis=1)).dropna()

        held = long_leg.astype(float)
        churn = ((held.diff().abs().sum(axis=1) / 2)
                 / held.sum(axis=1).replace(0, np.nan)).dropna().mean()

        ann = ls.mean() / h * 252
        vol = ls.std(ddof=1) / np.sqrt(h) * np.sqrt(252)
        ann_turn = (churn or 0) * (252 / h) * 2
        net = ann - ann_turn * COST_ONE_WAY

        out[h] = {"ic": ic.mean(), "t": t, "gross": ann,
                  "sharpe": ann / vol if vol else np.nan,
                  "turn": ann_turn, "net": net}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="20200101",
                   help="in-sample is before this date, out-of-sample after")
    p.add_argument("--horizons", default="5,20",
                   help="holding periods to test (1-day is not tradable, see 04)")
    p.add_argument("--adv-window", type=int, default=20)
    args = p.parse_args()

    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    split = pd.Timestamp(args.split)

    rule("Loading")
    panels = features.load_panels()
    market = load_market()
    if market is None:
        print("  No market data. Run 01_backfill.py --with-market first.")
        return 1

    close = market["close"].reindex(index=panels["foreign_shares"].index,
                                    columns=panels["foreign_shares"].columns)
    adv = (market["value_traded"]
           .reindex(index=close.index, columns=close.columns)
           .rolling(args.adv_window, min_periods=5).mean())

    print(f"  dates    : {close.index[0].date()} -> {close.index[-1].date()}")
    print(f"  in-sample: before {split.date()}   "
          f"out-of-sample: {split.date()} onward")
    print(f"  ADV window: {args.adv_window} days")

    ACCUM = [1, 5, 20]
    DENOM = ["shares", "adv"]
    LIQ = ["all", "adv_top50", "adv_top20"]
    grid = list(itertools.product(ACCUM, DENOM, LIQ))

    print(f"\n  variants: {len(grid)} constructions x {len(horizons)} horizons "
          f"= {len(grid) * len(horizons)} tests")
    print("  Ranked on in-sample. The out-of-sample column is the only one that")
    print("  counts. With this many tests, an in-sample winner is expected even")
    print("  from pure noise.")

    rows = []
    for i, (accum, denom, liq) in enumerate(grid, 1):
        label = f"{accum:>2}d/{denom:<6}/{liq:<9}"
        print(f"\r  evaluating {i}/{len(grid)}  {label}", end="", flush=True)
        try:
            ranked = build_variant(panels, market, accum, denom, liq, adv)
            ins = evaluate(ranked, close, horizons, hi=split)
            oos = evaluate(ranked, close, horizons, lo=split)
        except Exception as err:  # noqa: BLE001
            print(f"\n    ! {label}: {type(err).__name__}: {err}")
            continue

        for h in horizons:
            if ins.get(h) and oos.get(h):
                rows.append({"accum": accum, "denom": denom, "liq": liq, "h": h,
                             "ic_is": ins[h]["ic"], "t_is": ins[h]["t"],
                             "ic_oos": oos[h]["ic"], "t_oos": oos[h]["t"],
                             "net_is": ins[h]["net"], "net_oos": oos[h]["net"],
                             "sharpe_oos": oos[h]["sharpe"],
                             "turn": oos[h]["turn"]})
    print()

    if not rows:
        print("\n  No variant produced enough data to evaluate.")
        return 1

    df = pd.DataFrame(rows).sort_values("ic_is", key=abs, ascending=False)

    rule("Ranked by in-sample IC — read the OOS columns")
    print("  accum  denom   liquidity    h    IC(is)   t(is)    IC(oos)  t(oos)"
          "   net(oos)   turn")
    print("  -----  ------  ---------  ---  --------  ------  ---------  ------"
          "  ---------  -----")
    for _, r in df.iterrows():
        agree = "" if np.sign(r.ic_is) == np.sign(r.ic_oos) else "  <- sign flip"
        print(f"  {r.accum:>4}d  {r.denom:<6}  {r.liq:<9}  {r.h:>3}  "
              f"{r.ic_is:>+8.5f}  {r.t_is:>+6.2f}  {r.ic_oos:>+9.5f}  "
              f"{r.t_oos:>+6.2f}  {r.net_oos:>+8.2%}  {r.turn:>5.0f}x{agree}")

    rule("What survives")
    # In-sample significance is required too. A variant that is flat
    # in-sample and only appears out-of-sample was not validated by anything —
    # it is one of N draws that happened to land, which is exactly what the
    # split was meant to catch.
    survivors = df[(df.t_oos.abs() > 3)
                   & (df.t_is.abs() > 3)
                   & (np.sign(df.ic_is) == np.sign(df.ic_oos))
                   & (df.net_oos > 0)]
    if survivors.empty:
        print("  Nothing. No construction is significant out-of-sample, with a")
        print("  consistent sign, and net-positive after costs.")
        print()
        print("  That is a real answer, not a failure to find one. The flow is")
        print("  predictable — lag-20 autocorrelation at t=21 is not in doubt —")
        print("  but the market appears to price the predictable part. Options")
        print("  from here: a genuinely different signal (level rather than")
        print("  flow, limit-exhaustion, foreign-vs-domestic divergence), or")
        print("  conclude the disclosure is not an edge on its own.")
    else:
        print(f"  {len(survivors)} construction(s) significant out-of-sample with")
        print("  a consistent sign and positive net return:\n")
        for _, r in survivors.iterrows():
            print(f"    {r.accum}d accumulation / {r.denom} / {r.liq}, {r.h}d hold")
            print(f"      IC {r.ic_is:+.5f} (is) -> {r.ic_oos:+.5f} (oos, "
                  f"t={r.t_oos:+.2f}),  net {r.net_oos:+.2%}, "
                  f"Sharpe {r.sharpe_oos:+.2f}")
        print()
        print(f"  Caveat: {len(df)} tests were run. Even a 3-sigma out-of-sample")
        print("  result needs a fresh period before it is treated as real.")

    out = storage.config.FEATURE_DIR / "signal_sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n  full grid written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
