#!/usr/bin/env python3
"""Step 9b: settle which standard error is right, by measurement not assumption.

WHY THIS EXISTS
---------------
09_overlap_correction.py assumed that overlapping h-day forward returns must
make the daily IC series autocorrelated, and picked the non-overlapping
standard error on the strength of a calibration run against exactly that
assumption. On the real data the assumption failed: the measured lag-1
autocorrelation of the daily IC series is about zero at every horizon, and the
non-overlapping "correction" was doing nothing but multiplying the standard
error by sqrt(h) -- arithmetic, not evidence.

That is the error this script is here to avoid repeating. It does two things:

  1. --selftest now includes a WHITE-NOISE arm alongside the moving-average
     arm. Under white noise the naive standard error is correct and the
     non-overlapping one over-rejects the wrong way, rejecting far too rarely.
     Seeing both arms is what tells you which estimator to use, and neither
     arm alone can.

  2. On real data it prints the full autocorrelation profile of the daily IC
     series, lags 1..2h. Theory is specific here: if the overlap drives the
     series, the profile starts near (h-1)/h and falls linearly to zero at lag
     h. If instead it is flat and near zero from lag 1, the daily variation is
     cross-sectional estimation noise that is not shared across adjacent days,
     the naive standard error is appropriate, and no correction is warranted.

The profile is the evidence. Read it before trusting any t in this project.
"""
from __future__ import annotations

import argparse
import math
import random
import sys


def mean_t_naive(x):
    n = len(x)
    m = sum(x) / n
    var = sum((v - m) ** 2 for v in x) / (n - 1)
    se = math.sqrt(var / n)
    return m, (m / se if se else None)


def mean_t_nw(x, lags):
    n = len(x)
    m = sum(x) / n
    e = [v - m for v in x]
    s = sum(v * v for v in e) / n
    lags = max(0, min(lags, n - 2))
    for j in range(1, lags + 1):
        gj = sum(e[t] * e[t - j] for t in range(j, n)) / n
        s += 2.0 * (1.0 - j / (lags + 1.0)) * gj
    if s <= 0:
        return m, None
    se = math.sqrt(s / n)
    return m, (m / se if se else None)


def mean_t_nonoverlap(x, step):
    step = max(1, step)
    if step == 1:
        return mean_t_naive(x)
    m = sum(x) / len(x)
    ses = []
    for off in range(step):
        sub = x[off::step]
        if len(sub) < 3:
            continue
        sm = sum(sub) / len(sub)
        var = sum((v - sm) ** 2 for v in sub) / (len(sub) - 1)
        ses.append(math.sqrt(var / len(sub)))
    if not ses:
        return m, None
    ses.sort()
    k = len(ses)
    se = ses[k // 2] if k % 2 else 0.5 * (ses[k // 2 - 1] + ses[k // 2])
    return m, (m / se if se else None)


def autocorr(x, lag):
    n = len(x)
    if n <= lag + 2:
        return None
    m = sum(x) / n
    num = sum((x[t] - m) * (x[t - lag] - m) for t in range(lag, n))
    den = sum((v - m) ** 2 for v in x)
    return num / den if den else None


def selftest(reps=1200, n=1000, seed=20260821):
    rng = random.Random(seed)
    print("\nTwo nulls, both with a true mean of ZERO. A correct standard error")
    print("rejects about 5% of the time in BOTH.\n")
    print("  arm A  moving average of h shocks -- what overlapping windows give")
    print("         if the overlap dominates the daily variation")
    print("  arm B  white noise -- what you get if the daily variation is")
    print("         cross-sectional estimation error instead\n")
    print(f"  {reps:,} replications, {n:,} days each\n")
    print(f"  {'arm':<6}{'h':>4}{'naive':>10}{'Newey-West':>13}"
          f"{'non-overlap':>14}{'AC(1)':>9}")
    print("  " + "-" * 56)

    for arm in ("A: MA(h)", "B: white"):
        for h in (1, 5, 20, 60):
            rn = rw = ro = 0
            acs = []
            for _ in range(reps):
                if arm.startswith("A"):
                    sh = [rng.gauss(0, 1) for _ in range(n + h)]
                    x = [sum(sh[t:t + h]) / h for t in range(n)]
                else:
                    x = [rng.gauss(0, 1) for _ in range(n)]
                _, tn = mean_t_naive(x)
                _, tw = mean_t_nw(x, h - 1)
                _, to = mean_t_nonoverlap(x, h)
                rn += (tn is not None and abs(tn) > 1.96)
                rw += (tw is not None and abs(tw) > 1.96)
                ro += (to is not None and abs(to) > 1.96)
                a = autocorr(x, 1)
                if a is not None:
                    acs.append(a)
            ac = sum(acs) / len(acs) if acs else float("nan")
            print(f"  {arm:<6}{h:>4}{100*rn/reps:>9.1f}%{100*rw/reps:>12.1f}%"
                  f"{100*ro/reps:>13.1f}%{ac:>+9.3f}")
        print("  " + "-" * 56)

    print("\n  Arm A: naive over-rejects badly, non-overlap is right.")
    print("  Arm B: naive is right, non-overlap rejects far too RARELY --")
    print("         it inflates the standard error by sqrt(h) for no reason.")
    print("  Newey-West is the only column acceptable in both arms, because it")
    print("  estimates the autocovariance instead of assuming it. Use the")
    print("  measured AC(1) on real data to know which arm you are in.")


def real(args):
    import numpy as np  # noqa: F401
    from krxflow import features, storage
    import pandas as pd

    h = args.horizon
    print("\nLoading panels...")
    panels = features.load_panels(args.start, args.end)
    universe = features.universe_mask(panels)
    flow = features.build_flow(panels, denominator="shares").where(universe)
    flow, filt = features.apply_filters(
        flow, panels, drop_rebalance=True,
        rebalance_window=args.rebalance_window,
        drop_exdiv=not args.keep_exdiv, universe=universe)
    ranked = features.cross_sectional_rank(flow)

    df = storage.read_range("market", args.start, args.end,
                            columns=["trade_date", "ticker", "close"])
    if df.empty:
        sys.exit("no market store")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    close = df.pivot_table(index="trade_date", columns="ticker", values="close",
                           aggfunc="last", observed=True).sort_index() \
              .astype("float64").reindex(index=ranked.index, columns=ranked.columns)

    signal = ranked.shift(1)
    fwd = features.cross_sectional_rank(features.forward_returns(close, h))
    ic = signal.corrwith(fwd, axis=1).dropna()

    # Contiguity check: lag-1 is only lag-1 if the dates are adjacent.
    idx = ic.index
    pos = {d: i for i, d in enumerate(ranked.index)}
    gaps = sum(1 for a, b in zip(idx[:-1], idx[1:]) if pos[b] - pos[a] != 1)
    x = [float(v) for v in ic.to_numpy()]
    mean = sum(x) / len(x)
    std = math.sqrt(sum((v - mean) ** 2 for v in x) / (len(x) - 1))

    print(f"horizon {h}d   days {len(x):,}   non-adjacent day pairs {gaps:,}")
    if gaps:
        print("  WARNING: the series has gaps, so the lag axis below is not")
        print("  purely in trading days. Interpret with that in mind.")
    print(f"daily IC: mean {mean:+.5f}   std {std:.5f}")
    print()
    print(f"  Autocorrelation of the daily IC series. If the {h}-day overlap")
    print(f"  drives it, lag 1 starts near {(h-1)/h:+.2f} and falls to 0 by lag {h}.")
    print()
    print(f"  {'lag':>4}{'AC':>9}   {'overlap prediction':>19}   profile")
    print("  " + "-" * 62)
    for lag in range(1, 2 * h + 1):
        if h >= 20 and lag > 6 and lag % max(1, h // 5) and lag != h and lag != 2 * h:
            continue
        a = autocorr(x, lag)
        pred = max(0.0, (h - lag) / h)
        bar = "#" * int(round(abs(a) * 40)) if a is not None else ""
        sign = "" if (a is None or a >= 0) else "-"
        print(f"  {lag:>4}{a:>+9.3f}   {pred:>19.2f}   {sign}{bar}")
    print("  " + "-" * 62)

    _, tn = mean_t_naive(x)
    _, tw = mean_t_nw(x, h - 1)
    _, to = mean_t_nonoverlap(x, h)
    f = lambda v: "  n/a" if v is None else f"{v:+.2f}"
    print(f"  t naive {f(tn)}    t Newey-West {f(tw)}    t non-overlap {f(to)}")
    print()
    a1 = autocorr(x, 1)
    if a1 is not None and abs(a1) < 0.10:
        print("  VERDICT: lag-1 autocorrelation is near zero, so the overlap is")
        print("  NOT driving the daily variation. The naive standard error is")
        print("  appropriate and Newey-West agrees with it. The non-overlapping")
        print("  figure is over-conservative here by construction and should")
        print("  not be used.")
    else:
        print("  VERDICT: the series is autocorrelated. Use Newey-West; the")
        print("  naive standard error is too small.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--reps", type=int, default=1200)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--rebalance-window", type=int, default=2)
    p.add_argument("--keep-exdiv", action="store_true")
    a = p.parse_args()
    selftest(reps=a.reps) if a.selftest else real(a)


if __name__ == "__main__":
    main()
