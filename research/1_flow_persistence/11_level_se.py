"""Step 11: measure the standard error of the LEVEL signals. No new data.

WHY THIS IS SEPARATE FROM 09b
-----------------------------
09b measured the daily IC series of the FLOW signal and found its lag-1
autocorrelation is about zero, so the naive standard error was fine there and
the horizon table stands.

The level signals are the opposite case by construction. `level` is ownership
percent: it barely moves day to day, turnover is roughly 0.1x, and the forward
window at h=60 overlaps the next day's by 59 days. Both the signal and the
return are nearly the same tomorrow as today, so consecutive daily ICs should
be almost the same number. If so, `ic.std(ddof=1)/sqrt(n)` is badly too small
and the reported t = +66 on `level` is not 66 sigma of evidence.

The project notes already suspected this and estimated the inflation at
sqrt(1600/27) ~ 7.7x. That number was an assumption, never measured. 09b showed
the analogous assumption was wrong for flow, in the other direction. So it gets
measured here rather than argued.

WHAT IS REPORTED
----------------
For each signal, out-of-sample, at each horizon:

  t naive     what 09_level_signals.py prints
  t NW        Newey-West, Bartlett kernel, h-1 lags. Estimates the
              autocovariance instead of assuming it, and 09b's two-arm
              calibration showed it is the only estimator acceptable whether
              the series is autocorrelated or not.
  AC(1)       lag-1 autocorrelation of the daily IC series. The deciding
              measurement: near (h-1)/h means the overlap dominates, near zero
              means it does not.
  n_eff       n x (se_naive / se_NW)^2 -- how many independent observations the
              3,000-odd daily ICs are actually worth. This is the number the
              project notes guessed at.

A signal is only interesting if t NW clears the bar, not t naive.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
import pandas as pd

from krxflow import features, storage

SPLIT = pd.Timestamp("20200101")
HORIZONS = [20, 60]
Z_WIN = 252
AC_PROFILE_FOR = ("level", 60)      # print the full profile for the headline case
EMIT = {"schema": 1}


def nw_se(x, lags):
    """Bartlett-kernel HAC standard error of the sample mean."""
    n = len(x)
    m = sum(x) / n
    e = [v - m for v in x]
    s = sum(v * v for v in e) / n
    lags = max(0, min(lags, n - 2))
    for j in range(1, lags + 1):
        gj = sum(e[t] * e[t - j] for t in range(j, n)) / n
        s += 2.0 * (1.0 - j / (lags + 1.0)) * gj
    return math.sqrt(s / n) if s > 0 else None


def naive_se(x):
    n = len(x)
    m = sum(x) / n
    return math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1) / n)


def autocorr(x, lag):
    n = len(x)
    if n <= lag + 2:
        return None
    m = sum(x) / n
    num = sum((x[t] - m) * (x[t - lag] - m) for t in range(lag, n))
    den = sum((v - m) ** 2 for v in x)
    return num / den if den else None


def pv(df, col):
    return (df.pivot_table(index="trade_date", columns="ticker", values=col,
                           aggfunc="last", observed=True)
              .sort_index().astype("float64"))


def main() -> None:
    print("loading ...")
    p = features.load_panels()
    m = storage.read_range("market",
                           columns=["trade_date", "ticker", "close", "value_traded"])
    m["trade_date"] = pd.to_datetime(m["trade_date"])
    idx, cols = p["foreign_shares"].index, p["foreign_shares"].columns
    close = pv(m, "close").reindex(index=idx, columns=cols)
    adv = pv(m, "value_traded").reindex(index=idx, columns=cols) \
            .rolling(20, min_periods=5).mean()
    del m

    uni = features.universe_mask(p)
    lvl = p["foreign_pct"].where(uni)
    limit = p["foreign_limit_shares"]
    binding = (limit < p["shares_listed"] * 0.999)

    sig = {"level": lvl}
    mu = lvl.rolling(Z_WIN, min_periods=120).mean()
    sd = lvl.rolling(Z_WIN, min_periods=120).std()
    sig["level_z"] = (lvl - mu) / sd.where(sd > 0)
    sig["d60"] = lvl.diff(60)
    sig["d120"] = lvl.diff(120)
    sig["exhaust"] = (p["foreign_shares"] / limit.where(limit > 0) * 100) \
                     .where(uni & binding)

    # 09_level_signals.py reports its headline t = +66 on the liquid third, not
    # the full universe. Measure the same slice or the comparison is not one.
    adv_r = adv.rank(axis=1, pct=True)
    liquid = adv_r > 0.67
    sig = {k: v.where(liquid) for k, v in sig.items()}
    print("  universe: liquid top third by 20-day average traded value")

    oos = idx >= SPLIT
    print(f"  out-of-sample {idx[oos][0].date()} -> {idx[oos][-1].date()}  "
          f"({int(oos.sum()):,} days)\n")

    profile_series = None
    print("=" * 86)
    print(f"  {'signal':<10}{'h':>4}{'mean IC':>11}{'t naive':>10}{'t NW':>9}"
          f"{'AC(1)':>8}{'days':>7}{'n_eff':>8}   read as")
    print("  " + "-" * 82)

    # Rank the forward returns once per horizon, not once per signal.
    for h in HORIZONS:
        fwd_rank = features.cross_sectional_rank(features.forward_returns(close, h))
        for name, sig_panel in sig.items():
            r = features.cross_sectional_rank(sig_panel).shift(1).loc[oos]
            f = fwd_rank.reindex(index=r.index, columns=r.columns)
            ic = r.corrwith(f, axis=1).dropna()
            if len(ic) < 60:
                print(f"  {name:<10}{h:>4}   too few days ({len(ic)})")
                continue
            x = [float(v) for v in ic.to_numpy()]
            n = len(x)
            mean = sum(x) / n
            sn = naive_se(x)
            sw = nw_se(x, h - 1)
            a1 = autocorr(x, 1)
            tn = mean / sn if sn else None
            tw = mean / sw if sw else None
            neff = (n * (sn / sw) ** 2) if (sw and sn) else None

            if tw is None:
                read = "SE undefined"
            elif abs(tw) >= 3.0:
                read = "holds up"
            elif abs(tw) >= 1.96:
                read = "marginal"
            elif tn is not None and abs(tn) >= 1.96:
                read = "WAS significant, is NOT"
            else:
                read = "nothing either way"

            g = lambda v, nd=2: "   n/a" if v is None else f"{v:+.{nd}f}"
            print(f"  {name:<10}{h:>4}{mean:>+11.5f}{g(tn):>10}{g(tw):>9}"
                  f"{g(a1, 3):>8}{n:>7}"
                  f"{('   n/a' if neff is None else f'{neff:>8.0f}')}   {read}")

            EMIT.setdefault("signals", []).append(
                {"signal": name, "horizon": h, "mean_ic": mean,
                 "t_naive": tn, "t_nw": tw, "ac1": a1,
                 "days": n, "n_eff": neff, "reading": read})
            if (name, h) == AC_PROFILE_FOR:
                profile_series = x
        print("  " + "-" * 82)
        del fwd_rank

    if profile_series is not None:
        name, h = AC_PROFILE_FOR
        print(f"\n  Autocorrelation profile of the daily IC series, {name} at h={h}.")
        print(f"  If the {h}-day overlap and a near-static signal drive it, lag 1")
        print(f"  starts near {(h-1)/h:+.2f} and decays to zero around lag {h}.\n")
        print(f"  {'lag':>4}{'AC':>9}   {'overlap prediction':>19}   profile")
        print("  " + "-" * 62)
        for lag in [1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 80, 120]:
            a = autocorr(profile_series, lag)
            if a is None:
                continue
            pred = max(0.0, (h - lag) / h)
            bar = "#" * int(round(abs(a) * 40))
            sign = "" if a >= 0 else "-"
            print(f"  {lag:>4}{a:>+9.3f}   {pred:>19.2f}   {sign}{bar}")
        print("  " + "-" * 62)

    print("\n  n_eff is how many independent observations the daily IC series is")
    print("  actually worth. The project notes guessed ~27 out of ~1,600 for the")
    print("  60-day level signal; this is the measured value.")
    print("  Judge every level signal on t NW. t naive is printed only so the")
    print("  gap against 09_level_signals.py is visible.")

    if profile_series is not None:
        nm, hh = AC_PROFILE_FOR
        EMIT["ac_profile"] = {"signal": nm, "horizon": hh, "lags": {
            str(l): autocorr(profile_series, l)
            for l in [1, 2, 3, 5, 10, 20, 30, 40, 50, 60]}}
    dest = storage.config.DATA_DIR.parent / "results" / "level_se.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(EMIT, indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote {dest.relative_to(storage.config.DATA_DIR.parent)}")


if __name__ == "__main__":
    main()
