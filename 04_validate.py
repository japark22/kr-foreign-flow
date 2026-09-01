#!/usr/bin/env python3
"""Step 4: test the hypothesis.

    python 04_validate.py                      # persistence only
    python 04_validate.py --horizons 1,5,20    # + IC and long-short, needs market data

Answers, in order, the questions this project set out to test:

  1. Persistence   — does today's foreign flow predict tomorrow's flow?
  2. Predictive    — does it predict future *returns*? at which horizon?
  3. Denoised      — does the signal survive removing rebalance-driven flow?
  4. Incremental   — is it distinct from price momentum?
  5. (tradability is left for later; it needs cost assumptions)

Runs offline against what 01_backfill.py collected. Questions 2, 4 and 5 need
the market store, so run `01_backfill.py --with-market` first for those.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from krxflow import features, rebalance, storage

W = 78

# Machine-readable results. Every number the report page shows is written here
# by the measurement that produced it, so the page can never drift from the
# code: there is no step where a figure is copied by hand.
RESULTS = {"schema": 1}


def rule(title: str = "") -> None:
    print()
    print("=" * W)
    if title:
        print(f"  {title}")
        print("=" * W)


def bar(value: float, scale: float = 0.2, half: int = 12) -> str:
    n = max(-half, min(half, int(round(value / scale * half))))
    if n >= 0:
        return " " * half + "|" + "#" * n
    return " " * (half + n) + "#" * (-n) + "|"


def load_market_panels(start=None, end=None):
    df = storage.read_range("market", start, end)
    if df.empty:
        return None
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    def pivot(col):
        if col not in df.columns:
            return None
        return df.pivot_table(index="trade_date", columns="ticker", values=col,
                              aggfunc="last", observed=True).sort_index().astype("float64")

    return {"close": pivot("close"), "value_traded": pivot("value_traded")}


def autocorrelation(ranked: pd.DataFrame, max_lag: int, label: str) -> pd.Series:
    """Mean cross-sectional rank autocorrelation, with a t-stat per lag.

    Each lag gives one correlation per trading day. The t-stat is over those
    daily values, so it answers "is this reliably positive across days" rather
    than "is it big" — with 4,000 cross-sections, a small mean can still be
    overwhelming evidence, and it is worth seeing which.
    """
    means, tstats = {}, {}
    for lag in range(1, max_lag + 1):
        daily = ranked.corrwith(ranked.shift(-lag), axis=1).dropna()
        means[lag] = daily.mean()
        tstats[lag] = (daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily)))
                       if len(daily) > 2 and daily.std(ddof=1) > 0 else float("nan"))

    s = pd.Series(means, name=label)
    RESULTS.setdefault("autocorrelation", {})[label] = {
        "mean": {int(k): float(v) for k, v in s.items()},
        "t": {int(k): float(tstats[k]) for k in tstats},
    }

    print(f"\n  {label}")
    print("  lag    corr     t-stat    -0.2         0        +0.2")
    print("  ---   -------   -------   |-----------|-----------|")
    for lag, v in s.items():
        print(f"  {lag:>3}   {v:+.4f}   {tstats[lag]:>+7.1f}   {bar(v)}")
    return s


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default=None, help="YYYYMMDD")
    p.add_argument("--end", default=None, help="YYYYMMDD")
    p.add_argument("--max-lag", type=int, default=20)
    p.add_argument("--horizons", default="1,5,20",
                   help="forward-return horizons in trading days")
    p.add_argument("--quantiles", type=int, default=5)
    p.add_argument("--rebalance-window", type=int, default=2)
    p.add_argument("--keep-exdiv", action="store_true",
                   help="do NOT remove ex-dividend windows (to see their effect)")
    args = p.parse_args()

    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]

    rule("Loading")
    panels = features.load_panels(args.start, args.end)
    pct = panels["foreign_pct"]
    print(f"  dates   : {pct.index[0].date()} -> {pct.index[-1].date()}  ({len(pct):,} days)")
    print(f"  tickers : {pct.shape[1]:,} seen at least once")

    universe = features.universe_mask(panels)
    dropped = pct.shape[1] - int(universe.any().sum())
    print(f"  universe: {int(universe.any().sum()):,} usable "
          f"({dropped:,} dropped: foreign listings, pinned, or too thin)")

    ca = features.corporate_action_mask(panels)
    print(f"  corp actions masked: {int(ca.to_numpy().sum()):,} ticker-days")

    # ------------------------------------------------------------------ 1 --
    rule("1. Persistence — does foreign flow autocorrelate?")
    print("  Signal: Δ(foreign shares held) / shares outstanding, ranked")
    print("  cross-sectionally each day. If the hypothesis holds this decays")
    print("  over several days rather than dying at lag 1.")

    flow_raw = features.build_flow(panels, denominator="shares")
    flow_raw = flow_raw.where(universe)
    ranked_raw = features.cross_sectional_rank(flow_raw)
    ac_raw = autocorrelation(ranked_raw, args.max_lag, "RAW (rebalance dates still in)")

    # ----------------------------------------------------------------- 1b --
    rule("1b. Is the persistence real, or a tie artifact?")
    print("  A stock with no foreign trading has Δ = exactly 0. Every such name")
    print("  receives the same mid-rank, so a name that is inactive for weeks")
    print("  holds an identical rank for weeks. That alone would produce a")
    print("  decaying autocorrelation curve with no accumulation behind it —")
    print("  persistent INACTIVITY masquerading as persistent buying.")
    print()
    print("  The test: drop every exactly-zero observation and re-measure. If")
    print("  the curve survives among actively-traded names, the persistence is")
    print("  real. If it collapses, the headline number was an artifact.\n")

    valid = int(flow_raw.notna().to_numpy().sum())
    zeros = int((flow_raw == 0).to_numpy().sum())
    print(f"  observations         : {valid:,}")
    print(f"  exactly zero         : {zeros:,} ({100 * zeros / max(valid, 1):.1f}%)")
    print(f"  active (non-zero)    : {valid - zeros:,}")

    active = flow_raw.mask(flow_raw == 0)
    ranked_active = features.cross_sectional_rank(active)
    ac_active = autocorrelation(ranked_active, args.max_lag,
                                "ACTIVE ONLY (zero-flow observations dropped)")

    ratio = ac_active.loc[1] / ac_raw.loc[1] if ac_raw.loc[1] else float("nan")
    print(f"\n  lag-1: {ac_raw.loc[1]:+.4f} all -> {ac_active.loc[1]:+.4f} active "
          f"({ratio:.0%} retained)")
    if ac_active.loc[1] > 0.5 * ac_raw.loc[1]:
        print("  The signal survives on active names. The persistence is not")
        print("  simply an artifact of inactive stocks holding a tied rank.")
    else:
        print("  Most of the headline persistence came from zero-flow ties.")
        print("  Treat the RAW curve above as unreliable and use this one.")

    # ------------------------------------------------------------------ 3 --
    rule("3. Denoised — rebalance and ex-dividend windows removed")
    flow_clean, filt = features.apply_filters(
        flow_raw, panels,
        drop_rebalance=True, rebalance_window=args.rebalance_window,
        drop_exdiv=not args.keep_exdiv,
        universe=universe)

    total = filt["observations_in"]
    print(f"  observations in     : {total:,}")
    print(f"  removed, universe   : {filt['removed_universe']:,}")
    print(f"  removed, rebalance  : {filt['removed_rebalance']:,}")
    print(f"  removed, ex-dividend: {filt['removed_exdiv']:,}")
    print(f"  observations out    : {filt['observations_out']:,} "
          f"({100 * filt['observations_out'] / max(total, 1):.1f}% kept)")
    print(f"\n  {filt['exdiv_basis']}")
    print("  NOTE: rebalance dates are rule-derived, not official. See")
    print("        krxflow/rebalance.py and krxflow/exdiv.py before")
    print("        treating any of this as final.")

    ranked_clean = features.cross_sectional_rank(flow_clean)
    ac_clean = autocorrelation(ranked_clean, args.max_lag, "DENOISED")

    delta = ac_clean - ac_raw
    print(f"\n  mean change over lags 1-5: {delta.loc[1:5].mean():+.4f}")

    # ----------------------------------------------------------------- 3b --
    rule("3b. Are the flagged dates actually the noisy ones?")
    print("  If removing the filter windows barely moves the curve, there are")
    print("  two very different explanations and they need separating:")
    print()
    print("    (a) our derived dates are wrong, so we excluded ordinary days")
    print("        and left the real rebalance flow in the sample; or")
    print("    (b) the dates are right, and mechanical flow simply does not")
    print("        distort a cross-sectional RANK correlation much — ranks are")
    print("        robust to a few enormous trades by construction.")
    print()
    print("  Test: if the calendar is right, aggregate absolute flow should be")
    print("  visibly elevated on those days. If it is flat, the dates are wrong.\n")

    daily_abs = flow_raw.abs().mean(axis=1)
    baseline = daily_abs.median()

    eff_dates = rebalance.rebalance_dates(flow_raw.index)["effective_date"]
    pos = {d.date(): i for i, d in enumerate(flow_raw.index)}

    print(f"  {len(eff_dates)} rebalance effective dates in range, "
          f"baseline daily mean |flow| = {baseline:.3e}")
    print("\n  day vs effective   rel. to baseline")
    print("  ----------------   ----------------")
    for k in range(-5, 6):
        vals = [daily_abs.iloc[pos[e] + k] for e in eff_dates
                if e in pos and 0 <= pos[e] + k < len(daily_abs)]
        if not vals:
            continue
        rel = float(np.mean(vals)) / baseline
        RESULTS.setdefault("rebalance_profile", {})[str(k)] = rel
        RESULTS["rebalance_baseline"] = float(baseline)
        RESULTS["rebalance_dates"] = int(len(eff_dates))
        marker = "  <- effective date" if k == 0 else ""
        print(f"  {k:>+16}   {rel:>15.2f}x  {'#' * max(0, int((rel - 1) * 40))}{marker}")

    print("\n  A clear hump at 0 means the calendar is finding real events.")
    print("  A flat line near 1.00x means it is not, and explanation (a) holds:")
    print("  pin authoritative MSCI/FTSE dates in data/rebalance_overrides.csv")
    print("  before claiming the signal survives denoising.")

    # ------------------------------------------------------------------ 5 --
    inv = features.load_investor_panels(args.start, args.end)
    if inv:
        rule("5. Is the persistence specific to FOREIGN investors?")
        print("  This is the test that decides whether the project has an edge.")
        print()
        print("  Two hypotheses fit the curve above equally well:")
        print("    H1  foreign flow is persistent  -> Korea's daily foreign")
        print("        ownership disclosure is a genuine data edge")
        print("    H2  ALL large institutional flow is persistent, because big")
        print("        orders get split over days -> generic microstructure,")
        print("        no edge, and domestic institutional flow is disclosed")
        print("        in plenty of markets")
        print()
        print("  If 기관합계 traces the same curve as 외국인, H2 holds and")
        print("  \"foreign\" is not the special ingredient.\n")

        counts = inv.get("_counts")
        if counts is not None:
            print("  Tickers per day (truncation check):")
            for col in counts.columns:
                s = counts[col].dropna()
                print(f"    {col:<10} min {s.min():>6,.0f}  median {s.median():>6,.0f}  "
                      f"max {s.max():>6,.0f}")
            print("  Round numbers like 100 or 500 would mean KRX truncated the")
            print("  response to top-N, which makes it useless as a control.\n")

        curves = {}
        for investor in [i for i in inv if not i.startswith("_")]:
            panel = inv[investor]
            # Normalise by traded value so the measure is comparable across
            # names, then rank cross-sectionally as for the ownership signal.
            ranked_inv = features.cross_sectional_rank(panel)
            curves[investor] = autocorrelation(
                ranked_inv, args.max_lag, f"{investor} net-buying flow")

        if "외국인" in curves and "기관합계" in curves:
            f, k = curves["외국인"], curves["기관합계"]
            print("\n  lag   foreign   institution   ratio f/k")
            print("  ---   -------   -----------   ---------")
            shown = sorted(set(range(1, min(6, args.max_lag + 1))) | {args.max_lag})
            for lag in shown:
                if lag not in f.index or lag not in k.index:
                    continue
                ratio = f.loc[lag] / k.loc[lag] if k.loc[lag] else float("nan")
                print(f"  {lag:>3}   {f.loc[lag]:+.4f}   {k.loc[lag]:>+11.4f}   "
                      f"{ratio:>9.2f}")

            # The earlier version tested only whether the two curves DIFFER and
            # then announced that foreign flow was the more persistent one. It
            # never looked at which way the difference went. On the real data
            # institutional flow is roughly twice as persistent as foreign flow
            # at every lag, and that branch printed the opposite of the finding.
            gap = f.loc[1] - k.loc[1]
            near = abs(gap) < 0.25 * max(abs(f.loc[1]), 1e-9)
            print()
            if near:
                print("  The two are close. H2 is the better-supported reading:")
                print("  the persistence looks like large-order execution in")
                print("  general, not something foreign investors do uniquely.")
                print("  That does not kill the feature, but it does mean the")
                print("  pitch cannot be \"foreign flow is special\".")
            elif gap > 0:
                print("  Foreign flow is materially MORE persistent than domestic")
                print("  institutional flow. That is what H1 predicts and what")
                print("  would make Korea's disclosure a real edge.")
            else:
                print("  Domestic institutional flow is materially MORE persistent")
                print(f"  than foreign flow (ratio f/k = {f.loc[1]/k.loc[1]:.2f} at lag 1).")
                print("  H1 is refuted, and more strongly than H2 would have been:")
                print("  the premise was that foreign investors are slower to digest")
                print("  information and split larger orders. Domestic institutions")
                print("  do more of it. Persistence is a property of institutional")
                print("  execution, not of foreign investors, so the daily foreign")
                print("  disclosure has no special claim on it.")

        # Cross-check: our stock-based signal against the reported flow.
        if "외국인" in inv:
            rule("5b. Stock vs flow cross-check")
            print("  Our signal is a STOCK measure: Δ(shares held). KRX also")
            print("  reports foreign net buying directly, a FLOW measure. They")
            print("  should broadly agree. Where they do not, something is off —")
            print("  Korean trader-type data has a known misclassification issue")
            print("  for orders routed through domestic brokers.\n")
            fl = inv["외국인"]
            common_d = flow_raw.index.intersection(fl.index)
            common_t = flow_raw.columns.intersection(fl.columns)
            if len(common_d) > 20 and len(common_t) > 50:
                a = features.cross_sectional_rank(flow_raw.loc[common_d, common_t])
                b = features.cross_sectional_rank(fl.loc[common_d, common_t])
                daily = a.corrwith(b, axis=1).dropna()
                print(f"  overlapping days    : {len(common_d):,}")
                print(f"  overlapping tickers : {len(common_t):,}")
                RESULTS["stock_vs_flow"] = {
                    "mean_rank_corr": float(daily.mean()),
                    "p10": float(daily.quantile(0.10)),
                    "worst": float(daily.min()),
                    "days": int(len(common_d)), "tickers": int(len(common_t))}
                print(f"  mean daily rank corr: {daily.mean():+.4f}")
                print(f"  10th percentile     : {daily.quantile(0.10):+.4f}")
                print(f"  worst day           : {daily.min():+.4f}")
                if daily.mean() > 0.5:
                    print("\n  Strong agreement. The two independent measurements of")
                    print("  foreign activity corroborate each other.")
                else:
                    print("\n  Weak agreement. Investigate before trusting either:")
                    print("  settlement-date vs trade-date convention, lending")
                    print("  flows, or the broker misclassification issue.")
            else:
                print("  Not enough overlap yet — collect more investor-flow days.")
    else:
        rule("5. Foreign vs institutional control — SKIPPED")
        print("  No investor-flow data stored. This is the test that separates")
        print("  \"foreign flow is special\" from \"large orders get split\", so")
        print("  it matters more than any remaining refinement:")
        print("\n    caffeinate -i nohup python 01_backfill.py --start 2010-01-01 \\")
        print("      --with-investor --with-shorting > data/logs/extras.log 2>&1 &\n")

    # ------------------------------------------------------------- 2 & 4 --
    market = load_market_panels(args.start, args.end)
    if market is None or market.get("close") is None:
        rule("2 & 4. Predictive power — SKIPPED")
        print("  No market data stored. To answer whether the signal predicts")
        print("  returns, collect prices first:")
        print("\n    caffeinate -i python 01_backfill.py --start 2010-01-01 --with-market\n")
        rule("Summary")
        print("  Persistence measured. Return predictability still unknown.")
        return 0

    close = market["close"].reindex(index=flow_clean.index, columns=flow_clean.columns)

    rule("2. Predictive power — information coefficient by horizon")
    print("  Spearman IC = cross-sectional rank correlation between the flow")
    print("  signal on day t and the forward return over the next N days.")
    print("  Signal is lagged one day (D-1) so nothing looks ahead.\n")
    print("  horizon    mean IC     std     IC t-stat   IR (ann.)")
    print("  -------   --------   ------   ---------   ---------")

    signal = ranked_clean.shift(1)  # point-in-time: trade on the next day
    ic_table = {}
    for h in horizons:
        fwd = features.forward_returns(close, h)
        fwd_rank = features.cross_sectional_rank(fwd)
        ic = signal.corrwith(fwd_rank, axis=1).dropna()
        if len(ic) < 20:
            print(f"  {h:>5}d    (too few overlapping days)")
            continue
        t = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))
        ir = ic.mean() / ic.std(ddof=1) * np.sqrt(252 / h)
        ic_table[h] = ic
        RESULTS.setdefault("horizon_ic", {})[str(h)] = {
            "mean_ic": float(ic.mean()), "std": float(ic.std(ddof=1)),
            "t": float(t), "ir": float(ir), "days": int(len(ic))}
        print(f"  {h:>5}d   {ic.mean():+.5f}   {ic.std(ddof=1):.4f}   "
              f"{t:>+8.2f}   {ir:>+8.2f}")

    print("\n  Rule of thumb: |mean IC| above ~0.01 with a t-stat above 3 is a")
    print("  real effect in a cross-section this size.")

    rule(f"2b. Quantile long-short ({args.quantiles} buckets)")
    print("  Buy the top bucket of flow, sell the bottom, equal weight,")
    print("  rebalanced every `horizon` days. GROSS of costs.\n")
    print("  horizon   long-short/day   ann. return   ann. vol   Sharpe   hit rate")
    print("  -------   --------------   -----------   --------   ------   --------")

    gross = {}
    turnover = {}
    for h in horizons:
        fwd = features.forward_returns(close, h)
        q = signal.rank(axis=1, pct=True)
        long_leg = q >= 1 - 1 / args.quantiles
        short_leg = q <= 1 / args.quantiles
        top = fwd.where(long_leg).mean(axis=1)
        bot = fwd.where(short_leg).mean(axis=1)
        ls = (top - bot).dropna()
        if len(ls) < 20:
            print(f"  {h:>5}d    (too few overlapping days)")
            continue
        per_day = ls.mean() / h
        ann = per_day * 252
        vol = ls.std(ddof=1) / np.sqrt(h) * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else float("nan")
        gross[h] = ann
        print(f"  {h:>5}d   {per_day:>+13.5%}   {ann:>+10.2%}   {vol:>7.2%}   "
              f"{sharpe:>+6.2f}   {(ls > 0).mean():>7.1%}")

        # One-way turnover per rebalance: the share of each leg that changes.
        # Measured on the actual holdings, not assumed.
        held = long_leg.astype(float)
        churn = (held.diff().abs().sum(axis=1) / 2) / held.sum(axis=1).replace(0, np.nan)
        turnover[h] = churn.dropna().mean()

    # ------------------------------------------------------------------ 5 --
    rule("2c. Does it survive costs? (question 5)")
    print("  Korean securities transaction tax from 2026: 0.20% on every sale")
    print("  (KOSPI 0.05% + 농특세 0.15%; KOSDAQ 0.20%). Spread and impact come")
    print("  on top, and most of a 2,700-name universe is not liquid.")
    print()
    print("  Rebalancing at horizon h means roughly 252/h rebalances a year, and")
    print("  each one turns over the measured fraction of both legs below.\n")
    print("  horizon   turnover/rebal   annual turnover   gross   cost @30bp   net")
    print("  -------   --------------   ---------------   -----   ----------   ------")

    COST_ONE_WAY = 0.0030  # 20bp tax + ~10bp spread/impact, deliberately optimistic
    for h in horizons:
        if h not in gross or h not in turnover:
            continue
        rebals = 252 / h
        # both legs, one-way each side of the switch
        ann_turn = turnover[h] * rebals * 2
        cost = ann_turn * COST_ONE_WAY
        RESULTS.setdefault("costs", {})[str(h)] = {
            "turnover_per_rebalance": float(turnover[h]),
            "annual_turnover": float(ann_turn), "gross": float(gross[h]),
            "cost": float(cost), "net": float(gross[h] - cost),
            "cost_one_way": COST_ONE_WAY}
        print(f"  {h:>5}d   {turnover[h]:>13.1%}   {ann_turn:>14.1f}x   "
              f"{gross[h]:>+5.2%}   {cost:>9.2%}   {gross[h] - cost:>+6.2%}")

    print()
    print("  30bp one-way is generous for Korean small caps. If the net column")
    print("  is negative, the effect is not tradable in this construction — the")
    print("  question then is whether a slower or better-normalised construction")
    print("  survives, not whether this one can be tuned.")

    rule("4. Incremental — is this just momentum in disguise?")
    print("  Correlation between the flow signal and past-return momentum,")
    print("  measured in cross-sectional ranks. Low correlation means the")
    print("  signal carries information the momentum factor does not.\n")
    print("  momentum lookback   corr with flow signal")
    print("  -----------------   ---------------------")
    for lb in (5, 20, 60, 120):
        mom = features.cross_sectional_rank(close.pct_change(lb, fill_method=None))
        c = signal.corrwith(mom, axis=1).mean()
        RESULTS.setdefault("momentum_corr", {})[str(lb)] = float(c)
        print(f"  {lb:>15}d   {c:>+20.4f}")

    print("\n  Above ~0.3 in absolute terms would mean substantial overlap and")
    print("  the incremental-alpha test becomes the deciding one.")

    rule("Summary")
    if ic_table:
        best = max(ic_table, key=lambda h: abs(ic_table[h].mean()))
        ic = ic_table[best]
        t = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))
        print(f"  Strongest horizon: {best}d, mean IC {ic.mean():+.5f} (t={t:+.2f})")
    print(f"  Lag-1 flow autocorrelation, denoised: {ac_clean.loc[1]:+.4f}")
    print("\n  Caveats standing: rebalance dates are rule-derived; ex-dividend")
    print("  effects are NOT yet removed (needs OpenDART); free float is")
    print("  approximated by shares outstanding.")

    RESULTS["window"] = [str(pct.index[0].date()), str(pct.index[-1].date())]
    RESULTS["days"] = int(len(pct))
    RESULTS["tickers"] = int(universe.any().sum())
    RESULTS["observations"] = int(flow_raw.notna().to_numpy().sum())
    dest = storage.config.DATA_DIR.parent / "results" / "validate.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(RESULTS, indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote {dest.relative_to(storage.config.DATA_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
