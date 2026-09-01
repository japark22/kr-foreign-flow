"""Step 12: test the signal the way the brief actually framed it.

WHAT WAS TESTED BEFORE, AND WHY IT WAS THE WRONG BAR
---------------------------------------------------
Every step up to now asked whether foreign flow is a standalone edge: does it
beat costs on its own, does it survive a sweep on its own, does it work in the
liquid third on its own. The answer was no, repeatedly.

But the brief never asked for a standalone signal. It asked whether this is a
useful INPUT ALONGSIDE OTHER FEATURES. That is a different and much lower bar,
and it changes two things:

  THE THRESHOLD. An IC of 0.004 is useless alone. Combined, what matters is
  whether it carries information the other features do not already have. A
  small but ORTHOGONAL signal is a legitimate input; a large but redundant one
  is not.

  THE COST ARITHMETIC. The 2.4bp breakeven assumed the signal pays for its own
  turnover. As one input among several, what it must pay for is the MARGINAL
  turnover it adds to a book that was going to trade anyway. That is a smaller
  number, and it was never computed.

WHAT THIS MEASURES
------------------
1. LIQUIDITY SPLIT AT EVERY HORIZON. The existing split is at 20 days, where
   the effect is a reversal. The effect that is actually positive lives at
   1 day, and whether it survives among liquid names was never checked. That
   single number decides whether there is anything to combine.

2. ORTHOGONALITY. The signal is residualised each day against size, liquidity,
   short-term reversal, two momentum windows and volatility, by cross-sectional
   regression. If the IC survives residualisation, the information is its own.
   If it collapses, the feature is a repackaging of factors already owned.

3. MARGINAL CONTRIBUTION. A composite of those same standard factors stands in
   for "our other features" -- crude, but it is what can be built from public
   data. The signal is blended in at several weights and we ask whether IC
   improves by more than the marginal turnover costs.

STANDARD ERRORS
---------------
Every t is Newey-West with h-1 lags, not std/sqrt(n). 09b established why:
whether the naive standard error is right depends on how persistent the SIGNAL
is, and that differs between the flow and level families. Newey-West estimates
the autocovariance rather than assuming it, and was the only estimator that
held up under both a moving-average and a white-noise null.

NOTE ON THE NOISE FILTER
------------------------
The brief asked for index-rebalance and ex-dividend windows to be removed. Our
own test falsified the rule-derived rebalance calendar, so that filter is NOT
applied here -- applying a calendar we have shown to be wrong would discard 18%
of observations for nothing. This measures the unfiltered signal, and that is a
stated limitation rather than an oversight.

    python 12_combination.py --selftest    # planted answer, no data needed
    python 12_combination.py               # real data -> results/combination.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
import pandas as pd

HORIZONS = [1, 5, 20]
BLEND_WEIGHTS = [0.10, 0.25, 0.50]
COST_BPS_ONE_WAY = 20.0        # Korean securities transaction tax on sales, 2026
IC_FLOOR = 0.005               # below this a combined input is not worth wiring in
BAD_DAY = 0.50                 # a one-day move this large is a capital change
MAX_PLAUSIBLE_ANNUAL = 1.0     # a long-short book past this is a bug, not a result


# ---------------------------------------------------------------- statistics
def nw_t(x, lags):
    """Newey-West t of the mean. Returns (mean, t, se)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 30:
        return (float("nan"),) * 3
    m = x.mean()
    e = x - m
    s = float((e * e).sum() / n)
    lags = max(0, min(int(lags), n - 2))
    for j in range(1, lags + 1):
        s += 2.0 * (1 - j / (lags + 1.0)) * float((e[j:] * e[:-j]).sum() / n)
    if s <= 0:
        return m, float("nan"), float("nan")
    se = math.sqrt(s / n)
    return m, (m / se if se else float("nan")), se


def xrank(df):
    """Cross-sectional rank in [0,1], NaN preserved."""
    return df.rank(axis=1, pct=True)


def daily_ic(sig_rank, fwd_rank):
    """Per-day cross-sectional correlation of two rank panels."""
    return sig_rank.corrwith(fwd_rank, axis=1).dropna()


def residualise(sig, controls):
    """Cross-sectional OLS residual of sig on controls, one regression per day.

    All inputs are rank panels on the same index/columns. Rows with fewer than
    50 usable names are returned as NaN rather than fitted on a handful.
    """
    S = sig.to_numpy(dtype=float)
    # Hold the controls as a list of 2-D float32 arrays rather than one T x N x K
    # stack: the stack would allocate a second full copy of every control panel,
    # which on a sixteen-year panel is hundreds of megabytes for nothing.
    C = [c.to_numpy(dtype=np.float32) for c in controls]
    out = np.full_like(S, np.nan)
    T = S.shape[0]
    for t in range(T):
        y = S[t]
        X = np.column_stack([c[t] for c in C]).astype(float)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if ok.sum() < 50:
            continue
        yv = y[ok] - y[ok].mean()
        Xv = X[ok] - X[ok].mean(axis=0)
        try:
            beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
        except np.linalg.LinAlgError:
            continue
        out[t, ok] = yv - Xv @ beta
    return pd.DataFrame(out, index=sig.index, columns=sig.columns)


def leg_turnover(rank_panel, h, top=0.2):
    """Annualised one-way turnover of a long-short book rebalanced every h days.

    Fraction of each leg replaced at a rebalance, averaged over rebalances,
    times the number of rebalances in a year. Both legs counted.
    """
    r = rank_panel
    idx = list(range(0, len(r), h))
    if len(idx) < 3:
        return float("nan")
    churn = []
    prev_L = prev_S = None
    for i in idx:
        row = r.iloc[i]
        row = row[np.isfinite(row)]
        if row.size < 50:
            prev_L = prev_S = None
            continue
        L = set(row[row >= row.quantile(1 - top)].index)
        S = set(row[row <= row.quantile(top)].index)
        if prev_L is not None and L and S:
            churn.append(0.5 * (len(L - prev_L) / len(L) + len(S - prev_S) / len(S)))
        prev_L, prev_S = L, S
    if not churn:
        return float("nan")
    return float(np.mean(churn) * (252.0 / h) * 2.0)   # x2: both legs


def clean_forward(close, h):
    """Forward return with capital-change windows removed.

    The price panel is not split-adjusted. A 1:10 split reads as a -90% return,
    and a mean over 3,600 tickers and 4,000 days is dominated by a few hundred
    such events -- which is how an earlier version of this script reported a
    long-short book losing 1,160% a year. Rank correlations are immune because a
    jump moves one rank; means are not. Any forward window containing a day whose
    single-day move exceeds BAD_DAY is dropped rather than trusted.
    """
    r1 = close.pct_change(fill_method=None)
    bad = (r1.abs() > BAD_DAY).fillna(False)
    contaminated = bad.rolling(h, min_periods=1).max().shift(-h).fillna(1).astype(bool)
    return (close.shift(-h) / close - 1.0).mask(contaminated)


def ls_return(rank_panel, fwd, h, top=0.2, winsor=0.01):
    """Per-rebalance long-short return series, gross.

    Winsorised cross-sectionally each rebalance: even after capital-change
    windows are dropped, a handful of names per day carry returns large enough
    to set the mean on their own."""
    out = []
    for i in range(0, len(rank_panel) - h, h):
        row = rank_panel.iloc[i]
        f = fwd.iloc[i]
        ok = np.isfinite(row) & np.isfinite(f)
        if ok.sum() < 50:
            continue
        row, f = row[ok], f[ok]
        f = f.clip(f.quantile(winsor), f.quantile(1 - winsor))
        hi = row >= row.quantile(1 - top)
        lo = row <= row.quantile(top)
        if hi.sum() < 5 or lo.sum() < 5:
            continue
        out.append(float(f[hi].mean() - f[lo].mean()))
    return np.array(out)


# ---------------------------------------------------------------- self-test
def selftest():
    """Plant a small signal that is orthogonal to the controls by construction,
    plus a redundant one that is a pure function of a control. The measurement
    must keep the first through residualisation and destroy the second."""
    rng = np.random.default_rng(20260824)
    T, N, H = 900, 500, 5
    size = rng.normal(size=N)
    size_p = np.tile(size, (T, 1)) + rng.normal(scale=.05, size=(T, N))
    mom = rng.normal(size=(T, N))
    own = rng.normal(size=(T, N))                       # the orthogonal signal
    redundant = size_p + rng.normal(scale=.15, size=(T, N))

    ALPHA = 0.35
    idio = rng.normal(scale=1.0, size=(T + H, N))
    fwd = np.array([idio[t:t + H].sum(axis=0) for t in range(T)])
    fwd += ALPHA * own + 0.6 * size_p                   # size also pays, as in life

    def P(a):
        return pd.DataFrame(a, index=pd.RangeIndex(T), columns=[f"T{i}" for i in range(N)])

    ctrl = [xrank(P(size_p)), xrank(P(mom))]
    fr = xrank(P(fwd))
    print("\nPlanted: `own` carries alpha and is orthogonal to the controls.")
    print("         `redundant` is size in disguise and carries none of its own.\n")
    print(f"  {'signal':<12}{'raw IC':>10}{'t':>8}{'resid IC':>11}{'t':>8}{'kept':>8}")
    print("  " + "-" * 53)
    for nm, arr in (("own", own), ("redundant", redundant)):
        r = xrank(P(arr))
        a, ta, _ = nw_t(daily_ic(r, fr).to_numpy(), H - 1)
        res = residualise(r, ctrl)
        b, tb, _ = nw_t(daily_ic(res.rank(axis=1, pct=True), fr).to_numpy(), H - 1)
        kept = b / a if a else float("nan")
        print(f"  {nm:<12}{a:>+10.4f}{ta:>+8.2f}{b:>+11.4f}{tb:>+8.2f}{kept:>7.0%}")
    print("\n  `own` should keep most of its IC; `redundant` should lose nearly all")
    print("  of it. If that is not what you see, the residualisation is broken and")
    print("  no number from the real run means anything.\n")


# ---------------------------------------------------------------- real data
def run(args):
    from krxflow import features, storage

    print("loading ...")
    p = features.load_panels(args.start, args.end)
    idx, cols = p["foreign_shares"].index, p["foreign_shares"].columns
    m = storage.read_range("market", args.start, args.end,
                           columns=["trade_date", "ticker", "close",
                                    "value_traded", "market_cap"])
    if m.empty:
        sys.exit("no market store -- run 01_backfill.py --with-market first")
    m["trade_date"] = pd.to_datetime(m["trade_date"])

    def pv(col):
        return (m.pivot_table(index="trade_date", columns="ticker", values=col,
                              aggfunc="last", observed=True)
                 .sort_index().astype("float64").reindex(index=idx, columns=cols))

    close, vt, cap = pv("close"), pv("value_traded"), pv("market_cap")
    del m
    adv = vt.rolling(20, min_periods=5).mean()

    uni = features.universe_mask(p)
    flow = features.build_flow(p, denominator="shares").where(uni)
    sig = xrank(flow)

    # Controls: what a factor book would already own.
    ret1 = close.pct_change(fill_method=None)
    ctrl = {
        "size": xrank(np.log(cap.where(cap > 0))),
        "liquidity": xrank(np.log(adv.where(adv > 0))),
        "reversal_5d": xrank(close / close.shift(5) - 1.0),
        "momentum_60d": xrank(close.shift(5) / close.shift(65) - 1.0),
        "momentum_120d": xrank(close.shift(5) / close.shift(125) - 1.0),
        "volatility_20d": xrank(ret1.rolling(20, min_periods=10).std()),
    }
    ctrl = {k: v.where(uni) for k, v in ctrl.items()}
    composite = xrank(sum(v.fillna(0.5) for v in ctrl.values()) / len(ctrl)).where(uni)

    print(f"  {idx[0].date()} -> {idx[-1].date()}  {len(idx):,} days, "
          f"{int(uni.any().sum()):,} tickers")
    print("  residualising against " + ", ".join(ctrl) + " ...")
    resid = residualise(sig, list(ctrl.values()))
    resid_r = xrank(resid)

    adv_r = xrank(adv)
    terciles = {"thin": adv_r <= 1/3, "middle": (adv_r > 1/3) & (adv_r <= 2/3),
                "liquid": adv_r > 2/3}

    out = {"window": [str(idx[0].date()), str(idx[-1].date())],
           "days": int(len(idx)), "tickers": int(uni.any().sum()),
           "controls": list(ctrl),
           "cost_bps_one_way": COST_BPS_ONE_WAY,
           "note": "rebalance/ex-dividend filter NOT applied: the rule-derived "
                   "calendar was falsified by 04_validate.py section 3b",
           "liquidity_split": [], "orthogonality": [], "blend": []}

    # -- 1. liquidity split at every horizon, raw signal ---------------------
    print("\n" + "=" * 78)
    print("  1. Does the effect survive where you can actually trade?")
    print("  " + "-" * 74)
    print(f"  {'h':>3}{'universe':>10}{'IC':>11}{'t (NW)':>9}{'days':>7}   reading")
    for h in HORIZONS:
        fr = xrank(features.forward_returns(close, h))
        for nm, mask in [("all", None), *terciles.items()]:
            s = sig if mask is None else sig.where(mask)
            ic = daily_ic(s.shift(1), fr)
            mean, t, _ = nw_t(ic.to_numpy(), h - 1)
            read = ("tradable band" if nm == "liquid" and abs(t) >= 2 and mean > 0
                    else "zero" if abs(t) < 2 else "present")
            print(f"  {h:>2}d{nm:>10}{mean:>+11.5f}{t:>+9.2f}{len(ic):>7}   {read}")
            out["liquidity_split"].append(
                {"horizon": h, "universe": nm, "ic": mean, "t": t, "days": len(ic)})
        print("  " + "-" * 74)
        del fr

    # -- 2. orthogonality ----------------------------------------------------
    print("\n" + "=" * 78)
    print("  2. Is the information its own, or already owned by other factors?")
    print("  " + "-" * 74)
    print(f"  {'h':>3}{'raw IC':>11}{'t':>8}{'residual IC':>14}{'t':>8}{'kept':>8}   reading")
    for h in HORIZONS:
        fr = xrank(features.forward_returns(close, h))
        a, ta, _ = nw_t(daily_ic(sig.shift(1), fr).to_numpy(), h - 1)
        b, tb, _ = nw_t(daily_ic(resid_r.shift(1), fr).to_numpy(), h - 1)
        kept = (b / a) if (a and abs(a) > 1e-4) else float("nan")
        read = ("own information" if abs(tb) >= 2 and (not np.isfinite(kept) or abs(kept) > .5)
                else "repackaged factors" if abs(kept) < .3
                else "partly its own")
        kept_s = "    n/a" if not np.isfinite(kept) else f"{kept:>6.0%}"
        print(f"  {h:>2}d{a:>+11.5f}{ta:>+8.2f}{b:>+14.5f}{tb:>+8.2f}{kept_s:>8}   {read}")
        out["orthogonality"].append({"horizon": h, "ic_raw": a, "t_raw": ta,
                                     "ic_resid": b, "t_resid": tb, "kept": kept})
        del fr

    # -- 3. marginal contribution to a factor composite ----------------------
    print("\n" + "=" * 78)
    print("  3. Added to a book of standard factors, does it pay its own way?")
    print("  " + "-" * 74)
    print(f"  {'h':>3}{'weight':>8}{'IC':>10}{'t':>8}{'gross/yr':>10}"
          f"{'turnover':>10}{'marginal cost':>15}{'net gain':>10}")
    for h in HORIZONS:
        fwd = clean_forward(close, h)
        fr = xrank(fwd)
        base_r = composite.shift(1)
        base_ic, base_t, _ = nw_t(daily_ic(base_r, fr).to_numpy(), h - 1)
        base_turn = leg_turnover(base_r, h)
        base_ls = ls_return(base_r, fwd, h)
        base_gross = float(np.nanmean(base_ls)) * (252.0 / h) if base_ls.size else float("nan")
        if not np.isfinite(base_gross) or abs(base_gross) > MAX_PLAUSIBLE_ANNUAL:
            print(f"  {h:>2}d{'base':>8}{base_ic:>+10.4f}{base_t:>+8.2f}"
                  f"  UNUSABLE — mean return still contaminated at this horizon")
            base_gross = float("nan")
        else:
            print(f"  {h:>2}d{'base':>8}{base_ic:>+10.4f}{base_t:>+8.2f}"
                  f"{base_gross:>+9.2%}{base_turn:>9.1f}x{'-':>15}{'-':>10}")
        out["blend"].append({"horizon": h, "weight": 0.0, "ic": base_ic, "t": base_t,
                             "gross_annual": base_gross, "turnover": base_turn,
                             "marginal_cost": 0.0, "net_gain": 0.0})
        for w in BLEND_WEIGHTS:
            bl = xrank(w * sig.fillna(0.5) + (1 - w) * composite.fillna(0.5)).where(uni)
            bl_r = bl.shift(1)
            ic, t, _ = nw_t(daily_ic(bl_r, fr).to_numpy(), h - 1)
            turn = leg_turnover(bl_r, h)
            ls = ls_return(bl_r, fwd, h)
            gross = float(np.nanmean(ls)) * (252.0 / h) if ls.size else float("nan")
            mcost = (turn - base_turn) * COST_BPS_ONE_WAY / 1e4
            net = (gross - base_gross) - mcost
            if not np.isfinite(gross) or abs(gross) > MAX_PLAUSIBLE_ANNUAL:
                gross = net = float("nan")   # contaminated: do not report
            print(f"  {h:>2}d{w:>8.2f}{ic:>+10.4f}{t:>+8.2f}{gross:>+9.2%}"
                  f"{turn:>9.1f}x{mcost:>+14.2%}{net:>+10.2%}")
            out["blend"].append({"horizon": h, "weight": w, "ic": ic, "t": t,
                                 "gross_annual": gross, "turnover": turn,
                                 "marginal_cost": mcost, "net_gain": net})
        print("  " + "-" * 74)
        del fr, fwd

    # -- verdict -------------------------------------------------------------
    liq1 = next((r for r in out["liquidity_split"]
                 if r["horizon"] == 1 and r["universe"] == "liquid"), None)
    best = max((b for b in out["blend"] if b["weight"] > 0),
               key=lambda b: (b["net_gain"] if np.isfinite(b["net_gain"]) else -9))
    verdict = []
    if liq1 and abs(liq1["t"]) >= 2 and abs(liq1["ic"]) >= IC_FLOOR:
        verdict.append("The 1-day effect is present among liquid names.")
    else:
        verdict.append("The 1-day effect is absent among liquid names, which is "
                       "where a combined book would have to take it.")
    orth1 = next((r for r in out["orthogonality"] if r["horizon"] == 1), None)
    if orth1 and abs(orth1["t_resid"]) >= 2 and abs(orth1["kept"]) > .5:
        verdict.append("It carries information the standard factors do not.")
    else:
        verdict.append("Residualising against standard factors removes most of it.")
    if np.isfinite(best["net_gain"]) and best["net_gain"] > 0:
        verdict.append(f"Best blend adds {best['net_gain']:+.2%} a year net of "
                       f"marginal cost at weight {best['weight']:.2f}, "
                       f"horizon {best['horizon']}d.")
    else:
        verdict.append("No blend weight adds return net of the turnover it causes.")
    out["verdict"] = verdict

    print("\n" + "=" * 78)
    print("  VERDICT")
    for v in verdict:
        print("   -", v)
    print("=" * 78)

    dest = storage.config.DATA_DIR.parent / "results" / "combination.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote {dest.relative_to(storage.config.DATA_DIR.parent)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    a = ap.parse_args()
    selftest() if a.selftest else run(a)


if __name__ == "__main__":
    main()
