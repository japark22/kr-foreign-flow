"""Forward-return tracker: what riding the accumulation decile actually earned.

Method, stated plainly so the page can repeat it:
  every h sessions, rank the universe by its 20-session change in foreign
  ownership, buy the top decile equal-weighted, hold h sessions, sell.
  Benchmark is the equal-weighted universe over the same window, so the
  number shown is EXCESS return per round trip. Costs charge 0.50% per
  round trip: ~0.30% commission plus spread, 0.20% securities transaction
  tax on the sale. Formations do not overlap, so consecutive periods are
  nearly independent; the t-stat still uses Newey-West with a short lag.

This section exists because the research verdict ("no tradable edge") was
measured once, on history. Recomputing it on every refresh turns the claim
into a running experiment instead of a memory.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

COST_RT = 0.0050      # round trip: commission + spread + 0.20% sales tax
BAD_DAY = 0.50        # a one-day move this large is a capital change, not a price
TOP_Q = 0.10          # the accumulation decile
MIN_NAMES = 50        # skip a formation date with a thinner cross-section
SIG_WINDOW = 20       # the ranking signal: 20-session ownership change


def _nw_t(x: np.ndarray, lag: int) -> float:
    """Newey-West t-stat of the mean, Bartlett weights."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 8:
        return math.nan
    e = x - x.mean()
    s = float(e @ e) / n
    L = min(lag, n - 1)
    for k in range(1, L + 1):
        gk = float(e[k:] @ e[:-k]) / n
        s += 2.0 * (1.0 - k / (L + 1.0)) * gk
    se = math.sqrt(max(s, 1e-18) / n)
    return float(x.mean() / se)


def clean_forward(close: pd.DataFrame, h: int) -> pd.DataFrame:
    """h-session forward return, masked where any day in the window jumped
    more than BAD_DAY -- unadjusted prices make a split look like a crash."""
    r1 = close.pct_change(fill_method=None)
    bad = (r1.abs() > BAD_DAY).fillna(False)
    hit = bad.rolling(h, min_periods=1).max().shift(-h).fillna(1).astype(bool)
    return (close.shift(-h) / close - 1.0).mask(hit)


def ride(pct: pd.DataFrame, close: pd.DataFrame, mask: pd.DataFrame,
         horizons=(1, 5, 20)) -> dict:
    """Returns {"stats": [per-horizon dict], "curve": {"5": [points]}}."""
    sig = pct - pct.shift(SIG_WINDOW)
    out = {"stats": [], "curve": {}, "cost_rt_pct": COST_RT * 100,
           "top_pct": int(TOP_Q * 100), "sig_window": SIG_WINDOW}
    for h in horizons:
        fwd = clean_forward(close, h)
        rows = []
        for d in mask.index[SIG_WINDOW::h]:
            m = mask.loc[d]
            s = sig.loc[d].where(m)
            f = fwd.loc[d].where(m)
            ok = s.notna() & f.notna()
            if int(ok.sum()) < MIN_NAMES:
                continue
            sv, fv = s[ok], f[ok]
            cut = sv.quantile(1.0 - TOP_Q)
            top = fv[sv >= cut]
            if len(top) < 5:
                continue
            gross = float(top.mean() - fv.mean())
            rows.append((d, gross, gross - COST_RT))
        if len(rows) < 8:
            continue
        g = np.array([r[1] for r in rows])
        net = np.array([r[2] for r in rows])
        out["stats"].append({
            "h": h, "n": len(rows),
            "gross_bp": float(g.mean() * 1e4),
            "net_bp": float(net.mean() * 1e4),
            "t_gross": _nw_t(g, 2),
            "hit": float((g > 0).mean() * 100),
            "ann_net_pct": float(net.mean() * (250.0 / h) * 100),
        })
        cg, cn = np.cumsum(g), np.cumsum(net)
        out["curve"][str(h)] = [
            {"date": str(d.date()), "gross": float(a * 100), "net": float(b * 100)}
            for (d, _, _), a, b in zip(rows, cg, cn)]
    return out
