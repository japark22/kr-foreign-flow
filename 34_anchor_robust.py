"""Is the anchor a finding, or a weighting choice?

The headline -112 bp/SD is a day-level Fama-MacBeth estimate with min_n=25.
Moving that threshold moves the coefficient from -10 to -258, so before the
signal is decomposed into investor types the estimator itself has to be
pinned down. Korean earnings cluster into a handful of crowded days, so
"one vote per day" and "one vote per position" are measuring different
things, and only the second is what a book would earn.

Three estimators, same sample, same regressors:
  A  day-level FM            one vote per day, threshold swept
  B  day-level FM, weighted  days weighted by how many events they carry
  C  pooled OLS              one vote per event, SE clustered by day and
                             by reporting season

A real effect shows up in all three. If it lives only in A, it is a property
of the weighting, not of the market.
"""
import importlib.util

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("bfm", "21_baseline_fm.py")
bfm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bfm)

Y, BASE = "abn60", "i_flow20"
XS = ["surprise"] + list(bfm.CTRL) + [BASE]
IB = XS.index(BASE)


def season_id(dates: pd.Series) -> np.ndarray:
    """Korean filings arrive in four bursts a year; a season is the honest
    independent unit, far more so than a day."""
    d = pd.to_datetime(dates)
    return (d.dt.year * 4 + (d.dt.month - 1) // 3).to_numpy()


def day_blocks(sub: pd.DataFrame, min_n: int):
    out = []
    for day, g in sub.groupby("D"):
        g = g.dropna(subset=[Y] + XS)
        if len(g) < min_n:
            continue
        X = np.column_stack([bfm.rank_std(g[c]) for c in XS]
                            + [np.ones(len(g))])
        y = g[Y].to_numpy(dtype=float)
        lo, hi = np.percentile(y, [1, 99])
        out.append((day, X, np.clip(y, lo, hi)))
    return out


def fm(blocks, weighted: bool):
    b, w = [], []
    for _, X, y in blocks:
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        b.append(beta[IB])
        w.append(len(y))
    b, w = np.array(b), np.array(w, dtype=float)
    if len(b) < 8:
        return np.nan, np.nan, len(b)
    if weighted:
        s = b * (w / w.mean())
        return float(np.average(b, weights=w) * 1e4), \
            float(bfm.tracker._nw_t(s, 5)), len(b)
    return float(b.mean() * 1e4), float(bfm.tracker._nw_t(b, 5)), len(b)


def pooled(blocks):
    """One vote per event. SE clustered two ways, because a day is not an
    independent draw and a season is barely one either."""
    X = np.vstack([b[1] for b in blocks])
    y = np.concatenate([b[2] for b in blocks])
    dcl = np.concatenate([np.full(len(b[2]), i) for i, b in enumerate(blocks)])
    scl = np.concatenate([np.full(len(b[2]), s) for s, b in
                          zip(season_id(pd.Series([b[0] for b in blocks])),
                              blocks)])
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    out = {}
    for name, cl in (("day", dcl), ("season", scl)):
        meat = np.zeros((X.shape[1], X.shape[1]))
        for c in np.unique(cl):
            m = cl == c
            u = X[m].T @ e[m]
            meat += np.outer(u, u)
        V = XtX_inv @ meat @ XtX_inv
        se = float(np.sqrt(max(V[IB, IB], 1e-30)))
        out[name] = (float(beta[IB] * 1e4), float(beta[IB] / se),
                     int(len(np.unique(cl))))
    return out, len(y)


d = pd.read_parquet("results/event_panel.parquet")
d["D"] = pd.to_datetime(d["D"])
print(f"CTRL = {bfm.CTRL}\nbaseline {BASE}   outcome {Y}\n")

for tag, sub in (("provisional", d[d["kind"] == "provisional"]),
                 ("all events", d)):
    print(f"================ {tag} ================")
    print("  A  day-level FM, one vote per day")
    for mn in (5, 10, 15, 25, 40):
        bl = day_blocks(sub, mn)
        co, t, n = fm(bl, False)
        print(f"       min_n {mn:>3}   {co:+8.1f} bp/SD   t {t:+5.2f}"
              f"   days {n:,}")

    print("  B  day-level FM, days weighted by event count")
    for mn in (5, 25):
        bl = day_blocks(sub, mn)
        co, t, n = fm(bl, True)
        print(f"       min_n {mn:>3}   {co:+8.1f} bp/SD   t {t:+5.2f}"
              f"   days {n:,}")

    print("  C  pooled, one vote per event")
    for mn in (5, 25):
        bl = day_blocks(sub, mn)
        res, nobs = pooled(bl)
        for how, (co, t, ncl) in res.items():
            print(f"       min_n {mn:>3}   {co:+8.1f} bp/SD   t {t:+5.2f}"
                  f"   cluster {how} ({ncl:,})   events {nobs:,}")
    print()
