"""Which investor type carries the pre-event effect, measured honestly.

34_anchor_robust showed the -112 bp/SD headline was a weighting artefact:
it needs both a >=25-events-per-day filter and one-vote-per-day weighting.
One vote per position, clustered by reporting season, gives -61.9 bp/SD
(t -2.71) on provisional filings -- half the size, but independent of every
arbitrary knob, and that is the number worth decomposing.

Estimator, fixed before any type is looked at:
  sample      provisional filings only (the pooled panel is t -1.80)
  regressors  surprise + CTRL + baseline, rank-standardised within the day
  outcome     abn60, winsorised 1/99 within the day
  weighting   one vote per event
  inference   SE clustered on the reporting season (34 of them)

Eight types are searched, so the bar comes first: the baseline is shuffled
within each event day, the whole family re-estimated, the largest |t| kept.
The aggregate is run through the identical estimator as an anchor and the
run aborts if it does not come back at -61.9.

    python 35_decompose.py                 # 200 permutations
    python 35_decompose.py --perm 1000     # tighter bar
    python 35_decompose.py --rebuild       # re-read data/investor
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path("results/event_panel.parquet")
CACHE = Path("results/decompose_panel.parquet")
DETAIL = Path("data/investor")
OUT = Path("results/decompose.json")

FAMILY = ["금융투자", "보험", "투신", "사모", "은행", "기타금융", "연기금",
          "기타법인"]
REFERENCE = ["개인", "외국인"]
ALL_TYPES = FAMILY + REFERENCE

Y, ANCHOR, MIN_N = "abn60", "i_flow20", 5
ANCHOR_TARGET, ANCHOR_TOL = -61.9, 3.0


def load_reference():
    spec = importlib.util.spec_from_file_location("bfm", "21_baseline_fm.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def zwin(x, look=250, minp=120):
    h = x.shift(1)
    mu = h.rolling(look, min_periods=minp).mean()
    sd = h.rolling(look, min_periods=minp).std()
    return (x - mu) / sd.where(sd > 1e-12)


def flow_intensity(net, val, win):
    num = net.rolling(win, min_periods=max(5, win // 2)).sum()
    den = val.rolling(win, min_periods=max(5, win // 2)).sum()
    return zwin(num / den.where(den > 0))


def build_panel(start: str) -> pd.DataFrame:
    from krxflow import storage
    panel = pd.read_parquet(PANEL)
    panel["D"] = pd.to_datetime(panel["D"])

    print("loading market ...")
    m = storage.read_range("market", start, None,
                           columns=["trade_date", "ticker", "value_traded"])
    m["trade_date"] = pd.to_datetime(m["trade_date"])
    vt = (m.pivot_table(index="trade_date", columns="ticker",
                        values="value_traded", aggfunc="last", observed=True)
            .sort_index().astype("float64"))
    del m
    keep = set(panel["ticker"].unique()) & set(vt.columns)
    vt = vt[sorted(keep)]
    idx, cols = vt.index, vt.columns

    print("loading investor detail ...")
    files = [f for f in sorted(DETAIL.glob("*.parquet")) if f.stem in keep]
    frames = []
    for i, f in enumerate(files, 1):
        g = pd.read_parquet(f, columns=["trade_date", "ticker"] + ALL_TYPES)
        for c in ALL_TYPES:
            g[c] = g[c].astype("float32")
        frames.append(g)
        if i % 800 == 0:
            print(f"    {i:,}/{len(files):,}")
    long = pd.concat(frames, ignore_index=True)
    del frames
    long["trade_date"] = pd.to_datetime(long["trade_date"])

    rowpos, colpos = idx.get_indexer(panel["D"]), cols.get_indexer(panel["ticker"])
    ok = (rowpos >= 0) & (colpos >= 0)
    for c in ALL_TYPES:
        w = (long.pivot_table(index="trade_date", columns="ticker", values=c,
                              aggfunc="last", observed=True)
                 .sort_index().astype("float64")
                 .reindex(index=idx, columns=cols))
        f = flow_intensity(w, vt, 20).to_numpy()
        v = np.full(len(panel), np.nan)
        v[ok] = f[rowpos[ok], colpos[ok]]
        panel[f"x_{c}"] = v
        print(f"  {c:8s} coverage {np.isfinite(v).mean():.3f}")
        del w, f
    CACHE.parent.mkdir(exist_ok=True)
    panel.to_parquet(CACHE, index=False)
    print(f"cached {CACHE}")
    return panel


# ----------------------------------------------------------- estimator

def prepare(sub: pd.DataFrame, base: str, xs: list[str], bfm):
    """Stack the days into one pooled design, keeping the day boundaries so a
    permutation can reshuffle inside a day without crossing into another."""
    xo, xb, ys, sea, slices = [], [], [], [], []
    n = 0
    for day, g in sub.groupby("D"):
        g = g.dropna(subset=[Y] + xs)
        if len(g) < MIN_N:
            continue
        others = [c for c in xs if c != base]
        xo.append(np.column_stack([bfm.rank_std(g[c]) for c in others]
                                  + [np.ones(len(g))]))
        xb.append(bfm.rank_std(g[base]))
        y = g[Y].to_numpy(dtype=float)
        lo, hi = np.percentile(y, [1, 99])
        ys.append(np.clip(y, lo, hi))
        d = pd.Timestamp(day)
        sea.append(np.full(len(g), d.year * 4 + (d.month - 1) // 3))
        slices.append((n, n + len(g)))
        n += len(g)
    if not slices:
        return None
    return (np.vstack(xo), np.concatenate(xb), np.concatenate(ys),
            np.concatenate(sea), slices)


def fit(prep, rng=None):
    Xo, xb, y, sea, slices = prep
    x = xb
    if rng is not None:
        x = xb.copy()
        for s, e in slices:
            x[s:e] = rng.permutation(x[s:e])
    X = np.column_stack([x, Xo])
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(sea):
        m = sea == c
        u = X[m].T @ e[m]
        meat += np.outer(u, u)
    V = XtX_inv @ meat @ XtX_inv
    se = float(np.sqrt(max(V[0, 0], 1e-30)))
    return float(beta[0] * 1e4), float(beta[0] / se)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20160101")
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()

    bfm = load_reference()
    if CACHE.exists() and not a.rebuild:
        panel = pd.read_parquet(CACHE)
        panel["D"] = pd.to_datetime(panel["D"])
        print(f"using cached {CACHE.name} -- pass --rebuild to redo it")
    else:
        panel = build_panel(a.start)

    sub = panel[panel["kind"] == "provisional"]
    print(f"\nprovisional events {len(sub):,}   CTRL {list(bfm.CTRL)}")

    xs_a = ["surprise"] + list(bfm.CTRL) + [ANCHOR]
    pa = prepare(sub, ANCHOR, xs_a, bfm)
    ab, at = fit(pa)
    print(f"\nanchor  기관합계  {ab:+8.1f} bp/SD   t {at:+5.2f}"
          f"   events {len(pa[2]):,}   seasons {len(np.unique(pa[3]))}")
    if abs(ab - ANCHOR_TARGET) > ANCHOR_TOL:
        sys.exit(f"anchor off target ({ANCHOR_TARGET}) -- stopping")
    print("  reproduces 34_anchor_robust")

    print("\n--- family of 8, same estimator, each on its own sample ---")
    res, preps = {}, {}
    for c in FAMILY + REFERENCE:
        b = f"x_{c}"
        xs = ["surprise"] + list(bfm.CTRL) + [b]
        p = prepare(sub, b, xs, bfm)
        if p is None:
            print(f"  {c:10s} no usable days")
            continue
        preps[c] = p
        co, t = fit(p)
        res[c] = {"coef_bp": co, "t": t, "events": int(len(p[2])),
                  **({"reference": True} if c in REFERENCE else {})}
        tag = "  ref " if c in REFERENCE else "      "
        print(f"{tag}{c:10s} {co:+8.1f} bp/SD   t {t:+5.2f}"
              f"   events {len(p[2]):,}")

    fam = [c for c in FAMILY if c in preps]
    print(f"\nfamily-wise bar over {len(fam)} types, {a.perm} permutations ...")
    rng = np.random.default_rng(a.seed)
    maxes = np.empty(a.perm)
    for p in range(a.perm):
        maxes[p] = max(abs(fit(preps[c], rng)[1]) for c in fam)
        if (p + 1) % 25 == 0:
            print(f"    {p + 1}/{a.perm}   running bar "
                  f"{np.quantile(maxes[:p + 1], 0.95):.2f}")
    bar = float(np.quantile(maxes, 0.95))

    print(f"\nbar |t| >= {bar:.2f}")
    winners = []
    for c in fam:
        t = res[c]["t"]
        res[c]["p_fwer"] = float((maxes >= abs(t)).mean())
        if abs(t) >= bar:
            winners.append(c)
        print(f"  {'PASS' if abs(t) >= bar else '    '}  {c:10s} "
              f"t {t:+5.2f}   p {res[c]['p_fwer']:.3f}")

    print("\n" + (f"clears the bar: {', '.join(winners)}" if winners else
                  "no type clears the family bar -- the aggregate stays the "
                  "best available form of this signal"))

    OUT.write_text(json.dumps({
        "estimator": {"sample": "provisional", "y": Y, "min_n": MIN_N,
                      "weighting": "one vote per event",
                      "cluster": "reporting season",
                      "ctrl": list(bfm.CTRL)},
        "anchor": {"name": "기관합계", "coef_bp": ab, "t": at},
        "family": fam, "reference": REFERENCE, "bar": bar,
        "perm": a.perm, "seed": a.seed, "results": res, "winners": winners,
    }, ensure_ascii=False, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
