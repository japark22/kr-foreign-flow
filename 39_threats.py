"""Rule out the alternative explanations before the result is published.

The brief asked for two filters that were never built: index rebalancing
(MSCI, FTSE) and ex-dividend tax behaviour. Both are more than omissions --
they are rival explanations. Korean provisional filings cluster in the same
weeks as the quarterly index reviews, index-driven buying is mechanical by
construction, and post-rebalancing reversal runs about a quarter, which is
exactly the horizon where this effect appears and the only horizon where it
appears. If that is what we measured, the finding is not about earnings.

Four tests. The first needs no calendar at all, which matters because a
calendar I get wrong would quietly exonerate the threat:

  1  size terciles       MSCI Standard, FTSE Global and KOSPI 200 are
                         large-cap indices. Rebalancing pressure does not
                         reach small caps. If the effect survives there, the
                         rebalancing story cannot explain it.
  2  review windows      events whose pre-window overlaps a quarterly review
                         are dropped; the effect must survive on the rest
  3  December            the same, for the ex-dividend cluster at the
                         Korean fiscal year end
  4  market-wide waves   days when institutions were buying everything at
                         once, found from the data rather than a calendar,
                         so a wrong review date is still caught

Predictions if REBALANCING is the true cause: effect confined to large caps,
gone outside review windows, concentrated on wave days.
Predictions if the EARNINGS reading is right: present in every size tercile,
present outside review windows, present off the waves.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("results/decompose_panel.parquet")
CALSRC = Path("data/investor/005930.parquet")
OUT = Path("results/threats.json")
AGG, MIN_N = "i_flow20", 5
LOOK = 20                      # the baseline's own window, in trading days
REVIEW_PAD = 10                # trading days either side of an effective date
DEC_PAD = (10, 5)

spec = importlib.util.spec_from_file_location("bfm", "21_baseline_fm.py")
bfm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bfm)
CTRL = list(bfm.CTRL)


def prepare(sub, base, xs, y="abn60"):
    xo, xb, ys, sea, slices = [], [], [], [], []
    n = 0
    others = [c for c in xs if c != base]
    for day, g in sub.groupby("D"):
        g = g.dropna(subset=[y] + xs)
        if len(g) < MIN_N:
            continue
        xo.append(np.column_stack([bfm.rank_std(g[c]) for c in others]
                                  + [np.ones(len(g))]))
        xb.append(bfm.rank_std(g[base]))
        v = g[y].to_numpy(dtype=float)
        lo, hi = np.percentile(v, [1, 99])
        ys.append(np.clip(v, lo, hi))
        t = pd.Timestamp(day)
        sea.append(np.full(len(g), t.year * 4 + (t.month - 1) // 3))
        slices.append((n, n + len(g)))
        n += len(g)
    if not slices:
        return None
    return (np.vstack(xo), np.concatenate(xb), np.concatenate(ys),
            np.concatenate(sea), slices)


def fit(prep):
    Xo, xb, y, sea, _ = prep
    X = np.column_stack([xb, Xo])
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
    return {"bp": float(beta[0] * 1e4), "t": float(beta[0] / se),
            "events": int(len(y)), "seasons": int(len(np.unique(sea)))}


def cell(sub, tag, store=None, key=None):
    p = prepare(sub, AGG, ["surprise"] + CTRL + [AGG])
    if p is None or len(np.unique(p[3])) < 8:
        print(f"  {tag:<38}too few seasons")
        return None
    r = fit(p)
    print(f"  {tag:<38}{r['bp']:+8.1f} bp/SD   t {r['t']:+5.2f}"
          f"   events {r['events']:,}   seasons {r['seasons']}")
    if store is not None:
        store[key or tag] = r
    return r


def main() -> int:
    d = pd.read_parquet(CACHE)
    d["D"] = pd.to_datetime(d["D"])
    prov = d[d["kind"] == "provisional"].copy()

    cal = pd.DatetimeIndex(sorted(pd.read_parquet(
        CALSRC, columns=["trade_date"])["trade_date"].unique()))
    print(f"trading calendar {len(cal):,} days "
          f"{cal[0]:%Y-%m-%d} .. {cal[-1]:%Y-%m-%d}")

    # Event position on the trading calendar; the baseline reads the 20 days
    # before the event, so that is the interval a rival flow driver must miss.
    p_evt = cal.searchsorted(prov["D"].to_numpy())
    prov["p0"] = np.clip(p_evt - LOOK, 0, len(cal) - 1)
    prov["p1"] = np.clip(p_evt - 1, 0, len(cal) - 1)

    def last_of(year: int, month: int):
        s = cal[(cal.year == year) & (cal.month == month)]
        return int(cal.searchsorted(s[-1])) if len(s) else None

    reviews, decs = [], []
    for y in range(int(cal[0].year), int(cal[-1].year) + 1):
        for mo in (2, 5, 8, 11):          # MSCI quarterly index reviews
            q = last_of(y, mo)
            if q is not None:
                reviews.append(q)
        dz = last_of(y, 12)               # Korean annual dividend record date
        if dz is not None:
            decs.append(dz)
    print(f"review effective dates {len(reviews)}   "
          f"December record dates {len(decs)}")

    def hits(row_p0, row_p1, marks, pad):
        lo_pad, hi_pad = pad if isinstance(pad, tuple) else (pad, pad)
        a, b = row_p0.to_numpy(), row_p1.to_numpy()
        out = np.zeros(len(a), bool)
        for m in marks:
            out |= (a <= m + hi_pad) & (b >= m - lo_pad)
        return out

    prov["in_review"] = hits(prov["p0"], prov["p1"], reviews, REVIEW_PAD)
    prov["in_dec"] = hits(prov["p0"], prov["p1"], decs, DEC_PAD)

    # Market-wide waves: the cross-sectional average of the baseline on an
    # event day says how hard institutions were buying everything at once.
    day_mean = prov.groupby("D")[AGG].mean()
    prov["wave"] = prov["D"].map(day_mean >= day_mean.quantile(0.80))

    R = {"calendar": {"days": int(len(cal)),
                      "reviews": len(reviews), "december": len(decs),
                      "review_pad": REVIEW_PAD, "dec_pad": list(DEC_PAD)}}

    print("\nbaseline for reference")
    R["all"] = cell(prov, "all provisional events")

    print("\n1  size terciles (rebalancing cannot reach small caps)")
    q = prov.groupby("D")["c_size"].rank(pct=True)
    R["size"] = {}
    for tag, m in (("large (top third)", q >= 2 / 3),
                   ("mid", (q > 1 / 3) & (q < 2 / 3)),
                   ("small (bottom third)", q <= 1 / 3)):
        cell(prov[m], f"   {tag}", R["size"], tag)

    print("\n2  quarterly review windows")
    R["review"] = {}
    cell(prov[~prov["in_review"]], "   outside review windows",
         R["review"], "outside")
    cell(prov[prov["in_review"]], "   inside review windows",
         R["review"], "inside")

    print("\n3  December dividend cluster")
    R["december"] = {}
    cell(prov[~prov["in_dec"]], "   outside December", R["december"], "outside")
    cell(prov[prov["in_dec"]], "   inside December", R["december"], "inside")

    print("\n4  market-wide institutional waves")
    R["wave"] = {}
    cell(prov[~prov["wave"]], "   off the waves", R["wave"], "off")
    cell(prov[prov["wave"]], "   on the waves", R["wave"], "on")

    print("\n5  every filter at once (the strictest sample)")
    strict = prov[(~prov["in_review"]) & (~prov["in_dec"]) & (~prov["wave"])]
    R["strict"] = cell(strict, "   clean events only")

    # ---- verdict, written before the numbers were seen
    def ok(r):
        return r is not None and r["bp"] < 0 and r["t"] < -1.5

    # These splits are descriptive only. Gating on significance inside a
    # third of the sample asks whether a third of the data is significant,
    # not whether the effect varies with the moderator; that question is
    # settled by the interactions in 40_threats_formal, and the verdict
    # belongs there.
    small_ok = ok(R["size"].get("small (bottom third)"))
    outside_ok = ok(R["review"].get("outside"))
    verdict = ("descriptive splits only -- see 40_threats_formal for the "
               "interaction tests that decide whether these moderators "
               "actually change the effect")
    print(f"\nverdict: {verdict}")
    R["verdict"] = verdict
    R["gates"] = {"small_caps": bool(small_ok),
                  "outside_reviews": bool(outside_ok)}

    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
