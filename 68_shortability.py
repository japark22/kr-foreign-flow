#!/usr/bin/env python3
"""Does the constraint do the work? A test that needs no new data.

Korea suspended short selling twice, which is two observations, and no
amount of Korean history turns two into more. But between May 2021 and
November 2023 the rule was not uniform: index constituents could be sold
short and everything else could not. Within a single trading day, two
stocks then faced different rules. Comparing them removes the calendar
entirely -- the estimator already ranks inside the day, so the regime, the
market and the era are differenced out by construction.

Membership is proxied here by market-capitalisation rank inside each
board, because the index selects mostly on size and the exchange feed that
would give the real list is not responding. The proxy is noisy, which
pulls the estimate toward zero, so a result found with it is conservative.

The proxy cannot fake the placebo. During the two full suspensions nobody
could sell short, so membership must carry NO differential there whatever
the proxy's accuracy, and the same holds before 2020 and after March 2025
when everybody could. Five periods, one prediction:

    interaction negative in the mixed window, zero in the other four.

A negative sign means the signal is weaker where shorting is allowed,
which is the constraint hypothesis stated as an inequality.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from krxflow import storage

ROOT = Path(__file__).resolve().parent
V5 = ROOT / "results" / "event_panel_v5.parquet"
V6 = ROOT / "results" / "event_panel_v6.parquet"
OUT = ROOT / "results" / "shortability.json"
START = "20100101"
SIZES = {"KOSPI": 200, "KOSDAQ": 150}

REGIMES = [
    ("A  before 2020-03-16, everyone could short", None, "2020-03-16"),
    ("B  2020-03-16..2021-05-02, nobody", "2020-03-16", "2021-05-03"),
    ("C  2021-05-03..2023-11-05, members only", "2021-05-03", "2023-11-06"),
    ("D  2023-11-06..2025-03-30, nobody", "2023-11-06", "2025-03-31"),
    ("E  from 2025-03-31, everyone", "2025-03-31", None),
]


def load(name):
    spec = importlib.util.spec_from_file_location(name.split("_")[0],
                                                  ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


th = load("64_threats_flevel")
bfm = load("21_baseline_fm")
HARD = th.CTRL + th.LONG


def build_membership():
    if V6.exists():
        return pd.read_parquet(V6)
    print("building the size-rank proxy for index membership ...")
    mk = storage.read_range("foreign_ownership", START, None,
                            columns=["trade_date", "ticker", "market"])
    mk["trade_date"] = pd.to_datetime(mk["trade_date"])
    cap = storage.read_range("market", START, None,
                             columns=["trade_date", "ticker", "market_cap"])
    cap["trade_date"] = pd.to_datetime(cap["trade_date"])
    m = cap.merge(mk, on=["trade_date", "ticker"], how="inner")
    m["market"] = m["market"].astype(str)
    m = m[m["market"].isin(SIZES)].copy()
    m["rank"] = (m.groupby(["trade_date", "market"], observed=True)["market_cap"]
                 .rank(ascending=False, method="first"))
    m["cut"] = m["market"].map(SIZES).astype(float)
    m["member"] = (m["rank"] <= m["cut"]).astype(float)
    m["rank_gap"] = m["rank"] - m["cut"]        # 0 is the boundary
    print(f"  member share overall: {m['member'].mean():.3f}")

    ev = pd.read_parquet(V5)
    lag = m.copy()
    lag["join_date"] = lag["trade_date"]
    days = pd.DatetimeIndex(sorted(m["trade_date"].unique()))
    pos = days.searchsorted(ev["D"].to_numpy()) - 1
    pos = np.clip(pos, 0, len(days) - 1)
    ev["Dm1"] = days[pos]
    ev = ev.merge(lag[["join_date", "ticker", "member", "rank", "rank_gap", "market"]],
                  left_on=["Dm1", "ticker"], right_on=["join_date", "ticker"],
                  how="left").drop(columns=["join_date"])
    print(f"  matched membership on {ev['member'].notna().mean():.3f} of events")
    ev.to_parquet(V6, index=False)
    return ev


def design(d, base):
    """x, x*member, member, controls -- all ranked inside the day."""
    xs = ["surprise"] + HARD + [base]
    X, Y, S, T = [], [], [], []
    for day, g in d.groupby("D"):
        g = g.dropna(subset=["abn60", "member"] + xs)
        if len(g) < th.MIN_N or g["member"].nunique() < 2:
            continue
        xb = bfm.rank_std(g[base])
        mem = g["member"].to_numpy(dtype=float)
        cols = [xb, xb * mem, mem]
        cols += [bfm.rank_std(g[c]) for c in xs if c != base]
        cols += [np.ones(len(g))]
        X.append(np.column_stack(cols))
        v = g["abn60"].to_numpy(dtype=float)
        lo, hi = np.percentile(v, [1, 99])
        Y.append(np.clip(v, lo, hi))
        ts = pd.Timestamp(day)
        S.append(np.full(len(g), ts.year * 4 + (ts.month - 1) // 3))
        T.append(g["ticker"].to_numpy())
    if not X:
        return None
    return (np.vstack(X), np.concatenate(Y), np.concatenate(S),
            np.concatenate(T))


def fit(pack, want=1):
    X, y, S, T = pack
    XtX = np.linalg.pinv(X.T @ X)
    b = XtX @ (X.T @ y)
    e = y - X @ b
    u = X * e[:, None]
    df = pd.DataFrame(u)
    df["g"] = S
    G = df.groupby("g").sum().to_numpy()
    V = XtX @ (G.T @ G) @ XtX
    se = float(np.sqrt(max(V[want, want], 1e-30)))
    return {"main": float(b[0] * 1e4), "inter": float(b[1] * 1e4),
            "t": float(b[want] / se), "events": int(len(y)),
            "seasons": int(pd.Series(S).nunique()),
            "member_share": float(X[:, 2].mean())}


def show(tag, r):
    if r is None:
        print(f"  {tag:<46}  --")
        return
    print(f"  {tag:<46}main {r['main']:+7.1f}   "
          f"x member {r['inter']:+8.1f}   t {r['t']:+6.2f}   "
          f"n={r['events']:>6,}  members={r['member_share']:.2f}")


def main() -> int:
    ev = build_membership()
    d = ev[ev["kind"] == "periodic"].copy()
    d["fl_minus_i"] = (d.groupby("D")["f_level"].transform(lambda s: s.rank(pct=True))
                       - d.groupby("D")["i_flow20_v2"].transform(lambda s: s.rank(pct=True)))
    R = {}

    print("\nfull sample, every period  (the prediction is C and only C)")
    for tag, a_, b_ in REGIMES:
        sub = d
        if a_:
            sub = sub[sub["D"] >= a_]
        if b_:
            sub = sub[sub["D"] < b_]
        p = design(sub, "fl_minus_i")
        r = fit(p) if p else None
        R[tag] = r
        show(tag, r)

    print("\nnear the boundary only  (size held close, rule differs)")
    for width in (60, 100, 150):
        print(f"  [membership rank within {width} of the cut-off]")
        band = d[d["rank_gap"].abs() <= width]
        for tag, a_, b_ in REGIMES:
            sub = band
            if a_:
                sub = sub[sub["D"] >= a_]
            if b_:
                sub = sub[sub["D"] < b_]
            p = design(sub, "fl_minus_i")
            r = fit(p) if p else None
            R[f"band{width}_{tag}"] = r
            show("    " + tag, r)

    OUT.write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print("\nC negative and A, B, D, E near zero is the constraint hypothesis.")
    print("Anything negative in B or D is the proxy picking up index membership")
    print("itself, and then this design cannot separate the two.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
