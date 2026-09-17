#!/usr/bin/env python3
"""The two things most likely to kill the surviving result.

It cleared a family-wise threshold over 2,652 candidates, it is nearly
orthogonal to size, and it holds in both halves. Two threats remain, and
both are specific.

  1. The feature is slow. A company whose foreign ownership sits high in
     its own range tends to stay there for years, so the same company
     places the same bet every quarter. Season clusters absorb correlation
     inside a season and nothing across them. Firm clusters and two-way
     clusters are the honest standard errors here, and the number of
     effectively independent bets is reported next to the t.

  2. The feature is measured over 250 days. If foreign ownership tracks
     price, it is 250-day momentum wearing a different name. The controls
     stop at 60 days, which is not a test. Momentum over 120, 250 and 500
     days is built and added.

Nothing is claimed unless it survives both.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from krxflow import features, storage

ROOT = Path(__file__).resolve().parent
V4 = ROOT / "results" / "event_panel_v4.parquet"
V5 = ROOT / "results" / "event_panel_v5.parquet"
OUT = ROOT / "results" / "threats_flevel.json"
START = "20100101"
MIN_N = 25


def load(name):
    spec = importlib.util.spec_from_file_location(name.split("_")[0],
                                                  ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


bfm = load("21_baseline_fm")
CTRL = ["c_mom20", "c_mom60", "c_size", "c_vol", "c_turn"]
LONG = ["c_mom120", "c_mom250", "c_mom500"]


def add_long_momentum():
    if V5.exists():
        return pd.read_parquet(V5)
    p = features.load_panels(START, None)
    pct = p["foreign_pct"]
    idx, cols = pct.index, pct.columns
    m = storage.read_range("market", START, None,
                           columns=["trade_date", "ticker", "close"])
    m["trade_date"] = pd.to_datetime(m["trade_date"])
    close = (m.pivot_table(index="trade_date", columns="ticker", values="close",
                           aggfunc="last", observed=True)
               .sort_index().astype("float64")
               .reindex(index=idx, columns=cols))
    del m
    ev = pd.read_parquet(V4)
    colpos = {t: j for j, t in enumerate(cols)}
    ev = ev[ev["ticker"].isin(colpos)].reset_index(drop=True)
    pos = idx.searchsorted(ev["D"].to_numpy())
    jj = ev["ticker"].map(colpos).to_numpy()
    for w, nm in ((120, "c_mom120"), (250, "c_mom250"), (500, "c_mom500")):
        r = close / close.shift(w) - 1.0
        r = r.mask(r.abs() > 10.0)
        ev[nm] = r.to_numpy()[pos - 1, jj]
        print(f"  {nm:<10} coverage={ev[nm].notna().mean():.3f}  "
              f"corr(f_level)={ev[[nm, 'f_level']].corr().iloc[0, 1]:+.3f}")
    ev.to_parquet(V5, index=False)
    return ev


def prepare(d, base, ctrl, y="abn60"):
    xs = ["surprise"] + ctrl + [base]
    xo, xb, ys, sea, tic = [], [], [], [], []
    for day, g in d.groupby("D"):
        g = g.dropna(subset=[y] + xs)
        if len(g) < MIN_N:
            continue
        xo.append(np.column_stack([bfm.rank_std(g[c]) for c in xs if c != base]
                                  + [np.ones(len(g))]))
        xb.append(bfm.rank_std(g[base]))
        v = g[y].to_numpy(dtype=float)
        lo, hi = np.percentile(v, [1, 99])
        ys.append(np.clip(v, lo, hi))
        t = pd.Timestamp(day)
        sea.append(np.full(len(g), t.year * 4 + (t.month - 1) // 3))
        tic.append(g["ticker"].to_numpy())
    if not xo:
        return None
    return (np.vstack(xo), np.concatenate(xb), np.concatenate(ys),
            np.concatenate(sea), np.concatenate(tic))


def meat_by(u, key):
    df = pd.DataFrame({"u": u, "k": key})
    return float((df.groupby("k")["u"].sum() ** 2).sum())


def fit(prep):
    Xo, xb, y, sea, tic = prep
    # Frisch-Waugh: project the controls out of both sides
    Q, _ = np.linalg.qr(Xo)
    x = xb - Q @ (Q.T @ xb)
    yy = y - Q @ (Q.T @ y)
    den = float(x @ x)
    b = float(x @ yy) / den
    e = yy - x * b
    u = x * e
    pair = pd.Series(sea).astype(str) + "|" + pd.Series(tic).astype(str)
    m_s = meat_by(u, sea)
    m_f = meat_by(u, tic)
    m_sf = meat_by(u, pair.to_numpy())
    out = {"bp": b * 1e4, "events": len(y),
           "seasons": int(pd.Series(sea).nunique()),
           "firms": int(pd.Series(tic).nunique())}
    for tag, m in (("season", m_s), ("firm", m_f),
                   ("two_way", max(m_s + m_f - m_sf, 1e-30))):
        out[f"t_{tag}"] = b / (np.sqrt(m) / den)
    return out


def show(tag, r):
    if r is None:
        print(f"  {tag:<40}  --")
        return
    print(f"  {tag:<40}{r['bp']:+8.1f} bp/SD   "
          f"t(season) {r['t_season']:+6.2f}   "
          f"t(firm) {r['t_firm']:+6.2f}   "
          f"t(two-way) {r['t_two_way']:+6.2f}   "
          f"n={r['events']:>6,} firms={r['firms']:,}")


def main() -> int:
    print("building long-horizon momentum ...")
    d = add_long_momentum()
    d = d[d["kind"] == "periodic"].copy()
    d["fl_minus_i"] = (d.groupby("D")["f_level"].transform(lambda s: s.rank(pct=True))
                       - d.groupby("D")["i_flow20_v2"].transform(lambda s: s.rank(pct=True)))
    d["fl_x_surp"] = (d.groupby("D")["f_level"].transform(lambda s: s.rank(pct=True))
                      * d.groupby("D")["surprise"].transform(lambda s: s.rank(pct=True)))
    R = {}

    print("\nhow slow is the feature? (this is what firm clustering is for)")
    g = d.sort_values(["ticker", "D"])
    lag = g.groupby("ticker")["f_level"].shift(1)
    print(f"  corr(f_level, its own value at the previous filing) = "
          f"{g['f_level'].corr(lag):+.3f}")
    print(f"  events per firm: mean {len(d) / d['ticker'].nunique():.1f}")
    R["autocorr_across_filings"] = float(g["f_level"].corr(lag))

    print("\nthreat 1 -- standard errors that allow a firm to repeat itself")
    for b in ("f_level", "fl_minus_i", "fl_x_surp"):
        p = prepare(d, b, CTRL)
        R[f"base_{b}"] = fit(p) if p else None
        show(f"{b}, standard controls", R[f"base_{b}"])

    print("\nthreat 2 -- is it long-horizon momentum?")
    for b in ("f_level", "fl_minus_i", "fl_x_surp"):
        p = prepare(d, b, CTRL + LONG)
        R[f"long_{b}"] = fit(p) if p else None
        show(f"{b}, plus 120/250/500-day momentum", R[f"long_{b}"])

    print("\n  for reference, the momentum controls on their own:")
    for c in LONG:
        p = prepare(d, c, CTRL)
        show(f"{c} as the feature", fit(p) if p else None)

    print("\nboth threats at once, and both halves")
    med = d["D"].median()
    for tag, sub in ((f"before {pd.Timestamp(med).date()}", d[d["D"] < med]),
                     (f"from {pd.Timestamp(med).date()}", d[d["D"] >= med])):
        for b in ("fl_minus_i", "fl_x_surp"):
            p = prepare(sub, b, CTRL + LONG)
            r = fit(p) if p else None
            R[f"half_{tag}_{b}"] = r
            show(f"{b}, hard controls, {tag}", r)

    OUT.write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print("\nread t(two-way). it is the only one that lets a company repeat "
          "the same bet across seasons.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
