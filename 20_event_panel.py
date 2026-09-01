#!/usr/bin/env python3
"""Step 20: build the event panel once, so every later test is instant.

    set -a; source .env; set +a
    python 20_event_panel.py                 # provisional + periodic, cached
    python 20_event_panel.py --rebuild

WHY A PANEL
-----------
Every study so far reloaded ten years of price and ownership history, rebuilt
the same features, and threw them away. One cached table of one row per event
turns a four-minute run into a one-second one, which is what makes it possible
to test a hypothesis properly instead of testing it once.

WHAT EACH COLUMN IS FOR
-----------------------
surprise    the announcement reaction (D-1 close -> D+1 close) minus the mean
            reaction of every issuer filing that same day. Stands in for
            result-versus-consensus: it is the market's own verdict, and it is
            known before the D+1 close where a position would be entered.

BASELINES -- the thing under test. All measured strictly at D-1.
f_flow20/60 official exchange net foreign buying over the window, divided by
            the traded value over the same window, then z-scored against that
            issuer's own trailing year. This is REPORTED buying, not the
            holdings difference the earlier work used -- the two agree at only
            0.10, and the reported series is the one that means what it says.
f_hold20    the holdings-based version, kept so the two can be compared.
f_level     where foreign ownership sits inside its own trailing-year range.
            "Have they been buying" and "are they already full" are different
            questions and the brief's logic arguably means the second.
i_flow20    the same measure for domestic institutions. Our own work found
            institutional flow is 2.3x more persistent than foreign, so if the
            crowding logic holds for anyone it should hold for them too.

CONTROLS -- without these a baseline result is not credible.
c_mom20/60  the pre-event run-up. This is the price-based version of "the
            market already expected it", and it is the control that decides
            whether the ownership data adds anything at all.
c_size      log market cap. Post-earnings drift is stronger in small names and
            foreign ownership concentrates in large ones, so the two would be
            confounded without this.
c_vol       60-session realised volatility.
c_turn      20-session traded value over market cap.

OUTCOMES    market-adjusted returns from the D+1 close over 5, 20 and 60
            sessions, split-guarded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import tracker

ROOT = Path(__file__).resolve().parent
EVENTS = ROOT / "data" / "events" / "earnings_events.parquet"
OUT = ROOT / "results" / "event_panel.parquet"
MIN_ADV = 1e8
HORIZONS = (5, 20, 60)


def zwin(x: pd.DataFrame, look: int = 250, minp: int = 120) -> pd.DataFrame:
    """z against the issuer's own trailing history, excluding today."""
    h = x.shift(1)
    mu = h.rolling(look, min_periods=minp).mean()
    sd = h.rolling(look, min_periods=minp).std()
    return (x - mu) / sd.where(sd > 1e-12)


def flow_intensity(net: pd.DataFrame, val: pd.DataFrame, win: int) -> pd.DataFrame:
    num = net.rolling(win, min_periods=max(5, win // 2)).sum()
    den = val.rolling(win, min_periods=max(5, win // 2)).sum()
    return zwin(num / den.where(den > 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-06-01")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    if OUT.exists() and not a.rebuild:
        print(f"{OUT.name} exists -- pass --rebuild to redo it")
        return 0
    if not EVENTS.exists():
        sys.exit("run 15_earnings_events.py first")

    from krxflow import features, storage
    print("loading ownership panels ...")
    p = features.load_panels(a.start, None)
    pct, uni = p["foreign_pct"], features.universe_mask(p)
    idx, cols = pct.index, pct.columns

    print("loading market ...")
    m = storage.read_range("market", a.start, None,
                           columns=["trade_date", "ticker", "close",
                                    "value_traded", "market_cap"])
    m["trade_date"] = pd.to_datetime(m["trade_date"])

    def pv(df, c):
        return (df.pivot_table(index="trade_date", columns="ticker", values=c,
                               aggfunc="last", observed=True)
                  .sort_index().astype("float64")
                  .reindex(index=idx, columns=cols))

    close, vt, cap = pv(m, "close"), pv(m, "value_traded"), pv(m, "market_cap")
    del m

    print("loading official investor net buying ...")
    nets = {}
    for who, tag in (("외국인", "f"), ("기관합계", "i")):
        try:
            iv = storage.read_range("investor_flow", a.start, None,
                                    columns=["trade_date", "ticker",
                                             "investor", "net_value"])
            iv = iv[iv["investor"] == who]
            if iv.empty:
                print(f"  ! no rows for {who}")
                continue
            iv["trade_date"] = pd.to_datetime(iv["trade_date"])
            nets[tag] = pv(iv, "net_value")
            good = int(nets[tag].notna().sum().sum())
            print(f"  {who}: {good:,} ticker-days")
            del iv
        except Exception as exc:                              # noqa: BLE001
            print(f"  ! {who} unavailable: {exc}")

    adv = vt.rolling(20, min_periods=5).mean()
    mask = uni & (adv >= MIN_ADV) & pct.notna() & close.notna()

    feats = {}
    if "f" in nets:
        feats["f_flow20"] = flow_intensity(nets["f"], vt, 20)
        feats["f_flow60"] = flow_intensity(nets["f"], vt, 60)
    if "i" in nets:
        feats["i_flow20"] = flow_intensity(nets["i"], vt, 20)
    feats["f_hold20"] = zwin(pct - pct.shift(20))
    lo = pct.rolling(250, min_periods=120).min()
    hi = pct.rolling(250, min_periods=120).max()
    feats["f_level"] = (pct - lo) / (hi - lo).where((hi - lo) > 1e-9)
    r1 = close.pct_change(fill_method=None).mask(
        lambda x: x.abs() > 0.5)
    feats["c_mom20"] = close / close.shift(20) - 1.0
    feats["c_mom60"] = close / close.shift(60) - 1.0
    feats["c_size"] = np.log(cap.where(cap > 0))
    feats["c_vol"] = r1.rolling(60, min_periods=30).std()
    feats["c_turn"] = (adv / cap.where(cap > 0))
    for k in ("c_mom20", "c_mom60"):
        feats[k] = feats[k].mask(feats[k].abs() > 3.0)

    fwd = {h: tracker.clean_forward(close, h) for h in HORIZONS}
    mkt = {h: fwd[h].where(mask).mean(axis=1) for h in HORIZONS}
    react = close.shift(-1) / close.shift(1) - 1.0

    A = {k: v.to_numpy() for k, v in feats.items()}
    F = {h: fwd[h].to_numpy() for h in HORIZONS}
    M = {h: mkt[h].to_numpy() for h in HORIZONS}
    R, MK = react.to_numpy(), mask.to_numpy()
    colpos = {t: j for j, t in enumerate(cols)}

    ev = pd.read_parquet(EVENTS)
    ev = ev[~ev["is_correction"]].copy()
    ev["dt"] = pd.to_datetime(ev["rcept_dt"], format="%Y%m%d")
    ev = ev[ev["ticker"].isin(colpos)]
    ev["p"] = idx.searchsorted(ev["dt"].to_numpy())
    ev = ev[(ev["p"] >= 300) & (ev["p"] + max(HORIZONS) + 3 < len(idx))]
    ev = ev.sort_values(["kind", "ticker", "p"])
    ev = ev[~((ev["ticker"] == ev["ticker"].shift())
              & (ev["kind"] == ev["kind"].shift())
              & (ev["p"] - ev["p"].shift() <= 10))]
    print(f"\nevents to place: {len(ev):,}")

    rows = []
    for t_, p_, kind in zip(ev["ticker"].to_numpy(), ev["p"].to_numpy(),
                            ev["kind"].to_numpy()):
        j = colpos[t_]
        if not MK[p_ - 1, j] or not np.isfinite(R[p_, j]):
            continue
        e = p_ + 1
        row = {"ticker": t_, "D": idx[p_], "entry": int(e), "kind": kind,
               "react": float(R[p_, j])}
        for k, arr in A.items():
            row[k] = float(arr[p_ - 1, j])
        for h in HORIZONS:
            raw = F[h][e, j]
            row[f"abn{h}"] = float(raw - M[h][e]) if np.isfinite(raw) else np.nan
        rows.append(row)

    d = pd.DataFrame(rows)
    d["surprise"] = d["react"] - d.groupby(["D", "kind"])["react"].transform("mean")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT, index=False)

    print(f"\n  rows: {len(d):,}   {d['D'].min().date()} .. {d['D'].max().date()}")
    for k, n in d["kind"].value_counts().items():
        print(f"    {k:<12} {n:,}")
    print("\n  coverage (non-null share):")
    for c in sorted(c for c in d.columns
                    if c.startswith(("f_", "i_", "c_", "abn")) or c == "surprise"):
        print(f"    {c:<10} {d[c].notna().mean()*100:5.1f}%")
    print(f"\n  wrote {OUT.relative_to(ROOT)}  "
          f"({OUT.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
