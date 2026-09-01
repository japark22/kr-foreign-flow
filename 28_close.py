#!/usr/bin/env python3
"""Step 28: settle the level, and put an error bar on the filter.

    python 28_close.py

TWO QUESTIONS LEFT
------------------
1. Every provisional filer loses to its size-matched peers, and the loss grows
   almost linearly with the holding window: -112bp at twenty sessions, -208 at
   forty, -303 at sixty. That is about five basis points a session for the
   whole event universe, and it is either a real property of companies that
   choose to pre-announce or a fault in how this script handles returns. The
   event panel from step 20 computed its own market-adjusted returns through
   completely separate code. If the two agree, the level is real; if the panel
   says roughly zero, the portfolio code is wrong. Either way it is settled by
   comparison rather than by argument.

2. The filter gain -- quiet beats minus all beats -- is +8, +55 and +76bp per
   position at the three horizons. That is the number the whole project has
   been trying to produce, and it has never had a standard error. It gets one
   here, clustered by reporting season, because two positions opened in the
   same February are not independent observations of anything.

WHY THE LEVEL DOES NOT CONTAMINATE THE FILTER
---------------------------------------------
The filter gain is a difference between two subsets of the same universe on
the same dates. Whatever common level affects all provisional filers cancels
in that subtraction. So question 2 can be answered honestly even if question 1
turns out to be a fault -- which is why it is worth answering both separately
rather than waiting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import tracker

ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "results" / "event_panel.parquet"
RESULTS = ROOT / "results"
MIN_ADV = 1e8
N_BUCKETS = 10
HOLDS = (20, 40, 60)
BASE = "i_flow20"


def matched_excess(ret, cap, ok):
    out = np.full_like(ret, np.nan)
    for i in range(ret.shape[0]):
        m = ok[i] & np.isfinite(ret[i]) & np.isfinite(cap[i])
        if m.sum() < 50:
            continue
        c = cap[i][m]
        edges = np.quantile(c, np.linspace(0, 1, N_BUCKETS + 1)[1:-1])
        b = np.digitize(c, edges)
        r = ret[i][m]
        adj = np.empty_like(r)
        for g in range(N_BUCKETS):
            sel = b == g
            adj[sel] = r[sel] - r[sel].mean() if sel.sum() >= 3 else np.nan
        out[i][np.where(m)[0]] = adj
    return out


def positions(df, exc, n_dates, col, hold):
    """Cumulative matched excess per position, with its season label."""
    v, seas = [], []
    for t_, e, D in zip(df["ticker"], df["entry"], df["D"]):
        j = col.get(t_)
        if j is None:
            continue
        w = exc[int(e) + 1:min(int(e) + 1 + hold, n_dates), j]
        w = w[np.isfinite(w)]
        if len(w) >= hold // 2:
            v.append(float(w.sum()))
            seas.append(D.year * 4 + (D.month - 1) // 3)
    return np.array(v), np.array(seas)


def clustered(v, seas):
    """Mean and t, one observation per reporting season."""
    per = np.array([v[seas == s].mean() for s in np.unique(seas)])
    if len(per) < 8:
        return float(v.mean()), np.nan, len(per)
    return float(per.mean()), tracker._nw_t(per, 1), len(per)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-06-01")
    a = ap.parse_args()
    if not PANEL.exists():
        sys.exit("run 20_event_panel.py first")

    d = pd.read_parquet(PANEL)
    d = d[(d["kind"] == "provisional") & d["surprise"].notna()
          & d[BASE].notna()].copy()

    from krxflow import features, storage
    p = features.load_panels(a.start, None)
    pct, uni = p["foreign_pct"], features.universe_mask(p)
    m = storage.read_range("market", a.start, None,
                           columns=["trade_date", "ticker", "close",
                                    "value_traded", "market_cap"])
    m["trade_date"] = pd.to_datetime(m["trade_date"])

    def pv(c):
        return (m.pivot_table(index="trade_date", columns="ticker", values=c,
                              aggfunc="last", observed=True)
                  .sort_index().astype("float64")
                  .reindex(index=pct.index, columns=pct.columns))

    close, vt, cap = pv("close"), pv("value_traded"), pv("market_cap")
    del m
    adv = vt.rolling(20, min_periods=5).mean()
    ok = (uni & (adv >= MIN_ADV) & close.notna()).to_numpy()
    r = close.pct_change(fill_method=None).mask(lambda x: x.abs() > 0.5).to_numpy()
    print("building size-matched excess ...")
    exc = matched_excess(r, cap.to_numpy(), ok)
    n_dates = len(pct.index)
    col = {t: j for j, t in enumerate(pct.columns)}

    d["sur_r"] = d.groupby("D")["surprise"].rank(pct=True)
    d["base_r"] = d.groupby("D")[BASE].rank(pct=True)
    beat, quiet = d["sur_r"] >= 0.6, d["base_r"] <= 0.4
    crowd = d["base_r"] >= 0.6

    print("\n  1. THE LEVEL -- two independent code paths on the same events")
    print("     hold |  size-matched (this)  |  panel abn (step 20, market-adj)")
    out = {"level": {}, "filter": {}}
    for h in HOLDS:
        v, s = positions(d[beat], exc, n_dates, col, h)
        mn, t, ns = clustered(v, s)
        panel_col = f"abn{h}"
        if panel_col in d.columns:
            sub = d[beat & d[panel_col].notna()]
            pv_ = sub[panel_col].to_numpy()
            ps = (sub["D"].dt.year * 4 + (sub["D"].dt.month - 1) // 3).to_numpy()
            pmn, pt, _ = clustered(pv_, ps)
            print(f"     {h:>4} | {mn*1e4:+9.1f} bp  t {t:+5.2f} | "
                  f"{pmn*1e4:+9.1f} bp  t {pt:+5.2f}")
            out["level"][str(h)] = {"matched_bp": mn * 1e4, "matched_t": t,
                                    "panel_bp": pmn * 1e4, "panel_t": pt}
        else:
            print(f"     {h:>4} | {mn*1e4:+9.1f} bp  t {t:+5.2f} |  (no panel column)")
    lv = out["level"].get("60", {})
    if lv:
        gap = abs(lv["matched_bp"] - lv["panel_bp"])
        print(f"\n     the two differ by {gap:.0f} bp at 60 sessions.")
        print("     a small gap means the level is a real property of these")
        print("     issuers; a large one means this script's return handling")
        print("     disagrees with the panel and the level is an artefact.")
        out["level_gap_60_bp"] = gap

    print("\n  2. THE FILTER -- quiet beats minus all beats, per position,")
    print("     clustered by reporting season")
    print("     hold |  quiet beats |   all beats  |    gain   t     seasons")
    for h in HOLDS:
        vq, sq = positions(d[beat & quiet], exc, n_dates, col, h)
        va, sa = positions(d[beat], exc, n_dates, col, h)
        mq = {s: vq[sq == s].mean() for s in np.unique(sq)}
        ma = {s: va[sa == s].mean() for s in np.unique(sa)}
        common = sorted(set(mq) & set(ma))
        diff = np.array([mq[s] - ma[s] for s in common])
        t = tracker._nw_t(diff, 1) if len(diff) >= 8 else np.nan
        qm, _, _ = clustered(vq, sq)
        am, _, _ = clustered(va, sa)
        print(f"     {h:>4} | {qm*1e4:+9.1f} bp | {am*1e4:+9.1f} bp | "
              f"{diff.mean()*1e4:+8.1f} {t:+6.2f}  ({len(common)})")
        half = len(common) // 2
        d1, d2 = diff[:half], diff[half:]
        print(f"          halves: {d1.mean()*1e4:+.1f} bp / "
              f"{d2.mean()*1e4:+.1f} bp")
        out["filter"][str(h)] = {"quiet_bp": qm * 1e4, "all_bp": am * 1e4,
                                 "gain_bp": float(diff.mean() * 1e4), "t": t,
                                 "seasons": len(common),
                                 "halves": [float(d1.mean() * 1e4),
                                            float(d2.mean() * 1e4)]}

    print("\n  3. the mirror: crowded beats minus all beats (should be NEGATIVE)")
    for h in HOLDS:
        vc, sc = positions(d[beat & crowd], exc, n_dates, col, h)
        va, sa = positions(d[beat], exc, n_dates, col, h)
        mc = {s: vc[sc == s].mean() for s in np.unique(sc)}
        ma = {s: va[sa == s].mean() for s in np.unique(sa)}
        common = sorted(set(mc) & set(ma))
        diff = np.array([mc[s] - ma[s] for s in common])
        t = tracker._nw_t(diff, 1) if len(diff) >= 8 else np.nan
        print(f"     {h:>4} | {diff.mean()*1e4:+8.1f} bp  t {t:+6.2f}")
        out.setdefault("mirror", {})[str(h)] = {"bp": float(diff.mean() * 1e4),
                                                "t": t}

    g = out["filter"]
    ok_all = all(g[str(h)]["gain_bp"] > 0 for h in HOLDS)
    ok_sig = sum(1 for h in HOLDS if abs(g[str(h)].get("t", 0)) >= 2)
    ok_halves = all(min(g[str(h)]["halves"]) > 0 for h in HOLDS)
    print(f"\n  gain positive at all holds: {ok_all}")
    print(f"  positive in both halves at all holds: {ok_halves}")
    print(f"  holds with |t| >= 2: {ok_sig} of {len(HOLDS)}")
    out["verdict"] = ("FILTER CONFIRMED" if (ok_all and ok_halves and ok_sig >= 2)
                      else "FILTER DIRECTIONAL BUT NOT CONFIRMED")
    print(f"\n  verdict: {out['verdict']}")

    RESULTS.mkdir(exist_ok=True)
    dest = RESULTS / "close.json"
    dest.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
