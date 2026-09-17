#!/usr/bin/env python3
"""Accumulated position, not twenty-day flow.

Every feature tested so far is a flow measured over twenty days. Crowding is
not a flow; it is a position. Korea does not publish institutional holdings,
but integrating daily net buying over a long window reconstructs one:

    ipos_N = sum of institutional net value over N days / market cap

That is the same shape of quantity a quarterly holdings disclosure gives,
which makes it the variable a holdings-based method would use, and the one
an exchange that publishes actual custodian holdings could validate against.

The imputation rule and its eligible years are read from the audit written
by 55_inst_coverage.py rather than restated here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from krxflow import features, storage

ROOT = Path(__file__).resolve().parent
V3 = ROOT / "results" / "event_panel_v3.parquet"
V4 = ROOT / "results" / "event_panel_v4.parquet"
AUDIT = ROOT / "results" / "inst_coverage.json"
START = "20100101"
WINDOWS = (60, 120, 250)


def main() -> int:
    audit = json.loads(AUDIT.read_text())
    ok_years = sorted(set(audit["eligible_checked"]) |
                      set(audit["imputed_unchecked_pre"]))
    print("imputation years:", ok_years[0], "..", ok_years[-1],
          f"({len(ok_years)} years)")

    p = features.load_panels(START, None)
    pct = p["foreign_pct"]
    idx, cols = pct.index, pct.columns

    m = storage.read_range("market", START, None,
                           columns=["trade_date", "ticker", "value_traded",
                                    "market_cap"])
    m["trade_date"] = pd.to_datetime(m["trade_date"])

    def pv(df, c):
        return (df.pivot_table(index="trade_date", columns="ticker", values=c,
                               aggfunc="last", observed=True)
                  .sort_index().astype("float64")
                  .reindex(index=idx, columns=cols))

    vt, cap = pv(m, "value_traded"), pv(m, "market_cap")
    del m

    iv = storage.read_range("investor_flow", START, None,
                            columns=["trade_date", "ticker", "investor", "net_value"])
    iv = iv[iv["investor"] == "기관합계"].copy()
    iv["trade_date"] = pd.to_datetime(iv["trade_date"])
    net = pv(iv, "net_value")
    del iv

    fill = (vt.fillna(0) > 0) & net.isna()
    fill.loc[~np.isin(pd.DatetimeIndex(idx).year, ok_years), :] = False
    net = net.mask(fill, 0.0)

    capd = cap.where(cap > 0)
    extra = {}
    for w in WINDOWS:
        s = net.rolling(w, min_periods=int(w * 0.8)).sum()
        extra[f"ipos{w}"] = s / capd
        extra[f"ipos{w}_adv"] = s / (vt.rolling(w, min_periods=int(w * 0.8))
                                     .mean() * w).replace(0, np.nan)

    ev = pd.read_parquet(V3)
    colpos = {t: j for j, t in enumerate(cols)}
    ev = ev[ev["ticker"].isin(colpos)].reset_index(drop=True)
    pos = idx.searchsorted(ev["D"].to_numpy())
    jj = ev["ticker"].map(colpos).to_numpy()
    for k, panel in extra.items():
        ev[k] = panel.to_numpy()[pos - 1, jj]
        print(f"  {k:<14} coverage={ev[k].notna().mean():.3f}  "
              f"sd={ev[k].std():.4f}  "
              f"corr(mom20)={ev[[k, 'c_mom20']].corr().iloc[0, 1]:+.3f}  "
              f"corr(size)={ev[[k, 'c_size']].corr().iloc[0, 1]:+.3f}")

    ev.to_parquet(V4, index=False)
    print(f"\nwrote {V4.relative_to(ROOT)}  rows={len(ev):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
