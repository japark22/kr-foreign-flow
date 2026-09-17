#!/usr/bin/env python3
"""Recover the events that carry no institutional trading.

The exchange's investor-flow feed omits a ticker-day row when institutions
did not trade the name. The panel read that omission as unknown and dropped
the event, which cost about a quarter of the sample, concentrated in the
small, volatile names where the conditional tail effect was measured.

Three steps, in order. The script refuses to continue if the first fails.

  1. Rebuild i_flow20 from the raw store with no imputation and check it
     against the column already in the panel.
  2. Validate the imputation per year against an independent source, and
     impute only in the years that clear the declared threshold.
  3. Rebuild the institutional features, emitting no-participation as its
     own state rather than as a low value on a continuum.

Declared in PRE_REGISTRATION.md section 10 before this was run.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from krxflow import features, storage

ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "results" / "event_panel.parquet"
OUT = ROOT / "results" / "event_panel_v2.parquet"
REPORT = ROOT / "results" / "inst_coverage.json"

INST = ["금융투자", "보험", "투신", "사모", "은행", "기타금융", "연기금"]
ZERO_SHARE_MIN = 0.95
RECON_CORR_MIN = 0.999
START = "20100101"


def load_panel_builder():
    """Reuse the panel's own feature definitions rather than restate them."""
    spec = importlib.util.spec_from_file_location("ep", ROOT / "20_event_panel.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ep = load_panel_builder()
    report: dict = {"threshold": ZERO_SHARE_MIN}

    print("loading ownership and market panels ...")
    p = features.load_panels(START, None)
    pct = p["foreign_pct"]
    idx, cols = pct.index, pct.columns

    m = storage.read_range("market", START, None,
                           columns=["trade_date", "ticker", "value_traded"])
    m["trade_date"] = pd.to_datetime(m["trade_date"])

    def pv(df, c):
        return (df.pivot_table(index="trade_date", columns="ticker", values=c,
                               aggfunc="last", observed=True)
                  .sort_index().astype("float64")
                  .reindex(index=idx, columns=cols))

    vt = pv(m, "value_traded")
    del m

    print("loading institutional net buying ...")
    iv = storage.read_range("investor_flow", START, None,
                            columns=["trade_date", "ticker", "investor", "net_value"])
    iv = iv[iv["investor"] == "기관합계"].copy()
    iv["trade_date"] = pd.to_datetime(iv["trade_date"])
    net_raw = pv(iv, "net_value")
    del iv

    # ---- step 1: reconstruct without imputation, check against the panel
    print("\nstep 1  reconstruction check")
    i20_raw = ep.flow_intensity(net_raw, vt, 20)
    ev = pd.read_parquet(PANEL)
    colpos = {t: j for j, t in enumerate(cols)}
    ev = ev[ev["ticker"].isin(colpos)].reset_index(drop=True)
    pos = idx.searchsorted(ev["D"].to_numpy())
    jj = ev["ticker"].map(colpos).to_numpy()
    recon = i20_raw.to_numpy()[pos - 1, jj]

    have = ev["i_flow20"].notna().to_numpy() & np.isfinite(recon)
    c = float(np.corrcoef(ev.loc[have, "i_flow20"], recon[have])[0, 1])
    dmax = float(np.abs(ev.loc[have, "i_flow20"].to_numpy() - recon[have]).max())
    print(f"  overlapping cells {have.sum():,}   corr={c:.6f}   max|diff|={dmax:.3e}")
    report["reconstruction"] = {"cells": int(have.sum()), "corr": c,
                               "max_abs_diff": dmax}
    if c < RECON_CORR_MIN:
        print("  ABORT: reconstruction does not reproduce the panel column")
        REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        return 1
    print("  ok")

    # ---- step 2: validate the imputation per year
    print("\nstep 2  imputation validation")
    files = sorted(glob.glob(str(ROOT / "data" / "investor" / "*.parquet")))
    random.seed(0)
    det = pd.concat([pd.read_parquet(f)
                     for f in random.sample(files, min(400, len(files)))],
                    ignore_index=True)
    det["trade_date"] = pd.to_datetime(det["trade_date"])
    det["inst_sum"] = det[INST].fillna(0).sum(axis=1)

    keep_t = [t for t in net_raw.columns if t in set(det["ticker"])]
    agg = (net_raw[keep_t].stack(future_stack=True).rename("agg").reset_index()
           .rename(columns={"level_0": "trade_date", "level_1": "ticker"}))
    mg = det[["trade_date", "ticker", "inst_sum"]].merge(
        agg.dropna(subset=["agg"]), on=["trade_date", "ticker"],
        how="left", indicator=True)
    miss = mg[mg["_merge"] == "left_only"].assign(
        y=lambda d: d["trade_date"].dt.year)

    by_year = (miss.groupby("y")["inst_sum"]
               .agg(n="size", zero_share=lambda s: float((s == 0).mean())))
    print(by_year.round(4).to_string())
    eligible = sorted(int(y) for y in
                      by_year.index[by_year["zero_share"] >= ZERO_SHARE_MIN])
    checked_lo = int(by_year.index.min())
    pre_years = sorted({int(y) for y in pd.DatetimeIndex(idx).year
                        if int(y) < checked_lo})
    print(f"  eligible, checked        : {eligible}")
    print(f"  imputed, unchecked (pre) : {pre_years}")
    report["validation"] = {str(k): v for k, v in
                            by_year.to_dict(orient="index").items()}
    report["eligible_checked"] = eligible
    report["imputed_unchecked_pre"] = pre_years

    # ---- step 3: rebuild
    print("\nstep 3  rebuild")
    impute_years = sorted(set(eligible) | set(pre_years))
    yr = pd.DatetimeIndex(idx).year
    ok_rows = np.isin(yr, impute_years)

    fillable = (vt.fillna(0) > 0) & net_raw.isna()
    fillable.loc[~ok_rows, :] = False

    i20_filled = ep.flow_intensity(net_raw.mask(fillable, 0.0), vt, 20)
    active = (net_raw.notna() & (net_raw != 0)).astype("float64")
    no_inst = active.rolling(20, min_periods=20).sum() == 0

    ev["i_flow20_v2"] = i20_filled.to_numpy()[pos - 1, jj]
    ev["no_inst20"] = no_inst.to_numpy()[pos - 1, jj]
    ev["i_imputed"] = fillable.to_numpy()[pos - 1, jj]

    was = float(ev["i_flow20"].notna().mean())
    now = float(ev["i_flow20_v2"].notna().mean())
    print(f"  coverage  {was:.3f} -> {now:.3f}   "
          f"events {int(ev['i_flow20'].notna().sum()):,} -> "
          f"{int(ev['i_flow20_v2'].notna().sum()):,}")
    print(f"  no-participation state: {float(ev['no_inst20'].mean()):.3f} of events")

    both = ev["i_flow20"].notna() & ev["i_flow20_v2"].notna()
    d = float((ev.loc[both, "i_flow20"] - ev.loc[both, "i_flow20_v2"]).abs().max())
    print(f"  previously-present cells unchanged: max|diff|={d:.3e}")
    report.update({"coverage_before": was, "coverage_after": now,
                   "no_participation_share": float(ev["no_inst20"].mean()),
                   "unchanged_max_diff": d})

    by = ev.assign(y=ev["D"].dt.year).groupby("y").agg(
        n=("ticker", "size"),
        cov_before=("i_flow20", lambda s: s.notna().mean()),
        cov_after=("i_flow20_v2", lambda s: s.notna().mean()),
        none_share=("no_inst20", "mean"))
    print("\n" + by.round(3).to_string())

    ev.to_parquet(OUT, index=False)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT.relative_to(ROOT)}  and  {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
