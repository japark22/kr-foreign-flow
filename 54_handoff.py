"""The handoff: the signal as a daily panel, an event-level verification set,
and a one-page specification. Built so the receiving side can reproduce the
Korea numbers before comparing them with anything else -- a comparison that
starts from a re-implemented signal is a comparison of two implementations,
not of two markets.

Writes handoff/kr_positioning_daily.parquet, handoff/kr_positioning_events.parquet
and handoff/SIGNAL_SPEC.md. The parquets stay out of version control; the
specification is committed.

    python 54_handoff.py
"""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("handoff")
R = Path("results")
START = "20100101"


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def J(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> int:
    from krxflow import storage
    ep = load_mod("ep", "20_event_panel.py")          # the exact feature code
    OUT.mkdir(exist_ok=True)

    # ---------------------------------------------------------- daily panel
    print("market ...")
    m = storage.read_range("market", START, None,
                           columns=["trade_date", "ticker", "value_traded"])
    m["trade_date"] = pd.to_datetime(m["trade_date"])
    vt = (m.pivot_table(index="trade_date", columns="ticker", values="value_traded",
                        aggfunc="last", observed=True).sort_index().astype("float64"))
    del m
    idx, cols = vt.index, vt.columns
    print(f"  {len(idx):,} days x {len(cols):,} tickers")

    print("investor flow ...")
    iv = storage.read_range("investor_flow", START, None,
                            columns=["trade_date", "ticker", "investor", "net_value"])
    iv["trade_date"] = pd.to_datetime(iv["trade_date"])
    panels = {}
    for who, tag in (("기관합계", "inst_flow20"), ("외국인", "foreign_flow20")):
        sub = iv[iv["investor"] == who]
        net = (sub.pivot_table(index="trade_date", columns="ticker", values="net_value",
                               aggfunc="last", observed=True).sort_index()
                  .astype("float64").reindex(index=idx, columns=cols))
        panels[tag] = ep.flow_intensity(net, vt, 20).astype("float32")
        print(f"  {tag}: {int(panels[tag].notna().sum().sum()):,} ticker-days")
        del net
    del iv, vt

    long = (pd.concat({k: v.stack(future_stack=True) for k, v in panels.items()}, axis=1)
              .dropna(how="all").reset_index())
    long.columns = ["trade_date", "ticker", "inst_flow20", "foreign_flow20"]
    long = long[long["trade_date"] >= "2011-01-01"]
    daily_path = OUT / "kr_positioning_daily.parquet"
    long.to_parquet(daily_path, index=False)
    print(f"wrote {daily_path}  rows {len(long):,}  "
          f"{daily_path.stat().st_size / 1e6:.0f} MB")
    del long, panels

    # ---------------------------------------------------------- event set
    ev = pd.read_parquet(R / "event_panel.parquet")
    ev["D"] = pd.to_datetime(ev["D"])
    ev = ev.dropna(subset=["abn60", "surprise"]).copy()
    g = ev.groupby("D")
    ev["surprise_quintile"] = np.clip((g["surprise"].rank(pct=True) * 5).astype(int), 0, 4) + 1
    has = ev["i_flow20"].notna()
    ev.loc[has, "crowding_tercile"] = np.clip(
        (ev[has].groupby("D")["i_flow20"].rank(pct=True) * 3).astype(int), 0, 2) + 1
    keep = ["ticker", "D", "kind", "surprise", "i_flow20", "f_flow20",
            "surprise_quintile", "crowding_tercile", "abn5", "abn20", "abn60",
            "c_mom20", "c_mom60", "c_size", "c_vol", "c_turn"]
    ev = ev[keep].rename(columns={"D": "announcement_date",
                                  "i_flow20": "inst_flow20",
                                  "f_flow20": "foreign_flow20"})
    ev_path = OUT / "kr_positioning_events.parquet"
    ev.to_parquet(ev_path, index=False)
    print(f"wrote {ev_path}  rows {len(ev):,}")

    # ---------------------------------------------------------- spec
    ext, path, base, fin = J("extended.json"), J("path.json"), J("baseline.json"), J("final.json")
    W = ext.get("windows", {})
    full = W.get("full 2011-2026", {})
    foreign = ext.get("foreign", {}).get("f_flow20, abn60", {})
    sp = path.get("spread", {})
    bp = base.get("provisional", {}); ba = base.get("all", {})
    p10 = bp.get("p10", {}); p10a = ba.get("p10", {})
    est = fin.get("estimator", {})
    n_prov = int((ev["kind"] == "provisional").sum())

    spec = f"""# Korea pre-announcement positioning -- signal specification

Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC from the result files. One page. The two parquet files beside this document carry the signal (daily) and the verification set (per announcement).

## What the signal is

`inst_flow20`: net buying by domestic institutions (exchange category 기관합계, institutions in total) summed over the trailing 20 trading days, divided by value traded over the same 20 days, then standardised against the stock's own trailing 250-day history of that ratio (mean and standard deviation computed on days t-250..t-1, minimum 120 observations). Positive means institutions have been net buyers relative to the stock's own norm. `foreign_flow20` is the identical construction on the foreign-investor category (외국인, foreign investors) and is supplied as the comparison series that carries no forward information.

Construction code: `flow_intensity()` and `zwin()` in `20_event_panel.py`. Inputs are the exchange's daily investor-category net trading values and daily value traded. No look-ahead: every quantity at date t uses data through t.

## Files

| file | grain | rows | columns |
| --- | --- | --- | --- |
| `kr_positioning_daily.parquet` | ticker x trading day, 2011-01 onward | one row per ticker-day with at least one value | trade_date, ticker, inst_flow20, foreign_flow20 |
| `kr_positioning_events.parquet` | one row per earnings filing | {len(ev):,} ({n_prov:,} provisional) | ticker, announcement_date, kind, surprise, inst_flow20, foreign_flow20, surprise_quintile, crowding_tercile, abn5/20/60, controls |

`announcement_date` is the filing date; the signal is read as of that date (it uses the 20 days before). `surprise` is the announcement-day benchmark-adjusted return. `abn60` is the 60-trading-day return net of the equal-weight market starting the day after the filing, winsorised 1/99 within the day. `surprise_quintile` and `crowding_tercile` are ranks within the announcement day (1 = lowest).

## How it was estimated here

{est.get('weighting', 'one vote per event')}; regressors rank-standardised within the announcement day; controls {', '.join(fin.get('estimator', {}).get('regressors', '').split(' + ')[1:-1]) or 'c_mom20, c_mom60, c_size, c_vol, c_turn'}; standard errors clustered on the reporting season (year x quarter). Days with fewer than 5 usable filings dropped. Multiple specifications controlled with permutation family bars; portfolio contrasts checked against 200-draw placebo bands.

## Numbers to reproduce before comparing

| quantity | value |
| --- | --- |
| provisional filings, flow quintile 5 minus 1, cumulative return day -20..-1 | {sp.get('pre', {}).get('bp', 0):+,.0f} bp, t {sp.get('pre', {}).get('t', 0):+.1f} |
| same quintiles, day +1..+60 | {sp.get('post', {}).get('bp', 0):+,.0f} bp, t {sp.get('post', {}).get('t', 0):+.2f} |
| coefficient of inst_flow20 on abn60, provisional, full controls | {full.get('bp', 0):+.1f} bp/SD, t {full.get('t', 0):+.2f}, {full.get('events', 0):,} events, {full.get('seasons', 0)} seasons |
| same for foreign_flow20 | {foreign.get('bp', 0):+.1f} bp/SD, t {foreign.get('t', 0):+.2f} |
| top surprise quintile: 10th percentile of abn60, crowding tercile 3 minus 1, provisional | {p10.get('diff', 0) * 1e4:+,.0f} bp, t {p10.get('t', 0):+.2f}; placebo 5-95% [{p10.get('placebo_lo', 0) * 1e4:+,.0f}, {p10.get('placebo_hi', 0) * 1e4:+,.0f}] |
| same, all filings | {p10a.get('diff', 0) * 1e4:+,.0f} bp, t {p10a.get('t', 0):+.2f}; placebo [{p10a.get('placebo_lo', 0) * 1e4:+,.0f}, {p10a.get('placebo_hi', 0) * 1e4:+,.0f}] |

If the first two rows do not reproduce, the join or the window is off. If they do and the rest does not, the estimator differs, and the difference is the thing to discuss.

## What to test, and what not to expect

The signal is a record of the recent price move, not a forecast of it (row 1 against row 2). Tests on the conditional mean of post-announcement return will show little; they did here. The information, such as it is, sits in the lower tail of the conditional distribution: within the strongest surprises, crowded names have the same mean and hit rate as uncrowded names and a worse 10th percentile, concentrated in volatile names. Recommended comparison statistic: the conditional 5th and 10th percentiles of the post-announcement return by crowding tercile within surprise quintile, alongside the conditional mean.

## Resolution caveat for a comparison with quarterly holdings data

This is a daily flow. A quarterly holdings level (13F-type) is a different object: a stock of positions observed with a lag, not a 20-day change observed the same day. An identical methodology across the two requires a choice. Either coarsen this signal to quarter-end (for example, the cumulative institutional net buying over the quarter, scaled by quarterly value traded, read at quarter end) for a like-for-like test, or run each series in its native resolution and treat agreement in the tail statistic as the comparison. Running both is recommended; the coarsened version can be built from the daily file directly.

## Known limits

Single market, no holdout. The mean coefficient depends on the momentum controls (-25.1 at t -1.67 without them). The tail result rests on one pre-declared test plus robustness checks by half and by volatility tercile. Short-selling was banned 2020-03-16..2021-05-02 and 2023-11-06..2025-03-30; any short-interest overlay for Korea is missing for those windows.
"""
    (OUT / "SIGNAL_SPEC.md").write_text(spec, encoding="utf-8")
    print(f"wrote {OUT / 'SIGNAL_SPEC.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
