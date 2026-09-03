"""The picture the whole argument rests on: what happens before and after.

Every claim in this project has been a number at one horizon. "Institutions
crowd in beforehand and the position unwinds afterwards" is a statement about
a path, and the path has never been looked at. It is possible that the pre-
event window shows no run-up at all, in which case "unwind" is the wrong word
and the story needs rewriting even though the coefficient stands.

So: daily benchmark-adjusted returns in event time, from twenty trading days
before the announcement to sixty after, averaged inside quintiles of pre-event
institutional buying. The reversal, if it exists, is the crowded line rising
into the event and falling out of it while the quiet line does not.

Before any of that is reported the reconstruction is checked against the
panel's own abn60: same events, same window, and if the two disagree the run
stops rather than producing a chart of something else.

    python 46_event_path.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path("results/event_panel.parquet")
OUT = Path("results/path.json")
BASE, PRE, POST, NQ = "i_flow20", 20, 60, 5
START = "20100101"

spec = importlib.util.spec_from_file_location("bfm", "21_baseline_fm.py")
bfm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bfm)


def main() -> int:
    from krxflow import storage

    panel = pd.read_parquet(PANEL)
    panel["D"] = pd.to_datetime(panel["D"])
    prov = panel[panel["kind"] == "provisional"].dropna(
        subset=["abn60", BASE]).copy()
    print(f"provisional with a baseline: {len(prov):,}")

    print("loading closes ...")
    m = storage.read_range("market", START, None,
                           columns=["trade_date", "ticker", "close"])
    m["trade_date"] = pd.to_datetime(m["trade_date"])
    close = (m.pivot_table(index="trade_date", columns="ticker",
                           values="close", aggfunc="last", observed=True)
               .sort_index().astype("float64"))
    del m
    idx, cols = close.index, close.columns
    print(f"  {len(idx):,} days x {len(cols):,} tickers")

    # Daily returns, with the same absurd-move mask the panel uses, then
    # benchmark-adjusted by the equal-weight cross-section of that day.
    ret = close.pct_change(fill_method=None)
    ret = ret.mask(ret.abs() > 0.5)
    bench = ret.mean(axis=1)
    ex = ret.sub(bench, axis=0).to_numpy()
    del ret, close

    rp = idx.get_indexer(prov["D"])
    cp = cols.get_indexer(prov["ticker"])
    keep = (rp >= PRE) & (cp >= 0) & (rp + POST < len(idx))
    prov, rp, cp = prov[keep], rp[keep], cp[keep]
    print(f"  events with a complete window: {len(prov):,}")

    # Event-time matrix: rows are events, columns are day -PRE .. +POST.
    offs = np.arange(-PRE, POST + 1)
    rows = rp[:, None] + offs[None, :]
    path = ex[rows, cp[:, None]]
    print(f"  path matrix {path.shape}   missing {np.isnan(path).mean():.3f}")

    # --- reconstruction check against the panel, both plausible alignments
    print("\nchecking the reconstruction against the panel's abn60")
    ref = prov["abn60"].to_numpy(dtype=float)
    best = None
    for first in (0, 1):
        seg = path[:, PRE + first: PRE + first + POST]
        car = np.nansum(seg, axis=1)
        ok = np.isfinite(ref) & np.isfinite(car)
        c = float(np.corrcoef(car[ok], ref[ok])[0, 1])
        diff = float(np.mean(car[ok] - ref[ok]) * 1e4)
        print(f"  start at day {first:+d}   corr {c:.3f}   "
              f"mean difference {diff:+7.1f} bp")
        if best is None or c > best[1]:
            best = (first, c, diff)
    first, corr, diff = best
    if corr < 0.90 or abs(diff) > 100:
        raise SystemExit("reconstruction does not match the panel -- stopping "
                         "rather than charting a different quantity")
    print(f"  using day {first:+d} as the first forward day; "
          f"corr {corr:.3f}, difference {diff:+.1f} bp -- accepted")

    # --- quintiles of pre-event institutional buying, within the event day
    q = prov.groupby("D")[BASE].rank(pct=True)
    prov["q"] = np.clip((q * NQ).astype(int), 0, NQ - 1)
    prov["season"] = prov["D"].dt.year * 4 + (prov["D"].dt.month - 1) // 3

    curves = {}
    for k in range(NQ):
        sel = (prov["q"] == k).to_numpy()
        mean_daily = np.nanmean(path[sel], axis=0)
        curves[k] = {
            "car_bp": (np.nancumsum(mean_daily) * 1e4).tolist(),
            "events": int(sel.sum())}
        pre = float(np.nansum(mean_daily[:PRE]) * 1e4)
        post = float(np.nansum(mean_daily[PRE + first:]) * 1e4)
        print(f"  quintile {k + 1}   events {sel.sum():,}   "
              f"run-up {pre:+7.1f} bp   after {post:+7.1f} bp")

    # --- the spread, with an error bar that respects the season clustering
    top = (prov["q"] == NQ - 1).to_numpy()
    bot = (prov["q"] == 0).to_numpy()
    car_all = np.nansum(path[:, PRE + first:], axis=1)
    pre_all = np.nansum(path[:, :PRE], axis=1)
    sea = prov["season"].to_numpy()

    def spread(v):
        rows = []
        for s in np.unique(sea):
            a, b = v[top & (sea == s)], v[bot & (sea == s)]
            if len(a) and len(b):
                rows.append(a.mean() - b.mean())
        r = np.array(rows)
        return float(r.mean() * 1e4), float(bfm.tracker._nw_t(r, 1)), len(r)

    pb, pt, pn = spread(pre_all)
    ab, at, an = spread(car_all)
    print(f"\n  top minus bottom quintile, {pn} seasons")
    print(f"    run-up, day -20 to -1      {pb:+8.1f} bp   t {pt:+5.2f}")
    print(f"    after,  day {first:+d} to +{POST}      {ab:+8.1f} bp   t {at:+5.2f}")

    reversal = pb > 0 and ab < 0
    print("\n  " + ("run-up then reversal: the description holds"
                    if reversal else
                    "no run-up before the event -- 'unwind' is the wrong word "
                    "for this and the wording has to change"))

    OUT.write_text(json.dumps({
        "offsets": offs.tolist(), "first_forward_day": first,
        "quintiles": curves, "n_quintiles": NQ,
        "check": {"corr": corr, "mean_diff_bp": diff},
        "spread": {"pre": {"bp": pb, "t": pt},
                   "post": {"bp": ab, "t": at}, "seasons": pn},
        "reversal": bool(reversal),
        "events": int(len(prov)),
        "window": [str(prov["D"].min().date()), str(prov["D"].max().date())],
    }, ensure_ascii=False, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
