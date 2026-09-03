"""What the filter is worth to a book, in the language a book uses.

A coefficient of -46 bp per standard deviation does not answer the first
question anyone asks. This does, but it has to answer it honestly, and the
honest framing is not a standalone strategy. Holding sixty trading days means
turning the book over four times a year; at fifty basis points a round trip
that is two per cent of cost a year, more than the whole effect. Anything
presented as a standalone strategy here would be arithmetic theatre.

The effect is worth something as a filter on a book that already exists. Both
books -- the one that buys every strong surprise and the one that skips the
crowded ones -- turn over at the same rate and pay the same costs, so the cost
cancels in the difference and what remains is the filter's own contribution.
That is the number to report.

Returns are the sixty-day benchmark-adjusted returns already in the panel,
aggregated to reporting-season cohorts. Korean filings arrive in four bursts a
year, so a season cohort is very nearly a non-overlapping holding period, and
about thirty-four of them is the whole sample. Every ratio below is therefore
computed on thirty-four observations and should be read with that in mind:
these are informative magnitudes, not precise ones.

    python 43_book.py                    # top and bottom 30 per cent
    python 43_book.py --top 0.8 --crowd 0.8
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("results/decompose_panel.parquet")
OUT = Path("results/book.json")
BASE = "i_flow20"
PER_YEAR = 4.0                     # sixty trading days is about a quarter

spec = importlib.util.spec_from_file_location("bfm", "21_baseline_fm.py")
bfm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bfm)


def stats(r: np.ndarray, cost_bp: float = 0.0) -> dict:
    """Annualise a series of quarterly holding-period returns."""
    net = r - cost_bp / 1e4
    mean = float(net.mean() * PER_YEAR)
    vol = float(net.std(ddof=1) * np.sqrt(PER_YEAR))
    cum = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(cum)
    return {"ann_return_pct": mean * 100,
            "ann_vol_pct": vol * 100,
            "info_ratio": mean / vol if vol > 1e-12 else float("nan"),
            "hit_rate": float((net > 0).mean()),
            "worst_season_pct": float(net.min() * 100),
            "max_drawdown_pct": float((cum / peak - 1.0).min() * 100),
            "seasons": int(len(net))}


def line(tag: str, s: dict) -> None:
    print(f"  {tag:<26}{s['ann_return_pct']:+7.2f}%  vol {s['ann_vol_pct']:5.2f}%"
          f"   IR {s['info_ratio']:+5.2f}   hit {s['hit_rate']:.0%}"
          f"   worst {s['worst_season_pct']:+6.2f}%"
          f"   maxDD {s['max_drawdown_pct']:+6.2f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=float, default=0.7,
                    help="surprise percentile above which a name is bought")
    ap.add_argument("--crowd", type=float, default=0.7,
                    help="crowding percentile above which a name is skipped")
    a = ap.parse_args()

    d = pd.read_parquet(CACHE)
    d["D"] = pd.to_datetime(d["D"])
    s = d[d["kind"] == "provisional"].dropna(
        subset=["abn60", "surprise", BASE, "c_size", "c_turn"]).copy()
    s["season"] = s["D"].dt.year * 4 + (s["D"].dt.month - 1) // 3
    g = s.groupby("D")
    s["rs"] = g["surprise"].rank(pct=True)
    s["rc"] = g[BASE].rank(pct=True)

    naive = s[s["rs"] >= a.top]
    filt = naive[naive["rc"] < a.crowd]
    skipped = naive[naive["rc"] >= a.crowd]

    def cohort(x):
        return x.groupby("season")["abn60"].mean()

    cn, cf, ck = cohort(naive), cohort(filt), cohort(skipped)
    common = cn.index.intersection(cf.index).intersection(ck.index)
    cn, cf, ck = cn[common].to_numpy(), cf[common].to_numpy(), ck[common].to_numpy()

    print(f"provisional events {len(s):,}")
    print(f"  buy above surprise pct {a.top:.2f}, skip above crowding pct "
          f"{a.crowd:.2f}")
    print(f"  positions: every strong surprise {len(naive):,}   "
          f"kept {len(filt):,}   skipped {len(skipped):,}")
    print(f"  season cohorts {len(common)}   "
          f"median names per cohort: all {int(naive.groupby('season').size().median())}"
          f", kept {int(filt.groupby('season').size().median())}\n")

    print("gross, before costs (returns are already benchmark-adjusted)")
    R = {"params": {"top": a.top, "crowd": a.crowd},
         "counts": {"all": int(len(naive)), "kept": int(len(filt)),
                    "skipped": int(len(skipped)), "cohorts": int(len(common))},
         "gross": {}}
    for tag, r, key in (("every strong surprise", cn, "all"),
                        ("crowded skipped", cf, "filtered"),
                        ("the skipped names", ck, "skipped")):
        R["gross"][key] = stats(r)
        line(tag, R["gross"][key])

    print("\nthe filter's own contribution (costs cancel: same turnover)")
    diff = cf - cn
    t = float(bfm.tracker._nw_t(diff, 1))
    R["contribution"] = {
        "per_position_bp": float(diff.mean() * 1e4),
        "ann_pct": float(diff.mean() * PER_YEAR * 100),
        "t": t,
        "positive_seasons": float((diff > 0).mean()),
        "seasons": int(len(diff))}
    c = R["contribution"]
    print(f"  {'filtered - all':<26}{c['per_position_bp']:+7.1f} bp per position"
          f"   {c['ann_pct']:+6.2f}% a year   t {c['t']:+5.2f}"
          f"   positive in {c['positive_seasons']:.0%} of seasons")

    print("\nwhat the same books look like net of costs, four round trips a year")
    R["net"] = {}
    for cost in (0, 25, 50, 75):
        R["net"][str(cost)] = {"all": stats(cn, cost),
                               "filtered": stats(cf, cost)}
        n, f = R["net"][str(cost)]["all"], R["net"][str(cost)]["filtered"]
        print(f"  {cost:>3} bp/round trip   every surprise "
              f"{n['ann_return_pct']:+7.2f}%   crowded skipped "
              f"{f['ann_return_pct']:+7.2f}%   gap "
              f"{f['ann_return_pct'] - n['ann_return_pct']:+5.2f}%")

    print("\ncapacity of the names being bought")
    cap = np.exp(filt["c_size"].to_numpy())
    R["capacity"] = {
        "median_market_cap_krw_bn": float(np.median(cap) / 1e9),
        "p25_market_cap_krw_bn": float(np.percentile(cap, 25) / 1e9),
        "median_turnover_ratio": float(filt["c_turn"].median()),
        "median_names_per_cohort": int(filt.groupby("season").size().median())}
    k = R["capacity"]
    print(f"  median market cap        {k['median_market_cap_krw_bn']:,.0f} bn KRW"
          f"   (25th pct {k['p25_market_cap_krw_bn']:,.0f} bn)")
    print(f"  median daily turnover    {k['median_turnover_ratio']:.4f}"
          f" of market cap")
    print(f"  names held per cohort    {k['median_names_per_cohort']}")

    print("\nreading")
    if c["t"] > 2.0:
        verdict = ("the filter adds a measurable amount to an existing "
                   "surprise book at no extra cost")
    elif c["per_position_bp"] > 0:
        verdict = ("the filter points the right way but its contribution is "
                   "not separable from noise at this sample size")
    else:
        verdict = "the filter does not improve the book"
    print(f"  {verdict}")
    print("  Note: shorting the skipped names is not available for much of "
          "this period -- Korea banned short selling in 2020-21 and 2023-25 "
          "-- so the usable form is exclusion, not a long-short book.")
    R["verdict"] = verdict

    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
