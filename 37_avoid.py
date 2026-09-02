"""Does 연기금 crowding beat 기관합계 crowding as an avoidance signal?

The regression cannot separate the two -- put both in and neither survives.
The portfolio contrast says otherwise: crowded beats read -96.7 bp against
the pension baseline and -42.5 against the aggregate. Those two numbers were
computed on different samples, so the gap may be nothing at all.

This puts them on one sample and tests the difference directly. The paired
season differences are the unit: for each reporting season, the crowded-minus
-all contrast is computed under both baselines and subtracted. A placebo
baseline -- a random ranking, same dates, same counts -- is carried alongside
to show what the machinery reads when there is nothing there.

Nothing here is a new search. One comparison, declared: if the pension
baseline is genuinely sharper, the paired difference is negative with |t| > 2
and holds in both halves. Otherwise the aggregate is all we ever had.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("results/decompose_panel.parquet")
OUT = Path("results/avoid.json")
PENSION, AGG = "x_연기금", "i_flow20"
TOP, BOTTOM = 0.7, 0.3

spec = importlib.util.spec_from_file_location("bfm", "21_baseline_fm.py")
bfm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bfm)


def season(d):
    d = pd.to_datetime(d)
    return d.dt.year * 4 + (d.dt.month - 1) // 3


def seasonal(sub: pd.DataFrame, col: str, side: str) -> pd.Series:
    """Mean abn60 of the beats sitting on one side of a baseline, per season,
    minus the mean of all beats that season."""
    r = sub.groupby("D")[col].rank(pct=True)
    sel = sub[r >= TOP] if side == "crowded" else sub[r <= BOTTOM]
    a = sel.groupby("season")["abn60"].mean()
    b = sub.groupby("season")["abn60"].mean()
    common = a.index.intersection(b.index)
    return (a[common] - b[common]) * 1e4


def report(tag: str, s: pd.Series) -> dict:
    v = s.to_numpy()
    t = float(bfm.tracker._nw_t(v, 1))
    print(f"  {tag:<38}{v.mean():+8.1f} bp   t {t:+5.2f}   seasons {len(v)}")
    return {"bp": float(v.mean()), "t": t, "seasons": int(len(v))}


d = pd.read_parquet(CACHE)
d["D"] = pd.to_datetime(d["D"])
sub = d[d["kind"] == "provisional"].dropna(
    subset=["abn60", "surprise", PENSION, AGG]).copy()
sub["season"] = season(sub["D"])
sub["rs"] = sub.groupby("D")["surprise"].rank(pct=True)
beats = sub[sub["rs"] >= TOP].copy()

rng = np.random.default_rng(20260902)
beats["x_placebo"] = rng.standard_normal(len(beats))

print(f"one sample: {len(sub):,} provisional events, "
      f"{len(beats):,} beats, {beats['season'].nunique()} seasons\n")

out = {}
print("crowded beats - all beats")
series = {}
for tag, col in (("연기금", PENSION), ("기관합계", AGG),
                 ("placebo (random ranking)", "x_placebo")):
    series[tag] = seasonal(beats, col, "crowded")
    out[f"crowded/{tag}"] = report(f"  {tag}", series[tag])

print("\nquiet beats - all beats")
for tag, col in (("연기금", PENSION), ("기관합계", AGG),
                 ("placebo (random ranking)", "x_placebo")):
    out[f"quiet/{tag}"] = report(f"  {tag}", seasonal(beats, col, "quiet"))

print("\npaired difference, season by season (the actual question)")
diff = series["연기금"] - series["기관합계"]
out["pension_minus_aggregate"] = report("  연기금 - 기관합계 (crowded)", diff)

print("\n  split halves of that difference")
cut = int(np.median(diff.index.to_numpy()))
for tag, part in (("first half", diff[diff.index <= cut]),
                  ("second half", diff[diff.index > cut])):
    out[f"paired/{tag}"] = report(f"    {tag}", part)

verdict = ("연기금 is the sharper avoidance signal"
           if abs(out["pension_minus_aggregate"]["t"]) > 2.0
           and out["pension_minus_aggregate"]["bp"] < 0
           and out["paired/first half"]["bp"] < 0
           and out["paired/second half"]["bp"] < 0
           else "no separable gain over the aggregate")
print(f"\nverdict: {verdict}")
out["verdict"] = verdict
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(f"wrote {OUT}")
