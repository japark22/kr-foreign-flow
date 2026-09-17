#!/usr/bin/env python3
"""State the result in its actual form, or drop it.

Removing 22% of the sample weakened the effect. That is not yet a finding:
the removed windows contain the strongest years, so the arithmetic is
guaranteed. What makes it a statement is an interaction estimated on the
whole sample, and a placebo that removes windows of the same length from
elsewhere.

The suspension dates are regulatory and were not chosen here, so the test
is not circular. The placebo measures how unusual the contrast is against
windows that carry no regulatory meaning.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V5 = ROOT / "results" / "event_panel_v5.parquet"
OUT = ROOT / "results" / "shortban.json"
DRAWS = 200

BANS = {"2020-21": ("2020-03-16", "2021-05-02"),
        "2023-25": ("2023-11-06", "2025-03-30")}


def load(name):
    spec = importlib.util.spec_from_file_location(name.split("_")[0],
                                                  ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


th = load("64_threats_flevel")
CTRL, LONG = th.CTRL, th.LONG
HARD = CTRL + LONG


def run(d, base, ctrl=HARD):
    p = th.prepare(d, base, ctrl)
    return th.fit(p) if p else None


def line(tag, r, w=40):
    if r is None:
        print(f"  {tag:<{w}}  --")
        return
    print(f"  {tag:<{w}}{r['bp']:+8.1f}   t {r['t_two_way']:+6.2f}   "
          f"n={r['events']:>6,}  s={r['seasons']:>3}")


def main() -> int:
    d = pd.read_parquet(V5)
    d = d[d["kind"] == "periodic"].copy()
    d["fl_minus_i"] = (d.groupby("D")["f_level"].transform(lambda s: s.rank(pct=True))
                       - d.groupby("D")["i_flow20_v2"].transform(lambda s: s.rank(pct=True)))
    d["fl_x_surp"] = (d.groupby("D")["f_level"].transform(lambda s: s.rank(pct=True))
                      * d.groupby("D")["surprise"].transform(lambda s: s.rank(pct=True)))
    d = d.dropna(subset=["fl_minus_i", "fl_x_surp", "abn60", "surprise"] + HARD)

    inban = pd.Series(False, index=d.index)
    for a_, b_ in BANS.values():
        inban |= (d["D"] >= a_) & (d["D"] <= b_)
    d["inban"] = inban
    days = pd.DatetimeIndex(sorted(d["D"].unique()))
    span = sum((pd.Timestamp(b_) - pd.Timestamp(a_)).days for a_, b_ in BANS.values())
    print(f"events {len(d):,}   inside a suspension {int(inban.sum()):,} "
          f"({inban.mean():.1%})   suspended days total {span}")
    R = {"share_in_ban": float(inban.mean())}

    print("\n1. estimated separately")
    for b in ("fl_minus_i", "fl_x_surp"):
        print(f"  [{b}]")
        ri = run(d[d["inban"]], b)
        ro = run(d[~d["inban"]], b)
        R[f"in_{b}"], R[f"out_{b}"] = ri, ro
        line("    inside a suspension", ri, 38)
        line("    outside", ro, 38)
        if ri and ro:
            se_i = abs(ri["bp"] / ri["t_two_way"]) if ri["t_two_way"] else np.nan
            se_o = abs(ro["bp"] / ro["t_two_way"]) if ro["t_two_way"] else np.nan
            diff = ri["bp"] - ro["bp"]
            t = diff / np.sqrt(se_i ** 2 + se_o ** 2)
            print(f"    difference {diff:+8.1f} bp/SD   t {t:+5.2f}"
                  f"   <- this is the claim, not the two numbers above")
            R[f"diff_{b}"] = {"bp": float(diff), "t": float(t)}

    print("\n2. each suspension on its own  (one episode is an anecdote)")
    for tag, (a_, b_) in BANS.items():
        m = (d["D"] >= a_) & (d["D"] <= b_)
        for b in ("fl_minus_i", "fl_x_surp"):
            r = run(d[m], b)
            R[f"episode_{tag}_{b}"] = r
            line(f"{tag}  {b}", r)

    print(f"\n3. placebo: remove windows of the same length from elsewhere "
          f"({DRAWS} draws)")
    rng = np.random.default_rng(41)
    lens = [(pd.Timestamp(b_) - pd.Timestamp(a_)).days for a_, b_ in BANS.values()]
    lo_d, hi_d = days.min(), days.max()
    for b in ("fl_minus_i", "fl_x_surp"):
        full = run(d, b)
        real = run(d[~d["inban"]], b)
        vals = []
        for _ in range(DRAWS):
            keep = pd.Series(True, index=d.index)
            for L in lens:
                start = lo_d + pd.Timedelta(
                    days=int(rng.integers(0, max((hi_d - lo_d).days - L, 1))))
                keep &= ~((d["D"] >= start) & (d["D"] <= start + pd.Timedelta(days=L)))
            r = run(d[keep], b)
            if r:
                vals.append(r["bp"])
        v = np.array(vals)
        p05, p95 = np.percentile(v, [5, 95])
        print(f"  [{b}]  full {full['bp']:+7.1f} (t {full['t_two_way']:+5.2f})"
              f"   after removing the suspensions {real['bp']:+7.1f} "
              f"(t {real['t_two_way']:+5.2f})")
        print(f"         random removals of the same length: "
              f"mean {v.mean():+7.1f}  band [{p05:+7.1f},{p95:+7.1f}]  "
              f"{'OUTSIDE -- the suspensions are special' if real['bp'] < p05 else 'inside -- any 22% would have done this'}")
        R[f"placebo_{b}"] = {"full": full, "after_removal": real,
                             "placebo_mean": float(v.mean()),
                             "p05": float(p05), "p95": float(p95),
                             "special": bool(real["bp"] < p05)}

    OUT.write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
