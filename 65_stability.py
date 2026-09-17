#!/usr/bin/env python3
"""Is 2020 a regime, or did the median happen to cut there?

Splitting a sample in half and finding one side alive is weak evidence
either way. Three sharper questions:

  1. What does the coefficient do year by year? A step is a regime. Noise
     around a positive mean is a sample that was cut badly.
  2. How much of the damage from the long-horizon controls is the controls,
     and how much is the 12% of rows that the 500-day window costs? The
     ladder is run on ONE fixed sample so the two cannot be confused.
  3. Does the split year matter? Every break from 2013 to 2023 is tried and
     both sides reported, so the answer does not depend on one cut.

The short-selling suspensions are also removed, because the split date
landed two weeks after one of them began and that is worth ruling out.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V5 = ROOT / "results" / "event_panel_v5.parquet"
OUT = ROOT / "results" / "stability.json"

BANS = [("2020-03-16", "2021-05-02"), ("2023-11-06", "2025-03-30")]


def load(name):
    spec = importlib.util.spec_from_file_location(name.split("_")[0],
                                                  ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


th = load("64_threats_flevel")
CTRL, LONG = th.CTRL, th.LONG


def run(d, base, ctrl):
    p = th.prepare(d, base, ctrl)
    return th.fit(p) if p else None


def line(tag, r, width=34):
    if r is None:
        print(f"  {tag:<{width}}  --")
        return
    print(f"  {tag:<{width}}{r['bp']:+8.1f}   t {r['t_two_way']:+6.2f}   "
          f"n={r['events']:>6,}  firms={r['firms']:>5,}  s={r['seasons']:>3}")


def main() -> int:
    d = pd.read_parquet(V5)
    d = d[d["kind"] == "periodic"].copy()
    d["fl_minus_i"] = (d.groupby("D")["f_level"].transform(lambda s: s.rank(pct=True))
                       - d.groupby("D")["i_flow20_v2"].transform(lambda s: s.rank(pct=True)))
    d["fl_x_surp"] = (d.groupby("D")["f_level"].transform(lambda s: s.rank(pct=True))
                      * d.groupby("D")["surprise"].transform(lambda s: s.rank(pct=True)))
    R = {}

    # one fixed sample, so the control ladder is about controls only
    fixed = d.dropna(subset=["f_level", "i_flow20_v2", "surprise", "abn60"]
                     + CTRL + LONG)
    print(f"fixed sample for the ladder: {len(fixed):,} of {len(d):,}")

    print("\n1. control ladder on ONE sample  (bp/SD, two-way t)")
    ladders = [("standard controls", CTRL),
               ("+ 120-day momentum", CTRL + ["c_mom120"]),
               ("+ 120 and 250", CTRL + ["c_mom120", "c_mom250"]),
               ("+ 120, 250 and 500", CTRL + LONG)]
    for b in ("fl_minus_i", "fl_x_surp"):
        print(f"  [{b}]")
        for tag, ctrl in ladders:
            r = run(fixed, b, ctrl)
            R[f"ladder_{b}_{tag}"] = r
            line("    " + tag, r, 32)

    print("\n2. year by year, hard controls")
    for b in ("fl_minus_i", "fl_x_surp"):
        print(f"  [{b}]")
        rows = []
        for y in range(2011, 2027):
            sub = fixed[fixed["D"].dt.year == y]
            if len(sub) < 1500:
                continue
            r = run(sub, b, CTRL + LONG)
            if r:
                rows.append((y, r))
                bar = "#" * min(int(abs(r["bp"]) / 10), 40)
                sign = "+" if r["bp"] >= 0 else "-"
                print(f"    {y}  {r['bp']:+8.1f}  t {r['t_two_way']:+6.2f}  "
                      f"n={r['events']:>6,}  {sign}{bar}")
        R[f"yearly_{b}"] = {str(y): r for y, r in rows}
        pos = sum(1 for _, r in rows if r["bp"] > 0)
        print(f"    positive years: {pos} of {len(rows)}")

    print("\n3. break-year sweep, hard controls  (both sides, every cut)")
    for b in ("fl_minus_i", "fl_x_surp"):
        print(f"  [{b}]   {'cut':<6}{'before':>28}{'from':>28}")
        for y in range(2014, 2024):
            a = run(fixed[fixed["D"] < f"{y}-01-01"], b, CTRL + LONG)
            c = run(fixed[fixed["D"] >= f"{y}-01-01"], b, CTRL + LONG)
            fa = (f"{a['bp']:+8.1f} t {a['t_two_way']:+5.2f} n={a['events']:>6,}"
                  if a else "        --")
            fc = (f"{c['bp']:+8.1f} t {c['t_two_way']:+5.2f} n={c['events']:>6,}"
                  if c else "        --")
            print(f"         {y:<6}{fa:>28}{fc:>28}")
            R[f"break_{b}_{y}"] = {"before": a, "from": c}

    print("\n4. with the short-selling suspensions removed")
    mask = pd.Series(True, index=fixed.index)
    for a_, b_ in BANS:
        mask &= ~((fixed["D"] >= a_) & (fixed["D"] <= b_))
    clean = fixed[mask]
    print(f"  removed {len(fixed) - len(clean):,} events "
          f"({1 - len(clean) / len(fixed):.1%})")
    for b in ("fl_minus_i", "fl_x_surp"):
        r = run(clean, b, CTRL + LONG)
        R[f"noban_{b}"] = r
        line(b + ", suspensions removed", r)
        med = clean["D"].median()
        line("  before " + str(pd.Timestamp(med).date()),
             run(clean[clean["D"] < med], b, CTRL + LONG))
        line("  from " + str(pd.Timestamp(med).date()),
             run(clean[clean["D"] >= med], b, CTRL + LONG))

    OUT.write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
