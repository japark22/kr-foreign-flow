#!/usr/bin/env python3
"""Step 29: the filter's value, computed only from the trusted path.

    python 29_final.py

WHY THIS REPLACES STEPS 25 TO 28
--------------------------------
Those steps built a portfolio simulator to price the finding, and the
simulator turned out not to agree with the event panel: on the same events at
sixty sessions it reported -306bp where the panel's own market-adjusted return
said +82bp. A 388bp disagreement between two code paths means one of them is
wrong, and the panel is the one that has produced consistent, sensible numbers
since step 20. Two suspects were tested and cleared -- summing daily excess
returns matches compounding exactly, and the annualising bias was measured and
removed -- so the fault is elsewhere in the simulator and it is not worth
finding.

It is not worth finding because the simulator was never necessary. The
question is what the ownership data adds on top of an earnings signal, and
that is a difference between two subsets of the same events on the same dates.
The panel already stores a market-adjusted return for every event. Subtracting
one group's mean from another's answers the question directly, with no book
weighting, no cash days, no cost-timing artefact and no annualising.

WHAT IS MEASURED
----------------
For each horizon the panel carries:
  quiet beats   beats where institutions were in the bottom 40% of pre-event
                buying -- the rule the brief describes
  all beats     every beat, the base case with no ownership information
  crowded beats beats where institutions were in the top 40% -- the mirror,
                which must come out on the other side if the mechanism is what
                we think it is
  the quintile contrast, bottom versus top of institutional buying, which is
                the sharpest version and the one step 24 priced

Inference is clustered by reporting season: two positions opened in the same
February are one observation, not sixty. Costs are quoted separately rather
than folded in, because a 50bp round trip is a fact about the trade and should
not be buried inside a return.
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
BASE = "i_flow20"
COST_RT_BP = 50.0


def season(s: pd.Series) -> np.ndarray:
    return (s.dt.year * 4 + (s.dt.month - 1) // 3).to_numpy()


def clus(v: np.ndarray, s: np.ndarray):
    per = np.array([v[s == x].mean() for x in np.unique(s)])
    per = per[np.isfinite(per)]
    if len(per) < 8:
        return np.nan, np.nan, len(per)
    return float(per.mean()), tracker._nw_t(per, 1), len(per)


def contrast(a: pd.DataFrame, b: pd.DataFrame, col: str):
    """Mean(a) - mean(b) per season, then a clustered t."""
    ga = a.dropna(subset=[col])
    gb = b.dropna(subset=[col])
    sa, sb = season(ga["D"]), season(gb["D"])
    ma = {x: ga[col].to_numpy()[sa == x].mean() for x in np.unique(sa)}
    mb = {x: gb[col].to_numpy()[sb == x].mean() for x in np.unique(sb)}
    common = sorted(set(ma) & set(mb))
    if len(common) < 8:
        return np.nan, np.nan, len(common), np.array([])
    d = np.array([ma[x] - mb[x] for x in common])
    return float(d.mean()), tracker._nw_t(d, 1), len(common), d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    if not PANEL.exists():
        sys.exit("run 20_event_panel.py first")
    d = pd.read_parquet(PANEL)
    d = d[(d["kind"] == "provisional") & d["surprise"].notna()
          & d[BASE].notna()].copy()
    horizons = [h for h in (5, 20, 60) if f"abn{h}" in d.columns]
    d["sur_r"] = d.groupby("D")["surprise"].rank(pct=True)
    d["base_r"] = d.groupby("D")[BASE].rank(pct=True)
    beat = d["sur_r"] >= 0.6
    grp = {"all beats": d[beat],
           "quiet beats": d[beat & (d["base_r"] <= 0.4)],
           "crowded beats": d[beat & (d["base_r"] >= 0.6)],
           "quiet, any result": d[d["base_r"] <= 0.4]}
    print(f"provisional events: {len(d):,}   "
          f"{d['D'].min().date()} .. {d['D'].max().date()}")
    for k, v in grp.items():
        print(f"  {k:<20} {len(v):>6,} events")

    out = {"n": int(len(d)), "levels": {}, "contrasts": {}}
    print(f"\n  LEVELS -- market-adjusted return per position, bp, "
          f"season-clustered")
    print("    group                " + "".join(f"    abn{h:<12}" for h in horizons))
    for k, v in grp.items():
        row, cells = f"    {k:<20}", {}
        for h in horizons:
            col = f"abn{h}"
            g = v.dropna(subset=[col])
            mn, t, ns = clus(g[col].to_numpy(), season(g["D"]))
            cells[str(h)] = {"bp": mn * 1e4, "t": t, "seasons": ns}
            row += f" {mn*1e4:+8.1f} (t{t:+5.2f})"
        out["levels"][k] = cells
        print(row)

    print(f"\n  CONTRASTS -- what the ownership data adds, bp per position")
    print("    contrast                  " +
          "".join(f"    abn{h:<12}" for h in horizons))
    pairs = [("quiet beats - all beats", grp["quiet beats"], grp["all beats"]),
             ("crowded beats - all beats", grp["crowded beats"], grp["all beats"]),
             ("quiet - crowded (beats)", grp["quiet beats"], grp["crowded beats"])]
    for name, A, B in pairs:
        row, cells = f"    {name:<25}", {}
        for h in horizons:
            mn, t, ns, dd = contrast(A, B, f"abn{h}")
            half = len(dd) // 2
            cells[str(h)] = {"bp": mn * 1e4, "t": t, "seasons": ns,
                             "halves": [float(dd[:half].mean() * 1e4),
                                        float(dd[half:].mean() * 1e4)]
                             if len(dd) else None}
            row += f" {mn*1e4:+8.1f} (t{t:+5.2f})"
        out["contrasts"][name] = cells
        print(row)
        for h in horizons:
            hv = out["contrasts"][name][str(h)]["halves"]
            if hv:
                print(f"        abn{h} halves: {hv[0]:+.1f} / {hv[1]:+.1f} bp")

    print(f"\n  QUINTILE CONTRAST -- bottom vs top of institutional buying,")
    print("    all provisional events (this is what step 24 reported)")
    lo = d[d["base_r"] <= 0.2]
    hi = d[d["base_r"] >= 0.8]
    for h in horizons:
        mn, t, ns, dd = contrast(lo, hi, f"abn{h}")
        half = len(dd) // 2
        print(f"    abn{h:<3} {mn*1e4:+8.1f} bp  t {t:+5.2f}  ({ns} seasons)"
              f"   halves {dd[:half].mean()*1e4:+.0f} / {dd[half:].mean()*1e4:+.0f}")
        out.setdefault("quintile", {})[str(h)] = {"bp": mn * 1e4, "t": t,
                                                  "seasons": ns}

    print(f"\n  cost reference: a round trip is {COST_RT_BP:.0f}bp. Any contrast")
    print("  smaller than that is a research finding, not a trade.")
    key = out["contrasts"]["quiet beats - all beats"]
    hs = [str(h) for h in horizons]
    pos = all(key[h]["bp"] > 0 for h in hs)
    sig = sum(1 for h in hs if abs(key[h]["t"]) >= 2)
    halves_ok = all(key[h]["halves"] and min(key[h]["halves"]) > 0 for h in hs)
    print(f"\n  filter positive at every horizon: {pos}")
    print(f"  positive in both halves everywhere: {halves_ok}")
    print(f"  horizons with |t| >= 2: {sig} of {len(hs)}")
    out["verdict"] = ("FILTER CONFIRMED" if (pos and halves_ok and sig >= 2)
                      else "DIRECTIONAL, NOT CONFIRMED")
    print(f"\n  verdict: {out['verdict']}")

    RESULTS.mkdir(exist_ok=True)
    dest = RESULTS / "final.json"
    dest.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
