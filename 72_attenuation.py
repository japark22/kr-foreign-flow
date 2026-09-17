#!/usr/bin/env python3
"""Turn the chosen threshold into a declared statistic.

The contrast grows as the participation requirement rises, which is what
measurement error in a regressor does: a noisier crowding measure pulls the
estimate toward zero, and demanding more trading days makes the measure
cleaner. Read off a ladder, though, the rung is a choice, and a chosen rung
is not evidence.

The trend across the ladder is not a choice. It is one number -- the slope
of the contrast against the requirement -- and it can be compared with the
slope that random crowding produces on exactly the same ladder. If the
attenuation story is right the real slope is steeper than the placebo's,
and no rung has to be picked at all.

Reported alongside it: the same ladder for the placebo assignment, so the
comparison is like for like at every rung rather than only in aggregate.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V6 = ROOT / "results" / "event_panel_v6.parquet"
OUT = ROOT / "results" / "attenuation.json"
RUNGS = [1, 5, 10, 14, 18]
DRAWS = 200
NS, NC = 5, 3
CROWD = "i_flow20_v2"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name.split("_")[0],
                                                  ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


bfm = load_module("21_baseline_fm")


def contrast(sub, crowd_vals):
    """p10(crowded) - p10(quiet) inside the top surprise quintile,
    computed per reporting season then averaged, exactly as 50_baseline does."""
    sq = sub.groupby("D")["surprise"].rank(pct=True)
    cq = pd.Series(crowd_vals, index=sub.index).groupby(
        sub["D"].to_numpy()).rank(pct=True)
    s = np.clip((sq * NS).astype(int), 0, NS - 1).to_numpy()
    c = np.clip((cq * NC).astype(int), 0, NC - 1).to_numpy()
    top = s == NS - 1
    crowded, quiet = top & (c == NC - 1), top & (c == 0)
    y = sub["abn60"].to_numpy()
    season = sub["season"].to_numpy()
    rows = []
    for ss in np.unique(season):
        m = season == ss
        a, b = y[m & crowded], y[m & quiet]
        if len(a) >= 8 and len(b) >= 8:
            rows.append(np.percentile(a, 10) - np.percentile(b, 10))
    if len(rows) < 8:
        return np.nan, 0
    r = np.array(rows)
    return float(r.mean() * 1e4), len(r)


def main() -> int:
    d = pd.read_parquet(V6)
    d["season"] = d["D"].dt.year * 4 + (d["D"].dt.month - 1) // 3
    rng = np.random.default_rng(20260917)
    R = {"rungs": RUNGS, "draws": DRAWS}

    for kind in ("provisional", None):
        key = kind or "all"
        base = d if kind is None else d[d["kind"] == kind]
        base = base.dropna(subset=["abn60", "surprise", CROWD, "inst_days20"])
        print(f"\n{'=' * 76}\n{key} filings\n{'=' * 76}")
        print(f"  {'rung':<8}{'n':>9}{'real':>12}{'placebo mean':>15}"
              f"{'placebo 5-95%':>22}{'seasons':>9}")

        real, plac_mean, rows = [], [], {}
        for rung in RUNGS:
            sub = base[base["inst_days20"] >= rung]
            r, ns = contrast(sub, sub[CROWD].to_numpy())
            band = []
            for _ in range(DRAWS):
                v, _ = contrast(sub, rng.random(len(sub)))
                if np.isfinite(v):
                    band.append(v)
            b = np.array(band)
            lo, hi = np.percentile(b, [5, 95])
            real.append(r)
            plac_mean.append(float(b.mean()))
            rows[rung] = {"n": int(len(sub)), "real": r, "seasons": ns,
                          "placebo_mean": float(b.mean()),
                          "p05": float(lo), "p95": float(hi),
                          "outside": bool(r < lo or r > hi)}
            print(f"  {rung:<8}{len(sub):>9,}{r:>12.1f}{b.mean():>15.1f}"
                  f"{f'[{lo:+.0f}, {hi:+.0f}]':>22}{ns:>9}")

        x = np.array(RUNGS, dtype=float)
        sl_real = float(np.polyfit(x, np.array(real), 1)[0])
        sl_plac = float(np.polyfit(x, np.array(plac_mean), 1)[0])

        # the placebo's own slope distribution, one full ladder per draw
        sl_band = []
        for _ in range(DRAWS):
            ys = []
            for rung in RUNGS:
                sub = base[base["inst_days20"] >= rung]
                v, _ = contrast(sub, rng.random(len(sub)))
                ys.append(v)
            ys = np.array(ys, dtype=float)
            if np.isfinite(ys).all():
                sl_band.append(float(np.polyfit(x, ys, 1)[0]))
        sb = np.array(sl_band)
        lo, hi = np.percentile(sb, [5, 95])
        outside = sl_real < lo or sl_real > hi

        print(f"\n  slope of the contrast against the requirement")
        print(f"    real            {sl_real:+8.2f} bp per extra day required")
        print(f"    placebo ladders {sb.mean():+8.2f}   "
              f"5-95% [{lo:+.2f}, {hi:+.2f}]   {'OUTSIDE' if outside else 'inside'}")
        print(f"    {'attenuation is real' if outside and sl_real < lo else 'the trend is not distinguishable from chance'}")

        R[key] = {"ladder": rows, "slope_real": sl_real,
                  "slope_placebo_mean": float(sb.mean()),
                  "slope_p05": float(lo), "slope_p95": float(hi),
                  "slope_outside": bool(outside)}

    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2, default=str))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
