#!/usr/bin/env python3
"""Compress the finding to one statistic, then try to break it by period.

The tau grid says the conditional distribution is squeezed from the top
rather than shifted. One number carries that: the coefficient on the
interdecile spread,

    S = b(0.90) - b(0.10)

estimated as a single quantity so that its bootstrap and its placebo are
the bootstrap and placebo of the thing actually being claimed, not of two
separate coefficients read side by side.

Sub-periods are cut at the median event date and at 2018. Neither boundary
was chosen after seeing a result: the median is mechanical, and 2018 is
where the imputation check in PRE_REGISTRATION section 10 starts.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "results" / "event_panel_v3.parquet"
OUT = ROOT / "results" / "shape.json"
BOOTS, DRAWS = 300, 200
LO, HI = 0.10, 0.90


def load(name):
    spec = importlib.util.spec_from_file_location(name.split("_")[0],
                                                  ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


fr = load("38_final_results")
qr = load("58_qreg")
CTRL = fr.CTRL


def shape(d, base="i_flow20_v2", y="abn60", seed=3):
    prep = fr.prepare(d, base, ["surprise"] + CTRL + [base], y)
    if prep is None:
        return None
    Xo, xb, yy, sea, slices = prep
    X = np.column_stack([xb, Xo])
    seasons = np.unique(sea)
    rows = {s: np.where(sea == s)[0] for s in seasons}
    rng = np.random.default_rng(seed)

    def pair(Xm, ym):
        return (qr.qreg(Xm, ym, HI)[0] * 1e4, qr.qreg(Xm, ym, LO)[0] * 1e4)

    hi0, lo0 = pair(X, yy)
    s0 = hi0 - lo0

    bs = []
    for _ in range(BOOTS):
        pick = rng.choice(seasons, size=len(seasons), replace=True)
        idx = np.concatenate([rows[s] for s in pick])
        h, l = pair(X[idx], yy[idx])
        bs.append(h - l)
    sd = float(np.std(bs, ddof=1))

    pl = []
    for _ in range(DRAWS):
        xp = xb.copy()
        for a_, b_ in slices:
            xp[a_:b_] = rng.permutation(xp[a_:b_])
        h, l = pair(np.column_stack([xp, Xo]), yy)
        pl.append(h - l)
    p05, p95 = np.percentile(pl, [5, 95])

    return {"events": int(len(yy)), "seasons": int(len(seasons)),
            "b90": float(hi0), "b10": float(lo0), "spread": float(s0),
            "t": float(s0 / sd) if sd > 1e-12 else np.nan, "boot_sd": sd,
            "placebo_p05": float(p05), "placebo_p95": float(p95),
            "placebo_mean": float(np.mean(pl)),
            "inside_band": bool(p05 <= s0 <= p95)}


def show(tag, r):
    if r is None:
        print(f"  {tag:<40}  --")
        return
    mark = "inside band" if r["inside_band"] else "OUTSIDE band"
    print(f"  {tag:<40}{r['spread']:+8.1f} bp/SD   t {r['t']:+5.2f}   "
          f"[{r['placebo_p05']:+7.1f},{r['placebo_p95']:+7.1f}] {mark:<12} "
          f"n={r['events']:>6,} s={r['seasons']}")


def main() -> int:
    d = pd.read_parquet(PANEL)
    prov = d[d["kind"] == "provisional"]
    per = d[d["kind"] == "periodic"]
    med = prov["D"].median()
    print(f"median provisional event date: {pd.Timestamp(med).date()}\n")
    print(f"  {'spec':<40}{'b90 - b10':>12}")

    R = {"boots": BOOTS, "draws": DRAWS, "lo": LO, "hi": HI,
         "median_date": str(pd.Timestamp(med).date())}
    specs = [
        ("provisional, full", prov, "i_flow20_v2", "abn60"),
        ("provisional, participation 14+", prov[prov["inst_days20"] >= 14],
         "i_flow20_v2", "abn60"),
        (f"provisional, before {pd.Timestamp(med).date()}",
         prov[prov["D"] < med], "i_flow20_v2", "abn60"),
        (f"provisional, from {pd.Timestamp(med).date()}",
         prov[prov["D"] >= med], "i_flow20_v2", "abn60"),
        ("provisional, before 2018", prov[prov["D"] < "2018-01-01"],
         "i_flow20_v2", "abn60"),
        ("provisional, from 2018", prov[prov["D"] >= "2018-01-01"],
         "i_flow20_v2", "abn60"),
        ("provisional, foreign (null check)", prov, "f_flow20", "abn60"),
        ("periodic, full", per, "i_flow20_v2", "abn60"),
    ]
    for tag, sub, base, y in specs:
        r = shape(sub, base, y)
        show(tag, r)
        R[tag] = r

    # ---- tangible form: raw conditional deciles by crowding quintile
    print("\n  raw shape, provisional, within-day demeaned abn60 (bp)")
    p = prov.dropna(subset=["abn60", "i_flow20_v2"]).copy()
    p["dm"] = (p["abn60"] - p.groupby("D")["abn60"].transform("mean")) * 1e4
    p["q"] = p.groupby("D")["i_flow20_v2"].transform(
        lambda s: np.clip((s.rank(pct=True) * 5).astype(int), 0, 4) + 1)
    tbl = p.groupby("q")["dm"].agg(
        n="size", p10=lambda s: s.quantile(0.10), p50="median",
        p90=lambda s: s.quantile(0.90), mean="mean")
    tbl["p90_minus_p10"] = tbl["p90"] - tbl["p10"]
    print(tbl.round(1).to_string())
    R["raw_shape"] = tbl.round(2).to_dict(orient="index")

    OUT.write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
