"""The baseline expectation, as a distribution rather than a mean.

What was asked for was not a signal. It was a supportive feature that makes
the baseline expectation better understood -- and a baseline expectation is a
distribution. Every test in this project so far has asked whether crowding
moves the mean, and the mean barely moves. That is not the same as saying
crowding tells you nothing about what to expect.

Crowding should bite in the tail rather than at the centre. A name many
institutions are already holding into a disappointing print has an exit
problem the uncrowded name does not, and an exit problem shows up as a worse
tenth percentile and a lower hit rate long before it shows up as a lower
average. So the deliverable here is the conditional table itself -- what
happens after a surprise of each strength, split by how crowded the name was
-- with the two distributional statements tested against a placebo band.

Declared before the run, within the strongest surprises:
  H1  the crowded tenth percentile is worse than the quiet tenth percentile
  H2  the crowded hit rate is lower than the quiet hit rate

    python 50_baseline.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("results/decompose_panel.parquet")
PANEL = Path("results/event_panel.parquet")
OUT = Path("results/baseline.json")
CROWD, NS, NC, DRAWS = "i_flow20", 5, 3, 200

spec = importlib.util.spec_from_file_location("bfm", "21_baseline_fm.py")
bfm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bfm)


def load():
    src = CACHE if CACHE.exists() else PANEL
    d = pd.read_parquet(src)
    d["D"] = pd.to_datetime(d["D"])
    return d, src.name


def cell_stats(v: np.ndarray) -> dict:
    if len(v) < 30:
        return {"n": int(len(v))}
    return {"n": int(len(v)),
            "mean_bp": float(np.mean(v) * 1e4),
            "median_bp": float(np.median(v) * 1e4),
            "p10_bp": float(np.percentile(v, 10) * 1e4),
            "p25_bp": float(np.percentile(v, 25) * 1e4),
            "p75_bp": float(np.percentile(v, 75) * 1e4),
            "p90_bp": float(np.percentile(v, 90) * 1e4),
            "hit": float((v > 0).mean()),
            "big_loss": float((v < -0.10).mean())}


def season_diff(g, mask_a, mask_b, stat):
    """The statistic inside each reporting season, differenced, then a
    clustered t -- the seasons are the independent unit, not the events."""
    rows = []
    for s in np.unique(g["season"]):
        m = (g["season"] == s).to_numpy()
        a, b = g["abn60"].to_numpy()[m & mask_a], g["abn60"].to_numpy()[m & mask_b]
        if len(a) >= 8 and len(b) >= 8:
            rows.append(stat(a) - stat(b))
    r = np.array(rows, dtype=float)
    if len(r) < 8:
        return None
    return {"diff": float(r.mean()), "t": float(bfm.tracker._nw_t(r, 1)),
            "seasons": int(len(r))}


def main() -> int:
    d, src = load()
    print(f"reading {src}")
    for kind, label in (("provisional", "provisional filings"),
                        (None, "all filings")):
        sub = d if kind is None else d[d["kind"] == kind]
        sub = sub.dropna(subset=["abn60", "surprise", CROWD]).copy()
        if len(sub) < 2000:
            continue
        sub["season"] = sub["D"].dt.year * 4 + (sub["D"].dt.month - 1) // 3
        sq = sub.groupby("D")["surprise"].rank(pct=True)
        cq = sub.groupby("D")[CROWD].rank(pct=True)
        sub["s"] = np.clip((sq * NS).astype(int), 0, NS - 1)
        sub["c"] = np.clip((cq * NC).astype(int), 0, NC - 1)

        print(f"\n================ {label}: {len(sub):,} events ================")
        print("  subsequent 60-day abnormal return, bp")
        print(f"  {'surprise':<10}{'crowding':<10}{'n':>7}{'mean':>9}"
              f"{'median':>9}{'p10':>9}{'p90':>9}{'hit':>7}{'loss>10%':>10}")
        table = {}
        for si in range(NS):
            for ci in range(NC):
                v = sub[(sub["s"] == si) & (sub["c"] == ci)]["abn60"].to_numpy()
                st = cell_stats(v)
                table[f"s{si}c{ci}"] = st
                if "mean_bp" not in st:
                    continue
                print(f"  {si + 1:<10}{['quiet', 'mid', 'crowded'][ci]:<10}"
                      f"{st['n']:>7,}{st['mean_bp']:>9.1f}"
                      f"{st['median_bp']:>9.1f}{st['p10_bp']:>9.1f}"
                      f"{st['p90_bp']:>9.1f}{st['hit']:>7.1%}"
                      f"{st['big_loss']:>10.1%}")
        key = "provisional" if kind else "all"
        R = {key: {"table": table, "events": int(len(sub))}}

        # ---- the two declared statements, inside the strongest surprises
        top = (sub["s"] == NS - 1).to_numpy()
        crowded = top & (sub["c"] == NC - 1).to_numpy()
        quiet = top & (sub["c"] == 0).to_numpy()
        print(f"\n  within the strongest surprises: "
              f"crowded {crowded.sum():,}  quiet {quiet.sum():,}")

        stats = {"p10": lambda v: np.percentile(v, 10),
                 "hit rate": lambda v: (v > 0).mean(),
                 "mean": lambda v: v.mean()}
        rng = np.random.default_rng(20260903)
        for name, fn in stats.items():
            real = season_diff(sub, crowded, quiet, fn)
            if real is None:
                print(f"    {name:<10}too few comparable seasons")
                continue
            band = []
            for _ in range(DRAWS):
                r = rng.random(len(sub))
                fake_c = np.clip((pd.Series(r).groupby(
                    sub["D"].to_numpy()).rank(pct=True).to_numpy() * NC
                ).astype(int), 0, NC - 1)
                pc = top & (fake_c == NC - 1)
                pq = top & (fake_c == 0)
                res = season_diff(sub, pc, pq, fn)
                if res:
                    band.append(res["diff"])
            b = np.array(band)
            lo, hi = np.percentile(b, [5, 95])
            scale = 1e4 if name != "hit rate" else 100.0
            unit = "bp" if name != "hit rate" else "pp"
            outside = real["diff"] < lo or real["diff"] > hi
            print(f"    {name:<10}crowded minus quiet "
                  f"{real['diff'] * scale:+8.1f} {unit}"
                  f"   t {real['t']:+5.2f}   placebo 5-95% "
                  f"[{lo * scale:+.1f}, {hi * scale:+.1f}]"
                  f"   {'OUTSIDE' if outside else 'inside'}")
            R[key][name] = {"diff": real["diff"], "t": real["t"],
                            "seasons": real["seasons"],
                            "placebo_lo": float(lo), "placebo_hi": float(hi),
                            "outside_band": bool(outside)}
        if OUT.exists():
            prev = json.loads(OUT.read_text())
            prev.update(R)
            R = prev
        OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
