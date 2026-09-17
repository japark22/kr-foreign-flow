#!/usr/bin/env python3
"""Search every feature and every pairwise difference as ONE family.

Running many tests and reporting the best one is how this project has been
misled three times. The alternative is not to run fewer tests -- it is to
run all of them and control the family.

Every cell is a coefficient on abn{5,20,60} for provisional and periodic
filings, over singles, all pairwise differences, and surprise interactions.
The null distribution is the maximum absolute t over the WHOLE grid under a
within-day permutation, so a cell is reported only if it beats what the best
of roughly two thousand cells achieves by chance.

Permutation follows Freedman and Lane: the controls are projected out once,
and the residualised outcome is shuffled within each day. The features stay
fixed, so the correlation between cells is preserved and the family maximum
accounts for it.

Coverage differs across features, so the grid runs in two blocks -- one over
the full window, one over the sub-type era -- and each block is internally
complete. Coverage is printed with every result.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

ROOT = Path(__file__).resolve().parent
V4 = ROOT / "results" / "event_panel_v4.parquet"
DEC = ROOT / "results" / "decompose_panel.parquet"
OUT = ROOT / "results" / "search.json"
DRAWS = 500
MIN_N = 25
HORIZONS = ["abn5", "abn20", "abn60"]


def load(name):
    spec = importlib.util.spec_from_file_location(name.split("_")[0],
                                                  ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


bfm = load("21_baseline_fm")
CTRL = ["c_mom20", "c_mom60", "c_size", "c_vol", "c_turn"]

CORE = ["f_flow20", "f_flow60", "f_hold20", "f_level", "i_flow20_v2",
        "inst_days20", "inst_abs20", "no_inst20",
        "ipos60", "ipos120", "ipos250",
        "ipos60_adv", "ipos120_adv", "ipos250_adv"]


def build(d, names):
    """Rank-standardise inside the day, project out the controls, stack."""
    cols = ["surprise"] + CTRL
    keep = d.dropna(subset=names + cols + ["abn60"]).copy()
    ys, fs, sea, slices = {h: [] for h in HORIZONS}, [], [], []
    n = 0
    for day, g in keep.groupby("D"):
        g = g.dropna(subset=HORIZONS, how="all")
        if len(g) < MIN_N:
            continue
        C = np.column_stack([bfm.rank_std(g[c]) for c in cols]
                            + [np.ones(len(g))])
        Q, _ = np.linalg.qr(C)
        F = np.column_stack([bfm.rank_std(g[c]) for c in names])
        fs.append(F - Q @ (Q.T @ F))
        for h in HORIZONS:
            v = g[h].to_numpy(dtype=float)
            bad = ~np.isfinite(v)
            if bad.all():
                v = np.zeros(len(g))
            else:
                v = np.where(bad, np.nanmedian(v), v)
                lo, hi = np.percentile(v, [1, 99])
                v = np.clip(v, lo, hi)
            ys[h].append(v - Q @ (Q.T @ v))
        t = pd.Timestamp(day)
        sea.append(np.full(len(g), t.year * 4 + (t.month - 1) // 3))
        slices.append((n, n + len(g)))
        n += len(g)
    if not slices:
        return None
    return (np.vstack(fs), {h: np.concatenate(ys[h]) for h in HORIZONS},
            np.concatenate(sea), slices, len(keep))


def coefs(F, y, order, starts, denom):
    """Coefficient and season-clustered t for every column at once."""
    num = F.T @ y
    b = num / denom
    E = y[:, None] - F * b[None, :]
    U = (F * E)[order]
    S = np.add.reduceat(U, starts, axis=0)
    meat = (S ** 2).sum(0)
    se = np.sqrt(np.maximum(meat, 1e-30)) / denom
    return b, b / se


def run_block(d, names, tag, rng):
    built = build(d, names)
    if built is None:
        return [], []
    F, ys, sea, slices, nrows = built
    order = np.argsort(sea, kind="stable")
    ss = sea[order]
    starts = np.concatenate([[0], np.where(np.diff(ss) != 0)[0] + 1])
    denom = (F ** 2).sum(0)
    scale = float(np.median(denom[np.isfinite(denom)])) if len(denom) else 1.0
    good = denom > 1e-6 * max(scale, 1.0)
    if not good.all():
        print(f"  dropping {int((~good).sum())} degenerate columns "
              f"(nothing left after the controls)")
        F = F[:, good]
        names = [nm for nm, g in zip(names, good) if g]
        denom = denom[good]

    cells, maxes = [], []
    for h in HORIZONS:
        y = ys[h]
        b, t = coefs(F, y, order, starts, denom)
        for j, nm in enumerate(names):
            cells.append({"block": tag, "horizon": h, "feature": nm,
                          "bp": float(b[j] * 1e4), "t": float(t[j]),
                          "rows": int(len(y)), "seasons": int(len(starts))})
        draw_max = np.zeros(DRAWS)
        for i in range(DRAWS):
            yp = y.copy()
            for a_, b_ in slices:
                yp[a_:b_] = rng.permutation(yp[a_:b_])
            _, tp = coefs(F, yp, order, starts, denom)
            draw_max[i] = np.nanmax(np.abs(tp))
        maxes.append(draw_max)
    return cells, maxes


def main() -> int:
    d = pd.read_parquet(V4)
    dec = pd.read_parquet(DEC)
    xs = [c for c in dec.columns if c.startswith("x_")]
    d = d.merge(dec[["ticker", "D", "kind"] + xs], on=["ticker", "D", "kind"],
                how="left")
    d["no_inst20"] = d["no_inst20"].astype(float)

    rng = np.random.default_rng(23)
    all_cells, all_max = [], []

    for kind in ("provisional", "periodic"):
        sub = d[d["kind"] == kind]
        for btag, base in (("full-window", CORE), ("sub-type era", CORE + xs)):
            have = [c for c in base if c in sub.columns
                    and sub[c].notna().mean() > 0.25]
            names, expr = [], {}
            for c in have:
                names.append(c)
                expr[c] = (c, None)
            for a, b in itertools.combinations(have, 2):
                nm = f"{a}-{b}"
                names.append(nm)
                expr[nm] = (a, b)
            work = sub.copy()
            for nm, (a, b) in expr.items():
                if b is None:
                    continue
                za = work.groupby("D")[a].transform(lambda s: s.rank(pct=True))
                zb = work.groupby("D")[b].transform(lambda s: s.rank(pct=True))
                work[nm] = za - zb
            for c in have:
                nm = f"{c}*surprise"
                names.append(nm)
                work[nm] = (work.groupby("D")[c].transform(lambda s: s.rank(pct=True))
                            * work.groupby("D")["surprise"]
                                  .transform(lambda s: s.rank(pct=True)))
            tag = f"{kind}/{btag}"
            print(f"\n{tag}: {len(names)} features, "
                  f"{len(names) * len(HORIZONS)} cells")
            cells, maxes = run_block(work, names, tag, rng)
            print(f"  rows used {cells[0]['rows']:,}  "
                  f"seasons {cells[0]['seasons']}" if cells else "  empty")
            all_cells += cells
            all_max += maxes

    M = np.max(np.vstack(all_max), axis=0) if all_max else np.array([np.nan])
    thr = float(np.percentile(M, 95))
    print(f"\n{'=' * 72}")
    print(f"cells searched: {len(all_cells):,}   draws: {DRAWS}")
    print(f"family-wise threshold over the WHOLE grid: |t| > {thr:.2f}")
    print(f"  (a single test would use 1.96; the grid costs "
          f"{thr - 1.96:+.2f})")

    all_cells = [c for c in all_cells if np.isfinite(c["t"])]
    all_cells.sort(key=lambda c: -abs(c["t"]))
    print(f"\ntop 40 by |t|:")
    print(f"  {'t':>7} {'bp/SD':>9}  {'horizon':<7} {'block':<26} feature")
    for c in all_cells[:40]:
        mark = "  <-- SURVIVES" if abs(c["t"]) > thr else ""
        print(f"  {c['t']:>+7.2f} {c['bp']:>+9.1f}  {c['horizon']:<7} "
              f"{c['block']:<26} {c['feature']}{mark}")

    print("\nevery single feature, no combinations:")
    for c in sorted((c for c in all_cells
                     if "-" not in c["feature"] and "*" not in c["feature"]),
                    key=lambda c: -abs(c["t"]))[:20]:
        print(f"  {c['t']:>+7.2f} {c['bp']:>+9.1f}  {c['horizon']:<7} "
              f"{c['block']:<26} {c['feature']}")

    surv = [c for c in all_cells if abs(c["t"]) > thr]
    print(f"\nsurvivors: {len(surv)}")
    if not surv:
        print("  nothing in the grid beats what the grid achieves by chance.")

    OUT.write_text(json.dumps(
        {"draws": DRAWS, "cells": len(all_cells), "threshold_t": thr,
         "survivors": surv, "top50": all_cells[:50]},
        indent=2, ensure_ascii=False, default=str))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
