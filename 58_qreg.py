#!/usr/bin/env python3
"""The conditional distribution baseline, on the recovered sample.

Everything estimated so far has been a mean. The finding this project is
built on is not in the mean -- it is in the lower tail, where the mean and
the hit rate do not move. An estimator of the mean is the wrong instrument
for it, so the declared estimator is quantile regression across a tau grid.
See PRE_REGISTRATION.md sections 5 to 7.

No quantile-regression library is installed, so the fit is iteratively
reweighted least squares on the check function. That is only worth anything
if it is correct, so every fit reports the share of residuals below zero,
which must land on tau. A fit that misses its own quantile is reported as a
failure and its coefficients are not read.

Inference is a block bootstrap over reporting seasons. The placebo shuffles
the feature inside each day, 200 draws, and is reported as a band at every
tau -- a single draw has error as wide as the effect.

This is a COMPONENT test. The pre-registered primary is the composite
feature, which is not built yet. Nothing here is the primary result.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "results" / "event_panel_v3.parquet"
OUT = ROOT / "results" / "qreg.json"
TAUS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90]


def load(name):
    spec = importlib.util.spec_from_file_location(name.split("_")[0],
                                                  ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


fr = load("38_final_results")
CTRL = fr.CTRL


def qreg(X, y, tau, iters=80, tol=1e-9):
    """Quantile regression by iteratively reweighted least squares."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    floor = 1e-6 * (np.std(y) + 1e-12)
    eye = 1e-10 * np.eye(X.shape[1])
    for _ in range(iters):
        r = y - X @ beta
        w = np.where(r > 0, tau, 1.0 - tau) / np.maximum(np.abs(r), floor)
        Xw = X * w[:, None]
        try:
            nb = np.linalg.solve(X.T @ Xw + eye, Xw.T @ y)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(nb - beta)) < tol:
            beta = nb
            break
        beta = nb
    return beta


def fit_check(X, y, tau):
    b = qreg(X, y, tau)
    frac = float(np.mean((y - X @ b) < 0))
    return b, frac


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="provisional",
                    choices=["provisional", "periodic", "all"])
    ap.add_argument("--base", default="i_flow20_v2")
    ap.add_argument("--y", default="abn60",
                    choices=["abn5", "abn20", "abn60"])
    ap.add_argument("--boots", type=int, default=300)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--surprise-top", type=int, default=0,
                    help="keep only the top N-th quintile of surprise within the day")
    ap.add_argument("--deep", action="store_true",
                    help="restrict to names with participation on 14+ of 20 days")
    a = ap.parse_args()

    d = pd.read_parquet(PANEL)
    if a.kind != "all":
        d = d[d["kind"] == a.kind]
    if a.deep:
        d = d[d["inst_days20"] >= 14]
    if a.surprise_top:
        q = d.groupby(["D", "kind"])["surprise"].rank(pct=True)
        d = d[q > 1.0 - a.surprise_top / 5.0]
    print(f"{a.kind}  base={a.base}  deep={a.deep}  rows={len(d):,}")

    prep = fr.prepare(d, a.base, ["surprise"] + CTRL + [a.base], a.y)
    if prep is None:
        print("no usable cross-sections")
        return 1
    Xo, xb, y, sea, slices = prep
    X = np.column_stack([xb, Xo])
    seasons = np.unique(sea)
    rows_by_season = {s: np.where(sea == s)[0] for s in seasons}
    print(f"  events {len(y):,}   seasons {len(seasons)}")

    R = {"kind": a.kind, "base": a.base, "y": a.y, "deep": bool(a.deep),
         "events": int(len(y)), "seasons": int(len(seasons)),
         "boots": a.boots, "draws": a.draws, "taus": {}}

    # the mean, for the signature test in PRE_REGISTRATION section 2
    mean_fit = fr.fit(prep)
    print(f"\n  mean (season-clustered)   {mean_fit['bp']:+8.1f} bp/SD   "
          f"t {mean_fit['t']:+5.2f}")
    R["mean"] = mean_fit

    rng = np.random.default_rng(7)

    # point estimates first, so a failed fit is caught before any resampling
    point, frac_ok = {}, {}
    for tau in TAUS:
        b, frac = fit_check(X, y, tau)
        point[tau] = float(b[0] * 1e4)
        frac_ok[tau] = (frac, abs(frac - tau) < 0.02)
    if not all(v[1] for v in frac_ok.values()):
        for tau in TAUS:
            print(f"  tau {tau}: frac<0 = {frac_ok[tau][0]:.3f}")
        print("  FIT FAILED -- coefficients not read")
        return 1

    # bootstrap: one season draw shared by every tau
    boot = {tau: [] for tau in TAUS}
    for _ in range(a.boots):
        pick = rng.choice(seasons, size=len(seasons), replace=True)
        idxs = np.concatenate([rows_by_season[s] for s in pick])
        Xb, yb = X[idxs], y[idxs]
        for tau in TAUS:
            boot[tau].append(qreg(Xb, yb, tau)[0] * 1e4)
    sd = {tau: float(np.std(boot[tau], ddof=1)) for tau in TAUS}

    # placebo: one permutation shared by every tau, so a family-wise
    # maximum over the grid is a maximum over the same draw
    plac = {tau: [] for tau in TAUS}
    for _ in range(a.draws):
        xp = xb.copy()
        for s_, e_ in slices:
            xp[s_:e_] = rng.permutation(xp[s_:e_])
        Xp = np.column_stack([xp, Xo])
        for tau in TAUS:
            plac[tau].append(qreg(Xp, y, tau)[0] * 1e4)

    # family-wise: standardise each draw by the bootstrap scale, take the
    # maximum absolute value across the grid, and read its 95th percentile
    Tmax = []
    for i in range(a.draws):
        ts = [abs(plac[tau][i] - np.mean(plac[tau])) / sd[tau]
              for tau in TAUS if sd[tau] > 1e-12]
        Tmax.append(max(ts) if ts else np.nan)
    fwe = float(np.nanpercentile(Tmax, 95))

    print(f"\n  {'tau':>5} {'frac<0':>8} {'bp/SD':>10} {'t':>7} "
          f"{'placebo p05':>12} {'placebo p95':>12}  verdict")
    for tau in TAUS:
        t = point[tau] / sd[tau] if sd[tau] > 1e-12 else np.nan
        lo, hi = np.percentile(plac[tau], [5, 95])
        inside = lo <= point[tau] <= hi
        passes_fwe = abs(t) > fwe
        verdict = ("inside placebo band" if inside else
                   "outside band, clears family-wise" if passes_fwe else
                   "outside band, fails family-wise")
        print(f"  {tau:>5.2f} {frac_ok[tau][0]:>8.3f} {point[tau]:>+10.1f} "
              f"{t:>+7.2f} {lo:>+12.1f} {hi:>+12.1f}  {verdict}")
        R["taus"][str(tau)] = {"frac_below": frac_ok[tau][0], "fit_ok": True,
                               "bp": point[tau], "t": float(t), "boot_sd": sd[tau],
                               "placebo_p05": float(lo), "placebo_p95": float(hi),
                               "placebo_mean": float(np.mean(plac[tau])),
                               "inside_band": bool(inside),
                               "clears_fwe": bool(passes_fwe)}
    R["fwe_threshold_t"] = fwe
    print(f"\n  family-wise threshold across the grid: |t| > {fwe:.2f}")


    tag = (f"{a.kind}{'_deep' if a.deep else ''}"
       f"{'_sq' + str(a.surprise_top) if a.surprise_top else ''}_{a.base}_{a.y}")
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    prev[tag] = R
    OUT.write_text(json.dumps(prev, indent=2, ensure_ascii=False, default=str))
    print(f"wrote {OUT.relative_to(ROOT)}  [{tag}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
