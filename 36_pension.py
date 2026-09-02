"""Confirm or kill the pension-fund result.

35_decompose searched eight investor types under family-wise control and one
cleared the bar: 연기금 at -64.7 bp/SD, t -2.95 against a bar of 2.91. That
margin is thinner than the Monte Carlo error of a 200-permutation bar, and a
single surviving cell is exactly where a research programme fools itself.

Five checks, each with its sign written down before it is run. This is
confirmation of one hypothesis, not a second search, so nothing here reopens
the multiplicity problem the bar was built to control.

  1  the bar at 1,000 permutations        expect: still clears
  2  first half vs second half            expect: both negative
  3  abn5 / abn20 / abn60                 expect: abn60 strongest
  4  연기금 and 기관합계 together           expect: 연기금 holds, aggregate
                                          collapses toward zero
  5  event-weighted portfolio contrast    expect: beats that the pension fund
                                          was NOT buying beat all beats

Same estimator throughout: provisional filings, one vote per event,
regressors rank-standardised within the day, returns winsorised 1/99 within
the day, standard errors clustered on the reporting season.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("results/decompose_panel.parquet")
OUT = Path("results/pension.json")
PENSION, AGG, MIN_N = "x_연기금", "i_flow20", 5
FAMILY = ["금융투자", "보험", "투신", "사모", "은행", "기타금융", "연기금",
          "기타법인"]

spec = importlib.util.spec_from_file_location("bfm", "21_baseline_fm.py")
bfm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bfm)
CTRL = list(bfm.CTRL)


def season(d):
    d = pd.to_datetime(d)
    return d.dt.year * 4 + (d.dt.month - 1) // 3


def prepare(sub, base, xs, y):
    xo, xb, ys, sea, slices = [], [], [], [], []
    n = 0
    others = [c for c in xs if c != base]
    for day, g in sub.groupby("D"):
        g = g.dropna(subset=[y] + xs)
        if len(g) < MIN_N:
            continue
        xo.append(np.column_stack([bfm.rank_std(g[c]) for c in others]
                                  + [np.ones(len(g))]))
        xb.append(bfm.rank_std(g[base]))
        v = g[y].to_numpy(dtype=float)
        lo, hi = np.percentile(v, [1, 99])
        ys.append(np.clip(v, lo, hi))
        t = pd.Timestamp(day)
        sea.append(np.full(len(g), t.year * 4 + (t.month - 1) // 3))
        slices.append((n, n + len(g)))
        n += len(g)
    if not slices:
        return None
    return (np.vstack(xo), np.concatenate(xb), np.concatenate(ys),
            np.concatenate(sea), slices)


def fit(prep, rng=None, want=0):
    Xo, xb, y, sea, slices = prep
    x = xb
    if rng is not None:
        x = xb.copy()
        for s, e in slices:
            x[s:e] = rng.permutation(x[s:e])
    X = np.column_stack([x, Xo])
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(sea):
        m = sea == c
        u = X[m].T @ e[m]
        meat += np.outer(u, u)
    V = XtX_inv @ meat @ XtX_inv
    se = float(np.sqrt(max(V[want, want], 1e-30)))
    return float(beta[want] * 1e4), float(beta[want] / se)


def line(tag, r, extra=""):
    print(f"  {tag:<34}{r[0]:+8.1f} bp/SD   t {r[1]:+5.2f}   {extra}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260903)
    a = ap.parse_args()

    d = pd.read_parquet(CACHE)
    d["D"] = pd.to_datetime(d["D"])
    sub = d[d["kind"] == "provisional"].copy()
    print(f"provisional {len(sub):,} events   CTRL {CTRL}\n")
    out = {}

    xs_p = ["surprise"] + CTRL + [PENSION]
    base_prep = prepare(sub, PENSION, xs_p, "abn60")
    r0 = fit(base_prep)
    line("연기금 / abn60 (the claim)", r0, f"events {len(base_prep[2]):,}")
    out["headline"] = {"coef_bp": r0[0], "t": r0[1]}

    # 1 --------------------------------------------------------------
    print(f"\n1  family-wise bar at {a.perm} permutations "
          f"(expect: still clears)")
    preps = {}
    for c in FAMILY:
        p = prepare(sub, f"x_{c}", ["surprise"] + CTRL + [f"x_{c}"], "abn60")
        if p is not None:
            preps[c] = p
    rng = np.random.default_rng(a.seed)
    maxes = np.empty(a.perm)
    for i in range(a.perm):
        maxes[i] = max(abs(fit(preps[c], rng)[1]) for c in preps)
        if (i + 1) % 100 == 0:
            print(f"     {i + 1}/{a.perm}   bar "
                  f"{np.quantile(maxes[:i + 1], 0.95):.2f}")
    bar = float(np.quantile(maxes, 0.95))
    p_f = float((maxes >= abs(r0[1])).mean())
    print(f"   bar {bar:.2f}   observed {abs(r0[1]):.2f}   p {p_f:.4f}"
          f"   -> {'CLEARS' if abs(r0[1]) >= bar else 'FAILS'}")
    out["bar"] = {"bar": bar, "p_fwer": p_f, "perm": a.perm}

    # 2 --------------------------------------------------------------
    print("\n2  split halves (expect: both negative)")
    cut = sub["D"].quantile(0.5)
    for tag, s in (("first half", sub[sub["D"] <= cut]),
                   ("second half", sub[sub["D"] > cut])):
        p = prepare(s, PENSION, xs_p, "abn60")
        r = fit(p)
        line(f"   {tag}", r, f"events {len(p[2]):,}")
        out.setdefault("halves", {})[tag] = {"coef_bp": r[0], "t": r[1]}

    # 3 --------------------------------------------------------------
    print("\n3  horizons (expect: abn60 strongest)")
    for y in ("abn5", "abn20", "abn60"):
        p = prepare(sub, PENSION, ["surprise"] + CTRL + [PENSION], y)
        r = fit(p)
        line(f"   {y}", r)
        out.setdefault("horizons", {})[y] = {"coef_bp": r[0], "t": r[1]}

    # 4 --------------------------------------------------------------
    print("\n4  연기금 with 기관합계 (expect: pension holds, aggregate falls)")
    xs_j = ["surprise"] + CTRL + [AGG, PENSION]
    pj = prepare(sub, PENSION, xs_j, "abn60")
    rp = fit(pj, want=0)
    line("   연기금 (aggregate controlled)", rp, f"events {len(pj[2]):,}")
    pj2 = prepare(sub, AGG, xs_j, "abn60")
    ra = fit(pj2, want=0)
    line("   기관합계 (pension controlled)", ra)
    out["joint"] = {"pension": {"coef_bp": rp[0], "t": rp[1]},
                    "aggregate": {"coef_bp": ra[0], "t": ra[1]}}

    # 5 --------------------------------------------------------------
    print("\n5  portfolio contrast, one vote per position "
          "(expect: quiet beats > all beats)")
    s = sub.dropna(subset=["abn60", "surprise", PENSION]).copy()
    g = s.groupby("D")
    s["rs"] = g["surprise"].rank(pct=True)
    s["rp"] = g[PENSION].rank(pct=True)
    s["season"] = season(s["D"])
    beats = s[s["rs"] >= 0.7]
    quiet = beats[beats["rp"] <= 0.3]
    crowd = beats[beats["rp"] >= 0.7]

    def contrast(a_, b_):
        ma = a_.groupby("season")["abn60"].mean()
        mb = b_.groupby("season")["abn60"].mean()
        common = ma.index.intersection(mb.index)
        diff = (ma[common] - mb[common]).to_numpy()
        return (float(diff.mean() * 1e4),
                float(bfm.tracker._nw_t(diff, 1)), len(diff))

    for tag, x, y in (("quiet beats - all beats", quiet, beats),
                      ("crowded beats - all beats", crowd, beats),
                      ("quiet - crowded", quiet, crowd)):
        c, t, ns = contrast(x, y)
        print(f"   {tag:<28}{c:+8.1f} bp/position   t {t:+5.2f}"
              f"   seasons {ns}")
        out.setdefault("portfolio", {})[tag] = {"bp": c, "t": t,
                                                "seasons": ns}
    print(f"   sizes: all beats {len(beats):,}  quiet {len(quiet):,}  "
          f"crowded {len(crowd):,}")

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
