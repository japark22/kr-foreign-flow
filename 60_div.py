#!/usr/bin/env python3
"""Disagreement between investor types -- the one construction not yet tried.

Everything tested so far has been the same variable cut differently. This is
different information. Each type's twenty-day intensity contains the same
contemporaneous price-impact term, so a difference of standardised
intensities removes it to first order and leaves disagreement.

    DIV = z(a) - z(b),  z taken within (D, kind)

Qualification is applied before estimation, as declared in
PRE_REGISTRATION.md section 4: a feature correlating 0.20 or more with the
twenty-day run-up is momentum in costume and is dropped, whichever way the
result would have gone.

The bar for carrying a feature to another market is |t| >= 3 on this sample.
A smaller effect cannot be adjudicated in a smaller market, so confirming it
there is not possible and porting it is not research.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V3 = ROOT / "results" / "event_panel_v3.parquet"
DEC = ROOT / "results" / "decompose_panel.parquet"
OUT = ROOT / "results" / "div.json"
TAUS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90]
BOOTS, DRAWS = 200, 100
PORT_BAR = 3.0
MOM_MAX = 0.20


def load(name):
    spec = importlib.util.spec_from_file_location(name.split("_")[0],
                                                  ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


fr = load("38_final_results")
qr = load("58_qreg")
bfm = load("21_baseline_fm")
CTRL = fr.CTRL


def zwithin(d, col):
    def f(s):
        v = s.dropna()
        if len(v) < 10:
            return pd.Series(np.nan, index=s.index)
        out = pd.Series(np.nan, index=s.index)
        out.loc[v.index] = bfm.rank_std(v)
        return out
    return d.groupby(["D", "kind"])[col].transform(f)


def grid(d, base, tag, seed=5):
    prep = fr.prepare(d, base, ["surprise"] + CTRL + [base])
    if prep is None:
        print(f"  {tag}: no usable cross-sections")
        return None
    Xo, xb, y, sea, slices = prep
    X = np.column_stack([xb, Xo])
    seasons = np.unique(sea)
    rows = {s: np.where(sea == s)[0] for s in seasons}
    rng = np.random.default_rng(seed)

    mean = fr.fit(prep)
    print(f"\n  {tag}   events {len(y):,}   seasons {len(seasons)}")
    print(f"    mean  {mean['bp']:+8.1f} bp/SD   t {mean['t']:+5.2f}"
          f"   {'PORTABLE' if abs(mean['t']) >= PORT_BAR else 'below the port bar'}")

    point = {}
    for tau in TAUS:
        b, frac = qr.fit_check(X, y, tau)
        if abs(frac - tau) > 0.02:
            print(f"    tau {tau}: fit landed at {frac:.3f} -- FAILED")
            return None
        point[tau] = float(b[0] * 1e4)

    boot = {t_: [] for t_ in TAUS}
    for _ in range(BOOTS):
        pick = rng.choice(seasons, size=len(seasons), replace=True)
        idx = np.concatenate([rows[s] for s in pick])
        for t_ in TAUS:
            boot[t_].append(qr.qreg(X[idx], y[idx], t_)[0] * 1e4)
    sd = {t_: float(np.std(boot[t_], ddof=1)) for t_ in TAUS}

    plac = {t_: [] for t_ in TAUS}
    for _ in range(DRAWS):
        xp = xb.copy()
        for a_, b_ in slices:
            xp[a_:b_] = rng.permutation(xp[a_:b_])
        Xp = np.column_stack([xp, Xo])
        for t_ in TAUS:
            plac[t_].append(qr.qreg(Xp, y, t_)[0] * 1e4)

    Tmax = [max(abs(plac[t_][i] - np.mean(plac[t_])) / sd[t_]
                for t_ in TAUS if sd[t_] > 1e-12) for i in range(DRAWS)]
    fwe = float(np.percentile(Tmax, 95))

    res = {"mean": mean, "fwe_threshold_t": fwe, "taus": {}}
    for t_ in TAUS:
        tt = point[t_] / sd[t_] if sd[t_] > 1e-12 else np.nan
        lo, hi = np.percentile(plac[t_], [5, 95])
        inside = lo <= point[t_] <= hi
        flag = ("inside band" if inside
                else "outside, clears family-wise" if abs(tt) > fwe
                else "outside, fails family-wise")
        print(f"    tau {t_:.2f}  {point[t_]:+9.1f}  t {tt:+5.2f}  "
              f"[{lo:+7.1f},{hi:+7.1f}]  {flag}")
        res["taus"][str(t_)] = {"bp": point[t_], "t": float(tt),
                                "placebo_p05": float(lo), "placebo_p95": float(hi),
                                "inside_band": bool(inside)}
    return res


def main() -> int:
    d = pd.read_parquet(V3)
    dec = pd.read_parquet(DEC)
    xs = [c for c in dec.columns if c.startswith("x_")]
    d = d.merge(dec[["ticker", "D", "kind"] + xs], on=["ticker", "D", "kind"],
                how="left")

    print("building disagreement features ...")
    for c in ["i_flow20_v2", "f_flow20"] + xs:
        d[f"z_{c}"] = zwithin(d, c)

    variants = {
        "DIV inst - foreign":      ("z_i_flow20_v2", "z_f_flow20"),
        "DIV pension - private":   ("z_x_연기금", "z_x_사모"),
        "DIV trust - private":     ("z_x_투신", "z_x_사모"),
        "DIV inst - retail":       ("z_i_flow20_v2", "z_x_개인"),
        "DIV foreign - retail":    ("z_f_flow20", "z_x_개인"),
    }
    R = {"port_bar_t": PORT_BAR, "momentum_max_corr": MOM_MAX}

    print("\nqualification gate: |corr with the 20-day run-up| < 0.20")
    keep = {}
    for tag, (a, b) in variants.items():
        if a not in d.columns or b not in d.columns:
            print(f"  {tag:<26} missing input, skipped")
            continue
        name = "div_" + str(abs(hash(tag)) % 10**6)
        d[name] = d[a] - d[b]
        cov = float(d[name].notna().mean())
        c = float(d[[name, "c_mom20"]].corr().iloc[0, 1])
        ok = abs(c) < MOM_MAX
        print(f"  {tag:<26} corr={c:+.3f}  coverage={cov:.3f}  "
              f"{'qualifies' if ok else 'DROPPED as momentum in costume'}")
        R[tag] = {"corr_mom20": c, "coverage": cov, "qualifies": ok}
        if ok:
            keep[tag] = name

    prov = d[d["kind"] == "provisional"]
    for tag, name in keep.items():
        r = grid(prov, name, f"{tag}  [provisional]")
        if r:
            R[tag]["provisional"] = r

    OUT.write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print(f"\nnothing here is carried to another market unless |t| >= {PORT_BAR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
