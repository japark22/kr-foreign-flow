#!/usr/bin/env python3
"""Separate the allocation wave from the decision about the name.

Every orthogonalisation here has been against price. Daily institutional
buying was residualised on momentum, on size, on volatility -- never on
institutional buying itself. A day when institutions add to Korean equities
lifts nearly every name, and that is an allocation rather than a view. The
crowding measure is the sum of the two, and an information coefficient of
0.015 is what a real signal looks like after being averaged with something
that carries nothing.

The decomposition has to happen in the units the feature is actually built
from, which is a ratio of twenty-day sums rather than an average of daily
ratios. Let d be the daily institutional share of the issuer's traded value.
Because value times share is simply the net buying, the value-weighted mean
of d over the window is mechanically the feature's own numerator over its own
denominator:

    sum(val * d) / sum(val) = sum(net) / sum(val) = r

So a factor model on d, aggregated with value weights, splits r exactly:

    d[i,t] = a[i] + b[i] * F[t] + e[i,t]
    r_sys  = sum(val * (a + b F)) / sum(val)
    r_idio = sum(val * e)         / sum(val)
    r      = r_sys + r_idio                    to machine precision

Both legs are computed independently rather than one by subtraction, so the
identity is a test and not a tautology. The tolerance is 1e-9, not 0.999:
an exact identity that is only approximately satisfied is a broken one.

Each leg is then z-scored against its own trailing history exactly as the
feature is, so the legs are comparable to the column they came from.

Second construction, same daily panel. Accumulation that trends looks
different from accumulation that churns, and the variance ratio separates them
against a null of exactly one. Above one is a campaign, and a campaign ends
when the target position is filled, which is a structural reason for an unwind
rather than an assumed one.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from krxflow import features, storage

ROOT = Path(__file__).resolve().parent
V6 = ROOT / "results" / "event_panel_v6.parquet"
V7 = ROOT / "results" / "event_panel_v7.parquet"
OUT = ROOT / "results" / "flow_factor.json"
START = "20100101"
BETA_WIN, BETA_MIN = 250, 120
VR_WIN, VR_MIN, VR_Q = 60, 40, 5
MIN_ADV, TOL = 1e8, 1e-9


def load_module(name):
    spec = importlib.util.spec_from_file_location(name.split("_")[0],
                                                  ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ep = load_module("20_event_panel")


def main() -> int:
    audit = json.loads((ROOT / "results" / "inst_coverage.json").read_text())
    ok_years = sorted(set(audit["eligible_checked"]) |
                      set(audit["imputed_unchecked_pre"]))

    print("building the daily panel ...")
    p = features.load_panels(START, None)
    idx, cols = p["foreign_pct"].index, p["foreign_pct"].columns

    m = storage.read_range("market", START, None,
                           columns=["trade_date", "ticker", "value_traded"])
    m["trade_date"] = pd.to_datetime(m["trade_date"])
    val = (m.pivot_table(index="trade_date", columns="ticker",
                         values="value_traded", aggfunc="last", observed=True)
             .sort_index().astype("float64").reindex(index=idx, columns=cols))
    del m

    iv = storage.read_range("investor_flow", START, None,
                            columns=["trade_date", "ticker", "investor", "net_value"])
    iv = iv[iv["investor"] == "기관합계"].copy()
    iv["trade_date"] = pd.to_datetime(iv["trade_date"])
    net = (iv.pivot_table(index="trade_date", columns="ticker",
                          values="net_value", aggfunc="last", observed=True)
             .sort_index().astype("float64").reindex(index=idx, columns=cols))
    del iv

    fill = (val.fillna(0) > 0) & net.isna()
    fill.loc[~np.isin(pd.DatetimeIndex(idx).year, ok_years), :] = False
    net = net.mask(fill, 0.0)

    valp = val.where(val > 0)
    d = (net / valp).astype("float64")

    adv = val.rolling(20, min_periods=5).mean()
    liquid = adv >= MIN_ADV
    F = d.where(liquid).mean(axis=1, skipna=True)
    breadth = d.where(liquid).notna().sum(axis=1)
    disp = d.where(liquid).std(axis=1, skipna=True)
    print(f"  factor on {int(breadth.median()):,} names on a median day")

    print("rolling betas on the daily share ...")
    md = d.rolling(BETA_WIN, min_periods=BETA_MIN).mean()
    mdF = d.mul(F, axis=0).rolling(BETA_WIN, min_periods=BETA_MIN).mean()
    mF = F.rolling(BETA_WIN, min_periods=BETA_MIN).mean()
    vF = F.rolling(BETA_WIN, min_periods=BETA_MIN).var()
    beta = (mdF.sub(md.mul(mF, axis=0))).div(vF.where(vF > 1e-24), axis=0)
    beta = beta.clip(-5, 5)
    alpha = md.sub(beta.mul(mF, axis=0))
    del mdF, md

    fitted = alpha.add(beta.mul(F, axis=0))
    resid = d.sub(fitted)

    print("value-weighted twenty-day legs ...")
    # Every sum has to see the same days. A day the factor model cannot fit
    # leaves the two legs but stays in the whole unless it is masked out of
    # both, and a day that is in one and not the other breaks the identity
    # this script exists to test. The first pass missed exactly that: the
    # median error was 9e-18 and the worst was 0.74, which is the signature
    # of an alignment fault rather than of arithmetic.
    good = valp.notna() & fitted.notna() & net.notna()
    w = valp.where(good)
    den = w.rolling(20, min_periods=10).sum()
    den = den.where(den > 0)
    r_sys = (w * fitted).rolling(20, min_periods=10).sum() / den
    r_idio = (w * resid).rolling(20, min_periods=10).sum() / den
    r_all = net.where(good).rolling(20, min_periods=10).sum() / den
    print(f"  days the factor model could not fit, dropped from all four: "
          f"{1 - float(good.to_numpy().sum()) / float(valp.notna().to_numpy().sum()):.3f}")
    del fitted, resid, w, good

    print("\nself-check: the identity, at machine precision")
    diff = (r_sys + r_idio - r_all).abs()
    mask = r_all.notna() & r_sys.notna() & r_idio.notna()
    worst = float(diff.where(mask).max().max())
    med = float(diff.where(mask).stack().median())
    n = int(mask.to_numpy().sum())
    print(f"  cells {n:,}   median |error| {med:.3e}   worst {worst:.3e}")
    R = {"identity_cells": n, "identity_median_abs": med,
         "identity_worst_abs": worst, "tolerance": TOL}
    if not np.isfinite(worst) or worst > TOL:
        print(f"  STOP -- the legs do not sum to their own input within {TOL:g}")
        OUT.write_text(json.dumps(R, indent=2, ensure_ascii=False))
        return 1
    print("  ok")

    print("\nz-scoring each leg the way the feature is z-scored ...")
    i_idio20 = ep.zwin(r_idio)
    i_sys20 = ep.zwin(r_sys)
    i_all20 = ep.zwin(r_all)

    print("variance ratio of the daily share ...")
    d5 = d.rolling(VR_Q, min_periods=VR_Q).sum()
    v5 = d5.rolling(VR_WIN, min_periods=VR_MIN).var()
    v1 = d.rolling(VR_WIN, min_periods=VR_MIN).var()
    vr = (v5 / (VR_Q * v1.where(v1 > 1e-24))).clip(0, 6)
    del d5, v5, v1

    panels = {"flow_beta": beta, "i_idio20": i_idio20, "i_sys20": i_sys20,
              "i_all20": i_all20, "flow_vr": vr}

    ev = pd.read_parquet(V6)
    colpos = {t: j for j, t in enumerate(cols)}
    ev = ev[ev["ticker"].isin(colpos)].reset_index(drop=True)
    pos = idx.searchsorted(ev["D"].to_numpy())
    jj = ev["ticker"].map(colpos).to_numpy()
    for k, panel in panels.items():
        ev[k] = panel.to_numpy()[pos - 1, jj].astype("float32")
        print(f"  {k:<12} coverage={ev[k].notna().mean():.3f}  "
              f"sd={ev[k].std():.4f}  "
              f"corr(i_flow20_v2)={ev[[k, 'i_flow20_v2']].corr().iloc[0, 1]:+.4f}")

    print("\nsecond check: does the rebuilt whole match the panel's column?")
    b2 = ev["i_all20"].notna() & ev["i_flow20_v2"].notna()
    rho2 = float(ev.loc[b2, "i_all20"].corr(ev.loc[b2, "i_flow20_v2"]))
    print(f"  cells {int(b2.sum()):,}   correlation {rho2:.6f}")
    R["rebuild_vs_panel_corr"] = rho2
    if rho2 < 0.99:
        print("  STOP -- the rebuilt column is not the panel's column")
        OUT.write_text(json.dumps(R, indent=2, ensure_ascii=False))
        return 1
    print("  ok")

    pd.DataFrame({"trade_date": idx, "inst_factor": F.to_numpy(),
                  "inst_dispersion": disp.to_numpy(),
                  "breadth": breadth.to_numpy()}
                 ).to_parquet(ROOT / "results" / "inst_market_state.parquet",
                              index=False)
    R.update({"vr_median": float(ev["flow_vr"].median()),
              "vr_share_above_one": float((ev["flow_vr"] > 1).mean()),
              "beta_median": float(ev["flow_beta"].median()),
              "corr_idio_sys": float(ev[["i_idio20", "i_sys20"]]
                                     .corr().iloc[0, 1])})
    print(f"\n  variance ratio median {R['vr_median']:.3f}, "
          f"{R['vr_share_above_one']:.1%} above one")
    print(f"  flow beta median {R['beta_median']:.3f}")
    print(f"  correlation between the two legs {R['corr_idio_sys']:+.3f}")

    ev.to_parquet(V7, index=False)
    OUT.write_text(json.dumps(R, indent=2, ensure_ascii=False))
    print(f"\nwrote {V7.relative_to(ROOT)} and {OUT.relative_to(ROOT)}")
    print("no coefficients here -- this script builds and checks only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
