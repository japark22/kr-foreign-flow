#!/usr/bin/env python3
"""The same comparison, with the things crowding travels with taken out.

The bucket table has no controls. Its placebo shuffles crowding, so it tests
against no relationship at all -- not against no relationship beyond size,
volatility, turnover and momentum. If crowded names are systematically
larger or stronger, the whole downward shift could be those factors and the
placebo would never notice.

So the same question is asked inside the top surprise quintile with the
controls in place: the mean through the project's anchor estimator with
season and two-way clusters, and the tenth percentile through the quantile
regression that was checked against its own quantile earlier today. The
placebo shuffles the feature within the day, 200 draws, as everywhere else.

The controls include momentum at 120, 250 and 500 days, because the earlier
pass showed 250-day momentum takes about thirty percent of anything that
looks like this.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V6 = ROOT / "results" / "event_panel_v6.parquet"
OUT = ROOT / "results" / "controlled_table.json"
RUNG, DRAWS, BOOTS = 10, 200, 200
TAUS = [0.10, 0.25, 0.50]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name.split("_")[0],
                                                  ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


fr = load_module("38_final_results")
qr = load_module("58_qreg")
th = load_module("64_threats_flevel")
SOFT = list(fr.CTRL)
HARD = th.CTRL + th.LONG


def main() -> int:
    d = pd.read_parquet(V6)
    d = d[d["inst_days20"] >= RUNG].copy()
    d["sq"] = d.groupby("D")["surprise"].rank(pct=True)
    R = {"rung": RUNG, "draws": DRAWS}

    for kind in (None, "provisional"):
        key = kind or "all"
        base = d if kind is None else d[d["kind"] == kind]
        for scope, sub in (("top surprise quintile", base[base["sq"] > 0.8]),
                           ("every surprise level", base)):
            for cname, ctrl in (("standard", SOFT), ("plus long momentum", HARD)):
                sub2 = sub.dropna(subset=["abn60", "i_flow20_v2"] + ctrl)
                prep = fr.prepare(sub2, "i_flow20_v2",
                                  ["surprise"] + ctrl + ["i_flow20_v2"])
                if prep is None:
                    continue
                Xo, xb, y, sea, slices = prep
                X = np.column_stack([xb, Xo])
                tag = f"{key} / {scope} / {cname}"
                print(f"\n{tag}")
                print(f"  events {len(y):,}   seasons {len(np.unique(sea))}")

                m = fr.fit(prep)
                pack = th.prepare(sub2, "i_flow20_v2", ctrl)
                tw = th.fit(pack) if pack else None
                print(f"    mean       {m['bp']:+8.1f} bp/SD   "
                      f"t(season) {m['t']:+5.2f}"
                      + (f"   t(two-way) {tw['t_two_way']:+5.2f}" if tw else ""))
                R[f"{tag} / mean"] = {"bp": m["bp"], "t_season": m["t"],
                                      "t_two_way": tw["t_two_way"] if tw else None,
                                      "events": m["events"]}

                rng = np.random.default_rng(97)
                for tau in TAUS:
                    b, frac = qr.fit_check(X, y, tau)
                    if abs(frac - tau) > 0.02:
                        print(f"    tau {tau:.2f}   fit landed at {frac:.3f} "
                              f"-- not read")
                        continue
                    point = float(b[0] * 1e4)
                    bs = []
                    for _ in range(BOOTS):
                        pick = rng.choice(np.unique(sea), size=len(np.unique(sea)),
                                          replace=True)
                        idx = np.concatenate([np.where(sea == s)[0] for s in pick])
                        bs.append(qr.qreg(X[idx], y[idx], tau)[0] * 1e4)
                    sd = float(np.std(bs, ddof=1))
                    pl = []
                    for _ in range(DRAWS):
                        xp = xb.copy()
                        for a_, b_ in slices:
                            xp[a_:b_] = rng.permutation(xp[a_:b_])
                        pl.append(qr.qreg(np.column_stack([xp, Xo]), y, tau)[0] * 1e4)
                    lo, hi = np.percentile(pl, [5, 95])
                    outside = point < lo or point > hi
                    print(f"    tau {tau:.2f}   {point:+8.1f} bp/SD   "
                          f"t {point / sd:+5.2f}   "
                          f"placebo [{lo:+.1f}, {hi:+.1f}]   "
                          f"{'OUTSIDE' if outside else 'inside'}")
                    R[f"{tag} / tau{tau}"] = {
                        "bp": point, "t": float(point / sd),
                        "p05": float(lo), "p95": float(hi),
                        "outside_band": bool(outside)}

    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2, default=str))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print("\nif the mean and tau 0.10 hold up with long momentum in, the bucket")
    print("table is measuring crowding. if they collapse, it was measuring the")
    print("company, not the positioning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
