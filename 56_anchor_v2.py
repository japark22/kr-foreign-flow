#!/usr/bin/env python3
"""Re-anchor on the recovered sample, and separate the reasons it moved.

Filling the omitted cells changes two things at once: the 20-day window of
events that were already in the sample, and the set of events itself. A
single headline number cannot tell those apart, so each is isolated:

  A  original column, original sample          reproduction control
  B  recovered column, full sample             the new headline
  C  B minus the events that exist only        pre-registration 10b
     because of unchecked pre-2018 imputation
  D  recovered column, but only the events     isolates the window
     that were already in the sample           recomputation
  E  continuous effect among participating     pre-registration 10c
     names, and no-participation as its own
     state

The estimator is imported, not restated.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

def rank_corr(a, b):
    """Spearman without the dependency: rank, then correlate."""
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "results" / "event_panel_v2.parquet"
OUT = ROOT / "results" / "anchor_v2.json"


def load(name):
    spec = importlib.util.spec_from_file_location(name.split("_")[0],
                                                  ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


fr = load("38_final_results")
CTRL, MIN_N = fr.CTRL, fr.MIN_N


def run(sub, base, tag):
    sub = sub.copy()
    p = fr.prepare(sub, base, ["surprise"] + CTRL + [base])
    if p is None:
        return {"tag": tag, "bp": None}
    r = fr.fit(p)
    r["tag"] = tag
    return r


def line(r):
    if r.get("bp") is None:
        return f"  {r['tag']:<46}  --"
    return (f"  {r['tag']:<46}{r['bp']:+8.1f} bp/SD   t {r['t']:+5.2f}"
            f"   events {r['events']:>7,}   seasons {r['seasons']:>3}")


def main() -> int:
    d = pd.read_parquet(PANEL)
    d["i_v2"] = d["i_flow20_v2"]
    d["i_v1"] = d["i_flow20"]
    d["no_inst"] = d["no_inst20"].astype(float)
    R = {"min_n": int(MIN_N)}

    # ---- how much did the already-present cells actually move?
    print("change on cells that were already present")
    both = d["i_v1"].notna() & d["i_v2"].notna()
    delta = (d.loc[both, "i_v2"] - d.loc[both, "i_v1"])
    q = delta.abs().quantile([0.5, 0.9, 0.99, 1.0])
    print(f"  cells {int(both.sum()):,}   "
          f"|delta| p50={q[0.5]:.4f} p90={q[0.9]:.4f} "
          f"p99={q[0.99]:.4f} max={q[1.0]:.4f}")
    print(f"  share moved more than 1e-6: {float((delta.abs() > 1e-6).mean()):.4f}")

    # the estimator sees ranks, so rank agreement is what matters
    rs = []
    for _, g in d[both].groupby("D"):
        if len(g) < MIN_N:
            continue
        rs.append(rank_corr(g["i_v1"], g["i_v2"]))
    rs = np.array([x for x in rs if np.isfinite(x)])
    print(f"  within-day rank correlation: mean={rs.mean():.4f} "
          f"p05={np.percentile(rs, 5):.4f}  days={len(rs):,}")
    R["cell_change"] = {"cells": int(both.sum()),
                        "abs_delta_p50": float(q[0.5]),
                        "abs_delta_p99": float(q[0.99]),
                        "share_moved": float((delta.abs() > 1e-6).mean()),
                        "rank_corr_mean": float(rs.mean()),
                        "rank_corr_p05": float(np.percentile(rs, 5))}

    new_pre = (d["i_v1"].isna() & d["i_v2"].notna()
               & (d["D"] < "2018-01-01"))
    print(f"\n  events added by unchecked pre-2018 imputation: {int(new_pre.sum()):,}")
    print(f"  events added in total: "
          f"{int((d['i_v1'].isna() & d['i_v2'].notna()).sum()):,}")

    # ---- the ladder
    for kind_tag, sub in (("all events", d),
                          ("provisional", d[d["kind"] == "provisional"]),
                          ("periodic", d[d["kind"] == "periodic"])):
        print(f"\n=== {kind_tag} ===")
        rows = [
            run(sub, "i_v1", "A  original column, original sample"),
            run(sub, "i_v2", "B  recovered column, full sample"),
            run(sub[~new_pre.reindex(sub.index, fill_value=False)],
                "i_v2", "C  B minus unchecked pre-2018 additions"),
            run(sub[sub["i_v1"].notna()],
                "i_v2", "D  recovered column, original events only"),
            run(sub[~sub["no_inst20"].fillna(False)],
                "i_v2", "E1 continuous, participating names only"),
            run(sub, "no_inst", "E2 no-participation state, full sample"),
            run(sub, "f_flow20", "F  foreign, unchanged (invariance check)"),
        ]
        for r in rows:
            print(line(r))
        R[kind_tag] = {r["tag"][:2].strip(): r for r in rows}

    OUT.write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
