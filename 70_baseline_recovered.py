#!/usr/bin/env python3
"""The published baseline, re-measured on the recovered sample.

The conditional table in the record was built by 50_baseline.py, which
compares percentiles inside each reporting season and therefore keeps all
twenty thousand events. Earlier today the same claim was re-tested with a
quantile regression that requires twenty-five names in every daily
cross-section, and inside the top surprise quintile most days do not have
them -- that run kept 2,393 events, roughly a ninth of the original, and
calling the result a failure to reproduce was wrong.

This runs the original estimator, unchanged, twice on the recovered panel:
once with the column as it was, which must return the published numbers,
and once with the recovered column, which is the actual question. Anything
that differs between the two is the recovered sample and nothing else.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V6 = ROOT / "results" / "event_panel_v6.parquet"
OUT = ROOT / "results" / "baseline_recovered.json"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name.split("_")[0],
                                                  ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tercile_report(col):
    d = pd.read_parquet(V6)
    for kind in ("provisional", "periodic"):
        sub = d[d["kind"] == kind].dropna(subset=["abn60", "surprise", col])
        cq = sub.groupby("D")[col].rank(pct=True)
        c = np.clip((cq * 3).astype(int), 0, 2)
        zeros = float((sub[col] == 0).mean())
        counts = pd.Series(c).value_counts().sort_index().to_dict()
        print(f"    {kind:<12} n={len(sub):>7,}  exact zeros={zeros:.3f}  "
              f"terciles={counts}")


def main() -> int:
    bl = load_module("50_baseline")
    results = {}
    for col in ("i_flow20", "i_flow20_v2"):
        print(f"\n{'#' * 72}")
        print(f"#  crowding column: {col}")
        print(f"{'#' * 72}")
        print("  tie mass and tercile balance before anything is estimated:")
        tercile_report(col)

        tmp = ROOT / "results" / f"_baseline_{col}.json"
        if tmp.exists():
            tmp.unlink()
        bl.CACHE = V6
        bl.PANEL = V6
        bl.OUT = tmp
        bl.CROWD = col
        bl.main()
        results[col] = json.loads(tmp.read_text())
        tmp.unlink()

    print(f"\n{'=' * 72}\nside by side, within the strongest surprises\n{'=' * 72}")
    print(f"  {'sample':<14}{'statistic':<12}{'i_flow20':>28}{'i_flow20_v2':>28}")
    for key in ("provisional", "all"):
        for stat in ("p10", "mean", "hit rate"):
            cells = []
            for col in ("i_flow20", "i_flow20_v2"):
                r = results[col].get(key, {}).get(stat)
                if not r:
                    cells.append("--")
                    continue
                scale = 100.0 if stat == "hit rate" else 1e4
                unit = "pp" if stat == "hit rate" else "bp"
                mark = "OUT" if r["outside_band"] else "in "
                cells.append(f"{r['diff'] * scale:+8.1f}{unit} t{r['t']:+5.2f} "
                             f"{mark} s{r['seasons']}")
            print(f"  {key:<14}{stat:<12}{cells[0]:>28}{cells[1]:>28}")

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print("\nthe first column must match the published record. if it does,")
    print("the second column is the answer and the earlier verdict was mine,")
    print("not the data's.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
