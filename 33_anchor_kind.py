"""Locate the cell the known -117.7 bp/SD figure came from.

The decomposition has to be anchored to a number it can reproduce. The
pooled panel gives -78.8; the ladder in 21_baseline_fm was run per event
kind, so the target is probably one kind, not the pool. This scans kind x
horizon with the imported estimator and prints the grid -- nothing here is
a new test, it is bookkeeping to find which sample the headline refers to.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("bfm", "21_baseline_fm.py")
bfm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bfm)

d = pd.read_parquet("results/event_panel.parquet")
d["D"] = pd.to_datetime(d["D"])
print(f"CTRL = {bfm.CTRL}")
print(f"panel {len(d):,} events\n")

xs_of = lambda base: ["surprise"] + list(bfm.CTRL) + [base]

for base in ("i_flow20", "f_flow20"):
    print(f"=== baseline {base} ===")
    print(f"  {'sample':<14}{'abn5':>18}{'abn20':>18}{'abn60':>18}")
    for tag, sub in (("all", d),
                     ("periodic", d[d["kind"] == "periodic"]),
                     ("provisional", d[d["kind"] == "provisional"])):
        cells = []
        for y in ("abn5", "abn20", "abn60"):
            r = bfm.fama_macbeth(sub, y, xs_of(base), min_n=25, lag=5)[base]
            cells.append(f"{r['bp']:+8.1f} ({r['t']:+5.2f})"
                         if np.isfinite(r["bp"]) else "        --      ")
        n_days = r["days"]
        print(f"  {tag:<14}" + "".join(f"{c:>18}" for c in cells)
              + f"   days {n_days:,}")
    print()

# min_n matters when events cluster into a few crowded days.
print("=== i_flow20 / abn60 sensitivity to min_n ===")
for tag, sub in (("all", d),
                 ("periodic", d[d["kind"] == "periodic"]),
                 ("provisional", d[d["kind"] == "provisional"])):
    row = []
    for mn in (5, 10, 25, 50):
        r = bfm.fama_macbeth(sub, "abn60", xs_of("i_flow20"),
                             min_n=mn, lag=5)["i_flow20"]
        row.append(f"mn{mn}: {r['bp']:+7.1f} ({r['t']:+5.2f}) d{r['days']:,}")
    print(f"  {tag:<14}" + "   ".join(row))
