#!/usr/bin/env python3
"""The baseline once the quiet side means what it is supposed to mean.

Filling the omitted cells with zero puts every name institutions never
touched into the quiet tercile. Quiet then means two different things at
once: institutions were present and did not crowd, or institutions were not
there at all. The comparison group is contaminated, and a contaminated
comparison group shrinks the contrast whether or not the contrast is real.

PRE_REGISTRATION.md section 10(c), committed this morning before any
estimate was produced, says the no-participation state is carried
separately and the continuous feature is estimated on the cells with
participation. This applies that rule.

It is not a free pass. Restricting the sample widens the placebo band, so
the test gets harder as the ladder climbs, and the band is reported at
every rung.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V6 = ROOT / "results" / "event_panel_v6.parquet"
OUT = ROOT / "results" / "baseline_participation.json"
TMP = ROOT / "results" / "_part_panel.parquet"
RUNGS = [0, 1, 5, 10, 14, 18]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name.split("_")[0],
                                                  ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    bl = load_module("50_baseline")
    base = pd.read_parquet(V6)
    out = {}

    for rung in RUNGS:
        d = base[base["inst_days20"] >= rung].copy()
        tag = f"inst_days20 >= {rung}"
        n_prov = int((d["kind"] == "provisional").sum())
        print(f"\n{'#' * 72}\n#  {tag}    rows {len(d):,}  "
              f"(provisional {n_prov:,})\n{'#' * 72}")
        d.to_parquet(TMP, index=False)

        tmp_json = ROOT / "results" / "_part_out.json"
        if tmp_json.exists():
            tmp_json.unlink()
        bl.CACHE = TMP
        bl.PANEL = TMP
        bl.OUT = tmp_json
        bl.CROWD = "i_flow20_v2"
        bl.main()
        out[tag] = json.loads(tmp_json.read_text())
        tmp_json.unlink()

    TMP.unlink(missing_ok=True)

    print(f"\n{'=' * 84}")
    print("p10, crowded minus quiet, inside the strongest surprises")
    print(f"{'=' * 84}")
    print(f"  {'rung':<22}{'provisional':>30}{'all filings':>30}")
    for rung in RUNGS:
        tag = f"inst_days20 >= {rung}"
        cells = []
        for key in ("provisional", "all"):
            r = out[tag].get(key, {}).get("p10")
            if not r:
                cells.append("--")
                continue
            mark = "OUT" if r["outside_band"] else "in "
            cells.append(f"{r['diff'] * 1e4:+8.1f}bp t{r['t']:+5.2f} {mark} "
                         f"band[{r['placebo_lo'] * 1e4:+6.0f},"
                         f"{r['placebo_hi'] * 1e4:+6.0f}]")
        print(f"  {tag:<22}{cells[0]:>30}{cells[1]:>30}")

    print("\nmean and hit rate must stay inside their bands -- that is the")
    print("signature of a tail effect rather than a return predictor.")
    for stat in ("mean", "hit rate"):
        print(f"  [{stat}]")
        for rung in RUNGS:
            tag = f"inst_days20 >= {rung}"
            r = out[tag].get("provisional", {}).get(stat)
            if r:
                scale = 100.0 if stat == "hit rate" else 1e4
                mark = "OUTSIDE" if r["outside_band"] else "inside"
                print(f"    {tag:<22}{r['diff'] * scale:+8.1f}  "
                      f"t {r['t']:+5.2f}  {mark}")

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
