"""Assemble one publication file. The page reads this and nothing else.

Six scripts each wrote their own results file. A page that reaches into six
files, or worse takes numbers from a console, is a page that will one day
show a figure its own author has retracted -- which is exactly what happened
with -117.7. So the numbers are gathered once, here, with their provenance
and their limitations attached as data rather than as prose someone can
forget to update.

The limitations are part of the payload on purpose. A reader who sees the
headline must see, in the same object, that it rests on a single horizon and
that the long side does not work.

    python 41_publish.py            # refuse to publish stale inputs
    python 41_publish.py --force    # publish anyway, and say so on the page
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

R = Path("results")
CACHE = R / "decompose_panel.parquet"
OUT = R / "published.json"

INPUTS = {
    "final": "final.json",
    "decompose": "decompose.json",
    "pension": "pension.json",
    "avoid": "avoid.json",
    "threats": "threats.json",
    "threats_formal": "threats_formal.json",
}

CLAIM = (
    "On Korean provisional earnings announcements, institutional net buying "
    "in the twenty trading days before the announcement predicts the next "
    "sixty trading days of size-adjusted return negatively. The usable form "
    "is avoidance, not accumulation: among the strongest surprises, those the "
    "institutions had already crowded into underperform the rest."
)

LIMITS = [
    "The effect appears at sixty trading days and at no shorter horizon "
    "(abn5 -6.2, abn20 -1.0). Only three horizons were examined, so the "
    "search was narrow, but a result that lives at one horizon is weaker "
    "than one that builds across them.",
    "Half-sample stability cannot be claimed. The aggregate series reads "
    "-39.9 then -77.7 across the two halves and the pension series reads "
    "-83.1 then -48.1 -- opposite patterns, which is what thirty-four "
    "seasons buys you.",
    "Surprise is proxied by the announcement-day price reaction, which is "
    "entangled with the flow being measured. A true consensus-based surprise "
    "is the single largest available improvement and requires paid data.",
    "The pension fund cannot be claimed as a sharper baseline than the "
    "aggregate: the paired seasonal difference is -31.3 bp at t -1.47.",
    "The long side does not work. A random ranking separates the quiet "
    "surprises better than the real one does.",
]

REJECTED = [
    "Foreign ownership flow predicts post-event returns "
    "(+21.8 bp/SD, t +1.06)",
    "The effect holds across filing types "
    "(periodic filings: -19.8 bp/SD, t -1.31)",
    "Buying the quiet surprises earns a premium "
    "(placebo ranking reads higher)",
    "The pension fund is a separable carrier of the aggregate signal "
    "(both collapse when entered together)",
    "Index rebalancing explains the effect "
    "(size interaction +64.3, t +1.14 -- wrong sign)",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if not CACHE.exists():
        raise SystemExit("panel cache missing -- run 35_decompose.py --rebuild")
    base = CACHE.stat().st_mtime

    data, stale, missing = {}, [], []
    for key, name in INPUTS.items():
        p = R / name
        if not p.exists():
            missing.append(name)
            continue
        if p.stat().st_mtime < base:
            stale.append(name)
        data[key] = json.loads(p.read_text())

    if missing:
        raise SystemExit(f"missing result files: {', '.join(missing)}")
    if stale and not a.force:
        raise SystemExit(f"older than the panel: {', '.join(stale)}\n"
                         f"re-run those scripts, or pass --force")

    f = data["final"]
    tf = data["threats_formal"]
    th = data["threats"]

    doc = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "stale_inputs": stale,
        "claim": CLAIM,
        "estimator": f["estimator"],
        "coverage": f["coverage"],
        "headline": {
            "full_sample": f["headline"]["institutional_provisional"],
            "after_threats": th.get("strict"),
            "periodic": f["headline"]["institutional_periodic"],
            "all_events": f["headline"]["institutional_all_events"],
            "foreign": f["headline"]["foreign_provisional"],
        },
        "horizons": f["horizons"],
        "halves": f["halves"],
        "why_the_earlier_number_was_larger": {
            "note": "The retracted -117.7 bp/SD came from weighting each day "
                    "equally after discarding days carrying fewer than "
                    "twenty-five announcements. Both choices push the same "
                    "way; the sweep below is the evidence.",
            "sweep": f["min_n_sweep"],
        },
        "decomposition": f["decomposition"],
        "portfolio": f["portfolio"],
        "threats": {
            "interactions": {k: v for k, v in tf.items()
                             if isinstance(v, dict)},
            "subsamples": {k: th[k] for k in
                           ("size", "review", "december", "wave", "strict")
                           if k in th},
            "verdict": tf.get("verdict"),
        },
        "limitations": LIMITS,
        "rejected": REJECTED,
    }

    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2))
    h = doc["headline"]
    print(f"claim      {CLAIM[:70]}...")
    print(f"full       {h['full_sample']['bp']:+8.1f} bp/SD  "
          f"t {h['full_sample']['t']:+5.2f}")
    if h["after_threats"]:
        print(f"clean      {h['after_threats']['bp']:+8.1f} bp/SD  "
              f"t {h['after_threats']['t']:+5.2f}")
    print(f"limits     {len(LIMITS)} stated,  rejected {len(REJECTED)}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
