#!/usr/bin/env python3
"""Two rule changes, opposite directions, everything else held still.

Comparing index members with everyone else across whole regimes failed,
and the reason is visible in the output: the differential is there in
every regime, including the ones where the rule was the same for both
groups. Membership carries a size effect that is present always, so it
cannot be told apart from a rule that is present sometimes.

A constant differences away. Inside a short window around a switch date
the two groups keep their sizes, their sectors and their index status;
the only thing that moves is the rule. The estimate of interest is then
the triple interaction

    x  x  member  x  after the switch

and the constant membership effect sits in the x-by-member term where it
belongs.

Two switches, and the hypothesis predicts opposite signs:

    2021-05-03  members become shortable    -> triple interaction negative
    2023-11-06  members lose shortability   -> triple interaction positive

Getting both signs right by accident is hard, which is the point. Events
whose sixty-day window would straddle a switch are dropped, so no outcome
is measured across a rule change.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V6 = ROOT / "results" / "event_panel_v6.parquet"
OUT = ROOT / "results" / "switch_did.json"
REPS = 2000

SWITCHES = [("2021-05-03", "members become shortable", -1),
            ("2023-11-06", "members lose shortability", +1)]
PLACEBO = ["2019-05-03", "2022-05-03", "2018-11-06", "2022-11-06"]


def load(name):
    spec = importlib.util.spec_from_file_location(name.split("_")[0],
                                                  ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


th = load("64_threats_flevel")
bv = load("67_ban_vs_era")
bfm = load("21_baseline_fm")
HARD = th.CTRL + th.LONG


def window(d, switch, months, gap_days=90):
    s = pd.Timestamp(switch)
    lo, hi = s - pd.DateOffset(months=months), s + pd.DateOffset(months=months)
    pre = d[(d["D"] >= lo) & (d["D"] < s - pd.Timedelta(days=gap_days))].copy()
    post = d[(d["D"] >= s) & (d["D"] <= hi)].copy()
    pre["post"] = 0.0
    post["post"] = 1.0
    return pd.concat([pre, post], ignore_index=True)


def design(d, base):
    """x, x*member, x*post, x*member*post, then the level terms, per day."""
    xs = ["surprise"] + HARD + [base]
    X, Y, S = [], [], []
    for day, g in d.groupby("D"):
        g = g.dropna(subset=["abn60", "member", "post"] + xs)
        if len(g) < th.MIN_N or g["member"].nunique() < 2:
            continue
        x = bfm.rank_std(g[base])
        mem = g["member"].to_numpy(dtype=float)
        pst = g["post"].to_numpy(dtype=float)
        cols = [x, x * mem, x * pst, x * mem * pst, mem, mem * pst]
        cols += [bfm.rank_std(c) for c in (g[c] for c in xs if c != base)]
        cols += [np.ones(len(g))]
        X.append(np.column_stack(cols))
        v = g["abn60"].to_numpy(dtype=float)
        lo, hi = np.percentile(v, [1, 99])
        Y.append(np.clip(v, lo, hi))
        ts = pd.Timestamp(day)
        S.append(np.full(len(g), ts.year * 12 + ts.month))
    if len(X) < 8:
        return None
    return np.vstack(X), np.concatenate(Y), np.concatenate(S)


def run(d, base, months, switch):
    w = window(d, switch, months)
    pack = design(w, base)
    if pack is None:
        return None
    X, y, S = pack
    bp, t, _, _ = bv.cluster_fit(X, y, S, 3)
    t0, p, ng = bv.wild_bootstrap(X, y, S, 3, reps=REPS)
    npre = int((w["post"] == 0).sum())
    npost = int((w["post"] == 1).sum())
    return {"bp": bp * 1e4, "t": t, "wild_p": p, "clusters": ng,
            "events": int(len(y)), "pre": npre, "post": npost}


def show(tag, r, want=None):
    if r is None:
        print(f"  {tag:<44}  too thin")
        return
    ok = "" if want is None else (
        "  as predicted" if np.sign(r["bp"]) == want and r["wild_p"] < 0.10
        else "  not as predicted")
    print(f"  {tag:<44}{r['bp']:+8.1f}   t {r['t']:+6.2f}   "
          f"p {r['wild_p']:.3f}   n={r['events']:>6,}  months={r['clusters']}{ok}")


def main() -> int:
    d = pd.read_parquet(V6)
    d = d[d["kind"] == "periodic"].copy()
    d["fl_minus_i"] = (d.groupby("D")["f_level"].transform(lambda s: s.rank(pct=True))
                       - d.groupby("D")["i_flow20_v2"].transform(lambda s: s.rank(pct=True)))
    R = {}

    for switch, note, want in SWITCHES:
        print(f"\n{switch}   {note}   predicted sign "
              f"{'negative' if want < 0 else 'positive'}")
        for months in (9, 12, 18, 24):
            r = run(d, "fl_minus_i", months, switch)
            R[f"{switch}_{months}m"] = r
            show(f"  window +/- {months} months", r, want)

    print("\nplacebo switches, where no rule changed")
    for switch in PLACEBO:
        r = run(d, "fl_minus_i", 12, switch)
        R[f"placebo_{switch}"] = r
        show(f"  {switch}, +/- 12 months", r)

    print("\nreading it: the two real switches should carry opposite signs,")
    print("and the placebos should carry none.")
    OUT.write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str))
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
