#!/usr/bin/env python3
"""Put the one prediction that worked through the gate the feature passed.

Crowding bites harder where institutional accumulation has been trending than
where it churns: the interaction is -25.9 bp per standard deviation with a
wild cluster bootstrap p of 0.0195 over 49 clusters, and it was predicted in
advance rather than found. That is the first pre-declared prediction to
survive its first test today, which is a reason to test it harder rather than
a reason to report it.

Three of them were declared. Two failed. A Bonferroni threshold for three is
0.0167 and 0.0195 does not clear it, so this is reported as not clearing
Bonferroni whatever else it does.

Four ways to break it:

  A  is the variance ratio a disguise? It is a property of the flow series,
     but turnover and volatility are properties of the same trading. The
     interaction is re-estimated with the ratio residualised on turnover,
     volatility, size and the count of participation days, so what is left
     is the shape of the accumulation and not how much of it there was.
  B  does it hold in both halves?
  C  does it survive every estimator variant the feature was frozen on?
  D  is it in the tail or the whole distribution? A conditioning variable
     that only moves the mean is less useful than one that moves the shape.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V7 = ROOT / "results" / "event_panel_v7.parquet"
OUT = ROOT / "results" / "vr_gate.json"
RUNG, REPS = 10, 2000
BANS = [("2020-03-16", "2021-05-02"), ("2023-11-06", "2025-03-30")]
BONF = 0.05 / 3


def load_module(name):
    spec = importlib.util.spec_from_file_location(name.split("_")[0],
                                                  ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


th = load_module("64_threats_flevel")
bv = load_module("67_ban_vs_era")
qr = load_module("58_qreg")
bfm = load_module("21_baseline_fm")
HARD = th.CTRL + th.LONG


def build(d, vr_col, rank_y=False, y="abn60"):
    X, Y, S, T = [], [], [], []
    need = [y, "surprise", "i_flow20_v2", vr_col] + HARD
    for day, g in d.groupby("D"):
        g = g.dropna(subset=need)
        if len(g) < th.MIN_N:
            continue
        x = bfm.rank_std(g["i_flow20_v2"])
        v = bfm.rank_std(g[vr_col])
        cols = [x, x * v, v]
        cols += [bfm.rank_std(g[c]) for c in ["surprise"] + HARD]
        cols += [np.ones(len(g))]
        X.append(np.column_stack(cols))
        if rank_y:
            Y.append(bfm.rank_std(g[y]))
        else:
            v2 = g[y].to_numpy(dtype=float)
            lo, hi = np.percentile(v2, [1, 99])
            Y.append(np.clip(v2, lo, hi))
        ts = pd.Timestamp(day)
        S.append(np.full(len(g), ts.year * 4 + (ts.month - 1) // 3))
        T.append(g["ticker"].to_numpy())
    return (np.vstack(X), np.concatenate(Y), np.concatenate(S),
            np.concatenate(T))


def report(tag, pack, scale, want=1, boot=False):
    X, Y, S, T = pack
    bp, t, _, _ = bv.cluster_fit(X, Y, S, want)
    line = (f"  {tag:<40}{bp * scale:+9.2f}   t {t:+6.2f}   "
            f"n={len(Y):>6,}  s={pd.Series(S).nunique():>3}")
    out = {"coef": bp * scale, "t_season": t, "events": int(len(Y)),
           "seasons": int(pd.Series(S).nunique())}
    if boot:
        t0, p, ng = bv.wild_bootstrap(X, Y, S, want, reps=REPS)
        out["wild_p"] = p
        line += f"   p {p:.4f}{'  clears Bonferroni' if p < BONF else ''}"
    print(line)
    return out


def main() -> int:
    d = pd.read_parquet(V7)
    d = d[d["inst_days20"] >= RUNG].copy()
    d = d[~(((d["D"] >= BANS[0][0]) & (d["D"] <= BANS[0][1])) |
            ((d["D"] >= BANS[1][0]) & (d["D"] <= BANS[1][1])))]
    R = {"bonferroni": BONF}

    print("A. is the variance ratio a disguise for how much trading there was?")
    for c in ("c_turn", "c_vol", "c_size", "inst_days20", "c_mom20"):
        r = float(d[["flow_vr", c]].corr().iloc[0, 1])
        print(f"   corr(flow_vr, {c:<12}) {r:+.3f}")
        R[f"corr_{c}"] = r

    resid_cols = ["c_turn", "c_vol", "c_size", "inst_days20"]
    sub = d.dropna(subset=["flow_vr"] + resid_cols).copy()
    parts = []
    for day, g in sub.groupby("D"):
        if len(g) < 10:
            continue
        C = np.column_stack([bfm.rank_std(g[c]) for c in resid_cols]
                            + [np.ones(len(g))])
        Q, _ = np.linalg.qr(C)
        v = bfm.rank_std(g["flow_vr"])
        parts.append(pd.Series(v - Q @ (Q.T @ v), index=g.index))
    d["flow_vr_clean"] = pd.concat(parts).reindex(d.index)
    print(f"   residualised on turnover, volatility, size and participation; "
          f"coverage {d['flow_vr_clean'].notna().mean():.3f}")

    print("\n   the interaction, before and after")
    R["raw"] = report("crowding x variance ratio", build(d, "flow_vr"),
                      1e4, boot=True)
    R["clean"] = report("crowding x cleaned variance ratio",
                        build(d, "flow_vr_clean"), 1e4, boot=True)

    print("\nB. both halves, on the cleaned ratio")
    pack = build(d, "flow_vr_clean")
    S = pack[2]
    seasons = np.unique(S)
    cut = seasons[len(seasons) // 2]
    for tag, m in (("first", S < cut), ("second", S >= cut)):
        sl = tuple(a[m] for a in pack)
        R[f"half_{tag}"] = report(f"  {tag} half", sl, 1e4)

    print("\nC. every estimator variant the feature was frozen on")
    variants = [("raw return, 60 day", "abn60", False, 1e4),
                ("rank outcome, 60 day", "abn60", True, 1.0),
                ("rank outcome, 20 day", "abn20", True, 1.0),
                ("raw return, 20 day", "abn20", False, 1e4)]
    ok = True
    for tag, y, rank_y, scale in variants:
        if y not in d.columns:
            continue
        r = report(f"  {tag}", build(d, "flow_vr_clean", rank_y, y), scale)
        R[f"variant_{tag}"] = r
        ok &= abs(r["t_season"]) >= 2.0 and r["coef"] < 0

    print("\nD. where in the distribution does the conditioning act?")
    X, Y, S, T = build(d, "flow_vr_clean")
    for tau in (0.10, 0.25, 0.50, 0.75, 0.90):
        b, frac = qr.fit_check(X, Y, tau)
        if abs(frac - tau) > 0.02:
            print(f"   tau {tau:.2f}  fit landed at {frac:.3f} -- not read")
            continue
        print(f"   tau {tau:.2f}   interaction {b[1] * 1e4:+9.1f} bp/SD")
        R[f"tau_{tau}"] = float(b[1] * 1e4)

    same_sign = (R["half_first"]["coef"] < 0) and (R["half_second"]["coef"] < 0)
    print(f"\n  cleaned interaction survives: "
          f"{'yes' if R['clean']['coef'] < 0 and abs(R['clean']['t_season']) >= 2 else 'NO'}")
    print(f"  halves agree in sign: {'yes' if same_sign else 'NO'}")
    print(f"  every variant negative and |t| >= 2: {'yes' if ok else 'NO'}")
    print(f"  clears Bonferroni for three declared tests "
          f"(p < {BONF:.4f}): "
          f"{'yes' if R['clean'].get('wild_p', 1) < BONF else 'NO'}")
    R["passed_all"] = bool(same_sign and ok and R["clean"]["coef"] < 0
                           and R["clean"].get("wild_p", 1) < BONF)

    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2, default=str))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
