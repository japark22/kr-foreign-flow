#!/usr/bin/env python3
"""Two survivors, combined the honest way.

The exhaustive search left two things standing and they are different
information: how much institutions bought into the event, which is a flow,
and where foreign ownership sits inside its own year, which is a level.
Their correlation is small, so a combination should carry more than either.

Weights are information-coefficient based and are fixed on the first half of
the seasons only, then applied unchanged to the second. An in-sample
composite is reported beside it so the cost of honesty is visible rather
than hidden.

The level leg was concentrated in the short-selling suspensions, so the
combination is also measured with those windows removed. If it only works
there, it is a regime bet and is labelled one.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V6 = ROOT / "results" / "event_panel_v6.parquet"
OUT = ROOT / "results" / "composite.json"
RUNG, DRAWS = 10, 200
LEGS = ["i_flow20_v2", "f_level"]
BANS = [("2020-03-16", "2021-05-02"), ("2023-11-06", "2025-03-30")]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name.split("_")[0],
                                                  ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


fr = load_module("38_final_results")
th = load_module("64_threats_flevel")
bfm = load_module("21_baseline_fm")
HARD = th.CTRL + th.LONG


def residualise(d, cols):
    """Rank inside the day, project the controls out of the legs and the
    outcome, and return the pieces the information coefficient needs."""
    L, Y, S, T = [], [], [], []
    for day, g in d.groupby("D"):
        g = g.dropna(subset=["abn60", "surprise"] + cols + HARD)
        if len(g) < th.MIN_N:
            continue
        C = np.column_stack([bfm.rank_std(g[c]) for c in ["surprise"] + HARD]
                            + [np.ones(len(g))])
        Q, _ = np.linalg.qr(C)
        F = np.column_stack([bfm.rank_std(g[c]) for c in cols])
        y = g["abn60"].to_numpy(dtype=float)
        lo, hi = np.percentile(y, [1, 99])
        y = np.clip(y, lo, hi)
        L.append(F - Q @ (Q.T @ F))
        Y.append(y - Q @ (Q.T @ y))
        ts = pd.Timestamp(day)
        S.append(np.full(len(g), ts.year * 4 + (ts.month - 1) // 3))
        T.append(g["ticker"].to_numpy())
    return (np.vstack(L), np.concatenate(Y), np.concatenate(S),
            np.concatenate(T))


def coef(x, y, sea, tic):
    den = float(x @ x)
    b = float(x @ y) / den
    e = y - x * b
    u = x * e
    out = {"bp": b * 1e4, "events": int(len(y))}
    for tag, key in (("season", sea), ("firm", tic)):
        s = pd.DataFrame({"u": u, "k": key}).groupby("k")["u"].sum()
        out[f"t_{tag}"] = b / (float(np.sqrt((s ** 2).sum())) / den)
    pair = pd.Series(sea).astype(str) + "|" + pd.Series(tic).astype(str)
    sp = pd.DataFrame({"u": u, "k": pair}).groupby("k")["u"].sum()
    m = (pd.DataFrame({"u": u, "k": sea}).groupby("k")["u"].sum() ** 2).sum() \
        + (pd.DataFrame({"u": u, "k": tic}).groupby("k")["u"].sum() ** 2).sum() \
        - (sp ** 2).sum()
    out["t_two_way"] = b / (float(np.sqrt(max(m, 1e-30))) / den)
    return out


def ic_weights(L, Y, mask):
    """Information coefficient per leg, scaled by its own dispersion."""
    w = []
    for j in range(L.shape[1]):
        x, y = L[mask, j], Y[mask]
        ic = float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-12 else 0.0
        w.append(ic / max(x.std(), 1e-12))
    w = np.array(w)
    return w / (np.abs(w).sum() or 1.0)


def show(tag, r, extra=""):
    print(f"  {tag:<40}{r['bp']:+8.1f} bp/SD   t(season) {r['t_season']:+5.2f}"
          f"   t(two-way) {r['t_two_way']:+5.2f}   n={r['events']:>6,}{extra}")


def main() -> int:
    d = pd.read_parquet(V6)
    d = d[d["inst_days20"] >= RUNG]
    R = {"rung": RUNG, "legs": LEGS}

    for scope, sub in (("all filings", d),
                       ("suspensions removed", d[~(
                           ((d["D"] >= BANS[0][0]) & (d["D"] <= BANS[0][1])) |
                           ((d["D"] >= BANS[1][0]) & (d["D"] <= BANS[1][1])))])):
        print(f"\n{'=' * 78}\n{scope}\n{'=' * 78}")
        L, Y, S, T = residualise(sub, LEGS)
        print(f"  events {len(Y):,}   seasons {len(np.unique(S))}")
        c = np.corrcoef(L[:, 0], L[:, 1])[0, 1]
        print(f"  correlation between the two legs after controls: {c:+.3f}")

        for j, leg in enumerate(LEGS):
            show(f"{leg} alone", coef(L[:, j], Y, S, T))

        seasons = np.unique(S)
        cut = seasons[len(seasons) // 2]
        first, second = S < cut, S >= cut

        w_in = ic_weights(L, Y, np.ones(len(Y), dtype=bool))
        w_1st = ic_weights(L, Y, first)
        print(f"  weights, whole sample   "
              f"{dict(zip(LEGS, np.round(w_in, 3)))}")
        print(f"  weights, first half only "
              f"{dict(zip(LEGS, np.round(w_1st, 3)))}")

        comp_in = L @ w_in
        show("composite, in-sample weights", coef(comp_in, Y, S, T),
             "   <- optimistic")
        comp_oos = L[second] @ w_1st
        r_oos = coef(comp_oos, Y[second], S[second], T[second])
        show("composite, weights from the first half",
             r_oos, "   <- the honest one")

        for j, leg in enumerate(LEGS):
            show(f"  {leg}, second half only",
                 coef(L[second, j], Y[second], S[second], T[second]))

        rng = np.random.default_rng(31)
        pl = []
        for _ in range(DRAWS):
            idx = rng.permutation(np.where(second)[0])
            pl.append(coef(L[idx] @ w_1st, Y[second], S[second], T[second])["bp"])
        lo, hi = np.percentile(pl, [5, 95])
        print(f"  placebo band for the honest composite "
              f"[{lo:+.1f}, {hi:+.1f}]   "
              f"{'OUTSIDE' if r_oos['bp'] < lo or r_oos['bp'] > hi else 'inside'}")

        R[scope] = {"leg_corr": float(c), "weights_first_half": w_1st.tolist(),
                    "oos": r_oos, "placebo_p05": float(lo),
                    "placebo_p95": float(hi)}

    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2, default=str))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
