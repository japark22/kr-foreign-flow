"""Test the threats as interactions, not as split samples.

39_threats split the panel and read the pieces. That was the wrong shape of
test twice over. Splitting into thirds cuts t by the square root of three, so
a gate on significance asks whether a third of the data is significant, not
whether the effect differs by size -- which is the actual question. And the
eye cannot tell -48.0 from -44.0 without a standard error on the difference.

Each threat is therefore a single interaction on the full sample, where the
quantity of interest is estimated directly and carries its own error:

  size      baseline x (size rank - 1/2).  Index pressure reaches large caps
            and not small ones, so if rebalancing is the cause the
            interaction is negative and large. If the earnings reading is
            right the interaction is indistinguishable from zero.
  review    baseline x (event sits in a quarterly review window)
  december  baseline x (event sits in the dividend record cluster)
  wave      baseline x (institutions were buying market-wide)

For the last three the main coefficient is now the effect OFF the window,
which is the number that has to survive, and the interaction says how much
the window adds.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("results/decompose_panel.parquet")
CALSRC = Path("data/investor/005930.parquet")
OUT = Path("results/threats_formal.json")
AGG, MIN_N, LOOK = "i_flow20", 5, 20
REVIEW_PAD, DEC_PAD = 10, (10, 5)

spec = importlib.util.spec_from_file_location("bfm", "21_baseline_fm.py")
bfm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bfm)
CTRL = list(bfm.CTRL)


def build(sub: pd.DataFrame, extra: str | None, y="abn60"):
    """Stack days into one design: baseline first, then its interaction with
    `extra`, then the controls, the moderator itself, and a constant."""
    xs = ["surprise"] + CTRL + [AGG]
    need = [y] + xs + ([extra] if extra else [])
    X, Y, S = [], [], []
    for day, g in sub.groupby("D"):
        g = g.dropna(subset=need)
        if len(g) < MIN_N:
            continue
        b = bfm.rank_std(g[AGG])
        cols = [b]
        if extra:
            m = g[extra].to_numpy(dtype=float)
            cols += [b * m, m]
        cols += [bfm.rank_std(g[c]) for c in xs if c != AGG]
        cols += [np.ones(len(g))]
        X.append(np.column_stack(cols))
        v = g[y].to_numpy(dtype=float)
        lo, hi = np.percentile(v, [1, 99])
        Y.append(np.clip(v, lo, hi))
        t = pd.Timestamp(day)
        S.append(np.full(len(g), t.year * 4 + (t.month - 1) // 3))
    return np.vstack(X), np.concatenate(Y), np.concatenate(S)


def fit(X, y, sea, want):
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(sea):
        m = sea == c
        u = X[m].T @ e[m]
        meat += np.outer(u, u)
    V = XtX_inv @ meat @ XtX_inv
    se = float(np.sqrt(max(V[want, want], 1e-30)))
    return {"bp": float(beta[want] * 1e4), "t": float(beta[want] / se)}


def main() -> int:
    d = pd.read_parquet(CACHE)
    d["D"] = pd.to_datetime(d["D"])
    prov = d[d["kind"] == "provisional"].copy()

    cal = pd.DatetimeIndex(sorted(pd.read_parquet(
        CALSRC, columns=["trade_date"])["trade_date"].unique()))
    p = cal.searchsorted(prov["D"].to_numpy())
    a0, a1 = np.clip(p - LOOK, 0, len(cal) - 1), np.clip(p - 1, 0, len(cal) - 1)

    def last_of(y_, m_):
        s = cal[(cal.year == y_) & (cal.month == m_)]
        return int(cal.searchsorted(s[-1])) if len(s) else None

    rev = [q for y_ in range(int(cal[0].year), int(cal[-1].year) + 1)
           for m_ in (2, 5, 8, 11) if (q := last_of(y_, m_)) is not None]
    dec = [q for y_ in range(int(cal[0].year), int(cal[-1].year) + 1)
           if (q := last_of(y_, 12)) is not None]

    def flag(marks, pad):
        lo, hi = pad if isinstance(pad, tuple) else (pad, pad)
        out = np.zeros(len(a0), bool)
        for m in marks:
            out |= (a0 <= m + hi) & (a1 >= m - lo)
        return out.astype(float)

    prov["m_review"] = flag(rev, REVIEW_PAD)
    prov["m_dec"] = flag(dec, DEC_PAD)
    dm = prov.groupby("D")[AGG].mean()
    prov["m_wave"] = prov["D"].map(dm >= dm.quantile(0.80)).astype(float)
    prov["m_size"] = prov.groupby("D")["c_size"].rank(pct=True) - 0.5

    for c in ("m_review", "m_dec", "m_wave"):
        print(f"  {c:<10}{prov[c].mean():.3f} of events")
    print()

    R = {}
    X, y, s = build(prov, None)
    R["plain"] = fit(X, y, s, 0)
    print(f"  {'baseline, no interaction':<34}"
          f"{R['plain']['bp']:+8.1f} bp/SD   t {R['plain']['t']:+5.2f}"
          f"   events {len(y):,}")

    print("\ninteractions (main = effect at the reference state)")
    for tag, col, note in (
            ("size", "m_size", "negative interaction => confined to large caps"),
            ("review window", "m_review", "main = outside the window"),
            ("December", "m_dec", "main = outside December"),
            ("market wave", "m_wave", "main = off the waves")):
        X, y, s = build(prov, col)
        main_, inter = fit(X, y, s, 0), fit(X, y, s, 1)
        R[tag] = {"main": main_, "interaction": inter}
        print(f"  {tag:<16}main {main_['bp']:+8.1f} (t {main_['t']:+5.2f})"
              f"   interaction {inter['bp']:+9.1f} (t {inter['t']:+5.2f})")
        print(f"                  {note}")

    it = R["size"]["interaction"]
    verdict = ("size interaction is not distinguishable from zero -- index "
               "rebalancing does not explain the effect"
               if abs(it["t"]) < 2.0 else
               "the effect varies with size -- index pressure remains a live "
               "explanation")
    print(f"\nverdict: {verdict}")
    R["verdict"] = verdict
    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
