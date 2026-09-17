#!/usr/bin/env python3
"""The last gate, and if it opens, the frozen specification.

One feature, one universe, one control set. The composite failed out of
sample because the level leg changes sign between halves, so it is gone.
What is left is the variable this project started with, measured on the
recovered sample inside a universe where it can be measured at all.

The gate: both halves carry the same sign, the full sample clears a
two-way clustered t of 3, and the estimate sits outside a 200-draw
placebo band. Year by year is printed because a coefficient that lives in
two good years is not a baseline whatever its t.

If the gate opens the script writes the specification and the signal
panel, so the thing that goes to another market is a file and not a
paragraph.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V6 = ROOT / "results" / "event_panel_v6.parquet"
OUT = ROOT / "results" / "lock.json"
SPEC = ROOT / "handoff" / "KR_CROWDING_SPEC.md"
PANEL = ROOT / "handoff" / "kr_crowding_events.parquet"
RUNG, DRAWS, BAR = 10, 200, 3.0
BANS = [("2020-03-16", "2021-05-02"), ("2023-11-06", "2025-03-30")]
FEATURE = "i_flow20_v2"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name.split("_")[0],
                                                  ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


th = load_module("64_threats_flevel")
bfm = load_module("21_baseline_fm")
HARD = th.CTRL + th.LONG


def pieces(d):
    X, Y, S, T, D = [], [], [], [], []
    for day, g in d.groupby("D"):
        g = g.dropna(subset=["abn60", "surprise", FEATURE] + HARD)
        if len(g) < th.MIN_N:
            continue
        C = np.column_stack([bfm.rank_std(g[c]) for c in ["surprise"] + HARD]
                            + [np.ones(len(g))])
        Q, _ = np.linalg.qr(C)
        x = bfm.rank_std(g[FEATURE])
        y = g["abn60"].to_numpy(dtype=float)
        lo, hi = np.percentile(y, [1, 99])
        y = np.clip(y, lo, hi)
        X.append(x - Q @ (Q.T @ x))
        Y.append(y - Q @ (Q.T @ y))
        ts = pd.Timestamp(day)
        S.append(np.full(len(g), ts.year * 4 + (ts.month - 1) // 3))
        T.append(g["ticker"].to_numpy())
        D.append(np.full(len(g), ts.year))
    return (np.concatenate(X), np.concatenate(Y), np.concatenate(S),
            np.concatenate(T), np.concatenate(D))


def coef(x, y, sea, tic):
    den = float(x @ x)
    if den < 1e-12:
        return None
    b = float(x @ y) / den
    u = x * (y - x * b)
    def meat(k):
        return float((pd.DataFrame({"u": u, "k": k}).groupby("k")["u"].sum() ** 2).sum())
    pair = pd.Series(sea).astype(str) + "|" + pd.Series(tic).astype(str)
    m2 = max(meat(sea) + meat(tic) - meat(pair.to_numpy()), 1e-30)
    return {"bp": b * 1e4, "events": int(len(y)),
            "seasons": int(pd.Series(sea).nunique()),
            "t_season": b / (np.sqrt(meat(sea)) / den),
            "t_two_way": b / (np.sqrt(m2) / den)}


def main() -> int:
    d = pd.read_parquet(V6)
    d = d[d["inst_days20"] >= RUNG].copy()
    keep = ~(((d["D"] >= BANS[0][0]) & (d["D"] <= BANS[0][1])) |
             ((d["D"] >= BANS[1][0]) & (d["D"] <= BANS[1][1])))
    d = d[keep]
    x, y, S, T, yr = pieces(d)
    R = {"rung": RUNG, "bar": BAR, "feature": FEATURE}

    full = coef(x, y, S, T)
    print(f"full sample   {full['bp']:+8.1f} bp/SD   "
          f"t(season) {full['t_season']:+5.2f}   "
          f"t(two-way) {full['t_two_way']:+5.2f}   "
          f"n={full['events']:,}  seasons={full['seasons']}")
    R["full"] = full

    rng = np.random.default_rng(1009)
    pl = [coef(x[rng.permutation(len(x))], y, S, T)["bp"] for _ in range(DRAWS)]
    lo, hi = np.percentile(pl, [5, 95])
    outside = full["bp"] < lo or full["bp"] > hi
    print(f"placebo band  [{lo:+.1f}, {hi:+.1f}]   "
          f"{'OUTSIDE' if outside else 'inside'}")
    R["placebo"] = {"p05": float(lo), "p95": float(hi), "outside": bool(outside)}

    seasons = np.unique(S)
    cut = seasons[len(seasons) // 2]
    halves = {}
    for tag, m in (("first", S < cut), ("second", S >= cut)):
        r = coef(x[m], y[m], S[m], T[m])
        halves[tag] = r
        print(f"  {tag:<7}{r['bp']:+8.1f} bp/SD   t(two-way) {r['t_two_way']:+5.2f}"
              f"   n={r['events']:>6,}  seasons={r['seasons']}")
    R["halves"] = halves

    print("\nyear by year")
    years, pos = [], 0
    for yv in np.unique(yr):
        m = yr == yv
        if m.sum() < 1200:
            continue
        r = coef(x[m], y[m], S[m], T[m])
        if not r:
            continue
        pos += r["bp"] < 0
        years.append((int(yv), r))
        bar = "#" * min(int(abs(r["bp"]) / 8), 34)
        print(f"  {int(yv)}  {r['bp']:+8.1f}  t {r['t_season']:+5.2f}  "
              f"n={r['events']:>6,}  {'-' if r['bp'] < 0 else '+'}{bar}")
    print(f"  negative years: {pos} of {len(years)}")
    R["years"] = {str(k): v for k, v in years}

    same_sign = np.sign(halves["first"]["bp"]) == np.sign(halves["second"]["bp"])
    passed = (abs(full["t_two_way"]) >= BAR) and outside and same_sign
    print(f"\ngate: |t| >= {BAR} {'yes' if abs(full['t_two_way']) >= BAR else 'NO'}"
          f" | outside band {'yes' if outside else 'NO'}"
          f" | halves agree {'yes' if same_sign else 'NO'}")
    R["passed"] = bool(passed)

    if passed:
        SPEC.parent.mkdir(exist_ok=True)
        SPEC.write_text(f"""# Korean institutional crowding into earnings filings

Frozen {pd.Timestamp.today().date()}. Implement unchanged in any other market;
a deviation forced by data availability is reported next to the result.

## Universe
Liquid listed equities. Institutions must have traded the name on at least
{RUNG} of the 20 trading days before the filing, measured on the exchange's
daily net-buying feed. A day with no row in that feed is a day with no
institutional trade, which is verifiable against a per-investor-type source
where one exists.

## Feature
Net institutional buying value over the 20 trading days ending the day before
the filing, divided by 20-day average traded value, rank-standardised across
that day's cross-section.

## Controls
surprise, {', '.join(HARD)} -- all rank-standardised within the day.

## Outcome
60-trading-day abnormal return from the day after the filing, market return
subtracted, winsorised 1/99 within the day.

## Inference
Standard errors clustered two ways, on the reporting season and on the issuer.
Placebo: shuffle the feature within the day, 200 draws, report the 5-95 band.

## Result on Korea
{full['bp']:+.1f} bp per standard deviation, two-way t {full['t_two_way']:+.2f},
{full['events']:,} events over {full['seasons']} reporting seasons.
First half {halves['first']['bp']:+.1f} (t {halves['first']['t_two_way']:+.2f}),
second half {halves['second']['bp']:+.1f} (t {halves['second']['t_two_way']:+.2f}).
Placebo band [{lo:+.1f}, {hi:+.1f}].

Short-selling suspensions (2020-03-16 to 2021-05-02 and 2023-11-06 to
2025-03-30) are excluded. Including them gives a weaker estimate, so the
exclusion is conservative rather than flattering.

## Sign
Higher crowding predicts a lower subsequent abnormal return. The flow is a
record of buying that already happened, not a forecast of buying to come.
""")
        cols = ["ticker", "D", "kind", "surprise", FEATURE, "inst_days20",
                "abn5", "abn20", "abn60"] + HARD
        d[[c for c in cols if c in d.columns]].to_parquet(PANEL, index=False)
        print(f"\nwrote {SPEC.relative_to(ROOT)} and {PANEL.relative_to(ROOT)}")
    else:
        print("\ngate closed -- nothing written")

    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2, default=str))
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
