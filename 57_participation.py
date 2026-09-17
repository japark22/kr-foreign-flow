#!/usr/bin/env python3
"""Participation depth: is no-participation real, and a corner of something?

The recovered sample produced a large effect on a binary state -- no
institutional trading at all in the twenty days into a provisional filing.
A binary is the degenerate corner of a continuous quantity, so it is tested
as one. Two questions, one construction:

  1. Is the level effect monotone in participation depth? A cliff at exactly
     zero with a flat interior is what an illiquidity artefact looks like.
  2. Does the directional coefficient scale with depth? The recovered events
     diluted the pooled coefficient; if the dilution is the mechanism rather
     than a failure, depth should order the coefficient.

Question 2 is POST HOC. It was formed after seeing the dilution and is
labelled as such wherever it is reported. It earns nothing unless it is
monotone and outside the placebo band.

Ranks are taken on the full daily cross-section and the subsample selects
which rows enter the regression, so bucket coefficients stay comparable and
thin days are not re-ranked on noise.

The decisive test is the matched contrast: same day, same filing kind,
nearest neighbour on size, volatility, turnover, price level and liquidity.
Linear controls do not absorb a small-cap tilt; matching does.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from krxflow import features, storage

ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "results" / "event_panel_v2.parquet"
OUT = ROOT / "results" / "participation.json"
V3 = ROOT / "results" / "event_panel_v3.parquet"
START = "20100101"
DRAWS = 200
MATCH_ON = ["c_size", "c_vol", "c_turn", "logpx", "advpct"]


def load(name):
    spec = importlib.util.spec_from_file_location(name.split("_")[0],
                                                  ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


fr = load("38_final_results")
ep = load("20_event_panel")
CTRL, MIN_N = fr.CTRL, fr.MIN_N


def season(ts):
    ts = pd.DatetimeIndex(ts)
    return ts.year * 4 + (ts.month - 1) // 3


def cluster_t(values, seasons):
    """Mean of a per-event quantity with season-clustered standard error."""
    df = pd.DataFrame({"v": values, "s": seasons}).dropna()
    if df["s"].nunique() < 8:
        return np.nan, np.nan, 0
    g = df.groupby("s")["v"].agg(["mean", "size"])
    w = g["size"] / g["size"].sum()
    m = float((g["mean"] * w).sum())
    se = float(np.sqrt(((w ** 2) * g["mean"].var(ddof=1)).sum()
                       * len(g) / max(len(g) - 1, 1)))
    se = float(g["mean"].std(ddof=1) / np.sqrt(len(g)))
    return m, (m / se if se > 1e-12 else np.nan), len(g)


def build_depth():
    """Participation depth per ticker-day, from genuine activity only."""
    p = features.load_panels(START, None)
    pct = p["foreign_pct"]
    idx, cols = pct.index, pct.columns

    m = storage.read_range("market", START, None,
                           columns=["trade_date", "ticker", "close", "value_traded"])
    m["trade_date"] = pd.to_datetime(m["trade_date"])

    def pv(df, c):
        return (df.pivot_table(index="trade_date", columns="ticker", values=c,
                               aggfunc="last", observed=True)
                  .sort_index().astype("float64")
                  .reindex(index=idx, columns=cols))

    close, vt = pv(m, "close"), pv(m, "value_traded")
    del m

    iv = storage.read_range("investor_flow", START, None,
                            columns=["trade_date", "ticker", "investor", "net_value"])
    iv = iv[iv["investor"] == "기관합계"].copy()
    iv["trade_date"] = pd.to_datetime(iv["trade_date"])
    net = pv(iv, "net_value")
    del iv

    active = (net.notna() & (net != 0)).astype("float64")
    adv = vt.rolling(20, min_periods=5).mean()
    out = {
        "inst_days20": active.rolling(20, min_periods=20).sum(),
        "inst_abs20": (net.abs().fillna(0).rolling(20, min_periods=20).sum()
                       / (adv * 20).where(adv > 0)),
        "logpx": np.log(close.where(close > 0)),
        "advpct": adv.rank(axis=1, pct=True),
        "zerovol20": (vt.fillna(0) == 0).astype("float64")
                     .rolling(20, min_periods=20).sum(),
    }
    return idx, cols, out


def main() -> int:
    ev = pd.read_parquet(PANEL)
    idx, cols, extra = build_depth()
    colpos = {t: j for j, t in enumerate(cols)}
    ev = ev[ev["ticker"].isin(colpos)].reset_index(drop=True)
    pos = idx.searchsorted(ev["D"].to_numpy())
    jj = ev["ticker"].map(colpos).to_numpy()
    for k, panel in extra.items():
        ev[k] = panel.to_numpy()[pos - 1, jj]
    ev["sea"] = season(ev["D"])
    ev.to_parquet(V3, index=False)

    R = {"draws": DRAWS, "min_n": int(MIN_N), "post_hoc": ["depth_ladder"]}
    prov = ev[ev["kind"] == "provisional"].copy()
    per = ev[ev["kind"] == "periodic"].copy()

    for tag, sub in (("provisional", prov), ("periodic", per)):
        print(f"\n{'=' * 68}\n{tag}\n{'=' * 68}")
        d = sub.dropna(subset=["abn60", "inst_days20"]).copy()
        zero = d["inst_days20"] == 0
        print(f"  events {len(d):,}   depth-zero {int(zero.sum()):,} "
              f"({zero.mean():.3f})")

        print("\n  raw group means, no adjustment")
        for name, m_ in (("depth 0", zero), ("depth > 0", ~zero)):
            g = d[m_]
            print(f"    {name:<10} n={len(g):>7,}  abn60 mean={g['abn60'].mean()*1e4:+8.1f} bp"
                  f"  median={g['abn60'].median()*1e4:+8.1f}"
                  f"  p10={g['abn60'].quantile(0.10)*1e4:+9.1f}"
                  f"  size={g['c_size'].mean():.2f}"
                  f"  advpct={g['advpct'].mean():.3f}"
                  f"  zerovol={g['zerovol20'].mean():.2f}")

        # ---------- 1. monotonicity in depth
        print("\n  1. level effect by participation depth  (cliff or slope?)")
        bins = [-0.1, 0.5, 3.5, 7.5, 13.5, 20.1]
        labs = ["0", "1-3", "4-7", "8-13", "14-20"]
        d["bucket"] = pd.cut(d["inst_days20"], bins=bins, labels=labs)
        # within-day demeaned outcome, so the comparison is same-day
        d["abn_dm"] = d["abn60"] - d.groupby("D")["abn60"].transform("mean")
        tbl = {}
        for b in labs:
            g = d[d["bucket"] == b]
            if len(g) < 200:
                continue
            m_, t_, ncl = cluster_t(g["abn_dm"].to_numpy() * 1e4, g["sea"].to_numpy())
            tbl[b] = {"n": len(g), "bp": m_, "t": t_, "seasons": ncl}
            print(f"    depth {b:<6} n={len(g):>7,}   {m_:+8.1f} bp   t {t_:+5.2f}")
        R[f"{tag}_depth_level"] = tbl

        # ---------- 2. directional coefficient by depth  (POST HOC)
        print("\n  2. directional coefficient by depth  [POST HOC]")
        lad = {}
        for b in labs[1:]:
            g = d[d["bucket"] == b]
            if len(g) < 500:
                continue
            p = fr.prepare(g, "i_flow20_v2", ["surprise"] + CTRL + ["i_flow20_v2"])
            if p is None:
                continue
            r = fr.fit(p)
            lad[b] = r
            print(f"    depth {b:<6} {r['bp']:+8.1f} bp/SD   t {r['t']:+5.2f}"
                  f"   events {r['events']:>7,}")
        R[f"{tag}_depth_slope"] = lad

        # ---------- 3. matched contrast
        print("\n  3. matched contrast, depth 0 vs nearest participating name")
        md = d.dropna(subset=MATCH_ON).copy()
        Z = md[MATCH_ON].to_numpy(dtype=float)
        Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 1e-12, Z.std(0), 1.0)
        tr = (md["inst_days20"] == 0).to_numpy()
        if tr.sum() < 100 or (~tr).sum() < 100:
            print("    too few on one side; skipped")
            continue
        w = (Z[tr].mean(0) - Z[~tr].mean(0)) / np.maximum(Z.var(0), 1e-12)
        w = w / np.linalg.norm(w)
        md["score"] = Z @ w
        y = md["abn60"].to_numpy() * 1e4
        day = md["D"].to_numpy()
        sea = md["sea"].to_numpy()

        def contrast(treated):
            diffs, seas = [], []
            for dd in np.unique(day):
                m_ = day == dd
                ti = np.where(m_ & treated)[0]
                ci = np.where(m_ & ~treated)[0]
                if len(ti) == 0 or len(ci) < 3:
                    continue
                cs = md["score"].to_numpy()[ci]
                order = np.argsort(cs)
                ci, cs = ci[order], cs[order]
                k = np.clip(np.searchsorted(cs, md["score"].to_numpy()[ti]),
                            0, len(ci) - 1)
                diffs.append(y[ti] - y[ci[k]])
                seas.append(sea[ti])
            if not diffs:
                return np.nan, np.nan, 0
            return cluster_t(np.concatenate(diffs), np.concatenate(seas))

        m_, t_, ncl = contrast(tr)
        print(f"    matched difference {m_:+8.1f} bp   t {t_:+5.2f}   seasons {ncl}")

        rng = np.random.default_rng(11)
        n_by_day = pd.Series(tr).groupby(day).sum()
        vals = []
        part_idx = {dd: np.where((day == dd) & ~tr)[0] for dd in np.unique(day)}
        for _ in range(DRAWS):
            fake = np.zeros(len(md), dtype=bool)
            for dd, k in n_by_day.items():
                pool = part_idx[dd]
                if k == 0 or len(pool) <= k + 3:
                    continue
                fake[rng.choice(pool, size=int(k), replace=False)] = True
            vals.append(contrast(fake)[0])
        v = np.array([x for x in vals if np.isfinite(x)])
        lo, hi = np.percentile(v, [5, 95])
        inside = lo <= m_ <= hi
        print(f"    placebo band [{lo:+8.1f}, {hi:+8.1f}]  "
              f"mean {v.mean():+7.1f}  draws {len(v)}")
        print(f"    VERDICT: {'INSIDE the band -- not established' if inside else 'outside the band'}")
        R[f"{tag}_matched"] = {"bp": m_, "t": t_, "seasons": ncl,
                               "placebo_p05": float(lo), "placebo_p95": float(hi),
                               "placebo_mean": float(v.mean()),
                               "draws": int(len(v)), "inside_band": bool(inside)}

    OUT.write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str))
    print(f"\nwrote {OUT.relative_to(ROOT)} and {V3.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
