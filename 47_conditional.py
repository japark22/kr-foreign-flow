"""What the coefficient is actually measuring.

46_event_path drew the unconditional picture and it does not show what this
project has been describing. Stocks institutions bought hardest into the
announcement rose 985 bp more than the ones they bought least, and then went
on to earn +37 bp more, at t +0.75. There is no unwind. Yet the regression
reports -42.9 at t -2.51 on the same events and the same returns.

The difference is the control set, and inside it c_mom20 and c_mom60 are the
run-up itself. So the coefficient is not "crowded positions unwind" -- it is
"among stocks that ran up the same amount, the ones institutions bought more
of do slightly worse". Whether that is a finding or an artefact of one control
set is what this file settles.

  1  the control ladder: add controls one group at a time and watch the
     coefficient. If momentum is what creates it, that is the name it has to
     carry.
  2  the same event-time path, but on flow orthogonalised to momentum inside
     the day. If the effect is real in residual space the reversal shows up
     there, and that is the honest picture to publish. If it does not, the
     coefficient is a fragile function of the controls and has to be reported
     that way.

    python 47_conditional.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path("results/event_panel.parquet")
OUT = Path("results/conditional.json")
BASE, PRE, POST, NQ, MIN_N = "i_flow20", 20, 60, 5, 5
START = "20100101"

spec = importlib.util.spec_from_file_location("bfm", "21_baseline_fm.py")
bfm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bfm)

LADDER = [
    ("baseline alone", []),
    ("+ surprise", ["surprise"]),
    ("+ size, vol, turnover", ["surprise", "c_size", "c_vol", "c_turn"]),
    ("+ momentum (the full set)",
     ["surprise", "c_size", "c_vol", "c_turn", "c_mom20", "c_mom60"]),
    ("momentum only", ["c_mom20", "c_mom60"]),
]


def pooled(sub, base, ctrl, y="abn60"):
    xs = [base] + list(ctrl)
    X, Y, S = [], [], []
    for day, g in sub.groupby("D"):
        g = g.dropna(subset=[y] + xs)
        if len(g) < MIN_N:
            continue
        cols = [bfm.rank_std(g[c]) for c in xs] + [np.ones(len(g))]
        X.append(np.column_stack(cols))
        v = g[y].to_numpy(dtype=float)
        lo, hi = np.percentile(v, [1, 99])
        Y.append(np.clip(v, lo, hi))
        t = pd.Timestamp(day)
        S.append(np.full(len(g), t.year * 4 + (t.month - 1) // 3))
    if not X:
        return None
    X, y_, sea = np.vstack(X), np.concatenate(Y), np.concatenate(S)
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y_)
    e = y_ - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(sea):
        m = sea == c
        u = X[m].T @ e[m]
        meat += np.outer(u, u)
    V = XtX_inv @ meat @ XtX_inv
    se = float(np.sqrt(max(V[0, 0], 1e-30)))
    return {"bp": float(b[0] * 1e4), "t": float(b[0] / se),
            "events": int(len(y_)), "seasons": int(len(np.unique(sea)))}


def main() -> int:
    from krxflow import storage

    d = pd.read_parquet(PANEL)
    d["D"] = pd.to_datetime(d["D"])
    prov = d[d["kind"] == "provisional"].dropna(subset=["abn60", BASE]).copy()
    print(f"provisional {len(prov):,}\n")
    R = {}

    print("1  control ladder, institutional flow on abn60")
    R["ladder"] = {}
    for tag, ctrl in LADDER:
        r = pooled(prov, BASE if "momentum only" not in tag else "c_mom20",
                   ctrl if "momentum only" not in tag else [BASE])
        if r is None:
            continue
        R["ladder"][tag] = r
        what = "run-up" if "momentum only" in tag else "flow"
        print(f"   {tag:<28}{what:<8}{r['bp']:+8.1f} bp/SD"
              f"   t {r['t']:+5.2f}   events {r['events']:,}")

    # 2 -- flow with the run-up projected out, inside each event day
    print("\n2  event-time path on flow orthogonalised to momentum")
    prov["resid"] = np.nan
    for day, g in prov.groupby("D"):
        g2 = g.dropna(subset=[BASE, "c_mom20", "c_mom60"])
        if len(g2) < MIN_N:
            continue
        M = np.column_stack([bfm.rank_std(g2["c_mom20"]),
                             bfm.rank_std(g2["c_mom60"]),
                             np.ones(len(g2))])
        f = bfm.rank_std(g2[BASE])
        prov.loc[g2.index, "resid"] = f - M @ (np.linalg.pinv(M) @ f)
    have = prov["resid"].notna()
    print(f"   residual defined for {have.mean():.3f} of events")

    m = storage.read_range("market", START, None,
                           columns=["trade_date", "ticker", "close"])
    m["trade_date"] = pd.to_datetime(m["trade_date"])
    close = (m.pivot_table(index="trade_date", columns="ticker",
                           values="close", aggfunc="last", observed=True)
               .sort_index().astype("float64"))
    del m
    idx, cols = close.index, close.columns
    ret = close.pct_change(fill_method=None)
    ret = ret.mask(ret.abs() > 0.5)
    ex = ret.sub(ret.mean(axis=1), axis=0).to_numpy()
    del ret, close

    p = prov[have].copy()
    rp, cp = idx.get_indexer(p["D"]), cols.get_indexer(p["ticker"])
    keep = (rp >= PRE) & (cp >= 0) & (rp + POST < len(idx))
    p, rp, cp = p[keep], rp[keep], cp[keep]
    offs = np.arange(-PRE, POST + 1)
    path = ex[rp[:, None] + offs[None, :], cp[:, None]]
    print(f"   events with a complete window {len(p):,}")

    q = p.groupby("D")["resid"].rank(pct=True)
    p["q"] = np.clip((q * NQ).astype(int), 0, NQ - 1)
    p["season"] = p["D"].dt.year * 4 + (p["D"].dt.month - 1) // 3

    R["path_resid"] = {}
    for k in range(NQ):
        sel = (p["q"] == k).to_numpy()
        md = np.nanmean(path[sel], axis=0)
        pre = float(np.nansum(md[:PRE]) * 1e4)
        post = float(np.nansum(md[PRE + 1:]) * 1e4)
        R["path_resid"][k] = {"car_bp": (np.nancumsum(md) * 1e4).tolist(),
                              "events": int(sel.sum()),
                              "run_up_bp": pre, "after_bp": post}
        print(f"   quintile {k + 1}   events {sel.sum():,}   "
              f"run-up {pre:+8.1f} bp   after {post:+7.1f} bp")

    top = (p["q"] == NQ - 1).to_numpy()
    bot = (p["q"] == 0).to_numpy()
    sea = p["season"].to_numpy()

    def spread(v):
        rows = [v[top & (sea == s)].mean() - v[bot & (sea == s)].mean()
                for s in np.unique(sea)
                if (top & (sea == s)).any() and (bot & (sea == s)).any()]
        r = np.array(rows)
        return float(r.mean() * 1e4), float(bfm.tracker._nw_t(r, 1)), len(r)

    pb, pt, _ = spread(np.nansum(path[:, :PRE], axis=1))
    ab, at, ns = spread(np.nansum(path[:, PRE + 1:], axis=1))
    R["spread_resid"] = {"pre": {"bp": pb, "t": pt},
                         "post": {"bp": ab, "t": at}, "seasons": ns}
    print(f"\n   top minus bottom, {ns} seasons")
    print(f"     run-up   {pb:+8.1f} bp   t {pt:+5.2f}")
    print(f"     after    {ab:+8.1f} bp   t {at:+5.2f}")

    verdict = ("the effect is visible in residual space -- the honest name is "
               "buying in excess of the run-up, and that is the picture to "
               "publish" if ab < 0 and at < -1.5 else
               "the effect is not visible even in residual space -- the "
               "coefficient depends on the control set and has to be reported "
               "as such")
    print(f"\nverdict: {verdict}")
    R["verdict"] = verdict
    R["offsets"] = offs.tolist()
    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
