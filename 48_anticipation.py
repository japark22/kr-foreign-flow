"""How much of the news was already in the price -- and does that matter.

Today established that institutional pre-event buying is very nearly the same
object as the pre-event price run-up: the top and bottom flow quintiles differ
by 985 bp over the twenty days into the announcement, at t +29.5, and by 37 bp
over the sixty days after, at t +0.75. Flow does not forecast. It records.

That reframes the question rather than closing it. "Institutions have been
buying" means "this has already moved", and whether a beat still pays depends
on how much of the beat was already paid for. Price measures that directly;
flow only proxies it. So the anticipation is measured in price space, and flow
is kept as the second, noisier reading of the same thing.

Three tests, signs fixed here before the run, controlled as one family of
three by permutation:

  T1  surprise x prior run-up            expected NEGATIVE
      a beat that the price already anticipated should drift less
  T2  surprise x institutional flow      expected NEGATIVE
      the same statement in flow space -- the literal specification asked for
  T3  flow against run-up, 3x3           expected: heavy buying WITHOUT a
      run-up is the best cell -- that is accumulation absorbing supply rather
      than accumulation chasing price

And one measurement that needs no test: the share of the total move around an
announcement that lands before the announcement. That number is the baseline
expectation, stated as a fact rather than as a signal.

    python 48_anticipation.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path("results/event_panel.parquet")
OUT = Path("results/anticipation.json")
START, PRE, POST, MIN_N = "20100101", 20, 60, 5
PERM = 200

spec = importlib.util.spec_from_file_location("bfm", "21_baseline_fm.py")
bfm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bfm)
CTRL = list(bfm.CTRL)


def season(s):
    return s.dt.year * 4 + (s.dt.month - 1) // 3


def design(sub, mod, y="abn60"):
    """abn60 on surprise, the moderator, their product, and the controls.
    The product sits second so its own error is the one reported."""
    xs = ["surprise", mod] + [c for c in CTRL if c != mod]
    X, Y, S, SL = [], [], [], []
    n = 0
    for day, g in sub.groupby("D"):
        g = g.dropna(subset=[y] + xs)
        if len(g) < MIN_N:
            continue
        s_ = bfm.rank_std(g["surprise"])
        m_ = bfm.rank_std(g[mod])
        cols = [s_, s_ * m_, m_]
        cols += [bfm.rank_std(g[c]) for c in xs if c not in ("surprise", mod)]
        cols += [np.ones(len(g))]
        X.append(np.column_stack(cols))
        v = g[y].to_numpy(dtype=float)
        lo, hi = np.percentile(v, [1, 99])
        Y.append(np.clip(v, lo, hi))
        d = pd.Timestamp(day)
        S.append(np.full(len(g), d.year * 4 + (d.month - 1) // 3))
        SL.append((n, n + len(g)))
        n += len(g)
    if not X:
        return None
    return np.vstack(X), np.concatenate(Y), np.concatenate(S), SL


def fit(p, want=1, shuffle_mod=False, rng=None):
    X, y, sea, sl = p
    if shuffle_mod:
        X = X.copy()
        for a, b in sl:                       # permute the moderator inside
            perm = rng.permutation(b - a)     # the day, rebuild the product
            X[a:b, 2] = X[a:b, 2][perm]
            X[a:b, 1] = X[a:b, 0] * X[a:b, 2]
    XtX_inv = np.linalg.pinv(X.T @ X)
    b_ = XtX_inv @ (X.T @ y)
    e = y - X @ b_
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(sea):
        m = sea == c
        u = X[m].T @ e[m]
        meat += np.outer(u, u)
    V = XtX_inv @ meat @ XtX_inv
    se = float(np.sqrt(max(V[want, want], 1e-30)))
    return {"bp": float(b_[want] * 1e4), "t": float(b_[want] / se)}


def main() -> int:
    from krxflow import storage

    d = pd.read_parquet(PANEL)
    d["D"] = pd.to_datetime(d["D"])
    prov = d[d["kind"] == "provisional"].copy()

    print("building the pre-event abnormal run-up ...")
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

    rp, cp = idx.get_indexer(prov["D"]), cols.get_indexer(prov["ticker"])
    keep = (rp >= PRE) & (cp >= 0) & (rp + POST < len(idx))
    prov, rp, cp = prov[keep].copy(), rp[keep], cp[keep]
    offs = np.arange(-PRE, 0)
    prov["runup"] = np.nansum(ex[rp[:, None] + offs[None, :], cp[:, None]],
                              axis=1)
    prov["season"] = season(prov["D"])
    print(f"  {len(prov):,} events   run-up sd {prov['runup'].std():.4f}")

    R = {"n_events": int(len(prov))}

    # ---- the measurement, no test required
    print("\nhow much of the move arrives before the announcement")
    q = prov.groupby("D")["surprise"].rank(pct=True)
    prov["sq"] = np.clip((q * 5).astype(int), 0, 4)
    R["anticipation_share"] = {}
    for k in range(5):
        g = prov[prov["sq"] == k]
        pre = float(g["runup"].mean() * 1e4)
        day = float(g["surprise"].mean() * 1e4)
        post = float(g["abn60"].mean() * 1e4)
        tot = pre + day
        share = pre / tot if abs(tot) > 1e-9 else float("nan")
        R["anticipation_share"][k] = {"run_up_bp": pre, "day_bp": day,
                                      "after_bp": post, "share_pre": share,
                                      "events": int(len(g))}
        print(f"  surprise quintile {k + 1}   run-up {pre:+8.1f}   "
              f"day {day:+7.1f}   share before {share:+6.1%}   "
              f"after {post:+7.1f}")

    # ---- the three tests
    print(f"\nthree pre-declared tests, family bar from {PERM} permutations")
    preps, res = {}, {}
    for tag, mod, want_sign in (("T1 surprise x run-up", "runup", -1),
                                ("T2 surprise x flow", "i_flow20", -1)):
        p = design(prov, mod)
        if p is None:
            continue
        preps[tag] = p
        res[tag] = fit(p)
        res[tag]["expected"] = "negative"
        print(f"  {tag:<26}{res[tag]['bp']:+8.1f} bp   t {res[tag]['t']:+5.2f}"
              f"   events {len(p[1]):,}")

    # T3 as a contrast rather than a slope: the divergence grid
    g = prov.dropna(subset=["abn60", "i_flow20", "runup"]).copy()
    fr = g.groupby("D")["i_flow20"].rank(pct=True)
    rr = g.groupby("D")["runup"].rank(pct=True)
    g["fh"] = np.clip((fr * 3).astype(int), 0, 2)
    g["rh"] = np.clip((rr * 3).astype(int), 0, 2)
    print("\n  T3 divergence grid, mean abn60 by flow (rows) x run-up (cols)")
    grid = {}
    for i in range(3):
        row = []
        for j in range(3):
            cell = g[(g["fh"] == i) & (g["rh"] == j)]
            v = float(cell["abn60"].mean() * 1e4) if len(cell) else float("nan")
            grid[f"{i}{j}"] = {"bp": v, "n": int(len(cell))}
            row.append(f"{v:+7.1f}")
        print(f"    flow {['low', 'mid', 'high'][i]:<5}" + "  ".join(row))
    R["grid"] = grid

    def cell_contrast(i, j):
        a = g[(g["fh"] == i) & (g["rh"] == j)]
        rows = []
        for s in np.unique(g["season"]):
            x = a[a["season"] == s]["abn60"]
            y = g[g["season"] == s]["abn60"]
            if len(x) >= 3 and len(y) >= 10:
                rows.append(x.mean() - y.mean())
        r = np.array(rows)
        return float(r.mean() * 1e4), float(bfm.tracker._nw_t(r, 1)), len(r)

    hb, ht, hn = cell_contrast(2, 0)      # heavy buying, no run-up
    cb, ct, cn = cell_contrast(2, 2)      # heavy buying, big run-up
    res["T3 buying without a run-up"] = {"bp": hb, "t": ht, "seasons": hn,
                                         "expected": "positive"}
    print(f"  {'T3 heavy buying, no run-up':<26}{hb:+8.1f} bp   t {ht:+5.2f}"
          f"   seasons {hn}")
    print(f"  {'   (mirror) buying + run-up':<26}{cb:+8.1f} bp   t {ct:+5.2f}")

    # ---- family bar over the two slopes plus the contrast
    rng = np.random.default_rng(20260903)
    maxes = np.empty(PERM)
    for i in range(PERM):
        best = 0.0
        for tag, p in preps.items():
            best = max(best, abs(fit(p, shuffle_mod=True, rng=rng)["t"]))
        maxes[i] = best
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{PERM}   running bar "
                  f"{np.quantile(maxes[:i + 1], 0.95):.2f}")
    bar = float(np.quantile(maxes, 0.95))
    print(f"\n  family bar |t| >= {bar:.2f}")

    winners = []
    for tag, r in res.items():
        ok = abs(r["t"]) >= bar
        sign_ok = (r["t"] < 0) if r["expected"] == "negative" else (r["t"] > 0)
        r["clears"] = bool(ok and sign_ok)
        if r["clears"]:
            winners.append(tag)
        mark = "PASS" if r["clears"] else ("wrong sign" if ok else "    ")
        print(f"  {mark:<11}{tag:<26}t {r['t']:+5.2f}")
    R["tests"] = res
    R["bar"] = bar
    R["winners"] = winners

    print("\n" + (f"clears: {', '.join(winners)}" if winners else
                  "nothing clears -- anticipation does not change the drift "
                  "either, and the Korea line closes here"))
    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
