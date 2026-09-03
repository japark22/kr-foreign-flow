"""Why is the tail worse -- a friction story, tested rather than told.

The tail finding says: after the strongest surprises, names institutions were
already crowded into have a tenth percentile 100-240 bp worse than uncrowded
names, with no difference in the median. The explanation offered was exit
friction -- when a crowded print disappoints, everyone leaves through the same
narrow door. That is a story until it makes predictions that could fail.

It makes two.

  P1  liquidity   the door is narrower where turnover is low. The gap should be
                  largest in the low-turnover tercile and near zero in the
                  high-turnover tercile. A gap that persists in liquid names is
                  evidence against the story, not for it.
  P2  short bans  short sellers are the natural counterparty to an unwind.
                  Korea banned short selling outright from 2020-03-16 to
                  2021-05-02 and from 2023-11-06 to 2025-03-30. During those
                  windows the gap should be larger. The partial-permission
                  period is counted as unbanned, which is the conservative
                  choice for this test.

Both signs are fixed here. Twelve or so seasons fall inside the bans, so P2 is
a low-power test and is reported as such; P1 has the full sample behind it and
is the one that decides.

    python 52_mechanism_tail.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("results/decompose_panel.parquet")
OUT = Path("results/mechanism_tail.json")
CROWD, NS, NC = "i_flow20", 5, 3
BANS = [("2020-03-16", "2021-05-02"), ("2023-11-06", "2025-03-30")]

spec = importlib.util.spec_from_file_location("bfm", "21_baseline_fm.py")
bfm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bfm)
p10 = lambda x: np.percentile(x, 10)


def season_diff(g, ma, mb, stat, min_n=8, min_seasons=6):
    rows, v = [], g["abn60"].to_numpy()
    for s in np.unique(g["season"]):
        m = (g["season"] == s).to_numpy()
        a, b = v[m & ma], v[m & mb]
        if len(a) >= min_n and len(b) >= min_n:
            rows.append(stat(a) - stat(b))
    r = np.array(rows, dtype=float)
    if len(r) < min_seasons:
        return None
    return {"bp": float(r.mean() * 1e4), "t": float(bfm.tracker._nw_t(r, 1)),
            "seasons": int(len(r))}


def line(tag, r):
    if r is None:
        print(f"  {tag:<36}too few comparable seasons")
    else:
        print(f"  {tag:<36}{r['bp']:+8.1f} bp   t {r['t']:+5.2f}"
              f"   seasons {r['seasons']}")


def main() -> int:
    d = pd.read_parquet(CACHE)
    d["D"] = pd.to_datetime(d["D"])
    ban = np.zeros(len(d), bool)
    for a, b in BANS:
        ban |= ((d["D"] >= a) & (d["D"] <= b)).to_numpy()
    d["ban"] = ban
    R = {}
    for kind in ("provisional", None):
        sub = d if kind is None else d[d["kind"] == kind]
        sub = sub.dropna(subset=["abn60", "surprise", CROWD, "c_turn", "c_vol"]).copy()
        sub["season"] = sub["D"].dt.year * 4 + (sub["D"].dt.month - 1) // 3
        g = sub.groupby("D")
        sub["s"] = np.clip((g["surprise"].rank(pct=True) * NS).astype(int), 0, NS - 1)
        sub["c"] = np.clip((g[CROWD].rank(pct=True) * NC).astype(int), 0, NC - 1)
        sub["l"] = np.clip((g["c_turn"].rank(pct=True) * 3).astype(int), 0, 2)
        top = sub[sub["s"] == NS - 1].copy()
        crowded = (top["c"] == NC - 1).to_numpy()
        quiet = (top["c"] == 0).to_numpy()
        key = kind or "all"
        label = kind or "all filings"
        R[key] = {}
        print(f"\n================ {label}: {len(top):,} strongest-surprise "
              f"events ================")

        print("P1  by turnover tercile  (declared: low worst, high near zero)")
        R[key]["by_turnover"] = {}
        for li, name in enumerate(("low turnover", "mid turnover", "high turnover")):
            m = (top["l"] == li).to_numpy()
            r = season_diff(top, crowded & m, quiet & m, p10, min_n=5)
            R[key]["by_turnover"][name] = r
            line(f"   {name}", r)

        print("\nP2  by short-sale regime  (declared: banned worse)")
        R[key]["by_ban"] = {}
        for name, m in (("short selling banned", top["ban"].to_numpy()),
                        ("short selling allowed", ~top["ban"].to_numpy())):
            r = season_diff(top, crowded & m, quiet & m, p10)
            R[key]["by_ban"][name] = r
            line(f"   {name}", r)
        n_ban = int(top["ban"].sum())
        print(f"   ({n_ban:,} of {len(top):,} events fall inside a ban)")

        print("\n    the whole story at once: low turnover AND banned")
        m = (top["l"] == 0).to_numpy() & top["ban"].to_numpy()
        r = season_diff(top, crowded & m, quiet & m, p10, min_n=4, min_seasons=4)
        R[key]["low_turn_banned"] = r
        line("   low turnover, banned", r)

        bt = R[key]["by_turnover"]
        lo, hi = bt.get("low turnover"), bt.get("high turnover")
        p1 = bool(lo and hi and lo["bp"] < hi["bp"] and abs(hi["t"]) < 1.5)
        bb = R[key]["by_ban"]
        b1, b0 = bb.get("short selling banned"), bb.get("short selling allowed")
        p2 = bool(b1 and b0 and b1["bp"] < b0["bp"])
        R[key]["passes"] = {"P1_liquidity": p1, "P2_ban": p2}
        print(f"\n   P1 {'PASS' if p1 else 'FAIL'}   P2 {'PASS' if p2 else 'FAIL'}"
              f"   (P2 is low-power: sign only)")

    big = R.get("all", {}).get("passes", {})
    verdict = ("exit friction is supported -- the gap lives where the door is "
               "narrow" if big.get("P1_liquidity") else
               "the friction story is not supported -- the tail gap does not "
               "track liquidity, and it is reported as a pattern without a "
               "mechanism")
    print(f"\nverdict: {verdict}")
    R["verdict"] = verdict
    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
