"""Step 9: ownership LEVEL signals, not flow.

Every construction tested so far measures a CHANGE in holdings. That family is
now well covered and fails structurally: the effect sits in illiquid names and
is exactly zero in the liquid third.

The original reasoning invoked 13F research, but 13F is built on ownership
LEVELS and slow changes in them, not daily order flow. That gap is untested.
Levels also attack what killed every flow variant: they move slowly, so
turnover is low, so a small edge can survive a 20bp transaction tax.

  level     ownership %, cross-sectionally ranked
  level_z   ownership vs its OWN trailing 1y mean/sd (scarcity, not size)
  d60,d120  slow change in ownership, percentage points
  exhaust   foreign-limit exhaustion, only for names with a binding cap

In-sample ranks and fixes the trade direction; out-of-sample decides. Both must
be significant with a consistent sign. The test count is printed.
"""
import itertools, sys
import numpy as np, pandas as pd
from krxflow import features, storage

SPLIT = pd.Timestamp("20200101")
HORIZONS = [20, 60]
COST = 0.0030
Z_WIN = 252

def pv(df, col):
    return (df.pivot_table(index="trade_date", columns="ticker", values=col,
            aggfunc="last", observed=True).sort_index().astype("float64"))

print("loading ...")
p = features.load_panels()
m = storage.read_range("market", columns=["trade_date","ticker","close","value_traded"])
m["trade_date"] = pd.to_datetime(m["trade_date"])
idx, cols = p["foreign_shares"].index, p["foreign_shares"].columns
close = pv(m,"close").reindex(index=idx, columns=cols)
adv = pv(m,"value_traded").reindex(index=idx, columns=cols).rolling(20, min_periods=5).mean()
del m

uni = features.universe_mask(p)
lvl = p["foreign_pct"].where(uni)
limit = p["foreign_limit_shares"]
binding = (limit < p["shares_listed"] * 0.999)

sig = {}
sig["level"]   = lvl
mu = lvl.rolling(Z_WIN, min_periods=120).mean()
sd = lvl.rolling(Z_WIN, min_periods=120).std()
sig["level_z"] = (lvl - mu) / sd.where(sd > 0)
sig["d60"]     = lvl.diff(60)
sig["d120"]    = lvl.diff(120)
ex = (p["foreign_shares"] / limit.where(limit > 0) * 100).where(uni & binding)
sig["exhaust"] = ex

print(f"  dates      : {idx[0].date()} -> {idx[-1].date()}  ({len(idx):,} days)")
print(f"  universe    : {int(uni.any().sum()):,} tickers")
print(f"  binding limit: {int(binding.any().sum()):,} tickers ever capped")
print(f"  in-sample before {SPLIT.date()}, out-of-sample after\n")

adv_rank = adv.rank(axis=1, pct=True)
LIQ = {"all": None, "liquid_top33": 0.67}
grid = list(itertools.product(sig.keys(), LIQ.keys(), HORIZONS))
print(f"  {len(grid)} tests ({len(sig)} signals x {len(LIQ)} universes x {len(HORIZONS)} horizons)\n")

def ev(s, h, msk, direction=None):
    """direction: +1 long the top decile, -1 long the bottom. Chosen in-sample
    from the sign of the IC, then applied unchanged out-of-sample. A negative-IC
    signal is perfectly tradable by inverting the legs; refusing to do that
    would discard a real reversal effect."""
    s = features.cross_sectional_rank(s).shift(1).loc[msk]
    f = features.forward_returns(close, h).reindex(index=s.index, columns=s.columns)
    ic = s.corrwith(features.cross_sectional_rank(f), axis=1).dropna()
    if len(ic) < 60: return None
    t = ic.mean()/(ic.std(ddof=1)/np.sqrt(len(ic)))
    dirn = direction if direction is not None else (1 if ic.mean() >= 0 else -1)
    q = s.rank(axis=1, pct=True); L = q >= 0.8
    ls = dirn*(f.where(L).mean(axis=1) - f.where(q <= 0.2).mean(axis=1)).dropna()
    med = dirn*(f.where(L).median(axis=1) - f.where(q <= 0.2).median(axis=1)).dropna()
    hold = L.astype(float)
    ch = ((hold.diff().abs().sum(axis=1)/2)/hold.sum(axis=1).replace(0,np.nan)).dropna().mean()
    turn = (ch or 0)*(252/h)*2
    ann = ls.mean()/h*252
    return {"ic":ic.mean(), "t":t, "gross":ann, "turn":turn, "dir":dirn,
            "net":ann-turn*COST, "med":med.mean()/h*252,
            "sh":ann/(ls.std(ddof=1)/np.sqrt(h)*np.sqrt(252)) if ls.std(ddof=1) else np.nan}

rows=[]
IS, OOS = idx < SPLIT, idx >= SPLIT
for name, liq, h in grid:
    s = sig[name]
    if LIQ[liq] is not None: s = s.where(adv_rank >= LIQ[liq])
    a = ev(s, h, IS)
    b = ev(s, h, OOS, direction=a["dir"]) if a else None
    if a and b:
        rows.append(dict(signal=name, universe=liq, h=h, ic_is=a["ic"], t_is=a["t"],
                         ic_oos=b["ic"], t_oos=b["t"], net=b["net"], med=b["med"],
                         turn=b["turn"], sh=b["sh"], dirn=a["dir"]))
print(f"  evaluated {len(rows)} of {len(grid)}")

d = pd.DataFrame(rows).sort_values("t_oos", key=abs, ascending=False)
print("\n  Direction is set by the in-sample IC sign, then held fixed out-of-sample.")
print("  net and median are for that direction, so a reversal signal is credited")
print("  properly rather than discarded.\n")
print("  signal    universe      h   IC(is)   t(is)   IC(oos)  t(oos)  net(oos)  median  turn  side")
print("  --------  ------------  --  -------  ------  --------  ------  --------  ------  ----  --------")
for _, r in d.iterrows():
    fl = "" if np.sign(r.ic_is) == np.sign(r.ic_oos) else " FLIP"
    d_lbl = "long-top" if r.dirn > 0 else "long-bot"
    print(f"  {r.signal:<8}  {r.universe:<12}  {r.h:>2}  {r.ic_is:+.4f}  {r.t_is:+6.2f}  "
          f"{r.ic_oos:+8.4f}  {r.t_oos:+6.2f}  {r.net:+7.2%}  {r.med:+6.2%}  "
          f"{r.turn:3.0f}x  {d_lbl}{fl}")

surv = d[(d.t_oos.abs()>3)&(d.t_is.abs()>3)&(np.sign(d.ic_is)==np.sign(d.ic_oos))&(d.net>0)]
print("\n=== survives: significant both periods, consistent sign, net positive ===")
if surv.empty:
    print("  Nothing. Level signals fail the same way flow signals did.")
    print("  Combined with step 7, the disclosure does not carry a tradable")
    print("  edge in any construction tested. That is the answer.")
else:
    for _, r in surv.iterrows():
        agree = "mean and median agree" if np.sign(r.net)==np.sign(r.med) else "MEAN/MEDIAN DISAGREE - skew"
        print(f"  {r.signal} / {r.universe} / {r.h}d: IC {r.ic_is:+.4f} -> {r.ic_oos:+.4f} "
              f"(t={r.t_oos:+.2f}), net {r.net:+.2%}, Sharpe {r.sh:+.2f}, {r.turn:.0f}x turn")
        print(f"      {agree}  (median long-short {r.med:+.2%})")
    print(f"\n  {len(d)} tests were run. Check the median column: if it disagrees")
    print("  with net, the return is skew and not an edge.")
out = storage.config.FEATURE_DIR / "level_signals.csv"
out.parent.mkdir(parents=True, exist_ok=True)
d.to_csv(out, index=False)
print(f"\n  grid -> {out}")
