"""Step 10: is the level signal real, or a size/quality proxy?

Step 9 reported IC +0.17 out-of-sample for raw ownership LEVEL inside the
liquid third, with ~zero turnover. An IC that large on a simple, widely known
variable does not happen. Three things say it is a factor tilt, not a signal:
zero turnover means a static sort, IC 0.17 with Sharpe 0.84 means the returns
are serially correlated rather than repeated independent bets, and the effect
is STRONGER out-of-sample (2020-2026), exactly when Korean mega-cap
semiconductors ran.

Inside the liquid third, "high foreign ownership" is largely "mega-cap blue
chip". So the portfolio is long size/quality and short small liquid names.

This script settles it:

  1. how correlated is each signal with market-cap rank
  2. IC after neutralising size (demean within size quintiles each day)
  3. IC of a pure size factor, as the benchmark to beat
  4. long-leg and short-leg contribution separately - a factor tilt earns
     nearly everything on one side
  5. correlation with price momentum, which step 4 only tested for flow
  6. real turnover, not rounded to zero

If level collapses under neutralisation and level_z / d60 / d120 do not, the
change-based signals are the honest candidates and the level result should be
discarded.
"""
import os
import numpy as np, pandas as pd
from krxflow import features, storage

SPLIT = pd.Timestamp(os.getenv("KRXFLOW_SPLIT","20200101")); HOR = int(os.getenv("KRXFLOW_HOR","60")); Z = 252

def pv(df, c):
    return (df.pivot_table(index="trade_date", columns="ticker", values=c,
            aggfunc="last", observed=True).sort_index().astype("float64"))

print("loading ...")
p = features.load_panels()
m = storage.read_range("market", columns=["trade_date","ticker","close","value_traded","market_cap"])
m["trade_date"] = pd.to_datetime(m["trade_date"])
i_, c_ = p["foreign_shares"].index, p["foreign_shares"].columns
close = pv(m,"close").reindex(index=i_, columns=c_)
adv = pv(m,"value_traded").reindex(index=i_, columns=c_).rolling(20, min_periods=5).mean()
mcap = pv(m,"market_cap").reindex(index=i_, columns=c_)
del m

uni = features.universe_mask(p)
liq = adv.rank(axis=1, pct=True) >= 0.67
lvl = p["foreign_pct"].where(uni & liq)
mu, sd = lvl.rolling(Z, min_periods=120).mean(), lvl.rolling(Z, min_periods=120).std()

S = {"level": lvl, "level_z": (lvl-mu)/sd.where(sd>0),
     "d60": lvl.diff(60), "d120": lvl.diff(120),
     "size(benchmark)": mcap.where(uni & liq)}

fwd = features.forward_returns(close, HOR)
fwd_r = features.cross_sectional_rank(fwd)
size_r = features.cross_sectional_rank(mcap.where(uni & liq))
mom_r = features.cross_sectional_rank(close.pct_change(120, fill_method=None).where(uni & liq))
sq = mcap.where(uni & liq).rank(axis=1, pct=True)   # size quintile buckets

def neutralise(s):
    """Remove the size tilt: within each day, demean the signal inside each
    size quintile. What survives is variation unrelated to how big the name is."""
    r = features.cross_sectional_rank(s)
    out = r.copy()
    for k in range(5):
        b = (sq > k/5) & (sq <= (k+1)/5) if k else (sq >= 0) & (sq <= 0.2)
        sub = r.where(b)
        out = out.mask(b, sub.sub(sub.mean(axis=1), axis=0))
    return out

def ic(sig, msk):
    s = sig.shift(1).loc[msk]
    x = s.corrwith(fwd_r.loc[msk], axis=1).dropna()
    if len(x) < 60: return (np.nan, np.nan)
    return x.mean(), x.mean()/(x.std(ddof=1)/np.sqrt(len(x)))

def legs(sig, msk):
    s = features.cross_sectional_rank(sig).shift(1).loc[msk]
    f = fwd.loc[msk].reindex(columns=s.columns)
    q = s.rank(axis=1, pct=True)
    base = f.where(s.notna()).mean(axis=1)
    lo = (f.where(q>=0.8).mean(axis=1) - base).dropna().mean()/HOR*252
    sh = (base - f.where(q<=0.2).mean(axis=1)).dropna().mean()/HOR*252
    hold = (q>=0.8).astype(float)
    t = ((hold.diff().abs().sum(axis=1)/2)/hold.sum(axis=1).replace(0,np.nan)).dropna().mean()
    return lo, sh, (t or 0)*(252/HOR)*2

IS, OOS = i_ < SPLIT, i_ >= SPLIT
print(f"  liquid third, {HOR}d horizon, in/out split {SPLIT.date()}\n")
print("  signal            corr(size)  after-neut  corr(mom)   IC raw(oos)    IC neut(oos)   kept")
print("  ----------------  ----------  ----------  ---------  -------------  -------------  -----")
for k, s in S.items():
    r = features.cross_sectional_rank(s)
    n = neutralise(s)
    cs = r.corrwith(size_r, axis=1).mean()
    cn = n.corrwith(size_r, axis=1).mean()     # self-check: should be near zero
    cm = r.corrwith(mom_r, axis=1).mean()
    a, ta = ic(r, OOS)
    b, tb = ic(n, OOS)
    keep = b/a if a and abs(a) > 1e-9 else np.nan
    print(f"  {k:<16}  {cs:>+10.3f}  {cn:>+10.3f}  {cm:>+9.3f}  {a:>+7.4f} t{ta:>+4.0f}  "
          f"{b:>+7.4f} t{tb:>+4.0f}  {keep:>5.0%}")

print("\n  'after-neut' is a self-check: it must be near zero, otherwise the")
print("  neutralisation did not remove the size tilt and the IC neut column")
print("  means nothing. Read it before reading anything else.")

print("\n  A factor tilt shows high corr(size) and loses most of its IC when")
print("  neutralised. A real signal keeps most of it.\n")
print("  signal            long leg   short leg   turnover   one-sided?")
print("  ----------------  --------   ---------   --------   ----------")
for k, s in S.items():
    lo, sh, t = legs(s, OOS)
    tot = abs(lo) + abs(sh)
    frac = abs(lo)/tot if tot else np.nan
    flag = "YES - tilt" if (frac > 0.75 or frac < 0.25) else "balanced"
    print(f"  {k:<16}  {lo:>+7.2%}   {sh:>+8.2%}   {t:>7.1f}x   {flag}")
print("\n  A genuine cross-sectional signal earns on both legs. A factor tilt")
print("  earns nearly all of it on one side.")
