import sys
import numpy as np, pandas as pd
from krxflow import features, storage

ACC, HOR, SPLIT = 20, 20, pd.Timestamp("20200101")

def pv(df, col):
    return (df.pivot_table(index="trade_date", columns="ticker", values=col,
            aggfunc="last", observed=True).sort_index().astype("float64"))

print("loading ...")
panels = features.load_panels()
m = storage.read_range("market", columns=["trade_date","ticker","close","value_traded"])
m["trade_date"] = pd.to_datetime(m["trade_date"])
idx, cols = panels["foreign_shares"].index, panels["foreign_shares"].columns
close = pv(m,"close").reindex(index=idx, columns=cols)
adv = pv(m,"value_traded").reindex(index=idx, columns=cols).rolling(20, min_periods=5).mean()
del m

d = panels["foreign_shares"].diff().mask(features.corporate_action_mask(panels))
d = d.rolling(ACC, min_periods=ACC//2).sum()
flow = ((d*close)/adv.where(adv>0)).where(features.universe_mask(panels))
sig = features.cross_sectional_rank(flow).shift(1)
fwd = features.forward_returns(close, HOR)
q = sig.rank(axis=1, pct=True)
ql, qh = fwd.quantile(.01,axis=1), fwd.quantile(.99,axis=1)
fwdw = fwd.clip(lower=ql, upper=qh, axis=0)
per = {"in-sample": sig.index < SPLIT, "out-of-sample": sig.index >= SPLIT}

print("\n=== 1. decile returns: is it monotone, and is the mean just skew? ===")
for name, msk in per.items():
    qp, fp, fwp = q.loc[msk], fwd.loc[msk], fwdw.loc[msk]
    print("\n " + name)
    print("  dec      mean      median   winsor")
    rows=[]
    for i in range(10):
        lo, hi = i/10, (i+1)/10
        sel = (qp>lo)&(qp<=hi) if i else (qp>=0)&(qp<=hi)
        s = fp.where(sel).stack()
        rows.append((i+1, s.mean(), s.median(), fwp.where(sel).stack().mean()))
        print("  %3d  %+8.3f%%  %+8.3f%%  %+7.3f%%" % (rows[-1][0], rows[-1][1]*100,
              rows[-1][2]*100, rows[-1][3]*100))
    t, b = rows[-1], rows[0]
    print("  top-bottom  mean %+.3f%%   median %+.3f%%   winsor %+.3f%%"
          % ((t[1]-b[1])*100, (t[2]-b[2])*100, (t[3]-b[3])*100))
    if np.isnan(t[1]) or np.isnan(t[2]):
        print("  -> no data in this period, skipping")
        continue
    if np.sign(t[1]-b[1]) != np.sign(t[2]-b[2]):
        print("  -> MEAN AND MEDIAN DISAGREE: the spread is skew, not an edge")
    else:
        print("  -> mean and median agree in sign")
    mn = np.array([r[1] for r in rows]); md = np.array([r[2] for r in rows])
    x = np.arange(1,11)
    print("  monotonicity  mean %+.2f  median %+.2f"
          % (np.corrcoef(x,mn)[0,1], np.corrcoef(x,md)[0,1]))

print("\n=== 2. does the sign flip with liquidity, in BOTH periods? ===")
ar = adv.rank(axis=1, pct=True)
print("  bucket            IC(is)     t      IC(oos)    t")
for lab, lo, hi in [("thin  b33",0.0,0.33), ("mid 33-67",0.33,0.67), ("liquid t33",0.67,1.01)]:
    sel = (ar>=lo)&(ar<hi)
    out=[]
    for name, msk in per.items():
        s = sig.where(sel).loc[msk]
        f = features.cross_sectional_rank(fwd.loc[msk].where(sel.loc[msk]))
        ic = s.corrwith(f, axis=1).dropna()
        out.append((ic.mean(), ic.mean()/(ic.std(ddof=1)/np.sqrt(len(ic)))) if len(ic)>60 else (np.nan,np.nan))
    flag = "  SIGN FLIP" if (not np.isnan(out[0][0]) and not np.isnan(out[1][0])
                             and np.sign(out[0][0])!=np.sign(out[1][0])) else ""
    print("  %-11s  %+9.5f  %+6.2f   %+9.5f  %+6.2f%s"
          % (lab, out[0][0], out[0][1], out[1][0], out[1][1], flag))

print("\n=== 3. is it stable year by year? ===")
ic = sig.corrwith(features.cross_sectional_rank(fwd), axis=1).dropna()
pos = neg = 0
print("  year        IC      t    days")
for y, s in ic.groupby(ic.index.year):
    if len(s) < 60: continue
    pos += s.mean() > 0; neg += s.mean() < 0
    print("  %d  %+9.5f  %+5.1f  %4d" % (y, s.mean(), s.mean()/(s.std(ddof=1)/np.sqrt(len(s))), len(s)))
print("  %d negative, %d positive" % (neg, pos))
if min(pos,neg)/max(pos+neg,1) > 0.3:
    print("  -> sign NOT stable across years: pooled t hides a mix, not one effect")
else:
    print("  -> sign consistent across years")
