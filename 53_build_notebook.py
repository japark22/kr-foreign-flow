"""Assemble the research record as a notebook: every figure drawn from the
result files at build time, every table generated from the same files, no
number typed by hand. Structure follows the house research-record format --
a question per step, a table, a chart -- and the text says only what the
numbers support.

    python 53_build_notebook.py
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path("results")
OUT = Path("reports/korea_positioning_record.ipynb")

# ---- palette (validated reference set) -------------------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURF = ("#0b0b0b", "#52514e", "#898781",
                                      "#e1e0d9", "#c3c2b7", "#fcfcfb")
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # ordinal
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#1c5cab",
       "#104281", "#0d366b"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "axes.titleweight": "medium", "axes.titlesize": 11,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
})


def J(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else {}


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src}


def code_img(title, b64):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "source": f"# {title}",
            "outputs": [{"output_type": "display_data",
                         "metadata": {},
                         "data": {"image/png": b64, "text/plain": "<Figure>"}}]}


def f1(v, sign=True): return ("&mdash;" if v is None else
                              (f"{v:+,.1f}" if sign else f"{v:,.1f}"))
def ft(v): return "&mdash;" if v is None else f"{v:+.2f}"
def se_of(r): return abs(r["bp"] / r["t"]) if r and r.get("t") else np.nan


# ============================================================== load results
path, ext, cond = J("path.json"), J("extended.json"), J("conditional.json")
base, tail, mech = J("baseline.json"), J("tail_robust.json"), J("mechanism_tail.json")
reg, fin, ant = J("regime.json"), J("final.json"), J("anticipation.json")
lvf, thr, dec = J("level_vs_flow.json"), J("threats_formal.json"), J("decompose.json")
book = J("book.json")

W = ext.get("windows", {})
full = W.get("full 2011-2026", {})
orig = W.get("original 2018-2026", {})
added = W.get("added 2011-2017", {})
foreign = ext.get("foreign", {}).get("f_flow20, abn60", {})
cov = fin.get("coverage", {})
bp = base.get("provisional", {}); ba = base.get("all", {})
tp = tail.get("provisional", {}); ta = tail.get("all", {})

# raw distribution for the density figure
cache = pd.read_parquet(R / "decompose_panel.parquet")
cache["D"] = pd.to_datetime(cache["D"])
prov = cache[cache["kind"] == "provisional"].dropna(
    subset=["abn60", "surprise", "i_flow20"]).copy()
g = prov.groupby("D")
prov["s"] = np.clip((g["surprise"].rank(pct=True) * 5).astype(int), 0, 4)
prov["c"] = np.clip((g["i_flow20"].rank(pct=True) * 3).astype(int), 0, 2)
top = prov[prov["s"] == 4]
quiet_v = top[top["c"] == 0]["abn60"].to_numpy() * 100
crowd_v = top[top["c"] == 2]["abn60"].to_numpy() * 100

cells = []

# ============================================================== title
cells.append(md(f"""# Korea: Pre-Announcement Positioning and the Earnings Baseline - Research Record

This is the working record of the positioning research. Every number and chart below is produced from the result files by the build script, not hand-entered. The question was whether Korea's daily investor-flow disclosure -- unusual among markets -- tells you anything about what happens after an earnings announcement, and if so, what.

| | |
| --- | --- |
| universe | provisional and periodic earnings filings, {cov.get('first','')} to {cov.get('last','')}, {cov.get('events_total',0):,} events |
| positioning measure | net institutional buying over the 20 trading days before the filing, scaled by turnover, standardised against the stock's own history |
| outcome | 60-trading-day return net of the equal-weight market, winsorised 1/99 within the day |
| estimator | one vote per event, regressors rank-standardised within the day, errors clustered on the reporting season |
| what the flow is | a record of the price move, not a forecast: quintile spread {path.get('spread',{}).get('pre',{}).get('bp',0):+,.0f} bp going in (t {path.get('spread',{}).get('pre',{}).get('t',0):+.1f}), {path.get('spread',{}).get('post',{}).get('bp',0):+,.0f} bp coming out (t {path.get('spread',{}).get('post',{}).get('t',0):+.2f}) |
| where the hypothesis holds | the tail, not the mean: after the strongest surprises, crowded names' 10th percentile is {bp.get('p10',{}).get('diff',0)*1e4:+,.0f} bp worse (t {bp.get('p10',{}).get('t',0):+.2f}), median and hit rate unchanged |
| foreign ownership flow | no forward information: {foreign.get('bp',0):+.1f} bp/SD, t {foreign.get('t',0):+.2f} |
"""))

cells.append(md("""## Data used, and which fields

| store | fields used | how the field is used |
| --- | --- | --- |
| market (KRX, daily) | trade_date, ticker, close, value_traded, market_cap | daily returns, 60-day forward returns, benchmark, turnover, size, volatility controls |
| investor_flow (KRX, daily) | trade_date, ticker, investor, net_value | net buying by investor group; the 20-day institutional baseline and its foreign counterpart |
| foreign_ownership (KRX, daily) | trade_date, ticker, foreign_pct | ownership level and its 20-day change, tested as level-type measures |
| investor detail (per ticker, 2018 on) | seven institutional sub-categories, other corporations, retail | decomposition of the aggregate into its components |
| earnings filings (DART) | rcept_dt, ticker, report_nm, kind | event dates; provisional announcements separated from periodic reports |
"""))

# ============================================================== PART A
cells.append(md("# Part A - What the daily flow disclosure actually is"))

# --- A1 event-time path
q = path.get("quintiles", {})
offs = path.get("offsets", list(range(-20, 61)))
first = path.get("first_forward_day", 1)
rows = []
for k in range(5):
    c = q.get(str(k), q.get(k, {}))
    car = c.get("car_bp", [])
    if not car:
        continue
    pre = car[19]                                 # cumulative through day -1
    post = car[-1] - car[20 + first - 1]
    rows.append((k + 1, c.get("events", 0), pre, post))
tbl = "\n".join(f"| {k} | {n:,} | {pre:+,.0f} | {post:+,.0f} |" for k, n, pre, post in rows)
sp = path.get("spread", {})
cells.append(md(f"""## Step 1 - The event-time path by pre-announcement flow

`data: market + investor_flow + earnings filings  (provisional announcements)`

**Question: does institutional buying before an announcement tell you what happens after it?**

Stocks are sorted each announcement day into quintiles of the 20-day institutional flow, and the benchmark-adjusted return is followed from 20 days before to 60 days after. The quintiles separate by {sp.get('pre',{}).get('bp',0):+,.0f} bp before the announcement (t {sp.get('pre',{}).get('t',0):+.1f}) and by {sp.get('post',{}).get('bp',0):+,.0f} bp after it (t {sp.get('post',{}).get('t',0):+.2f}). The flow records the move that has already happened. It does not forecast the one to come. This is the fact the rest of the record is organised around.

| flow quintile | events | run-up, day -20 to -1 (bp) | after, day +{first} to +60 (bp) |
| --- | --- | --- | --- |
{tbl}
"""))
fig, ax = plt.subplots(figsize=(9, 4.6))
for k in range(5):
    c = q.get(str(k), q.get(k, {}))
    car = c.get("car_bp", [])
    if not car:
        continue
    ax.plot(offs, car, color=RAMP[k], lw=2 if k in (0, 4) else 1.4,
            alpha=1 if k in (0, 4) else 0.85, solid_capstyle="round")
    if k in (0, 4):
        ax.annotate(f"quintile {k + 1}", (offs[-1], car[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=9)
ax.axvline(0, color=AXIS, lw=0.8)
ax.axhline(0, color=AXIS, lw=0.8)
ax.set_xlabel("trading days relative to the announcement")
ax.set_ylabel("cumulative benchmark-adjusted return, bp")
ax.set_title("Flow quintiles diverge before the announcement and run parallel after it")
ax.set_xlim(-20, 68)
ax.legend([plt.Line2D([], [], color=RAMP[i], lw=2) for i in range(5)],
          [f"Q{i+1}" + (" lowest buying" if i == 0 else " highest buying" if i == 4 else "")
           for i in range(5)], frameon=False, fontsize=8.5,
          loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5)
cells.append(code_img("Step 1 - event-time path by flow quintile", fig_to_b64(fig)))

# --- A2 foreign vs institutional
cells.append(md(f"""## Step 2 - The original question: foreign flow, and the domestic series next to it

`data: market + investor_flow + earnings filings  (provisional, 2011-2026)`

**Question: does the daily foreign-ownership disclosure carry forward information around earnings?**

It does not. Foreign flow reads {foreign.get('bp',0):+.1f} bp per standard deviation at t {foreign.get('t',0):+.2f} across {foreign.get('seasons',0)} reporting seasons and {foreign.get('events',0):,} events, and the result did not move when the sample was doubled. The domestic institutional series carries a negative coefficient of {full.get('bp',0):+.1f} bp/SD (t {full.get('t',0):+.2f}) in the full specification. The next step shows what that coefficient is made of.

| series | coefficient (bp/SD) | t | events | seasons |
| --- | --- | --- | --- | --- |
| foreign flow | {f1(foreign.get('bp'))} | {ft(foreign.get('t'))} | {foreign.get('events',0):,} | {foreign.get('seasons',0)} |
| domestic institutional flow | {f1(full.get('bp'))} | {ft(full.get('t'))} | {full.get('events',0):,} | {full.get('seasons',0)} |
"""))
fig, ax = plt.subplots(figsize=(8, 2.6))
names = ["foreign flow", "domestic institutional flow"]
vals = [foreign.get("bp", 0), full.get("bp", 0)]
ses = [se_of(foreign), se_of(full)]
cols = [MUTED, BLUE]
y = np.arange(2)
ax.barh(y, vals, height=0.42, color=cols, xerr=[2 * s for s in ses],
        error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3})
ax.axvline(0, color=AXIS, lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(names)
ax.set_xlabel("coefficient on 60-day benchmark-adjusted return, bp per SD (bars: +/-2 SE)")
ax.set_title("Only the domestic series reads negative, and only modestly")
ax.grid(axis="y", visible=False)
cells.append(code_img("Step 2 - foreign vs institutional", fig_to_b64(fig)))

# --- A3 control ladder
L = cond.get("ladder", {})
order = ["baseline alone", "+ surprise", "+ size, vol, turnover",
         "+ momentum (the full set)"]
tbl = "\n".join(f"| {k} | {f1(L[k]['bp'])} | {ft(L[k]['t'])} |" for k in order if k in L)
cells.append(md(f"""## Step 3 - What the institutional coefficient is made of

`data: market + investor_flow + earnings filings  (provisional, 2011-2026)`

**Question: is the negative coefficient a property of the flow, or of the controls?**

Controls are added one group at a time. Without momentum the coefficient is {f1(L.get('baseline alone',{}).get('bp'))} at t {ft(L.get('baseline alone',{}).get('t'))}, short of significance; the momentum controls take it to {f1(L.get('+ momentum (the full set)',{}).get('bp'))} at t {ft(L.get('+ momentum (the full set)',{}).get('t'))}. Since the 20-day run-up is what the flow mostly measures, the coefficient is best read as "among stocks that ran up the same amount, the ones institutions bought more of do slightly worse" -- a residual after momentum, not a forecast. Read alongside Step 1, the mean of the return is not where positioning tells you anything.

| specification | coefficient (bp/SD) | t |
| --- | --- | --- |
{tbl}
"""))
fig, ax = plt.subplots(figsize=(8, 3))
ks = [k for k in order if k in L]
vals = [L[k]["bp"] for k in ks]; ses = [se_of(L[k]) for k in ks]
y = np.arange(len(ks))
ax.barh(y, vals, height=0.5, color=[BLUE if "momentum" in k else "#86b6ef" for k in ks],
        xerr=[2 * s for s in ses], error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3})
ax.axvline(0, color=AXIS, lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(ks); ax.invert_yaxis()
ax.set_xlabel("coefficient on institutional flow, bp per SD (bars: +/-2 SE)")
ax.set_title("The coefficient reaches significance only once momentum is controlled")
ax.grid(axis="y", visible=False)
cells.append(code_img("Step 3 - control ladder", fig_to_b64(fig)))

# ============================================================== PART B
cells.append(md("# Part B - Where positioning does inform the baseline: the tail"))

# --- B1 baseline table + heatmap
T = bp.get("table", {})
def cellv(s, c, k): return T.get(f"s{s}c{c}", {}).get(k)
lab = ["quiet", "mid", "crowded"]
tbl = "\n".join(
    f"| {s+1} | {lab[c]} | {T.get(f's{s}c{c}',{}).get('n',0):,} | "
    f"{f1(cellv(s,c,'mean_bp'))} | {f1(cellv(s,c,'median_bp'))} | "
    f"{f1(cellv(s,c,'p10_bp'))} | {cellv(s,c,'hit') or 0:.1%} | {cellv(s,c,'big_loss') or 0:.1%} |"
    for s in (4, 0) for c in range(3))
cells.append(md(f"""## Step 4 - The baseline expectation, as a distribution

`data: market + investor_flow + earnings filings  (provisional, {bp.get('events',0):,} events)`

**Question: what should be expected after a surprise of a given strength, and does prior positioning change that?**

A baseline expectation is a distribution, not a mean. Each announcement is placed by surprise quintile (the announcement-day reaction) and by crowding tercile (institutional buying over the prior 20 days), and the distribution of the following 60-day return is tabulated. The mean and the hit rate barely move with crowding. The tenth percentile does: among the strongest surprises it runs from {f1(cellv(4,0,'p10_bp'))} bp for uncrowded names to {f1(cellv(4,2,'p10_bp'))} bp for crowded ones. Crowding changes what a good print earns when it goes wrong, not what it earns on average. Full table for the two extreme surprise quintiles:

| surprise quintile | crowding | n | mean (bp) | median (bp) | p10 (bp) | hit rate | loss > 10% |
| --- | --- | --- | --- | --- | --- | --- | --- |
{tbl}
"""))
M = np.array([[cellv(s, c, "p10_bp") or np.nan for c in range(3)] for s in range(5)])
fig, ax = plt.subplots(figsize=(6.4, 4.6))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ[::-1])
im = ax.imshow(M, cmap=cmap, aspect="auto", origin="lower")
for s in range(5):
    for c in range(3):
        v = M[s, c]
        if np.isfinite(v):
            lum = (v - np.nanmin(M)) / (np.nanmax(M) - np.nanmin(M) + 1e-9)
            ax.text(c, s, f"{v:+,.0f}", ha="center", va="center", fontsize=9.5,
                    color=SURF if lum < 0.45 else INK)
ax.set_xticks(range(3)); ax.set_xticklabels(lab)
ax.set_yticks(range(5)); ax.set_yticklabels([f"Q{s+1}" for s in range(5)])
ax.set_xlabel("pre-announcement institutional crowding")
ax.set_ylabel("surprise quintile  (Q5 strongest)")
ax.set_title("10th percentile of the 60-day return, bp  (darker = worse tail)")
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03); cb.outline.set_visible(False)
cells.append(code_img("Step 4 - baseline expectation heatmap (p10)", fig_to_b64(fig)))

# --- B2 distribution: quiet vs crowded within the strongest surprises
P = bp.get("p10", {}); H = bp.get("hit rate", {}); MM = bp.get("mean", {})
shape = tp.get("shape", {})
tbl = "\n".join(
    f"| {k} | {f1(shape[k]['bp'])} | {ft(shape[k]['t'])} |"
    for k in ("p5", "p10", "p25", "p50") if shape.get(k))
cells.append(md(f"""## Step 5 - After the strongest surprises: quiet against crowded

`data: market + investor_flow + earnings filings  (provisional, strongest surprise quintile: quiet {len(quiet_v):,}, crowded {len(crowd_v):,})`

**Question: where in the distribution does crowding bite?**

Declared before the run: crowded names should have a worse tenth percentile and a lower hit rate. The tenth percentile is {P.get('diff',0)*1e4:+,.0f} bp worse at t {P.get('t',0):+.2f}, outside a 200-draw placebo band of [{P.get('placebo_lo',0)*1e4:+,.0f}, {P.get('placebo_hi',0)*1e4:+,.0f}]. The hit rate is not different ({H.get('diff',0)*100:+.1f} pp, t {H.get('t',0):+.2f}) and neither is the mean ({MM.get('diff',0)*1e4:+,.0f} bp, t {MM.get('t',0):+.2f}). The effect is concentrated in the extreme tail and fades toward the centre. The same statement holds on the four-times-larger all-filings sample ({ba.get('p10',{}).get('diff',0)*1e4:+,.0f} bp, t {ba.get('p10',{}).get('t',0):+.2f}, also outside its band).

| percentile of the 60-day return | crowded minus quiet (bp) | t |
| --- | --- | --- |
{tbl}
"""))
def kde(v, xs, bw):
    return np.mean(np.exp(-0.5 * ((xs[:, None] - v[None, :]) / bw) ** 2), axis=1) / (bw * np.sqrt(2 * np.pi))
xs = np.linspace(-45, 45, 400)
bw = 2.2
kq, kc = kde(quiet_v, xs, bw), kde(crowd_v, xs, bw)
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.plot(xs, kq, color=BLUE, lw=2, label=f"uncrowded  (n {len(quiet_v):,})")
ax.plot(xs, kc, color=ORANGE, lw=2, label=f"crowded  (n {len(crowd_v):,})")
p10q, p10c = np.percentile(quiet_v, 10), np.percentile(crowd_v, 10)
ax.fill_between(xs, 0, kc, where=xs <= p10c, color=ORANGE, alpha=0.18, lw=0)
ax.fill_between(xs, 0, kq, where=xs <= p10q, color=BLUE, alpha=0.18, lw=0)
ax.axvline(p10q, color=BLUE, lw=1, ls="-", alpha=0.7)
ax.axvline(p10c, color=ORANGE, lw=1, ls="-", alpha=0.7)
ax.annotate(f"p10 uncrowded {p10q:+.1f}%", (p10q, kq.max() * 0.62), xytext=(-8, 0),
            textcoords="offset points", ha="right", color=INK2, fontsize=8.5)
ax.annotate(f"p10 crowded {p10c:+.1f}%", (p10c, kc.max() * 0.42), xytext=(-8, 0),
            textcoords="offset points", ha="right", color=INK2, fontsize=8.5)
ax.set_xlabel("60-day benchmark-adjusted return after a top-quintile surprise, %")
ax.set_ylabel("density")
ax.set_title("Same centre, worse left tail: crowding shows up where the print goes wrong")
ax.legend(frameon=False, loc="upper right")
ax.set_yticks([])
ax.text(0.02, 0.97,
        f"10th percentile, crowded minus uncrowded: {P.get('diff',0)*1e4:+,.0f} bp   t {P.get('t',0):+.2f}\n"
        f"outside a 200-draw placebo band [{P.get('placebo_lo',0)*1e4:+,.0f}, {P.get('placebo_hi',0)*1e4:+,.0f}]\n"
        f"median and hit rate: no difference",
        transform=ax.transAxes, va="top", fontsize=8.5, color=INK2, linespacing=1.5)
cells.append(code_img("Step 5 - distribution after strong surprises", fig_to_b64(fig)))

# --- B3 robustness
def rob_rows(src, label):
    out = []
    for grp in ("by_vol", "by_half"):
        for k, r in (src.get(grp) or {}).items():
            if r:
                out.append((f"{label}: {k}", r["bp"], se_of(r), r["t"], r["seasons"]))
    return out
rows = rob_rows(tp, "provisional") + rob_rows(ta, "all filings")
tbl = "\n".join(f"| {n} | {v:+,.1f} | {t:+.2f} | {s} |" for n, v, _, t, s in rows)
cells.append(md(f"""## Step 6 - Does the tail gap survive the obvious attacks?

`data: results of Step 5, split by volatility tercile and by half of the sample`

**Question: is it a volatility artefact, and does it hold over time?**

Two checks, signs declared in advance: the gap should be negative inside every volatility tercile and in both halves. It is negative in both halves in both samples, though in provisional filings it is concentrated in the second half; the larger all-filings sample is stable across both. It is not negative in every volatility tercile -- in provisional filings it is close to zero in low-volatility names and {f1((tp.get('by_vol') or {}).get('high vol',{}).get('bp'))} bp in high-volatility names (t {ft((tp.get('by_vol') or {}).get('high vol',{}).get('t'))}). That is a scope condition rather than a confound: the comparison inside the high-volatility tercile holds volatility fixed and the gap is still there. The finding belongs to volatile names.

| subsample | crowded minus quiet, p10 (bp) | t | seasons |
| --- | --- | --- | --- |
{tbl}
"""))
fig, ax = plt.subplots(figsize=(9, 4.4))
y = np.arange(len(rows))
for i, (n, v, se, t, s) in enumerate(rows):
    col = BLUE if n.startswith("provisional") else AQUA
    ax.errorbar(v, i, xerr=2 * se if np.isfinite(se) else 0, fmt="o", color=col,
                ms=7, mec=SURF, mew=1.5, ecolor=col, elinewidth=1.2, capsize=0)
ax.axvline(0, color=AXIS, lw=0.8)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5); ax.invert_yaxis()
ax.set_xlabel("crowded minus uncrowded 10th percentile, bp  (bars: +/-2 SE)")
ax.set_title("Negative in both halves; concentrated in volatile names")
ax.legend([plt.Line2D([], [], marker="o", color=BLUE, lw=0, ms=7),
           plt.Line2D([], [], marker="o", color=AQUA, lw=0, ms=7)],
          ["provisional filings", "all filings"], frameon=False, fontsize=8.5,
          loc="lower right")
ax.grid(axis="y", visible=False)
cells.append(code_img("Step 6 - robustness of the tail gap", fig_to_b64(fig)))

# --- B4 mechanism refuted
mp, ma = mech.get("provisional", {}), mech.get("all", {})
def mrow(src, grp, k): r = (src.get(grp) or {}).get(k); return (r["bp"], r["t"], r["seasons"]) if r else (None, None, None)
tbl = []
for k in ("low turnover", "mid turnover", "high turnover"):
    a, b = mrow(mp, "by_turnover", k), mrow(ma, "by_turnover", k)
    tbl.append(f"| {k} | {f1(a[0])} ({ft(a[1])}) | {f1(b[0])} ({ft(b[1])}) |")
for k in ("short selling banned", "short selling allowed"):
    a, b = mrow(mp, "by_ban", k), mrow(ma, "by_ban", k)
    tbl.append(f"| {k} | {f1(a[0])} ({ft(a[1])}) | {f1(b[0])} ({ft(b[1])}) |")
cells.append(md(f"""## Step 7 - A mechanism, proposed and refuted

`data: results of Step 5, split by turnover tercile and by short-selling regime`

**Question: is the worse tail an exit-friction effect?**

The natural story is that a crowded print that disappoints has everyone leaving through the same narrow door. It makes two predictions: the gap should be worst where turnover is low, and worse during the short-selling bans, when the natural counterparty to an unwind is absent. Neither holds across both samples. In provisional filings the gap is largest where turnover is *highest*, the opposite of a narrow-door story, and the two samples disagree on the sign of the ban effect. The tail pattern stands; its mechanism is open. No replacement story is offered here, because one built after seeing this table would be exactly the kind of result this record was written to avoid.

| split | provisional: p10 gap (t) | all filings: p10 gap (t) |
| --- | --- | --- |
{chr(10).join(tbl)}
"""))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=False)
for ax, (src, title) in zip(axes, ((mp, "provisional"), (ma, "all filings"))):
    ks = ["low turnover", "mid turnover", "high turnover"]
    vals = [(src.get("by_turnover") or {}).get(k, {}) for k in ks]
    v = [x.get("bp", np.nan) if x else np.nan for x in vals]
    se = [se_of(x) if x else np.nan for x in vals]
    ax.bar(range(3), v, width=0.5, color=BLUE,
           yerr=[2 * s if np.isfinite(s) else 0 for s in se],
           error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3})
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xticks(range(3)); ax.set_xticklabels(["low", "mid", "high"])
    ax.set_title(title); ax.set_xlabel("turnover tercile")
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("crowded minus uncrowded p10, bp")
fig.suptitle("A friction story predicts the worst gap at low turnover; the samples do not agree", y=1.02, fontsize=11, color=INK)
cells.append(code_img("Step 7 - mechanism test", fig_to_b64(fig)))

# ============================================================== PART C
cells.append(md("# Part C - What was corrected along the way"))

sw = fin.get("min_n_sweep", [])
tbl = "\n".join(f"| {r['min_n']} | {f1(r['bp'])} | {ft(r['t'])} | {r['days']:,} |" for r in sw)
cells.append(md(f"""## Step 8 - Why an earlier headline read -117.7

`data: market + investor_flow + earnings filings  (provisional, 2011-2026)`

**Question: how much of a coefficient can one default setting create?**

An earlier version of this research reported -117.7 bp/SD on the 2018-2026 window. It came from weighting each announcement day equally after discarding days carrying fewer than 25 filings; the same setting on the full 2011-2026 window below reads {next((r['bp'] for r in sw if r['min_n'] == 25), float('nan')):+.1f}. Korean earnings arrive in bursts, with day counts from one to several hundred, and the two choices compound. Sweeping the threshold moves the coefficient across the table below. Weighting each position equally and clustering on the reporting season -- the estimator used everywhere else in this record -- gives the figures in Part A. The retraction is recorded rather than removed.

| minimum filings per day | day-weighted coefficient (bp/SD) | t | days used |
| --- | --- | --- | --- |
{tbl}
"""))
fig, ax = plt.subplots(figsize=(8, 3.4))
xs_ = [r["min_n"] for r in sw]; vs = [r["bp"] for r in sw]
ax.plot(xs_, vs, color=BLUE, lw=2, marker="o", ms=7, mec=SURF, mew=1.5)
for r in sw:
    ax.annotate(f"t {r['t']:+.2f}", (r["min_n"], r["bp"]), xytext=(0, -14),
                textcoords="offset points", ha="center", fontsize=8, color=INK2)
ax.axhline(0, color=AXIS, lw=0.8)
ax.axhline(full.get("bp", 0), color=ORANGE, lw=1.2)
ax.annotate(f"event-weighted, season-clustered  {full.get('bp',0):+.1f}",
            (xs_[-1], full.get("bp", 0)), xytext=(0, 6), textcoords="offset points",
            ha="right", fontsize=8.5, color=INK2)
ax.set_xlabel("minimum announcements per day (day-weighted estimator)")
ax.set_ylabel("coefficient, bp per SD")
ax.set_title("One threshold moves the coefficient from near zero to below -110")
cells.append(code_img("Step 8 - the min_n sweep", fig_to_b64(fig)))

# --- C2 regime
swp = reg.get("sweep", {}); by = reg.get("by_year", {})
ch = reg.get("break_2018", {}).get("change", {})
tbl = "\n".join(f"| {y} | {f1(v['bp'])} | {ft(v['t'])} |" for y, v in sorted(swp.items(), key=lambda kv: int(kv[0])))
cells.append(md(f"""## Step 9 - Doubling the sample, and looking for a break

`data: market + investor_flow + earnings filings, extended from 2018 back to 2011`

**Question: does the effect hold in years never examined while the hypothesis was formed?**

The event set was extended from 2018 back to 2011, raising the reporting seasons from {orig.get('seasons',0)} to {full.get('seasons',0)}. The rule was fixed beforehand: use the maximum window the data allows and report both. The added 2011-2017 block reads {f1(added.get('bp'))} bp/SD (t {ft(added.get('t'))}) against {f1(orig.get('bp'))} (t {ft(orig.get('t'))}) for the original window; the change is {f1(ch.get('bp'))} bp at t {ft(ch.get('t'))}, not distinguishable from noise, and sweeping the break year finds no year that stands out from its neighbours. The full window is the honest estimate, and year-to-year stability cannot be claimed either way.

| break year | change in coefficient after the break (bp) | t |
| --- | --- | --- |
{tbl}
"""))
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), gridspec_kw={"width_ratios": [1, 1.4]})
ys = sorted(int(k) for k in swp)
axes[0].plot(ys, [swp[str(y)]["bp"] for y in ys], color=BLUE, lw=2, marker="o", ms=6, mec=SURF, mew=1.5)
axes[0].axhline(0, color=AXIS, lw=0.8)
axes[0].set_title("change after a break at each year"); axes[0].set_xlabel("break year"); axes[0].set_ylabel("bp per SD")
yy = sorted(int(k) for k in by)
vals = [by[str(y)]["bp"] for y in yy]
axes[1].bar(yy, vals, width=0.7, color=[BLUE if v < 0 else "#86b6ef" for v in vals])
axes[1].axhline(0, color=AXIS, lw=0.8)
axes[1].set_title("coefficient by year (each on four seasons, so each is noisy)")
axes[1].set_xlabel("year"); axes[1].grid(axis="x", visible=False)
cells.append(code_img("Step 9 - break-year sweep and year-by-year coefficient", fig_to_b64(fig)))

# --- C3 rejected
LV = lvf.get("ladder", {})
TH = thr
dt = dec.get("results", {})
pen = dt.get("연기금", {})
rej = [
    ("foreign flow predicts post-announcement returns", f"{f1(foreign.get('bp'))} bp/SD, t {ft(foreign.get('t'))}"),
    ("surprise x flow interaction (positioning conditions the drift)", f"{f1((ant.get('tests') or {}).get('T2 surprise x flow',{}).get('bp'))} bp, t {ft((ant.get('tests') or {}).get('T2 surprise x flow',{}).get('t'))}"),
    ("surprise x prior run-up interaction", f"{f1((ant.get('tests') or {}).get('T1 surprise x run-up',{}).get('bp'))} bp, t {ft((ant.get('tests') or {}).get('T1 surprise x run-up',{}).get('t'))}"),
    ("the effect survives in flow orthogonalised to momentum", f"top-minus-bottom after, {f1((cond.get('spread_resid') or {}).get('post',{}).get('bp'))} bp, t {ft((cond.get('spread_resid') or {}).get('post',{}).get('t'))}"),
    ("ownership level carries what flow does not", f"f_level {f1(LV.get('f_level',{}).get('bp'))} bp/SD, t {ft(LV.get('f_level',{}).get('t'))}; ladder ordering rho {lvf.get('ordering_rho',0):+.2f}"),
    ("index rebalancing explains the effect", f"size interaction {f1(TH.get('size',{}).get('interaction',{}).get('bp'))} bp, t {ft(TH.get('size',{}).get('interaction',{}).get('t'))} -- wrong sign"),
    ("the effect is amplified in market-wide buying waves", f"interaction {f1(TH.get('market wave',{}).get('interaction',{}).get('bp'))} bp, t {ft(TH.get('market wave',{}).get('interaction',{}).get('t'))} on the full sample"),
    ("pension funds are a separable carrier of the aggregate", f"alone {f1(pen.get('coef_bp'))} bp/SD (t {ft(pen.get('t'))}); entered with the aggregate, neither survives"),
    ("the long side (buying uncrowded surprises) earns a premium", "inside a 200-draw placebo band in both samples"),
    ("exit friction explains the tail gap", "predictions fail across samples (Step 7)"),
]
tbl = "\n".join(f"| {a} | {b} |" for a, b in rej)
cells.append(md(f"""## Step 10 - Tested and rejected

`data: all of the above`

**Question: what else was tried, and what did it say?**

Every hypothesis below had its sign written down before it was run. All of them failed. They are listed because a record of what did not work is what makes the two or three things that did work believable.

| claim | result |
| --- | --- |
{tbl}
"""))

# ============================================================== PART D
cells.append(md("# Part D - Design decisions and bottom line"))
cells.append(md(f"""## Every design decision, side by side

| decision | alternatives tested | chosen | why |
| --- | --- | --- | --- |
| weighting | one vote per day (min_n 5-40) / one vote per position | one vote per position | day-weighting is a function of an arbitrary threshold (Step 8) |
| inference | day clusters / Newey-West on days / season clusters | reporting season | Korean filings arrive in four bursts a year; the season is the independent unit |
| sample window | 2018-2026 / 2011-2026 | 2011-2026, both reported | maximum available, fixed before the extension; no break year stands out (Step 9) |
| positioning measure | 20d flow / 60d flow / 20d ownership change / ownership level | 20d institutional flow | none of the alternatives carries more; all are records of price (Steps 1, 10) |
| outcome statistic | mean / hit rate / percentiles | tenth percentile | the mean does not move with crowding; the tail does (Steps 4-5) |
| scope | all names / by volatility | volatile names | the gap is near zero in low-volatility names (Step 6) |
| multiplicity | none / Bonferroni / permutation | permutation family bars and 200-draw placebo bands | the search is explicit and the bar is empirical |
| mechanism | exit friction | none claimed | both predictions fail across samples (Step 7) |
"""))
K = book.get("gross", {}); C = book.get("contribution", {})
A_ = ant.get("anticipation_share", {})
pead_lo = (A_.get("0") or A_.get(0) or {}).get("after_bp", float("nan"))
pead_hi = (A_.get("4") or A_.get(4) or {}).get("after_bp", float("nan"))
cells.append(md(f"""## Bottom line

- **The daily flow disclosure is a record, not a forecast.** Flow quintiles separate by {sp.get('pre',{}).get('bp',0):+,.0f} bp before an announcement (t {sp.get('pre',{}).get('t',0):+.1f}) and by {sp.get('post',{}).get('bp',0):+,.0f} bp after it (t {sp.get('post',{}).get('t',0):+.2f}). This is why the foreign-ownership series carries nothing ({f1(foreign.get('bp'))} bp/SD, t {ft(foreign.get('t'))}), and why every forward-looking test built on flow failed. Positioning flow belongs in a model as a state variable, not as a predictive feature.
- **Positioning does inform the baseline -- in the tail.** After the strongest surprises, names institutions were already crowded into have a tenth percentile {bp.get('p10',{}).get('diff',0)*1e4:+,.0f} bp worse (t {bp.get('p10',{}).get('t',0):+.2f}, outside a 200-draw placebo band), with no difference in median or hit rate. The effect is concentrated in volatile names and holds in both halves of the larger sample. Its mechanism is open: an exit-friction account was tested and refuted.
- **As a filter on an existing surprise book**, excluding crowded names adds about {C.get('per_position_bp',0):+.0f} bp per position ({C.get('ann_pct',0):+.1f}% a year) at no extra turnover, t {C.get('t',0):+.2f} -- directional, not established. The value of the finding is in sizing the downside, not in the mean.
- **Earnings drift itself is present and monotone** across surprise quintiles ({pead_lo:+,.0f} bp to {pead_hi:+,.0f} bp over 60 days, weakest to strongest surprise); positioning does not add to its mean.
- **Limits stated plainly:** the sample is a single market with no holdout; the tail finding rests on one pre-declared test plus its robustness checks; the coefficient on the mean depends on the momentum controls; short-interest data, the natural level-type complement, is unusable in Korea for half the sample because of the 2020-21 and 2023-25 bans.
- **Next, in order:** the same tail statistic on a market whose disclosure is a level rather than a flow -- participant holdings and short balances -- where the record above says the information, if any, should live.
"""))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
n_img = sum(1 for c in cells if c["cell_type"] == "code")
print(f"wrote {OUT}   cells {len(cells)}   figures {n_img}   "
      f"{OUT.stat().st_size / 1e6:.1f} MB")
