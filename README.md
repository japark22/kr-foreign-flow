# Korean Foreign-Ownership Flow

Research into whether the persistence of daily foreign-investor flow in Korean
equities predicts future flow and future returns, and whether any of it is
tradable.

This repository holds the collection pipeline, the signal construction, the
validation code, and the results.

## Research at a glance

**Question.** Korea publishes every stock's foreign ownership daily, a disclosure most markets do not have. Does that flow persist, and does it, or the institutional flow published beside it, predict returns well enough to trade after Korean transaction tax?

**Phases and what each found.** Every number comes from a committed result file or a generated page.

| Phase | Question | Answer |
|---|---|---|
| 1. Flow persistence | Does foreign flow continue? Does it predict returns? | Yes, it persists: lag-1 autocorrelation is +0.088 and 21% of it remains at lag 20, the shape of large orders worked over days. Returns barely move: 1-day IC +0.0039 (t +3.90), reversing by 20 days (t −4.37). Breakeven one-way cost is **2.4 bp** against 20 bp of sales tax, so the effect is **not tradable** (§5). |
| 2. Positioning before earnings | Does institutional buying ahead of an earnings filing predict the reaction? | On provisional filings, institutional crowding in the 20 days before predicts **lower** 60-day size-adjusted returns: **−42.9 bp per SD, t −2.51**, over 17,841 events and 61 reporting seasons ([`results/published.json`](results/published.json), [event page](https://japark22.github.io/kr-foreign-flow/event.html)). An earlier −117.7 bp figure was retracted and the weighting choice behind it is documented. |
| 3–4. Robustness and recovered sample | Does it survive threats, multiplicity and a cleaner sample? | Frozen specification: **−30.3 bp per SD (two-way t −2.98)**, rank IC −0.0152 (t −3.10), over 53,176 events and 49 seasons. It holds across five estimator variants (t 2.82–3.10), each outside its placebo band, and is weaker in the second half (−17.1 bp, t −1.55) ([`handoff/KR_CROWDING_SPEC.md`](handoff/KR_CROWDING_SPEC.md)). |
| 5. Flow factor | Splitting flow into a market-wide allocation wave and a stock-specific decision | In progress ([`research/5_flow_factor`](research/5_flow_factor)). |

**How it was kept honest**

- The hypotheses were [pre-registered](PRE_REGISTRATION.md) before the data was examined.
- Multiplicity is controlled with permutation family-wise bars and placebo bands.
- Standard errors are clustered on the reporting season, and on the issuer where it matters.
- A pykrx float16 rounding trap was found and avoided (§3).
- Results feed generated pages, with no hand-copied numbers, and [`publish/80_audit.py`](publish/80_audit.py) re-derives every published figure before anything is shared.

**Live.** A [monitor page](https://japark22.github.io/kr-foreign-flow/monitor.html) of recent foreign positioning is refreshed every trading day by a [GitHub Actions job](.github/workflows/update.yml).

---

## 1. The research question

Most equity markets do not disclose foreign ownership per stock on a daily
basis. Korea does, and so does Taiwan. That disclosure is the whole reason this
project exists: it is a dataset a competitor cannot trivially replicate.

The hypothesis, stated plainly:

> Foreign-investor flows are persistent. Orders are large relative to daily
> liquidity, and institutions take time to digest new information, so buying or
> selling that starts today tends to continue over the following days.

If that is true, today's foreign flow carries information about the next few
days of flow, and therefore about price.

The parallel is 13F ownership research — inferring institutional intent from
disclosed holdings — except here the disclosure is **daily** rather than
quarterly, and covers a distinct investor class.

### The questions this has to answer

| # | Question | Status |
|---|---|---|
| 1 | Does foreign flow autocorrelate? | **Answered — yes** |
| 2 | Does it predict returns, and at what horizon? | **Marginal at 1 day, reverses by 20** |
| 3 | Does it survive removing mechanical, uninformed flow? | **Cannot answer — the calendar is wrong** |
| 4 | Is it distinct from momentum? | **Yes, correlation under 0.04** |
| 5 | Does it survive trading costs? | **No, in this construction** |
| 6 | **Is the persistence specific to foreign investors?** | Pending investor-flow data |

A feature is only usable if these come out together. Persistence alone proves
nothing about profit.

Question 6 was added after seeing the first results, and it is the one that
decides whether this project has an edge at all — see §6.

---

## 2. The signal

Per stock, per day:

```
normalised foreign flow
  = (Δ foreign shares held − change caused by corporate actions)
    / shares outstanding
```

Then rank-normalised across the cross-section each day. Ranking makes the
measure comparable across days and robust to the scale gap between a large cap
and a micro cap.

Supporting series: the ownership level itself, the change in ownership
percentage, and the foreign-limit exhaustion rate for capped names.

Every value is tagged with the time it was fetched and lagged one day, so the
backtest only ever sees what was knowable at the time.

---

## 3. Data

**Source:** KRX [12023] 외국인보유량(개별종목) — 전종목, retrieved through
`pykrx` against the KRX Data Marketplace.

**Coverage:** 2010-01-04 to present, all listed KOSPI and KOSDAQ names
including delisted ones, ~4,100 trading days, ~2,760 tickers currently.

**Supporting data:**

- Prices, volume, traded value, market cap — KRX, for returns and liquidity
- Fiscal year-end month per company — OpenDART, for ex-dividend windows
- MSCI / FTSE review dates — derived from published rules
- Net buying by investor type (외국인, 기관합계) — KRX, as a control group and
  an independent cross-check on the ownership series
- Short-sale balance and volume — KRX, to separate genuine buying from short
  covering

### One thing that would have quietly broken the study

`pykrx` casts `지분율` and `한도소진률` to `np.float16`
(`website/krx/market/wrap.py:478`). At the ownership levels we care about,
float16 resolution is:

| ownership level | float16 spacing |
|---|---|
| 5% | 0.0039 pp |
| 13% | 0.0078 pp |
| 55% | 0.0313 pp |
| 99% | 0.0625 pp |

A typical one-day change in foreign ownership is **0.01–0.05 pp**. At large
caps the rounding error is the same size as the signal — it would have erased
the thing we are measuring.

Measured on live data (2026-08-14, 2,763 names): median absolute error
**0.0026 pp**, worst **0.0307 pp**.

`보유수량`, `상장주식수` and `한도수량` arrive as exact `int64`, so the
ownership ratio is recomputed in float64 from those. The as-returned column is
kept under the name `foreign_pct_krx_lossy` and is never used.

---

## 4. Method

1. **Collect** a daily all-market snapshot. One immutable parquet file per
   trading day, written atomically.
2. **Normalise** by shares outstanding; strip changes caused by corporate
   actions so what remains is trading rather than arithmetic.
3. **Denoise** by excluding index-rebalance windows and ex-dividend windows.
4. **Validate**: flow autocorrelation curve, information coefficient by
   horizon, quantile long-short backtest, correlation against momentum.

### Why the measurement code can be trusted

Before running it on real data, the validator was run against synthetic data
containing a **known planted signal** — an AR(1) flow process with φ = 0.60 and
a fixed return sensitivity. It recovered:

| planted | recovered |
|---|---|
| lag-1 = φ = 0.60 | +0.5824 |
| lag-2 = φ² = 0.36 | +0.3498 |
| lag-3 = φ³ = 0.216 | +0.2085 |
| positive IC | +0.0107, t = +3.45 |

So when it reports a number on real data, the number means what it says.

---

## 5. Findings so far

Sample: 2010-01-04 to 2026-08-14, 4,091 trading days, 3,671 tickers,
**9,068,519 stock-day observations**.

### Persistence exists, and decays far more slowly than AR(1)

| lag | corr | | lag | corr |
|---|---|---|---|---|
| 1 | **+0.0883** | | 10 | +0.0386 |
| 2 | **+0.0877** | | 15 | +0.0263 |
| 3 | +0.0743 | | 20 | **+0.0186** |
| 5 | +0.0574 | | | |

The shape matters more than the level.

- A **lag-1 spike that dies at lag-2** would point at settlement or reporting
  mechanics, and we would have rejected the hypothesis.
- Under **AR(1)**, lag-2 would be lag-1² = 0.0077. Observed is 0.0877 — eleven
  times higher.
- What is actually there is **flat at the front, then a long tail**: 21% of the
  lag-1 value still remains at lag 20.

That is closer to power-law decay than exponential, and slowly-decaying
autocorrelation of this shape is the standard signature of **large orders being
split and worked over multiple days** in the market-microstructure literature.

It matches the original reasoning — large orders, slow information digestion —
more precisely than a simple positive autocorrelation would have.

### The obvious objection, tested and rejected

A stock with no foreign trading has Δ = exactly zero. Every such name receives
the same mid-rank, so a name inactive for weeks holds an identical rank for
weeks. That alone would produce a smooth decaying curve — persistent
**inactivity** dressed up as persistent buying. With ~2,700 listed names, many
thinly traded, this is a real risk.

Test: drop every exactly-zero observation and re-measure.

| | |
|---|---|
| observations | 9,068,519 |
| exactly zero | 1,251,656 (13.8%) |
| lag-1, all | +0.0883 |
| lag-1, active only | **+0.0869 (98% retained)** |

The whole curve survives. The persistence is not a ranking artifact.

### Flow predicts flow overwhelmingly. It barely predicts returns.

This contrast is the central finding.

| measurement | t-stat |
|---|---|
| flow → flow, lag 1 | **+64.7** |
| flow → flow, lag 20 | **+21.5** |
| flow → return, 1 day | **+3.90** |
| flow → return, 20 days | **−4.37** |

Tomorrow's flow is predictable beyond any doubt. Next month's flow is still
predictable at t=21. Tomorrow's *return* is barely predictable, and the sign
inverts by twenty days.

The natural reading is that the market already prices the predictable component
of foreign flow. Predictable order flow is predictable to everyone.

### Returns by horizon

| horizon | mean IC | t | long-short (gross) | Sharpe |
|---|---|---|---|---|
| 1 day | +0.00390 | **+3.90** | +6.22% | +0.93 |
| 5 days | −0.00031 | −0.32 | −0.52% | −0.08 |
| 20 days | −0.00392 | **−4.37** | −1.25% | −0.15 |

Two things to note. The 1-day IC clears significance but is a third of the
0.01 magnitude threshold. The t comes from the time series of 3,351 daily
cross-sectional ICs, not from the 7.4 million ticker-days behind them, so
it is not inflated by counting correlated stocks as independent — but a
tiny effect over 3,351 days still reaches t=4. And the sign inverts monotonically by 20 days, which is the signature of
temporary price pressure reverting, not of information.

### Costs settle it for this construction

Korean securities transaction tax from 2026 is 0.20% on every sale (KOSPI
0.05% + 농특세 0.15%; KOSDAQ 0.20%), before spread and impact.

| one-way cost | 1d net | 5d net | 20d net |
|---|---|---|---|
| 10 bp | −19.4% | −5.6% | −2.5% |
| 20 bp | −45.0% | −10.8% | −3.8% |
| 30 bp | −70.6% | −15.9% | −5.1% |

**Breakeven one-way cost for the 1-day effect is 2.4 bp**, in a market where
tax alone is 20 bp. Where the effect exists it cannot be traded; where it can
be traded there is no effect.

### The rebalance calendar is wrong — measured, not suspected

Removing rebalance windows moved the curve by +0.0004 while discarding 1.47
million observations. §3b tested whether the flagged dates carry unusual flow:

```
day vs effective   mean |flow| relative to baseline
      -5                      1.07x
      -1                      1.05x
      +0                      1.03x   <- effective date
      +1                      1.07x
```

Flat, and the effective date itself is *lower* than five days before it. The
rule-derived dates are not the real rebalance dates. **Everything labelled
"denoised" in this repo is therefore meaningless** until authoritative MSCI and
FTSE dates are pinned in `data/rebalance_overrides.csv`.

---

## 6. Known limitations

Stated plainly, because a result is only as good as what it admits.

**The overlapping-window objection was raised and does not hold — measured, not
assumed.** The IC at horizon h is computed on every trading day, so consecutive
20-day forward windows overlap by nineteen days. That normally makes the daily
IC series autocorrelated and `std/sqrt(n)` too small, which would have inflated
the 20-day and 60-day t-statistics by up to sqrt(h).

`09b_se_diagnostic.py` tests it directly. If the overlap drove the daily
variation, the lag-1 autocorrelation of the daily IC series would sit near
(h-1)/h = +0.95 at h = 20 and decay to zero by lag 20. The measured profile is
flat from lag 1: -0.033, +0.014, +0.038, and within noise of zero thereafter.
Newey-West with 19 lags gives t = -4.18 against the naive -4.37 — it estimates
the autocovariance, finds almost none, and barely moves the answer.

The reason is that an IC is a normalised cross-sectional correlation, not a
return. Shifting the window one day leaves the common component largely
cancelled between numerator and denominator, and what remains varies from day
to day without being shared across days.

A non-overlapping standard error (every h-th day) reports t = -0.98 at h = 20,
and that figure should NOT be used: the same script's calibration shows it
rejects a true null 0% of the time when the series is white noise, because it
pairs a full-sample mean with a subsample standard error. It is a fixed
sqrt(h) penalty rather than a correction. All published t-statistics here
stand as reported.

**The noise filter has not been shown to work.** Removing rebalance and
ex-dividend windows discards 18.2% of observations and moves the curve by
+0.0004 — essentially nothing. Two explanations, not yet separated:

- *(a)* the derived dates are wrong, so ordinary days were discarded while real
  rebalance flow stayed in the sample; or
- *(b)* the dates are right, and mechanical flow does not distort a
  cross-sectional **rank** correlation much, since ranks are robust to a few
  enormous trades by construction.

`04_validate.py` §3b separates them: if the calendar is right, aggregate
absolute flow should be visibly elevated on flagged days. Until that is run,
**no claim should be made that the signal survives denoising.**

**Index-rebalance dates are derived from rules, not from an official list.**
MSCI's published schedule covers forward dates only. Dates are generated from
the rules (MSCI: last trading day of Feb/May/Aug/Nov; FTSE GEIS: third Friday
of Mar/Jun/Sep/Dec) and snapped to KRX trading days. Both providers move dates.
Authoritative dates go in `data/rebalance_overrides.csv`.

**Ex-dividend dates are derived, not observed.** OpenDART publishes no
배당기준일 field — `alotMatter` gives amounts and 결산기준일 only. Dates are
inferred from each company's fiscal year end and flagged as a window rather
than a day. Since the 2024 배당절차 개선방안, companies may set 배당기준일 after
the AGM, so from 2024 the year-end concentration is dispersing and this
progressively under-captures the effect.

**Free float is approximated by shares outstanding.** The correct denominator
needs a vendor feed.

**Corporate actions are detected by proxy** — a change in shares outstanding —
rather than by parsing capital-change events.

**Persistence may not be specific to foreign investors.** This is the most
consequential open question, and it was not visible until the first results
came in. The slowly-decaying shape that supports the hypothesis is also the
standard signature of large orders being split over days — which is a property
of institutional execution in general, not of foreign investors in particular.

Two hypotheses fit the observed curve equally well:

- **H1** — foreign flow is persistent, and Korea's daily disclosure is a real
  data edge.
- **H2** — *all* large institutional flow is persistent, because big orders get
  split. Generic microstructure, no edge, and domestic institutional flow is
  disclosed in plenty of markets.

The test is a control group: run the identical measurement on 기관합계
(domestic institutional) net buying. If the two curves match, H2 holds and the
pitch cannot be "foreign flow is special". `04_validate.py` §5 implements this.
The measurement was verified on synthetic data in both directions — it correctly
concludes H1 when foreign φ=0.55 against institutional φ=0.10, and H2 when both
sit near 0.49.

**Return predictability is unknown.** Persistence is not profit. Order
splitting produces autocorrelation on its own; if there is no information
behind it, predictive power is zero. This is the same distinction that made 13F
research hard: holdings persistence is well documented, alpha is the harder
question.

---

## 7. Repository layout

Scripts keep their step numbers, so the order of the research is still readable. They are grouped by role:

```
krxflow/                  the library every script imports
  config.py               paths, credentials, schema, throttle settings
  calendar.py             KRX trading days
  collect.py              login, throttled and retrying fetchers
  storage.py              parquet layout, atomic writes, range reads
  dart.py                 OpenDART client (corp codes, fiscal months, dividends)
  rebalance.py, exdiv.py  index-rebalance and ex-dividend windows
  features.py             panels, universe, corporate-action scrub, the flow signal
  tracker.py, charts.py   forward-return tracker and SVG charts for the monitor
  paths.py                script("20_event_panel.py") -> its path, for scripts that load each other

pipeline/                 data: 00 smoke test, 01 backfill, 02 inspect, 03 DART,
                          05 daily update, 06 freshness, 15 earnings dates, 17 investor detail
research/
  1_flow_persistence/     04 hypothesis test, 07-12 signal sweep, level signals, standard errors
  2_earnings_positioning/ 16-40 event study, baseline regressions, multiplicity, threats
  3_robustness/           43-52 book value, extended sample, regimes, event path, tails
  4_recovered_sample/     55-79 recovered sample, participation, short-sale bans, final lock
  5_flow_factor/          81-83 allocation wave vs stock decision (in progress)
publish/                  13 research page, 14 monitor, 41-42 event page, 53 notebook,
                          54 handoff files, 80 pre-release audit
tools/                    scratch diagnostics
scripts/                  local scheduling wrappers (launchd)
docs/                     the published pages (GitHub Pages)
results/, reports/        generated result files, notebooks and reports
handoff/                  frozen specifications
```

---

## 8. Reproducing

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # makes krxflow importable from every folder

cp .env.example .env      # fill in KRX_ID, KRX_PW, DART_API_KEY
set -a && source .env && set +a

python pipeline/00_smoke_test.py
python pipeline/01_backfill.py --start 2010-01-01 --with-market
python pipeline/03_collect_dart.py
python research/1_flow_persistence/04_validate.py --horizons 1,5,20
```

The backfill is resumable: every trading day is its own file and finished days
are skipped, so an interrupted run picks up where it stopped.

Access requires a KRX Data Marketplace account. Since 2025-12-27 the site
requires a login; registration is free but must be done with an ID and password
rather than a social login, because the client posts credentials to the login
endpoint. An OpenDART key is free and issued on request.

---

## 9. Operational notes

**Scheduled runs.** The daily collection runs on GitHub Actions
([`.github/workflows/update.yml`](.github/workflows/update.yml), weekdays 20:05 KST).
It restores a rolling window of exchange data from the Actions cache, about 820
calendar days, which is what the monitor reads. It fetches the missing days and
rebuilds `docs/monitor.html`, then commits only that page and `results/monitor.json`.
Exchange data is never committed. On an empty cache the job backfills the window
first. Credentials come from repository secrets (`KRX_ID`, `KRX_PW`, `DART_API_KEY`).
The research results need the full 2010+ history, so they are rebuilt locally rather than
on the runner. `scripts/install_schedule.sh` still sets up the earlier launchd
jobs for a local machine. `pipeline/05_daily_update.py` works out what is missing and
fetches only that, so it is safe to fire on a schedule and safe to run by hand.

**Report cadence is weekly by design.** The validation statistics run over
4,000+ trading days, so one additional day shifts them by ~0.02%. Regenerating
the report daily would produce diffs dominated by rounding, which buries real
changes when they happen. The daily job's value is catching a collection
failure early, and the run log does that.

**Request throttling.** 0.7 s between requests plus jitter, exponential
backoff, and a stop after ten consecutive failures rather than hammering the
source. KRX moved to a login-only model in December 2025 citing server load
from bulk collection, so restraint here is not optional. Do not lower
`KRXFLOW_SLEEP`.

**Request timeouts are forced on.** `pykrx` issues its data requests with no
timeout (`website/comm/webio.py` lines 42, 50, 76, 84; only the login calls in
`auth.py` set one). A single stalled connection blocks indefinitely — the
process stays alive, burns no CPU, and never advances, and retry logic never
fires because no exception is raised. This was observed in practice: a
4,000-day backfill stopped dead at day 410. `krxflow.collect` installs a default
timeout (10 s connect, 60 s read) on every request that lacks one.

**Point-in-time integrity.** The raw layer is stored exactly as fetched, one
file per day. If the source revises a published value, that shows up as a
file-level difference rather than silently overwriting history.
`06_check_freshness.py` re-fetches a sample and diffs it against disk to check
whether revisions occur at all.

---

## 10. Data handling

Raw snapshots stay out of version control. Sixteen years is roughly 500 MB, and
it is licensed exchange data. `.gitignore` excludes `data/`, `.env` and
`.venv/`, and the push routine in `05_daily_update.py` independently re-checks
the staged file list and aborts if anything under `data/` or any `.env` appears.
Two independent guards, because a credential leak is not a recoverable mistake.

Credentials live in environment variables on the machine that runs the
pipeline, and are never committed.

---

## 11. Open items

1. Complete the price backfill; answer questions 2, 4 and 5.
2. Collect investor-type net buying and run the 기관합계 control (question 6).
   Until this is done, no claim should be made that the effect is specific to
   foreign investors.
3. Run `04_validate.py` §3b and settle whether the rebalance calendar is real.
4. Pin authoritative MSCI / FTSE dates in `data/rebalance_overrides.csv`.
5. Source per-company 배당기준일 for an exact ex-dividend filter.
6. Source free float for the correct denominator.
7. Define the production universe: market-cap and liquidity floors, KONEX
   in or out, treatment of preferred shares.
8. Decide target holding period, and the neutralisation basis (sector, size).
9. Extend to Taiwan, which discloses comparable data.
