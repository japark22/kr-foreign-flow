# Pre-registration — conditional positioning baseline

Written and committed before any coefficient in this study was computed.
The purpose of the file is to make it impossible to pick the test after
seeing the answer.

## 1. Taken as given, not re-tested

Established in the preceding work and carried in as a constraint:

- Disclosed daily flow is contemporaneous with price, not predictive of it.
  The top-minus-bottom flow quintile separates by +985 bp (t +29.5) over the
  20 trading days into an announcement and +37 bp (t +0.75) over the 60 after.
- Foreign net buying carries no forward information: +9.8 bp/SD, t +0.59,
  61 reporting seasons, 20,036 events.
- Domestic institutional net buying carries -42.9 bp/SD (t -2.51), and
  -33.5 bp/SD (t -2.15) after every threat filter.

## 2. Hypothesis

If flow is a record rather than a forecast, its usable content is not a
prediction of the mean but a description of the position the market carries
into the event.

PRIMARY: conditional on an earnings announcement, pre-event positioning
shifts the LOWER quantiles of the 60-day abnormal return while leaving the
median and the mean unchanged.

A result in which the mean moves as well is NOT a confirmation. It is a
different finding — an ordinary return predictor — and is to be reported as
such, under its own name.

## 3. Features

No feature may use information dated later than D-1. Every feature is defined
so that any market publishing daily net buying by investor type can implement
it without reinterpretation.

AP_k   Abnormal positioning, investor type k.
       Within each (D, kind) cross-section, regress rank-standardised
       k_flow20 on [c_mom20, c_mom60, |c_mom20|, c_size, c_vol, c_turn];
       AP_k is the rank-standardised residual. |c_mom20| is included to
       absorb the U-shaped absolute-flow artefact identified earlier as a
       volatility proxy. The R-squared of this regression is itself a
       reported quantity: it measures how much of disclosed positioning is
       mechanically explained by contemporaneous price action, and is the
       quantitative answer to why foreign flow carries nothing.

DIV    Cross-type disagreement. DIV = z(i_flow20) - z(f_flow20), z taken
       within (D, kind). Both legs contain the same price-impact term, so
       the difference removes it to first order. Sub-type variants are
       formed the same way from the x_* columns.

PERS   Shape of accumulation. Over [D-20, D-1], the share of days with
       positive net buying, centred: (days_positive - 10) / 10.

BLOCK  Concentration of accumulation.
       max(|daily net|) / sum(|daily net|) over the same window.

IDIO   Own-history standardisation.
       (flow20 - mean_250(flow20)) / sd_250(flow20), per ticker.

SHORT  Short-balance change over [D-20, D-1] divided by shares listed, and
       the level at D-1. Korean short-selling was suspended 2020-03-16 to
       2021-05-02 and 2023-11-06 to 2025-03-30; those windows are excluded
       and the exclusion is reported on the face of any result using SHORT.
       The complete test of this leg belongs in a market whose short data
       is not interrupted.

COMPOSITE  Information-coefficient weighted combination of the qualifying
       features, with weights fixed in-sample and held fixed out-of-sample.

## 4. Disqualification rule, applied before estimation

Any feature with |corr(feature, c_mom20)| >= 0.20 is dropped as momentum in
costume. Dropped features are named in the output; the rule is not relaxed
after seeing which features it removes.

## 5. Outcome and estimator

Outcome: abn60, winsorised 1/99 within (D, kind).

Estimator: quantile regression at tau in {0.05, 0.10, 0.25, 0.50, 0.75, 0.90}.
The mean is estimated alongside it for the signature test in section 2, using
the one-vote-per-position, season-clustered specification already adopted as
this project's anchor.

Rationale: the effect under test is a change in the shape of the conditional
distribution. An estimator of the mean is the wrong instrument for it, and a
bucket table is a coarse approximation of it.

## 6. Inference

- Standard errors: block bootstrap over reporting seasons, 2000 replications.
  Korean filings arrive in roughly four bursts a year, so the season is the
  independent unit, not the day and not the event.
- Placebo: 200 draws, shuffling the feature within (D, kind), recording the
  full distribution of the contrast at each tau. A single draw is not a
  placebo; it has error as wide as the effect.
- Family-wise control: across the feature-by-tau grid, the maximum |t| under
  permutation is recorded; the primary statistic must exceed the 95th
  percentile of that maximum distribution.

## 7. Decision rules

PASS on the primary hypothesis requires ALL of:
  a. |t| >= 2 at tau = 0.10 for COMPOSITE within the top surprise quintile,
     under season-clustered bootstrap;
  b. the point estimate outside the 5th-95th percentile placebo band;
  c. clearing the family-wise threshold in section 6;
  d. |t| < 2 at tau = 0.50 and |t| < 2 on the mean.

Failing (d) while passing (a)-(c) is reported as an ordinary return
predictor, not as a tail result.

Failing (b) is a negative baseline and is reported as a negative baseline.

## 8. Sample

Full available window for each feature, 2011-03 to 2026-06, no start year
chosen after inspecting results. Coverage is reported per feature. Both
filing kinds are reported separately and pooled; pooled is primary. Where a
feature's coverage begins later than the panel, the restricted window is
stated next to every number derived from it.

## 9. Cross-market clause

Taiwan and Hong Kong implement these definitions unchanged. Window lengths,
quantile grid, thresholds and universe rules are not re-tuned per market.
A deviation forced by data availability is reported as a deviation, in the
same place as the result, and never absorbed silently.
