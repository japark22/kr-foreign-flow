# Korean institutional crowding into earnings filings

Frozen 2026-09-17. Implement unchanged elsewhere; a
deviation forced by data availability is reported beside the result.

## Universe
Liquid listed equities where institutions traded the name on at least 10
of the 20 trading days before the filing. The column that carries this count is
`inst_days20`, and the filter is `inst_days20 >= 10`; the feature column is
`i_flow20_v2`. A day with no row in the exchange's
investor-type feed is a day with no institutional trade; that reading was
verified against a per-sub-type source, which reproduces the aggregate exactly
and shows the omitted cells are zero in 97.6 to 100 percent of cases by year.

## Feature
Net institutional buying value over the 20 trading days ending the day before
the filing, divided by 20-day average traded value, rank-standardised across
that day's cross-section.

## Controls
surprise, c_mom20, c_mom60, c_size, c_vol, c_turn, c_mom120, c_mom250, c_mom500, each rank-standardised within the day.

## Outcome
60-trading-day abnormal return from the day after the filing, market return
removed. Reported two ways: winsorised 1/99 within the day, and replaced by
its within-day rank.

## Inference
Two-way clustered standard errors, on the reporting season and on the issuer.
Placebo: the feature shuffled within the day, 200 draws, 5-95 band.

## Result on Korea
-30.3 bp per standard deviation of crowding
(two-way t -2.98), and as a rank information coefficient
-0.0152 (two-way t -3.10).
53,176 events over 49 reporting seasons.

Across five estimator variants the two-way t lies between
2.82 and 3.10; every variant sits outside its placebo band and
both halves of the sample agree in sign in every variant. First half
-57.1 bp (t -2.87),
second half -17.1 bp
(t -1.55) -- the same sign throughout, weaker
lately, and that asymmetry is part of the claim rather than a footnote.

Short-selling suspensions (2020-03-16 to 2021-05-02, 2023-11-06 to 2025-03-30)
are excluded. Including them weakens the estimate, so the exclusion is
conservative.

## How to read the size
An information coefficient near 0.015 is small. This is not a
standalone strategy. It is a weak, nearly orthogonal input -- correlation with
20-day momentum about -0.11 and with size about +0.06 -- of the kind that earns
its place inside a composite rather than on its own.

## Sign
More crowding predicts a lower subsequent abnormal return. The disclosed flow
is a record of buying that already happened, not a forecast of buying to come:
the top-minus-bottom flow quintile separates by +985 bp over the 20 days into
the announcement and by +37 bp over the 60 days after it.

## What was tested and did not survive
Foreign flow, at every quantile (largest |t| 1.41). Cross-type disagreement,
five constructions. A no-participation state, which dies under matched
controls. The position of foreign ownership in its own 250-day range, which is
real but confined to the short-selling suspensions and reverses sign between
halves. A conditional tail statistic on the strongest surprises, whose
published magnitude was inflated by comparing percentiles between groups of
unequal size.
