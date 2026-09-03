# Korea pre-announcement positioning -- signal specification

Generated 2026-09-03 09:32 UTC from the result files. One page. The two parquet files beside this document carry the signal (daily) and the verification set (per announcement).

## What the signal is

`inst_flow20`: net buying by domestic institutions (exchange category 기관합계, institutions in total) summed over the trailing 20 trading days, divided by value traded over the same 20 days, then standardised against the stock's own trailing 250-day history of that ratio (mean and standard deviation computed on days t-250..t-1, minimum 120 observations). Positive means institutions have been net buyers relative to the stock's own norm. `foreign_flow20` is the identical construction on the foreign-investor category (외국인, foreign investors) and is supplied as the comparison series that carries no forward information.

Construction code: `flow_intensity()` and `zwin()` in `20_event_panel.py`. Inputs are the exchange's daily investor-category net trading values and daily value traded. No look-ahead: every quantity at date t uses data through t.

## Files

| file | grain | rows | columns |
| --- | --- | --- | --- |
| `kr_positioning_daily.parquet` | ticker x trading day, 2011-01 onward | one row per ticker-day with at least one value | trade_date, ticker, inst_flow20, foreign_flow20 |
| `kr_positioning_events.parquet` | one row per earnings filing | 122,643 (23,351 provisional) | ticker, announcement_date, kind, surprise, inst_flow20, foreign_flow20, surprise_quintile, crowding_tercile, abn5/20/60, controls |

`announcement_date` is the filing date; the signal is read as of that date (it uses the 20 days before). `surprise` is the announcement-day benchmark-adjusted return. `abn60` is the 60-trading-day return net of the equal-weight market starting the day after the filing, winsorised 1/99 within the day. `surprise_quintile` and `crowding_tercile` are ranks within the announcement day (1 = lowest).

## How it was estimated here

one vote per event; regressors rank-standardised within the announcement day; controls c_mom20, c_mom60, c_size, c_vol, c_turn; standard errors clustered on the reporting season (year x quarter). Days with fewer than 5 usable filings dropped. Multiple specifications controlled with permutation family bars; portfolio contrasts checked against 200-draw placebo bands.

## Numbers to reproduce before comparing

| quantity | value |
| --- | --- |
| provisional filings, flow quintile 5 minus 1, cumulative return day -20..-1 | +985 bp, t +29.5 |
| same quintiles, day +1..+60 | +37 bp, t +0.75 |
| coefficient of inst_flow20 on abn60, provisional, full controls | -42.9 bp/SD, t -2.51, 17,841 events, 61 seasons |
| same for foreign_flow20 | +9.8 bp/SD, t +0.59 |
| top surprise quintile: 10th percentile of abn60, crowding tercile 3 minus 1, provisional | -236 bp, t -3.24; placebo 5-95% [-213, +27] |
| same, all filings | -117 bp, t -2.02; placebo [-54, +97] |

If the first two rows do not reproduce, the join or the window is off. If they do and the rest does not, the estimator differs, and the difference is the thing to discuss.

## What to test, and what not to expect

The signal is a record of the recent price move, not a forecast of it (row 1 against row 2). Tests on the conditional mean of post-announcement return will show little; they did here. The information, such as it is, sits in the lower tail of the conditional distribution: within the strongest surprises, crowded names have the same mean and hit rate as uncrowded names and a worse 10th percentile, concentrated in volatile names. Recommended comparison statistic: the conditional 5th and 10th percentiles of the post-announcement return by crowding tercile within surprise quintile, alongside the conditional mean.

## Resolution caveat for a comparison with quarterly holdings data

This is a daily flow. A quarterly holdings level (13F-type) is a different object: a stock of positions observed with a lag, not a 20-day change observed the same day. An identical methodology across the two requires a choice. Either coarsen this signal to quarter-end (for example, the cumulative institutional net buying over the quarter, scaled by quarterly value traded, read at quarter end) for a like-for-like test, or run each series in its native resolution and treat agreement in the tail statistic as the comparison. Running both is recommended; the coarsened version can be built from the daily file directly.

## Known limits

Single market, no holdout. The mean coefficient depends on the momentum controls (-25.1 at t -1.67 without them). The tail result rests on one pre-declared test plus robustness checks by half and by volatility tercile. Short-selling was banned 2020-03-16..2021-05-02 and 2023-11-06..2025-03-30; any short-interest overlay for Korea is missing for those windows.
