"""Ex-dividend noise flags.

WHY THIS IS DERIVED RATHER THAN FETCHED
---------------------------------------
Ex-dividend dates are a noise source: foreign holders move
positions around them for tax reasons, and that shows up in the ownership
series as flow that is not a view on the company.

There is no OpenDART endpoint that returns 배당기준일 or an ex-dividend date.
`alotMatter` gives dividend *amounts* and `stlm_dt` (결산기준일) only. So the
dates here are derived from the fiscal year end.

THE CONVENTION, AND WHY WE USE A WINDOW
---------------------------------------
Historically almost all Korean listed companies closed their books on
31 December, 배당기준일 was the fiscal year end, and — with T+2 settlement and
a year-end market holiday — the ex-dividend effect landed on the last trading
day or two of December, with the rebound in early January.

Two things make a single exact date the wrong tool:

  1. The exact day depends on where holidays fall in a given year and on the
     settlement cycle in force at the time.
  2. Since the 2024 배당절차 개선방안 (FSC), companies may set 배당기준일 after
     the AGM. Adoption is partial and growing, so from 2024 onward the
     year-end concentration is genuinely breaking up and a December-only flag
     progressively under-captures the effect.

So we flag a *window* rather than a day. Losing five trading days a year out
of ~245 is a cheap price for not mis-timing the exclusion by one day, which
would leave the largest contaminated observation in the sample.

WHAT WOULD MAKE THIS EXACT
--------------------------
Per-company 배당기준일, which needs either a vendor feed (FnGuide) or parsing
현금·현물배당결정 disclosures out of DART's 공시검색. Worth doing before any
result is treated as final, and noted as an open item in the project docs.
"""
from __future__ import annotations

import pandas as pd


def year_end_window(trading_days, before: int = 3, after: int = 2) -> pd.Series:
    """Market-wide flag around each fiscal year end (December closers).

    Costs zero API calls and captures the dominant pre-2024 effect.

    Args:
        before: trading days flagged at the end of December
        after:  trading days flagged at the start of January
    """
    days = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in trading_days))
    mask = pd.Series(False, index=days)

    for year in sorted({d.year for d in days}):
        dec = days[(days.year == year) & (days.month == 12)]
        if len(dec):
            mask.loc[dec[-before:]] = True

        jan = days[(days.year == year) & (days.month == 1)]
        if len(jan):
            mask.loc[jan[:after]] = True

    return mask


def fiscal_month_windows(trading_days, fiscal_months: pd.Series,
                         before: int = 3, after: int = 2) -> pd.DataFrame:
    """Per-ticker flag, using each company's own fiscal year end.

    Args:
        fiscal_months: Series indexed by ticker, values = 결산월 ("12", "03", ...)
                       as returned by `dart.fiscal_month_table`.

    Returns:
        Boolean panel (index=date, columns=ticker). True = inside that
        company's ex-dividend window.
    """
    days = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in trading_days))
    tickers = list(fiscal_months.index)
    out = pd.DataFrame(False, index=days, columns=tickers)

    months = (pd.to_numeric(fiscal_months, errors="coerce")
              .fillna(12).astype(int).clip(1, 12))

    for month in sorted(months.unique()):
        cols = months[months == month].index.tolist()
        if not cols:
            continue

        flagged = pd.DatetimeIndex([])
        for year in sorted({d.year for d in days}):
            in_month = days[(days.year == year) & (days.month == month)]
            if len(in_month):
                flagged = flagged.union(in_month[-before:])

            nxt_year, nxt_month = (year, month + 1) if month < 12 else (year + 1, 1)
            after_month = days[(days.year == nxt_year) & (days.month == nxt_month)]
            if len(after_month):
                flagged = flagged.union(after_month[:after])

        out.loc[flagged, cols] = True

    return out


def coverage_note(fiscal_months: pd.Series | None) -> str:
    """One-line honest description of which flag is in force."""
    if fiscal_months is None or fiscal_months.empty:
        return ("ex-dividend: market-wide December window only "
                "(no DART fiscal-month data; non-December closers are NOT flagged)")

    months = pd.to_numeric(fiscal_months, errors="coerce").fillna(12).astype(int)
    non_dec = int((months != 12).sum())
    return (f"ex-dividend: per-company fiscal-year-end windows "
            f"({len(months):,} companies, {non_dec:,} non-December). "
            f"Post-2024 배당기준일 dispersion is NOT captured.")
