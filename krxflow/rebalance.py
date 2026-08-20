"""Index-rebalance calendar (MSCI / FTSE) for the noise filter.

WHY THIS EXISTS
---------------
Passive money tracking MSCI and FTSE trades in size on a handful of known
dates. Those trades are mechanical, not informed, and they show up in the
foreign-ownership series as large one-day jumps. Left in, they contaminate
exactly the autocorrelation we are trying to measure.

ACCURACY WARNING  <-- read before publishing any result
------------------------------------------------------
The dates below are generated from the published *rules*, not from an
official historical date list:

  MSCI  — quarterly index reviews in Feb / May / Aug / Nov. Changes take
          effect as of the close of the last business day of the review
          month, so the rebalance trade prints on that day.

  FTSE  — GEIS semi-annual reviews in Mar / Sep, quarterly reviews in
          Jun / Dec. Changes take effect after the close of the third
          Friday of the review month.

Both are then snapped to the nearest actual KRX trading day.

This is a good approximation and is fine for a first pass, but MSCI and FTSE
do move dates, and MSCI's own schedule shows at least one review (May 2027)
landing on a date the simple rule does not predict. Before any result goes to
final, pin the real dates in `rebalance_overrides.csv` (see
`load_overrides`) and re-run.

MSCI publishes forward dates at:
  https://app2.msci.com/eqb/pressreleases/archive/ir_dates.csv
"""
from __future__ import annotations

import calendar as _cal
import datetime as dt
from pathlib import Path

import pandas as pd

from . import config

MSCI_MONTHS = (2, 5, 8, 11)
FTSE_MONTHS = (3, 6, 9, 12)

OVERRIDE_FILE = config.DATA_DIR / "rebalance_overrides.csv"


def _last_calendar_day(year: int, month: int) -> dt.date:
    return dt.date(year, month, _cal.monthrange(year, month)[1])


def _third_friday(year: int, month: int) -> dt.date:
    first = dt.date(year, month, 1)
    # weekday(): Monday=0 ... Friday=4
    offset = (4 - first.weekday()) % 7
    return first + dt.timedelta(days=offset + 14)


def _snap(target: dt.date, trading_days: list[dt.date], how: str = "prev") -> dt.date | None:
    """Move a calendar date onto an actual KRX trading day."""
    if how == "prev":
        candidates = [d for d in trading_days if d <= target]
        return candidates[-1] if candidates else None
    candidates = [d for d in trading_days if d >= target]
    return candidates[0] if candidates else None


def load_overrides() -> pd.DataFrame:
    """Optional CSV of authoritative dates that replaces the rule-based guess.

    Format:
        provider,effective_date,note
        MSCI,2024-05-31,May 2024 SAIR
        FTSE,2024-09-20,Sep 2024 semi-annual
    """
    if not OVERRIDE_FILE.exists():
        return pd.DataFrame(columns=["provider", "effective_date", "note"])
    df = pd.read_csv(OVERRIDE_FILE)
    df["effective_date"] = pd.to_datetime(df["effective_date"]).dt.date
    return df


def rebalance_dates(trading_days: list) -> pd.DataFrame:
    """Effective dates for MSCI and FTSE reviews, snapped to trading days.

    Args:
        trading_days: KRX trading days (anything pandas can parse as a date).

    Returns:
        DataFrame with columns provider, effective_date, source, note.
    """
    days = sorted({pd.Timestamp(d).date() for d in trading_days})
    if not days:
        return pd.DataFrame(columns=["provider", "effective_date", "source", "note"])

    first_year, last_year = days[0].year, days[-1].year
    rows: list[dict] = []

    for year in range(first_year, last_year + 1):
        for month in MSCI_MONTHS:
            snapped = _snap(_last_calendar_day(year, month), days, how="prev")
            if snapped and days[0] <= snapped <= days[-1]:
                rows.append({
                    "provider": "MSCI",
                    "effective_date": snapped,
                    "source": "rule",
                    "note": f"{year}-{month:02d} review, last trading day of month",
                })
        for month in FTSE_MONTHS:
            snapped = _snap(_third_friday(year, month), days, how="prev")
            if snapped and days[0] <= snapped <= days[-1]:
                kind = "semi-annual" if month in (3, 9) else "quarterly"
                rows.append({
                    "provider": "FTSE",
                    "effective_date": snapped,
                    "source": "rule",
                    "note": f"{year}-{month:02d} {kind}, third Friday",
                })

    df = pd.DataFrame(rows)

    overrides = load_overrides()
    if not overrides.empty:
        # An override wins for its (provider, month): drop the rule-based row.
        df["_ym"] = pd.to_datetime(df["effective_date"]).dt.to_period("M")
        overrides["_ym"] = pd.to_datetime(overrides["effective_date"]).dt.to_period("M")
        overrides["source"] = "override"
        keys = set(zip(overrides["provider"], overrides["_ym"]))
        df = df[~df.apply(lambda r: (r["provider"], r["_ym"]) in keys, axis=1)]
        df = pd.concat([df, overrides], ignore_index=True)
        df = df.drop(columns=["_ym"])

    return df.sort_values("effective_date").reset_index(drop=True)


def rebalance_mask(trading_days: list, window: int = 2) -> pd.Series:
    """Boolean series over trading days: True = inside a rebalance window.

    Args:
        window: trading days flagged either side of each effective date.
                window=2 flags a 5-day window (t-2 .. t+2).
    """
    days = sorted({pd.Timestamp(d).date() for d in trading_days})
    index = {d: i for i, d in enumerate(days)}
    mask = pd.Series(False, index=pd.to_datetime(days))

    for eff in rebalance_dates(days)["effective_date"]:
        i = index.get(eff)
        if i is None:
            continue
        lo, hi = max(0, i - window), min(len(days) - 1, i + window)
        mask.iloc[lo:hi + 1] = True

    return mask
