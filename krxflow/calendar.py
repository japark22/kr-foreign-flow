"""Korean trading-day calendar.

Trading days are derived from KRX itself (via pykrx) rather than from a
hardcoded holiday list, so Korean public holidays, temporary market closures
and the odd extra half-day are all handled correctly.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from . import config

_CACHE = config.DATA_DIR / "trading_days.json"


def _to_yyyymmdd(d) -> str:
    if isinstance(d, str):
        return d.replace("-", "")
    return d.strftime("%Y%m%d")


def trading_days(start, end, use_cache: bool = True) -> list[str]:
    """Return KRX trading days in [start, end] as YYYYMMDD strings.

    Uses pykrx's business-day helper, which reads the OHLCV history of a
    long-listed ticker and takes the dates that actually printed.
    """
    from pykrx import stock

    start_s, end_s = _to_yyyymmdd(start), _to_yyyymmdd(end)

    cached: dict[str, list[str]] = {}
    if use_cache and _CACHE.exists():
        try:
            cached = json.loads(_CACHE.read_text())
        except (json.JSONDecodeError, OSError):
            cached = {}

    key = f"{start_s}-{end_s}"
    if key in cached:
        return cached[key]

    days = stock.get_previous_business_days(fromdate=start_s, todate=end_s)
    out = [pd.Timestamp(d).strftime("%Y%m%d") for d in days]

    if use_cache:
        cached[key] = out
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE.write_text(json.dumps(cached, indent=0))

    return out


def latest_trading_day(lookback_days: int = 14) -> str:
    """Most recent completed trading day, as YYYYMMDD.

    KRX publishes the foreign-ownership snapshot for day D after the close of
    D, so for a point-in-time pipeline the newest safely-available date is the
    last trading day strictly before today.
    """
    today = dt.date.today()
    days = trading_days(today - dt.timedelta(days=lookback_days), today, use_cache=False)
    today_s = today.strftime("%Y%m%d")
    past = [d for d in days if d < today_s]
    if not past:
        raise RuntimeError(
            f"No trading day found in the last {lookback_days} days before {today_s}. "
            "Is the KRX connection working?"
        )
    return past[-1]
