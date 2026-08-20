"""Parquet storage layout.

One immutable file per trading day per store:

    data/raw/foreign_ownership/2026/foreign_ownership_20260814.parquet
    data/raw/market/2026/market_20260814.parquet

Rationale (from the project spec): keep the raw layer exactly as fetched and
snapshot-per-date, so that (a) a backfill is trivially resumable, (b) a KRX
restatement is visible as a file-level diff rather than silently overwriting
history, and (c) any point-in-time reconstruction is just a file filter.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config

STORES = {
    "foreign_ownership": config.RAW_FOREIGN,
    "market": config.RAW_MARKET,
    "investor_flow": config.RAW_INVESTOR,
    "shorting": config.RAW_SHORTING,
}

# Requests KRX needs per market per trading day, used for time estimates.
REQUESTS_PER_DAY = {
    "foreign_ownership": 1,
    "market": 2,          # market cap + OHLCV
    "investor_flow": len(config.INVESTORS),
    "shorting": 2,        # balance + volume
}


def path_for(store: str, date: str) -> Path:
    base = STORES[store]
    return base / date[:4] / f"{store}_{date}.parquet"


def exists(store: str, date: str) -> bool:
    p = path_for(store, date)
    return p.exists() and p.stat().st_size > 0


def write(store: str, date: str, df: pd.DataFrame) -> Path:
    p = path_for(store, date)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False, compression="snappy")
    tmp.replace(p)  # atomic: a killed run never leaves a half-written file
    return p


CATEGORICAL = ("market", "ticker", "source")


def read_range(store: str, start: str | None = None, end: str | None = None,
               columns: list[str] | None = None) -> pd.DataFrame:
    """Load every stored day for a store, optionally bounded by YYYYMMDD.

    Sixteen years of all-market snapshots is roughly 10 million rows, so this
    reads only the requested columns and stores the repeated string columns as
    categoricals. Without that, `ticker` alone costs ~600 MB as Python objects
    and a laptop starts swapping.
    """
    base = STORES[store]
    if not base.exists():
        return pd.DataFrame()

    picked = []
    for f in sorted(base.glob("*/*.parquet")):
        date = f.stem.split("_")[-1]
        if start and date < start:
            continue
        if end and date > end:
            continue
        picked.append(f)

    if not picked:
        return pd.DataFrame()

    df = pd.concat(
        (pd.read_parquet(f, columns=columns) for f in picked),
        ignore_index=True,
    )

    for col in CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def stored_dates(store: str) -> list[str]:
    base = STORES[store]
    if not base.exists():
        return []
    return sorted(f.stem.split("_")[-1] for f in base.glob("*/*.parquet"))


# Re-exported so scripts can reach paths without a second import.
__all__ = ["STORES", "REQUESTS_PER_DAY", "path_for", "exists", "write",
           "read_range", "stored_dates", "config"]
