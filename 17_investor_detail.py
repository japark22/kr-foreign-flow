#!/usr/bin/env python3
"""Step 17: daily net buying by investor TYPE, per issuer.

    set -a; source .env; set +a          # pykrx needs the KRX login
    python 17_investor_detail.py --limit 30    # trial: time and size it first
    python 17_investor_detail.py               # full universe, resumable

WHY THIS EXISTS
---------------
Everything measured so far used ONE aggregate series: total foreign ownership.
Two problems with that, and this collection fixes both at once.

1. MEASUREMENT. The holdings series agrees with the exchange's own reported
   foreign net buying at only 0.10. If holdings change is a 0.10-correlated
   proxy for actual buying, attenuation alone would shrink a true IC of ~0.04
   to the ~0.004 we measured. This endpoint returns the reported net buying
   itself, so the proxy can be replaced by the thing it was proxying for.

2. AGGREGATION. "Foreign" blends informed money with mechanical money -- index
   and passive flows sit in the same number as discretionary funds, and they
   plausibly cancel. The detail view splits the market into 금융투자, 보험,
   투신, 사모, 은행, 기타금융, 연기금, 기타법인, 개인, 외국인, 기타외국인.
   사모 is the closest domestic analogue to a hedge fund; 연기금 is its
   opposite (slow, mandate-driven). Those should not be averaged together.

Values are net buying in KRW for the day (buy minus sell), per issuer.

One API call covers one ticker's whole history, cached to
data/investor/<ticker>.parquet, so a stopped run resumes for free and a
re-run costs nothing.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "investor"
SLEEP = 0.25
START = "20180101"


def universe(limit: int | None) -> list[str]:
    """Every ticker the ownership store has ever seen, most liquid first.

    Delisted names are kept: dropping them would build survivorship into
    every study that uses this data.
    """
    from krxflow import storage
    dates = storage.stored_dates("foreign_ownership")
    if not dates:
        sys.exit("no ownership store yet -- run 01_backfill.py first")
    df = storage.read_range("foreign_ownership", dates[0], None,
                            columns=["ticker"])
    seen = df["ticker"].astype(str).value_counts()          # days present
    tickers = [t for t in seen.index if len(t) == 6 and not t.startswith("9")]
    return tickers[:limit] if limit else tickers


def fetch(ticker: str, start: str, end: str):
    import pykrx.stock as s
    for attempt in (1, 2):
        try:
            df = s.get_market_trading_value_by_date(start, end, ticker,
                                                    detail=True)
            if df is None or df.empty:
                return None
            df = df.reset_index()
            df.columns = [str(c) for c in df.columns]
            first = df.columns[0]
            df = df.rename(columns={first: "trade_date"})
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            if "전체" in df.columns and (df["전체"].abs().sum() == 0):
                df = df.drop(columns=["전체"])
            df.insert(1, "ticker", ticker)
            for c in df.columns:
                if c not in ("trade_date", "ticker"):
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
            return df
        except Exception as err:                            # noqa: BLE001
            if attempt == 2:
                print(f"    ! {ticker}: {type(err).__name__}: {err}")
                return None
            time.sleep(1.5)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only the N most-present tickers (trial run)")
    ap.add_argument("--start", default=START)
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch tickers already cached")
    a = ap.parse_args()

    if not os.getenv("KRX_ID"):
        print("  KRX_ID is not in the environment. Run:  set -a; source .env; set +a")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp.today().strftime("%Y%m%d")
    tickers = universe(a.limit)
    print(f"universe: {len(tickers):,} tickers   window {a.start}..{end}")

    t0 = time.time()
    done = fetched = failed = 0
    cols_seen = set()
    for i, t in enumerate(tickers, 1):
        cache = OUT / f"{t}.parquet"
        if cache.exists() and not a.refresh:
            done += 1
            continue
        time.sleep(SLEEP)
        df = fetch(t, a.start, end)
        if df is None or df.empty:
            failed += 1
            continue
        df.to_parquet(cache, index=False)
        cols_seen.update(df.columns)
        fetched += 1
        done += 1
        if fetched % 25 == 0:
            el = time.time() - t0
            rate = fetched / el
            left = (len(tickers) - i) / rate if rate else 0
            print(f"  {i:,}/{len(tickers):,}  fetched {fetched:,}  "
                  f"failed {failed}  {rate:.1f}/s  ~{left/60:.0f} min left")

    files = sorted(OUT.glob("*.parquet"))
    mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"\n  cached tickers: {len(files):,}   disk {mb:,.0f} MB")
    print(f"  fetched this run: {fetched:,}   failed: {failed:,}   "
          f"elapsed {(time.time()-t0)/60:.1f} min")
    if cols_seen:
        keep = [c for c in cols_seen if c not in ("trade_date", "ticker")]
        print(f"  investor columns: {', '.join(sorted(keep))}")
    if a.limit and files:
        full = len(universe(None))
        print(f"\n  extrapolated to the full {full:,} tickers: "
              f"~{mb/len(files)*full:,.0f} MB, "
              f"~{(time.time()-t0)/max(fetched,1)*full/60:,.0f} min")
        print("  re-run without --limit to collect the rest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
