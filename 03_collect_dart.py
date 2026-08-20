#!/usr/bin/env python3
"""Step 3: collect the OpenDART reference data the noise filter needs.

    python 03_collect_dart.py --check          # verify the key works (1 call)
    python 03_collect_dart.py                  # corp-code map + 결산월 per ticker

What it fetches, and why:

  corpCode   one call, maps 6-digit KRX tickers to DART's 8-digit corp_code.
             Everything else in DART is keyed on corp_code.

  company    one call per company, gives acc_mt (결산월). This is what turns
             the ex-dividend filter from a blunt "every December" flag into
             a per-company one. Most companies close in December; the minority
             that do not are exactly the ones a December-only flag misses.

Both are cached to disk, so re-running costs nothing. OpenDART allows 20,000
calls/day and our universe is ~2,800 companies, so this fits in one day.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from krxflow import config, dart, storage


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true",
                   help="verify the API key with a single call, then stop")
    p.add_argument("--refresh", action="store_true",
                   help="re-download the corp-code map (picks up new listings)")
    p.add_argument("--limit", type=int, default=None,
                   help="only fetch 결산월 for the first N tickers (trial run)")
    args = p.parse_args()

    config.ensure_dirs()

    # ------------------------------------------------------------------- 1 --
    print("=" * 74)
    print("  OpenDART reference data")
    print("=" * 74)

    try:
        codes = dart.corp_codes(refresh=args.refresh)
    except dart.DartError as err:
        print(f"\nFAILED: {err}")
        return 1
    except Exception as err:  # noqa: BLE001
        print(f"\nFAILED: {type(err).__name__}: {err}")
        return 1

    print(f"  corp-code map: {len(codes):,} listed companies")
    print(f"  sample:")
    print(codes.head(3).to_string(index=False))

    if args.check:
        print("\n  Key works. Run without --check to collect 결산월.")
        return 0

    # ------------------------------------------------------------------- 2 --
    dates = storage.stored_dates("foreign_ownership")
    if not dates:
        print("\n  No ownership data stored — cannot tell which tickers matter.")
        print("  Run 01_backfill.py first.")
        return 1

    recent = storage.read_range("foreign_ownership", start=dates[-1],
                                columns=["ticker"])
    tickers = sorted(recent["ticker"].astype(str).unique())
    if args.limit:
        tickers = tickers[:args.limit]

    print(f"\n  universe from {dates[-1]}: {len(tickers):,} tickers")

    matched = set(codes["stock_code"]) & set(tickers)
    print(f"  matched to DART: {len(matched):,} "
          f"({len(tickers) - len(matched):,} not found — usually preferred "
          f"shares or foreign listings)")

    table = dart.fiscal_month_table(tickers)
    if table.empty:
        print("\n  No fiscal-month data collected.")
        return 1

    months = pd.to_numeric(table["acc_mt"], errors="coerce").fillna(12).astype(int)
    print(f"\n  결산월 collected for {len(table):,} companies")
    print("\n  month   companies")
    print("  -----   ---------")
    for month, n in months.value_counts().sort_index().items():
        flag = "  <- the ones a December-only filter would miss" if month != 12 else ""
        print(f"  {month:>5}   {n:>9,}{flag}")

    print(f"\n  API calls used this run: {dart.calls_made():,}")
    print(f"  cached at: {config.DATA_DIR / 'dart'}")
    print("\n  The ex-dividend filter in 04_validate.py now uses these dates.")
    print("  Next:  python 04_validate.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
