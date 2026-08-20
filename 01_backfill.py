#!/usr/bin/env python3
"""Step 2: backfill daily foreign-ownership history.

    python 01_backfill.py --start 2023-01-01
    python 01_backfill.py --start 2010-01-01 --end 2026-08-14
    python 01_backfill.py --start 2023-01-01 --with-market   # + close/volume/cap

Resumable by design: each trading day is written as its own parquet file and
already-present days are skipped, so you can kill this at any point (Ctrl-C)
and re-run to pick up exactly where it stopped. That also makes it the same
script used for the scheduled 3x/week update — just point it at a recent
start date and it fills whatever is missing.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time

import pandas as pd

from krxflow import calendar as kcal
from krxflow import collect, config, storage


FETCHERS = {
    "foreign_ownership": lambda d, m: collect.fetch_foreign_ownership(d, m),
    "market": lambda d, m: collect.fetch_market_snapshot(d, m),
    "investor_flow": lambda d, m: collect.fetch_investor_flow(d, m),
    "shorting": lambda d, m: collect.fetch_shorting(d, m),
}


def parse_args() -> argparse.Namespace:
    today = dt.date.today()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default=(today - dt.timedelta(days=365 * 3)).isoformat(),
                   help="first date, YYYY-MM-DD (default: 3 years ago)")
    p.add_argument("--end", default=None,
                   help="last date, YYYY-MM-DD (default: last completed trading day)")
    p.add_argument("--markets", default=",".join(config.MARKETS),
                   help="comma-separated: KOSPI,KOSDAQ,KONEX")
    p.add_argument("--with-market", action="store_true",
                   help="also collect close/volume/traded value/market cap "
                        "(needed for returns and the ADV denominator; roughly "
                        "doubles the runtime)")
    p.add_argument("--with-investor", action="store_true",
                   help="also collect net buying by investor type "
                        f"({', '.join(config.INVESTORS)}) — 기관합계 is the "
                        "control group for the persistence result")
    p.add_argument("--with-shorting", action="store_true",
                   help="also collect short balance and short volume, to "
                        "separate genuine buying from short covering")
    p.add_argument("--redo", action="store_true",
                   help="re-fetch days that are already stored")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after N days (useful for a quick trial run)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    markets = [m.strip().upper() for m in args.markets.split(",") if m.strip()]
    config.ensure_dirs()

    if not collect.login():
        print("Could not establish a KRX session. Run 00_smoke_test.py first.")
        return 1

    end = args.end.replace("-", "") if args.end else kcal.latest_trading_day()
    start = args.start.replace("-", "")

    print(f"Resolving trading days {start} -> {end} ...")
    days = kcal.trading_days(start, end)
    if not days:
        print("No trading days in that range.")
        return 1

    stores = ["foreign_ownership"]
    if args.with_market:
        stores.append("market")
    if args.with_investor:
        stores.append("investor_flow")
    if args.with_shorting:
        stores.append("shorting")

    todo = [d for d in days
            if args.redo or any(not storage.exists(s, d) for s in stores)]

    if args.limit:
        todo = todo[:args.limit]

    print(f"  {len(days):,} trading days in range")
    print(f"  markets: {', '.join(markets)}")

    # Report per store. A combined "already stored" count is meaningless: with
    # ownership complete and prices empty it reads as 0 done when in fact half
    # the work is finished.
    total_requests = 0
    for store in stores:
        have = sum(1 for d in days if storage.exists(store, d))
        need = len(days) if args.redo else len(days) - have
        total_requests += need * len(markets) * storage.REQUESTS_PER_DAY[store]
        print(f"  {store:<18} {have:>6,} stored, {need:>6,} to fetch")

    if not todo:
        print("\nNothing to do — already up to date.")
        return 0

    est_min = total_requests * (config.REQUEST_SLEEP_SEC + 0.15) / 60
    print(f"\n  ~{total_requests:,} requests, rough estimate {est_min:,.0f} min\n")

    started = time.monotonic()
    written = failed = consecutive_failures = 0

    for i, date in enumerate(todo, 1):
        elapsed = time.monotonic() - started
        rate = i / elapsed if elapsed > 0 else 0
        eta_min = (len(todo) - i) / rate / 60 if rate > 0 else 0
        prefix = f"[{i:>5}/{len(todo)}] {date}"

        try:
            for store in stores:
                if storage.exists(store, date) and not args.redo:
                    continue

                fetcher = FETCHERS[store]
                parts = []
                for market in markets:
                    df = fetcher(date, market)
                    if not df.empty:
                        parts.append(df)

                if not parts:
                    print(f"{prefix}  {store}: empty (non-trading day?) — skipped")
                    continue

                combined = pd.concat(parts, ignore_index=True)
                storage.write(store, date, combined)
                written += 1
                # Print for whichever store actually did work, so a run that
                # only fills the price store still shows progress instead of
                # sitting silent for hours.
                print(f"{prefix}  {store:<18} {len(combined):>5,} rows   "
                      f"ETA {eta_min:5.1f} min", flush=True)

        except KeyboardInterrupt:
            print("\nInterrupted. Progress is saved — re-run the same command "
                  "to resume from here.")
            return 130
        except Exception as err:  # noqa: BLE001
            failed += 1
            consecutive_failures += 1
            print(f"{prefix}  ERROR {type(err).__name__}: {err}", flush=True)
            # Count CONSECUTIVE failures, not total. A 4,000-day run should not
            # abort because twenty days scattered across sixteen years each hit
            # a transient timeout; it should abort when KRX has clearly stopped
            # answering.
            if consecutive_failures >= 10:
                print("\n10 failures in a row — KRX is not answering. Stopping "
                      "rather than hammering it. Re-run the same command later "
                      "to resume.")
                return 1
        else:
            consecutive_failures = 0

    mins = (time.monotonic() - started) / 60
    print(f"\nDone. {written:,} days written, {failed} failed, {mins:.1f} min elapsed.")
    print("Next:  python 02_inspect.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
