#!/usr/bin/env python3
"""Step 6: measure how fresh KRX foreign-ownership data actually is.

    python 06_check_freshness.py

Two questions, both answered by measurement rather than by reading a spec that
does not exist. KRX does not publish an update schedule for this statistic.

  A. PUBLICATION LAG — on any given day, how recent is the newest snapshot
     KRX will serve? This is what decides whether collecting daily is worth
     anything over collecting weekly, and it decides the D-lag the signal has
     to assume to stay point-in-time.

  B. RESTATEMENT — does KRX ever revise a day's numbers after first publishing
     them? This has not been checked, and it matters more than it sounds. If
     history gets silently revised, then a backtest reading today's stored
     values is using numbers nobody could have seen at the time, and every
     result is optimistic by an unknown amount.

     We can test it because the raw layer stored each day exactly as fetched
     during the backfill. Re-fetching a sample and diffing against what is on
     disk is a direct check.

Makes roughly 20 requests.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import zoneinfo

import pandas as pd

from krxflow import calendar as kcal
from krxflow import collect, config, storage

KST = zoneinfo.ZoneInfo("Asia/Seoul")
W = 76

KEY_COLUMNS = ["shares_listed", "foreign_shares", "foreign_limit_shares"]


def rule(title: str) -> None:
    print()
    print("=" * W)
    print(f"  {title}")
    print("=" * W)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe-days", type=int, default=6,
                   help="how many recent calendar days to probe for availability")
    p.add_argument("--restate-sample", type=int, default=6,
                   help="how many stored days to re-fetch and diff")
    args = p.parse_args()

    config.ensure_dirs()

    now_kst = dt.datetime.now(KST)
    now_local = dt.datetime.now()
    print("=" * W)
    print("  KRX foreign-ownership freshness check")
    print("=" * W)
    print(f"  now, Seoul  : {now_kst:%Y-%m-%d %H:%M} KST")
    print(f"  now, local  : {now_local:%Y-%m-%d %H:%M}")
    print(f"  KRX session : 09:00-15:30 KST")

    if not collect.login():
        print("\n  KRX login failed.")
        return 1

    # ------------------------------------------------------------------ A --
    rule("A. Publication lag — what is the newest day KRX will serve?")

    today = now_kst.date()
    probe = [today - dt.timedelta(days=i) for i in range(args.probe_days)]

    print("  date         weekday   KOSPI rows   status")
    print("  ----------   -------   ----------   ------------------------")

    newest_available = None
    for d in probe:
        ds = d.strftime("%Y%m%d")
        try:
            df = collect.fetch_foreign_ownership(ds, "KOSPI")
            n = len(df)
        except Exception as err:  # noqa: BLE001
            print(f"  {d}   {d:%a}       {'-':>10}   error: {type(err).__name__}")
            continue

        if n == 0:
            note = "no data (holiday, weekend, or not yet published)"
        else:
            note = "available"
            if newest_available is None:
                newest_available = d
        print(f"  {d}   {d:%a}       {n:>10,}   {note}")

    print()
    if newest_available is None:
        print("  Nothing available in the probe window — unexpected, investigate.")
    else:
        lag_days = (today - newest_available).days
        if newest_available == today:
            print(f"  Newest = TODAY ({newest_available}). At {now_kst:%H:%M} KST the")
            print("  same-day snapshot is already served, so a run after the close")
            print("  captures the current session.")
        else:
            print(f"  Newest = {newest_available}, {lag_days} calendar day(s) behind today.")
            print(f"  Run this again after 17:00 KST to see whether today's snapshot")
            print("  appears later in the day — that pins the publication hour.")

    # ------------------------------------------------------------------ B --
    rule("B. Restatement — does KRX revise days it already published?")

    stored = storage.stored_dates("foreign_ownership")
    if len(stored) < 10:
        print("  Not enough stored history to check. Run 01_backfill.py first.")
        return 0

    # Sample across the history: recent days are the most likely to be revised,
    # old ones establish whether anything drifts long after the fact.
    picks: list[str] = []
    for frac in (1.0, 0.995, 0.98, 0.9, 0.5, 0.05):
        i = min(len(stored) - 1, int(len(stored) * frac))
        if stored[i] not in picks:
            picks.append(stored[i])
    picks = picks[:args.restate_sample]

    print(f"  Re-fetching {len(picks)} stored day(s) and diffing against disk.\n")
    print("  date         rows on disk   rows now   differing rows   verdict")
    print("  ----------   ------------   --------   --------------   -------------")

    any_diff = False
    for ds in picks:
        on_disk = storage.read_range("foreign_ownership", ds, ds)
        on_disk = on_disk[on_disk["market"] == "KOSPI"] if "market" in on_disk else on_disk
        try:
            fresh = collect.fetch_foreign_ownership(ds, "KOSPI")
        except Exception as err:  # noqa: BLE001
            print(f"  {ds}   {len(on_disk):>12,}   {'error':>8}   "
                  f"{'-':>14}   {type(err).__name__}")
            continue

        a = on_disk.set_index("ticker")[KEY_COLUMNS].astype("float64").sort_index()
        b = fresh.set_index("ticker")[KEY_COLUMNS].astype("float64").sort_index()
        common = a.index.intersection(b.index)
        diff = (a.loc[common] != b.loc[common]).any(axis=1)
        n_diff = int(diff.sum())
        any_diff |= n_diff > 0

        verdict = "identical" if n_diff == 0 else f"REVISED"
        print(f"  {ds}   {len(a):>12,}   {len(b):>8,}   {n_diff:>14,}   {verdict}")

        if n_diff:
            sample = diff[diff].index[:3]
            for t in sample:
                print(f"      {t}: stored {a.loc[t, 'foreign_shares']:,.0f} "
                      f"-> now {b.loc[t, 'foreign_shares']:,.0f}")

    print()
    if any_diff:
        print("  KRX REVISES published values.")
        print("  Consequence: the stored snapshot is the point-in-time truth and")
        print("  must be used as-is for backtesting. Never re-fetch and overwrite")
        print("  history — that would leak information backwards. The immutable")
        print("  per-day files already protect this; keep it that way.")
    else:
        print("  No revisions found in this sample. Published values look stable,")
        print("  which means a re-fetch would be harmless — but the immutable")
        print("  raw layer stays the safer default, and costs nothing.")

    rule("What this means for the update schedule")
    print("  Collection cadence should match the publication cadence: if a new")
    print("  snapshot lands every trading day, collect every trading day. The")
    print("  cost is two requests.")
    print()
    print("  Report cadence is a different question. The validation statistics")
    print(f"  are computed over {len(stored):,} trading days, so one more day moves")
    print(f"  them by roughly 1/{len(stored):,} = {100/len(stored):.3f}% of their weight.")
    print("  Regenerating the full report daily would produce a diff dominated by")
    print("  rounding. Weekly is already generous; the daily job's real value is")
    print("  catching a collection failure early, which the log does.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
