#!/usr/bin/env python3
"""Step 1: prove the whole chain works end to end on this machine.

Run this before anything else:

    python 00_smoke_test.py

It checks, in order: Python version -> packages -> credentials -> KRX login ->
trading calendar -> one real day of foreign-ownership data -> data sanity ->
parquet write/read. Every check prints PASS or FAIL with a specific reason, so
a failure tells you exactly what to fix.

Nothing is downloaded in bulk; this makes about five requests total.
"""
from __future__ import annotations

import platform
import sys
import traceback

RESULTS: list[tuple[str, bool, str]] = []
WIDTH = 74


def record(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def section(title: str) -> None:
    print()
    print(title)
    print("-" * WIDTH)


def main() -> int:
    print("=" * WIDTH)
    print("  Korea foreign-ownership pipeline — smoke test")
    print("=" * WIDTH)

    # ------------------------------------------------------------------ 1 --
    section("1. Environment")
    print(f"  python   : {sys.version.split()[0]}  ({sys.executable})")
    print(f"  platform : {platform.platform()}")
    print(f"  machine  : {platform.machine()}")
    record("Python >= 3.10", sys.version_info >= (3, 10),
           f"found {sys.version_info.major}.{sys.version_info.minor}")

    # ------------------------------------------------------------------ 2 --
    section("2. Packages")
    try:
        import contextlib
        import io

        import pandas as pd
        import pyarrow  # noqa: F401

        # pykrx prints a Korean "login failed" line at import time when
        # KRX_ID/KRX_PW are unset. Swallow it here so it does not look like a
        # real failure — credentials get their own check in section 3.
        with contextlib.redirect_stdout(io.StringIO()):
            import pykrx

        record("pandas", True, pd.__version__)
        record("pyarrow", True, pyarrow.__version__)
        record("pykrx", True, getattr(pykrx, "__version__", "unknown"))
    except ImportError as err:
        record("imports", False, str(err))
        print("\n  Fix:  pip install -r requirements.txt")
        return summarise()

    from krxflow import calendar as kcal
    from krxflow import collect, config, storage

    # ------------------------------------------------------------------ 3 --
    section("3. Credentials")
    status = config.credentials_status()
    for key, value in status.items():
        print(f"  {key:<14} {value}")
    have_krx = status["KRX_ID"] != "NOT SET" and status["KRX_PW"] != "NOT SET"
    record("KRX_ID / KRX_PW present", have_krx,
           "" if have_krx else "export them, then re-run")
    if status["DART_API_KEY"] == "NOT SET":
        print("  note: DART_API_KEY is optional for this smoke test "
              "(needed later for dividends / corporate actions).")
    if not have_krx:
        return summarise()

    # ------------------------------------------------------------------ 4 --
    section("4. KRX login")
    try:
        logged_in = collect.login()
    except Exception as err:  # noqa: BLE001
        logged_in = False
        print(f"  exception: {type(err).__name__}: {err}")
    record("authenticated KRX session", logged_in,
           "" if logged_in else "check the ID/password, or log in at krx.co.kr once "
                                "to clear any forced password change")
    if not logged_in:
        return summarise()

    # ------------------------------------------------------------------ 5 --
    section("5. Trading calendar")
    try:
        date = kcal.latest_trading_day()
        record("latest completed trading day", True, date)
    except Exception as err:  # noqa: BLE001
        record("latest completed trading day", False, f"{type(err).__name__}: {err}")
        traceback.print_exc()
        return summarise()

    # ------------------------------------------------------------------ 6 --
    section(f"6. Foreign-ownership snapshot for {date}")
    frames = {}
    for market in config.MARKETS:
        try:
            df = collect.fetch_foreign_ownership(date, market)
            ok = not df.empty
            frames[market] = df
            record(f"{market} snapshot", ok,
                   f"{len(df):,} tickers" if ok else "empty response")
        except Exception as err:  # noqa: BLE001
            record(f"{market} snapshot", False, f"{type(err).__name__}: {err}")
            traceback.print_exc()

    if not any(len(d) for d in frames.values()):
        return summarise()

    combined = pd.concat([d for d in frames.values() if not d.empty], ignore_index=True)

    # ------------------------------------------------------------------ 7 --
    section("7. Data sanity")
    record("ticker count is plausible (>2000)", len(combined) > 2000, f"{len(combined):,}")
    record("no duplicate tickers", combined["ticker"].duplicated().sum() == 0,
           f"{combined['ticker'].duplicated().sum()} dupes")
    record("shares_listed all positive",
           bool((combined["shares_listed"].fillna(0) > 0).all()),
           f"{int((combined['shares_listed'].fillna(0) <= 0).sum())} bad rows")
    within = combined["foreign_pct"].dropna()
    record("foreign_pct within 0-100", bool(((within >= 0) & (within <= 100)).all()),
           f"min {within.min():.4f}  max {within.max():.4f}")

    print("\n  Largest foreign holdings by ownership %:")
    top = combined.nlargest(5, "foreign_pct")[
        ["ticker", "market", "shares_listed", "foreign_shares", "foreign_pct"]
    ]
    print(top.to_string(index=False))

    samsung = combined[combined["ticker"] == "005930"]
    if not samsung.empty:
        row = samsung.iloc[0]
        print(f"\n  Spot check 005930 (Samsung Electronics):")
        print(f"    shares listed  {int(row['shares_listed']):,}")
        print(f"    foreign held   {int(row['foreign_shares']):,}")
        print(f"    ownership      {row['foreign_pct']:.6f} %")
        record("Samsung ownership in a believable band (30-70%)",
               30 <= row["foreign_pct"] <= 70, f"{row['foreign_pct']:.4f}%")

    # ------------------------------------------------------------------ 8 --
    section("8. float16 precision check (why we recompute the ratio)")
    cmp = combined.dropna(subset=["foreign_pct", "foreign_pct_krx_lossy"]).copy()
    cmp["abs_err_pp"] = (cmp["foreign_pct"] - cmp["foreign_pct_krx_lossy"]).abs()
    worst = cmp.nlargest(5, "abs_err_pp")[
        ["ticker", "foreign_pct", "foreign_pct_krx_lossy", "abs_err_pp"]
    ]
    print("  pykrx returns 지분율 as np.float16. Exact vs as-returned:")
    print(worst.to_string(index=False,
                          float_format=lambda v: f"{v:.6f}"))
    print(f"\n  median absolute error : {cmp['abs_err_pp'].median():.6f} pp")
    print(f"  worst absolute error  : {cmp['abs_err_pp'].max():.6f} pp")
    print("  A typical one-day change in foreign ownership is 0.01-0.05 pp,")
    print("  so the as-returned column is unusable for the signal. We use")
    print("  foreign_shares / shares_listed (both exact int64) instead.")
    record("recomputed foreign_pct differs from lossy column",
           cmp["abs_err_pp"].max() > 0, "confirms the float16 rounding is real")

    # ------------------------------------------------------------------ 9 --
    section("9. Parquet write / read round-trip")
    try:
        config.ensure_dirs()
        path = storage.write("foreign_ownership", date, combined)
        back = pd.read_parquet(path)
        ok = len(back) == len(combined)
        size_kb = path.stat().st_size / 1024
        record("wrote and re-read parquet", ok,
               f"{path}  ({size_kb:,.0f} KB, {len(back):,} rows)")
        print(f"\n  Stored columns: {list(back.columns)}")
        est_mb_per_year = size_kb * 245 / 1024
        print(f"\n  Size estimate: ~{est_mb_per_year:,.0f} MB per year of history "
              f"for this store.")
    except Exception as err:  # noqa: BLE001
        record("parquet round-trip", False, f"{type(err).__name__}: {err}")
        traceback.print_exc()

    return summarise()


def summarise() -> int:
    section("Summary")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"  {passed}/{len(RESULTS)} checks passed")
    if failed:
        print("\n  Failed:")
        for name in failed:
            print(f"    - {name}")
        print("\n  Fix the failures above and re-run.")
        return 1
    print("\n  All good. Next step:")
    print("    python 01_backfill.py --start 2023-01-01   # ~3 years, resumable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
