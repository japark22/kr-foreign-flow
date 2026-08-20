#!/usr/bin/env python3
"""Step 5: the scheduled run. One command, safe to run any number of times.

    python 05_daily_update.py                 # collect whatever is missing
    python 05_daily_update.py --with-market   # also prices
    python 05_daily_update.py --report        # + regenerate the validation report
    python 05_daily_update.py --report --push # + commit and push the report

Designed to be driven by launchd (see install_schedule.sh). Every run:

  1. works out which trading days are missing and fetches only those
  2. optionally regenerates reports/validation.md
  3. optionally commits and pushes — code and reports only, never data
  4. appends one line to data/logs/updates.log either way

Idempotent by construction: if nothing is missing it does nothing and says so.
That is what makes it safe to fire on a schedule and safe to run by hand.
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

from krxflow import calendar as kcal
from krxflow import collect, config, storage

import pandas as pd

LOG = config.LOG_DIR / "updates.log"
REPORTS = config.ROOT / "reports"


def log_line(message: str) -> None:
    config.ensure_dirs()
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG.open("a") as fh:
        fh.write(f"{stamp}  {message}\n")
    print(message)


def collect_missing(markets: list[str], stores: list[str], lookback_days: int) -> int:
    """Fetch any trading day we do not already have. Returns days written."""
    end = kcal.latest_trading_day()
    start = (dt.date.today() - dt.timedelta(days=lookback_days)).strftime("%Y%m%d")
    days = kcal.trading_days(start, end, use_cache=False)

    todo = [d for d in days if any(not storage.exists(s, d) for s in stores)]
    if not todo:
        log_line(f"up to date through {end} — nothing to fetch")
        return 0

    log_line(f"fetching {len(todo)} missing day(s): {todo[0]}..{todo[-1]}")

    written = 0
    for date in todo:
        for store in stores:
            if storage.exists(store, date):
                continue
            fetcher = (collect.fetch_foreign_ownership if store == "foreign_ownership"
                       else collect.fetch_market_snapshot)
            parts = [df for market in markets
                     if not (df := fetcher(date, market)).empty]
            if parts:
                storage.write(store, date, pd.concat(parts, ignore_index=True))
                if store == "foreign_ownership":
                    written += 1

    log_line(f"wrote {written} day(s) through {end}")
    return written


def data_status() -> dict:
    """Coverage and gap summary for the report header."""
    dates = storage.stored_dates("foreign_ownership")
    market_dates = storage.stored_dates("market")
    if not dates:
        return {"days": 0}

    expected = kcal.trading_days(dates[0], dates[-1])
    missing = sorted(set(expected) - set(dates))

    return {
        "days": len(dates),
        "first": dates[0],
        "last": dates[-1],
        "missing": missing,
        "market_days": len(market_dates),
        "market_last": market_dates[-1] if market_dates else None,
    }


def write_report(args_extra: list[str]) -> Path | None:
    """Run the validator and wrap its output as a standalone English report."""
    REPORTS.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, str(config.ROOT / "04_validate.py"), *args_extra],
        capture_output=True, text=True, cwd=config.ROOT,
    )
    if proc.returncode != 0:
        tail = (proc.stderr.strip().splitlines() or ["unknown error"])[-1]
        log_line(f"validation FAILED: {tail}")
        return None

    st = data_status()
    if not st["days"]:
        log_line("report skipped: no data")
        return None

    gaps = st["missing"]
    gap_line = (f"**{len(gaps)} missing trading day(s)** — "
                f"{', '.join(gaps[:8])}{' …' if len(gaps) > 8 else ''}"
                if gaps else "No gaps.")

    market_line = (f"{st['market_days']:,} days through {st['market_last']}"
                   if st["market_days"] else
                   "not collected — IC, long-short and momentum tests are skipped")

    body = f"""# Korean Foreign-Ownership Flow — Validation Report

Generated {dt.datetime.now():%Y-%m-%d %H:%M} local time.

## What this tests

Whether the persistence of daily foreign-investor flow in Korean equities
predicts future flow and future returns, and whether it survives the removal of
mechanical, uninformed trading around index rebalances and ex-dividend dates.

## Data status

| | |
|---|---|
| Ownership snapshots | {st['days']:,} trading days, {st['first']} to {st['last']} |
| Market data | {market_line} |
| Coverage | {gap_line} |
| Source | KRX [12023] 외국인보유량(개별종목), all listed names |

Foreign ownership percentages are recomputed in float64 from exact integer
share counts. The percentage column KRX returns is float16 and is carried only
as a labelled lossy field; at 55% ownership its resolution is 0.031pp, which is
larger than a typical one-day change.

## Results

```
{proc.stdout}
```

## Standing caveats

These are known limitations, not oversights. Each is tracked as an open item.

1. **Index-rebalance dates are rule-derived.** MSCI and FTSE effective dates are
   generated from published rules (MSCI: last trading day of Feb/May/Aug/Nov;
   FTSE GEIS: third Friday of Mar/Jun/Sep/Dec) and snapped to KRX trading days.
   Both providers move dates. Pin authoritative dates in
   `data/rebalance_overrides.csv` before treating results as final.

2. **Ex-dividend dates are derived, not observed.** OpenDART publishes no
   배당기준일 field. Dates are inferred from each company's fiscal year end and
   flagged as a window. Since the 2024 배당절차 개선방안, companies may set
   배당기준일 after the AGM, so from 2024 onward the dispersion is not captured.

3. **Free float is approximated by shares outstanding.** The correct denominator
   needs a vendor feed.

4. **Corporate actions are detected by a proxy** — a change in shares
   outstanding — rather than by parsing the actual capital-change events.

5. **Trading costs are not modelled.** Long-short figures are gross.

## Reproducing

```bash
python 01_backfill.py --start 2010-01-01 --with-market
python 03_collect_dart.py
python 04_validate.py --horizons 1,5,20
```
"""
    path = REPORTS / "validation.md"
    path.write_text(body)
    log_line(f"wrote {path.relative_to(config.ROOT)}"
             + (f" ({len(gaps)} gaps)" if gaps else ""))
    return path


def git(*cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *cmd], cwd=config.ROOT,
                          capture_output=True, text=True)


def push_report() -> bool:
    """Commit and push code + reports. .gitignore keeps data/ and .env out."""
    if not (config.ROOT / ".git").exists():
        log_line("push skipped: not a git repository (run git init first)")
        return False

    status = git("status", "--porcelain")
    if not status.stdout.strip():
        log_line("push skipped: nothing changed")
        return False

    for target in ("reports", "krxflow", "scripts", "*.py",
                   "README.md", "requirements.txt", ".gitignore"):
        if target.startswith("*") or (config.ROOT / target).exists():
            git("add", target)

    # Belt and braces: never let credentials or data through, even if
    # .gitignore were edited by accident.
    staged = git("diff", "--cached", "--name-only").stdout.split()
    leaked = [f for f in staged
              if f.startswith("data/") or f.endswith(".env") or f == ".env"]
    if leaked:
        git("reset")
        log_line(f"push ABORTED: refused to commit {leaked}")
        return False

    stamp = dt.date.today().isoformat()
    commit = git("commit", "-m", f"Update validation report {stamp}")
    if commit.returncode != 0 and "nothing to commit" not in commit.stdout:
        log_line(f"commit failed: {commit.stdout.strip()[:200]}")
        return False

    pushed = git("push")
    if pushed.returncode != 0:
        log_line(f"push failed: {pushed.stderr.strip()[:200]}")
        return False

    log_line(f"pushed report for {stamp}")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--with-market", action="store_true",
                   help="also collect prices / volume / market cap")
    p.add_argument("--lookback", type=int, default=30,
                   help="how many calendar days back to check for gaps")
    p.add_argument("--markets", default=",".join(config.MARKETS))
    p.add_argument("--report", action="store_true",
                   help="regenerate reports/validation.md")
    p.add_argument("--push", action="store_true",
                   help="git commit and push the report (implies --report)")
    p.add_argument("--horizons", default="1,5,20")
    args = p.parse_args()

    if args.push:
        args.report = True

    markets = [m.strip().upper() for m in args.markets.split(",") if m.strip()]
    stores = ["foreign_ownership"] + (["market"] if args.with_market else [])

    config.ensure_dirs()
    log_line(f"--- run start (stores={','.join(stores)}) ---")

    if not collect.login():
        log_line("ABORT: KRX login failed")
        return 1

    try:
        collect_missing(markets, stores, args.lookback)
    except Exception as err:  # noqa: BLE001
        log_line(f"ABORT: collection failed — {type(err).__name__}: {err}")
        return 1

    if args.report:
        write_report(["--horizons", args.horizons])

    if args.push:
        push_report()

    log_line("--- run end ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
