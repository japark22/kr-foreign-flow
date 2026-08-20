"""Fetch daily snapshots from KRX with polite throttling and retries."""
from __future__ import annotations

import os

import datetime as dt
import random
import time

import pandas as pd
import requests

from . import config

_last_request_at = 0.0

# --------------------------------------------------------------- timeouts ----
# pykrx issues its data requests with NO timeout (website/comm/webio.py lines
# 42, 50, 76, 84 call session.get/post without one; only the login calls in
# auth.py set timeout=15). A single stalled TCP connection to KRX therefore
# blocks forever: the process stays alive, burns no CPU, and never advances.
# The retry/backoff below cannot help, because no exception is ever raised.
#
# Observed in practice — a 4,000-day backfill stopped dead at day 410 with the
# process still running.
#
# Rather than fork pykrx, install a default timeout on every request that does
# not specify one. A hang now raises after the read timeout, which turns an
# indefinite stall into an ordinary retryable failure.
CONNECT_TIMEOUT = float(os.getenv("KRXFLOW_CONNECT_TIMEOUT", "10"))
READ_TIMEOUT = float(os.getenv("KRXFLOW_READ_TIMEOUT", "60"))

_patched = False


def install_default_timeout() -> None:
    """Idempotently give every requests call a default timeout."""
    global _patched
    if _patched:
        return

    original = requests.Session.request

    def with_timeout(self, method, url, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = (CONNECT_TIMEOUT, READ_TIMEOUT)
        return original(self, method, url, **kwargs)

    requests.Session.request = with_timeout
    _patched = True


install_default_timeout()


def _throttle() -> None:
    """Space out requests so KRX does not start rejecting us."""
    global _last_request_at
    wait = config.REQUEST_SLEEP_SEC + random.uniform(0, config.REQUEST_JITTER_SEC)
    elapsed = time.monotonic() - _last_request_at
    if elapsed < wait:
        time.sleep(wait - elapsed)
    _last_request_at = time.monotonic()


def _with_retry(fn, *args, what: str = "request", **kwargs):
    last_err: Exception | None = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        _throttle()
        try:
            return fn(*args, **kwargs)
        except Exception as err:  # noqa: BLE001 - pykrx raises a grab-bag of types
            last_err = err
            if attempt == config.MAX_RETRIES:
                break
            delay = config.BACKOFF_BASE_SEC * (2 ** (attempt - 1))
            print(f"    ! {what} failed ({type(err).__name__}: {err}). "
                  f"retry {attempt}/{config.MAX_RETRIES - 1} in {delay:.0f}s")
            time.sleep(delay)
    raise RuntimeError(f"{what} failed after {config.MAX_RETRIES} attempts") from last_err


def login() -> bool:
    """Establish the authenticated KRX session pykrx 1.2.8+ requires.

    Returns True on success. Prints a specific diagnosis on failure rather
    than a bare exception, because almost every first-run problem lands here.
    """
    from pykrx.website.comm import auth

    if not (config.KRX_ID and config.KRX_PW):
        print("KRX_ID / KRX_PW are not set in the environment.")
        print("  export KRX_ID='your-krx-id'")
        print("  export KRX_PW='your-krx-password'")
        return False

    session = auth.build_krx_session(config.KRX_ID, config.KRX_PW)
    return session is not None


def fetch_foreign_ownership(date: str, market: str) -> pd.DataFrame:
    """All-ticker foreign-ownership snapshot for one trading day.

    Source: KRX [12023] 외국인보유량(개별종목) - 전종목.

    Returns a tidy frame with exact integer share counts. The percentage
    columns KRX/pykrx hand back are float16 and are carried through only as
    `*_krx_lossy`; `foreign_pct` is recomputed here in float64.
    """
    from pykrx import stock

    raw = _with_retry(
        stock.get_exhaustion_rates_of_foreign_investment_by_ticker,
        date,
        market,
        False,
        what=f"foreign ownership {market} {date}",
    )

    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.rename(columns=config.FOREIGN_COLUMN_MAP).reset_index()
    df = df.rename(columns={"티커": "ticker", "index": "ticker"})

    # Keep the lossy columns as float32 so parquet round-trips cleanly, but
    # keep the name honest about what they are.
    for col in ("foreign_pct_krx_lossy", "limit_exhaustion_pct_krx_lossy"):
        if col in df.columns:
            df[col] = df[col].astype("float32")

    for col in ("shares_listed", "foreign_shares", "foreign_limit_shares"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # The number we actually build the signal from. Kept in float64 (not the
    # nullable dtype) so downstream maths and nlargest behave predictably.
    shares = df["shares_listed"].astype("float64")
    held = df["foreign_shares"].astype("float64")
    df["foreign_pct"] = (held / shares.where(shares > 0)) * 100.0
    df["foreign_pct"] = df["foreign_pct"].astype("float64")

    df.insert(0, "trade_date", pd.to_datetime(date, format="%Y%m%d").date())
    df.insert(1, "market", market)
    df["fetched_at_utc"] = dt.datetime.now(dt.timezone.utc)
    df["source"] = "KRX/12023"

    return df.sort_values("ticker").reset_index(drop=True)


def fetch_investor_flow(date: str, market: str,
                        investors: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Net buying per ticker, per investor type, for one trading day.

    Source: KRX 투자자별 순매수. This is the FLOW counterpart to the ownership
    STOCK measure, and it serves two purposes the ownership data cannot:

      1. CONTROL GROUP. 기관합계 (domestic institutions) lets us ask whether
         persistence is specific to foreign investors or is simply what large
         orders look like when they are split over several days. If the two
         curves match, "foreign" is not the edge — order splitting is.

      2. CROSS-CHECK. Δ(holdings) and reported net buying should broadly agree.
         Where they diverge, something is off — Korean trader-type data has a
         known misclassification issue for orders routed through domestic
         brokers, and this is how we would see it.

    NOTE: KRX documents this endpoint as returning 순매수 상위종목 ("top net
    purchase stocks"). Whether it returns the full cross-section or a truncated
    top-N is not documented. `01_backfill.py` reports the row count per day so
    truncation would be visible immediately — if counts sit at a round number
    like 100 or 500, the data is truncated and unusable as a control group.
    """
    from pykrx import stock

    investors = investors or config.INVESTORS
    frames = []

    for investor in investors:
        raw = _with_retry(
            stock.get_market_net_purchases_of_equities_by_ticker,
            date, date, market, investor,
            what=f"investor flow {investor} {market} {date}",
        )
        if raw is None or raw.empty:
            continue

        df = raw.rename(columns=config.INVESTOR_COLUMN_MAP).reset_index()
        df = df.rename(columns={"티커": "ticker", "index": "ticker"})
        df["investor"] = investor
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    for col in ("sell_volume", "buy_volume", "net_volume",
                "sell_value", "buy_value", "net_value"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    out.insert(0, "trade_date", pd.to_datetime(date, format="%Y%m%d").date())
    out.insert(1, "market", market)
    out["fetched_at_utc"] = dt.datetime.now(dt.timezone.utc)
    out["source"] = "KRX/investor-net-purchase"

    return out.sort_values(["investor", "ticker"]).reset_index(drop=True)


def fetch_shorting(date: str, market: str) -> pd.DataFrame:
    """Short-sale balance and short volume per ticker for one trading day.

    Why this matters here: foreign investors account for the large majority of
    Korean short selling, so a rise in foreign holdings may be short covering
    rather than fresh conviction. Without this series the two are
    indistinguishable. Korea's short-selling bans also make useful natural
    experiments.
    """
    from pykrx import stock

    balance = _with_retry(
        stock.get_shorting_balance_by_ticker,
        date, market,
        what=f"short balance {market} {date}",
    )
    volume = _with_retry(
        stock.get_shorting_volume_by_ticker,
        date, market,
        what=f"short volume {market} {date}",
    )

    if (balance is None or balance.empty) and (volume is None or volume.empty):
        return pd.DataFrame()

    parts = []
    if balance is not None and not balance.empty:
        parts.append(balance.rename(columns=config.SHORTING_COLUMN_MAP))
    if volume is not None and not volume.empty:
        vol = volume.rename(columns=config.SHORTING_COLUMN_MAP)
        # Drop names that collide with the balance frame.
        vol = vol[[c for c in vol.columns
                   if not parts or c not in parts[0].columns]]
        parts.append(vol)

    joined = parts[0]
    for extra in parts[1:]:
        joined = joined.join(extra, how="outer")

    df = joined.reset_index().rename(columns={"티커": "ticker", "index": "ticker"})
    df.insert(0, "trade_date", pd.to_datetime(date, format="%Y%m%d").date())
    df.insert(1, "market", market)
    df["fetched_at_utc"] = dt.datetime.now(dt.timezone.utc)
    df["source"] = "KRX/shorting"

    return df.sort_values("ticker").reset_index(drop=True)


def fetch_market_snapshot(date: str, market: str) -> pd.DataFrame:
    """Close / volume / traded value / market cap for one trading day.

    Needed for the backtest's return series and for liquidity normalisation
    (the ADV denominator in the signal definition).
    """
    from pykrx import stock

    cap = _with_retry(
        stock.get_market_cap_by_ticker,
        date,
        market,
        what=f"market cap {market} {date}",
    )
    ohlcv = _with_retry(
        stock.get_market_ohlcv_by_ticker,
        date,
        market,
        what=f"ohlcv {market} {date}",
    )

    if cap is None or cap.empty:
        return pd.DataFrame()

    cap = cap.rename(columns=config.MARKET_COLUMN_MAP)
    if ohlcv is not None and not ohlcv.empty:
        ohlcv = ohlcv.rename(columns=config.MARKET_COLUMN_MAP)
        keep = [c for c in ("open", "high", "low") if c in ohlcv.columns]
        cap = cap.join(ohlcv[keep], how="left")

    df = cap.reset_index().rename(columns={"티커": "ticker", "index": "ticker"})
    df.insert(0, "trade_date", pd.to_datetime(date, format="%Y%m%d").date())
    df.insert(1, "market", market)
    df["fetched_at_utc"] = dt.datetime.now(dt.timezone.utc)
    df["source"] = "KRX/12021+12001"

    return df.sort_values("ticker").reset_index(drop=True)
