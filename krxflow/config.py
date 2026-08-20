"""Central configuration for the Korea foreign-ownership pipeline.

Credentials are read from environment variables ONLY. Never hardcode them here
and never commit a filled-in .env file.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- paths ----
# Project root = the directory containing this package.
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("KRXFLOW_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
FEATURE_DIR = DATA_DIR / "features"
LOG_DIR = DATA_DIR / "logs"

# Raw sub-stores. One parquet file per trading day per store.
RAW_FOREIGN = RAW_DIR / "foreign_ownership"
RAW_MARKET = RAW_DIR / "market"  # close / volume / value / market cap
RAW_INVESTOR = RAW_DIR / "investor_flow"  # net buying by investor type
RAW_SHORTING = RAW_DIR / "shorting"  # short balance and short volume

# Investor types to collect. 기관합계 is the control group: if domestic
# institutional flow shows the same persistence as foreign flow, then the
# effect is generic order-splitting rather than anything specific to foreign
# investors, and the premise of the project weakens.
#
# Full set KRX offers: 금융투자 / 보험 / 투신 / 사모 / 은행 / 기타금융 /
# 연기금 / 기관합계 / 기타법인 / 개인 / 외국인 / 기타외국인 / 전체
INVESTORS = ("외국인", "기관합계")

# ---------------------------------------------------------- credentials ----
KRX_ID = os.getenv("KRX_ID")
KRX_PW = os.getenv("KRX_PW")
DART_API_KEY = os.getenv("DART_API_KEY")

# ------------------------------------------------------------- universe ----
MARKETS = ("KOSPI", "KOSDAQ")  # KONEX excluded by default; universe scope is an open item

# ------------------------------------------------------------ throttling ----
# KRX will start refusing requests if hammered. These defaults are deliberately
# polite: a full-market snapshot is one request, so a 3-year backfill of two
# markets is ~1,500 requests.
REQUEST_SLEEP_SEC = float(os.getenv("KRXFLOW_SLEEP", "0.7"))
REQUEST_JITTER_SEC = 0.3
MAX_RETRIES = 4
BACKOFF_BASE_SEC = 3.0

# --------------------------------------------------------------- schema ----
# Raw column names as pykrx returns them -> our stored names.
#
# IMPORTANT: pykrx casts 지분율 and 한도소진률 to np.float16
# (pykrx/website/krx/market/wrap.py). At 55% ownership float16 resolution is
# 0.031 percentage points, which is LARGER than a typical one-day change in
# foreign ownership. Those two columns are therefore stored with a _krx_lossy
# suffix and MUST NOT be used to build the signal.
#
# 보유수량 / 상장주식수 / 한도수량 arrive as exact int64, so the usable
# ownership ratio is recomputed in float64 downstream.
FOREIGN_COLUMN_MAP = {
    "상장주식수": "shares_listed",
    "보유수량": "foreign_shares",
    "지분율": "foreign_pct_krx_lossy",
    "한도수량": "foreign_limit_shares",
    "한도소진률": "limit_exhaustion_pct_krx_lossy",
    "한도소진율": "limit_exhaustion_pct_krx_lossy",  # pykrx spells it both ways
}

INVESTOR_COLUMN_MAP = {
    "종목명": "name",
    "매도거래량": "sell_volume",
    "매수거래량": "buy_volume",
    "순매수거래량": "net_volume",
    "매도거래대금": "sell_value",
    "매수거래대금": "buy_value",
    "순매수거래대금": "net_value",
}

SHORTING_COLUMN_MAP = {
    "공매도잔고": "short_balance_shares",
    "공매도금액": "short_balance_value",
    "상장주식수": "shares_listed",
    "시가총액": "market_cap",
    "비중": "short_balance_pct_krx_lossy",
    "공매도": "short_volume",
    "매수": "total_volume",
}

MARKET_COLUMN_MAP = {
    "종가": "close",
    "시가": "open",
    "고가": "high",
    "저가": "low",
    "거래량": "volume",
    "거래대금": "value_traded",
    "시가총액": "market_cap",
    "상장주식수": "shares_listed",
    "등락률": "pct_change_krx_lossy",
}


def ensure_dirs() -> None:
    for d in (RAW_FOREIGN, RAW_MARKET, RAW_INVESTOR, RAW_SHORTING,
              FEATURE_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def credentials_status() -> dict[str, str]:
    """Human-readable, non-leaking view of which credentials are present."""

    def mask(value: str | None) -> str:
        if not value:
            return "NOT SET"
        if len(value) <= 4:
            return "set (****)"
        return f"set ({value[:2]}{'*' * (len(value) - 4)}{value[-2:]})"

    return {
        "KRX_ID": mask(KRX_ID),
        "KRX_PW": "set (********)" if KRX_PW else "NOT SET",
        "DART_API_KEY": mask(DART_API_KEY),
    }
