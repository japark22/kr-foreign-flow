"""OpenDART client — corporate actions and dividends.

Used for the denoising step: we need to know which changes in foreign holdings
were caused by corporate events rather than by anyone deciding to buy.

WHAT DART DOES AND DOES NOT GIVE US
-----------------------------------
`alotMatter` (배당에 관한 사항) reports dividend amounts per company per fiscal
year, and `stlm_dt` (결산기준일). It does **not** report 배당기준일 or an
ex-dividend date. There is no OpenDART endpoint that does.

So ex-dividend dates are *derived*, not fetched — see `krxflow/exdiv.py`.

RATE LIMIT
----------
OpenDART allows 20,000 calls per key per day. `corpCode` is one call for every
listed company; `company` is one call per company. Everything here is cached to
disk so a re-run costs nothing.
"""
from __future__ import annotations

import io
import json
import os
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd
import requests

from . import config

BASE = "https://opendart.fss.or.kr/api"
CACHE = config.DATA_DIR / "dart"

DAILY_CALL_LIMIT = 20_000
SLEEP_SEC = float(os.getenv("DART_SLEEP", "0.12"))

# 사업보고서 / 반기 / 1분기 / 3분기
REPRT_ANNUAL = "11011"

_calls_made = 0


class DartError(RuntimeError):
    pass


def _key() -> str:
    key = os.getenv("DART_API_KEY")
    if not key:
        raise DartError(
            "DART_API_KEY is not set.\n"
            "  1. get a key at https://opendart.fss.or.kr (개인, free)\n"
            "  2. put it in .env as DART_API_KEY=...\n"
            "  3. set -a && source .env && set +a"
        )
    return key.strip()


def _get(endpoint: str, **params) -> requests.Response:
    global _calls_made
    if _calls_made >= DAILY_CALL_LIMIT:
        raise DartError(f"Hit the {DAILY_CALL_LIMIT:,}/day OpenDART limit. Resume tomorrow.")

    params["crtfc_key"] = _key()
    time.sleep(SLEEP_SEC)
    resp = requests.get(f"{BASE}/{endpoint}", params=params, timeout=30)
    _calls_made += 1
    resp.raise_for_status()
    return resp


# DART status codes worth naming, so failures are readable.
STATUS_MESSAGES = {
    "010": "등록되지 않은 키입니다. 키를 확인하세요.",
    "011": "사용할 수 없는 키입니다. 오픈API에 등록되었으나 일시적으로 사용 중지된 키입니다.",
    "012": "접근할 수 없는 IP입니다.",
    "013": "조회된 데이터가 없습니다.",
    "020": "요청 제한을 초과했습니다 (일 20,000건).",
    "100": "필드의 부적절한 값입니다.",
    "800": "시스템 점검 중입니다.",
    "900": "정의되지 않은 오류입니다.",
    "901": "사용자 계정의 개인정보 보유기간이 만료되었습니다.",
}


def _check(payload: dict, context: str) -> None:
    status = payload.get("status")
    if status in (None, "000"):
        return
    if status == "013":  # no data is normal for many company-years
        return
    raise DartError(
        f"{context}: DART status {status} — "
        f"{STATUS_MESSAGES.get(status, payload.get('message', 'unknown'))}"
    )


# ------------------------------------------------------------ corp codes ----
def corp_codes(refresh: bool = False) -> pd.DataFrame:
    """Every DART-registered company: corp_code, corp_name, stock_code.

    One API call. The response is a zip containing a single XML file.
    Cached to disk; pass refresh=True to re-download (new listings appear here).
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / "corp_codes.parquet"

    if cached.exists() and not refresh:
        return pd.read_parquet(cached)

    resp = _get("corpCode.xml")

    if resp.content[:2] != b"PK":  # not a zip -> DART returned an error document
        try:
            _check(json.loads(resp.text), "corpCode")
        except json.JSONDecodeError:
            pass
        raise DartError(f"corpCode did not return a zip. First bytes: {resp.content[:200]!r}")

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = zf.namelist()[0]
        xml = zf.read(name)

    rows = []
    for item in ET.fromstring(xml).iter("list"):
        def text(tag: str) -> str:
            node = item.find(tag)
            return (node.text or "").strip() if node is not None else ""

        rows.append({
            "corp_code": text("corp_code"),
            "corp_name": text("corp_name"),
            "stock_code": text("stock_code"),
            "modify_date": text("modify_date"),
        })

    df = pd.DataFrame(rows)
    # Only listed companies have a stock_code; those are the ones we can join.
    df = df[df["stock_code"].str.len() == 6].reset_index(drop=True)
    df.to_parquet(cached, index=False)
    return df


def ticker_to_corp_code(refresh: bool = False) -> dict[str, str]:
    df = corp_codes(refresh=refresh)
    return dict(zip(df["stock_code"], df["corp_code"]))


# ---------------------------------------------------------------- company ----
def company(corp_code: str) -> dict:
    """기업개황 — includes acc_mt (결산월), which drives the ex-dividend date."""
    payload = _get("company.json", corp_code=corp_code).json()
    _check(payload, f"company {corp_code}")
    return payload


def fiscal_month_table(tickers: list[str], refresh: bool = False) -> pd.DataFrame:
    """Fiscal year-end month per ticker. One API call per company, cached.

    Most Korean companies close in December, and their ex-dividend date sits
    just before the year-end. The minority with other fiscal months need their
    own dates, which is the whole reason we fetch this.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / "fiscal_months.parquet"

    known = pd.read_parquet(cached) if (cached.exists() and not refresh) else \
        pd.DataFrame(columns=["ticker", "corp_code", "corp_name", "acc_mt"])

    mapping = ticker_to_corp_code()
    todo = [t for t in tickers if t in mapping and t not in set(known["ticker"])]

    if not todo:
        return known

    print(f"  fetching 결산월 for {len(todo):,} companies "
          f"(~{len(todo) * SLEEP_SEC / 60:.0f} min)")

    rows = []
    for i, ticker in enumerate(todo, 1):
        try:
            info = company(mapping[ticker])
        except (DartError, requests.RequestException) as err:
            print(f"    ! {ticker}: {err}")
            continue

        if info.get("status") == "013":
            continue

        rows.append({
            "ticker": ticker,
            "corp_code": mapping[ticker],
            "corp_name": info.get("corp_name", ""),
            "acc_mt": info.get("acc_mt", ""),
        })

        if i % 200 == 0:
            print(f"    {i:,}/{len(todo):,}")
            pd.concat([known, pd.DataFrame(rows)], ignore_index=True).to_parquet(
                cached, index=False)

    out = pd.concat([known, pd.DataFrame(rows)], ignore_index=True)
    out.to_parquet(cached, index=False)
    return out


# --------------------------------------------------------------- dividend ----
def dividends(corp_code: str, year: int, reprt_code: str = REPRT_ANNUAL) -> pd.DataFrame:
    """배당에 관한 사항 for one company-year. Empty frame when nothing filed.

    Note: gives amounts and 결산기준일 (stlm_dt), NOT 배당기준일.
    """
    payload = _get("alotMatter.json", corp_code=corp_code,
                   bsns_year=str(year), reprt_code=reprt_code).json()
    _check(payload, f"alotMatter {corp_code} {year}")

    items = payload.get("list") or []
    if not items:
        return pd.DataFrame()

    df = pd.DataFrame(items)
    df["bsns_year"] = year
    return df


def calls_made() -> int:
    return _calls_made
