#!/usr/bin/env python3
"""Step 15: collect earnings-announcement dates from OpenDART.

    python 15_earnings_events.py --check          # verify the key (1 call)
    python 15_earnings_events.py                  # collect since 2018-01
    python 15_earnings_events.py --start 202401   # shorter window

Two kinds of event, kept separate because they behave differently:

  provisional  잠정실적 공정공시 (I002) -- the market-moving release, filed
               close to the quarter end. Mostly larger companies file these.
  periodic     사업/반기/분기보고서 (A001/A002/A003) -- universal coverage,
               but filed weeks later and usually pre-empted by the above.

The study defines the event as rcept_dt (filing date). Filing TIME is not in
the API, so whether the market reacted on D or D+1 is ambiguous -- the study
enters positions at the close of D+1 for that reason.

Monthly results are cached in data/events/, so a stopped run resumes free.
OpenDART allows 20,000 calls/day; a full 2018-to-now run uses well under that.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "events"
BASE = "https://opendart.fss.or.kr/api/list.json"
SLEEP = 0.12
PAGE = 100

# pblntf_detail_ty codes; if DART rejects one (status 100/101), the collector
# falls back to the broad pblntf_ty and filters by report name locally.
KINDS = {
    "provisional": {"detail": "I002", "broad": "I",
                    "match": lambda nm: ("실적" in nm
                                         and ("잠정" in nm or "공정공시" in nm)
                                         and "전망" not in nm)},
    "periodic": {"detail": "A001,A002,A003", "broad": "A",
                 "match": lambda nm: ("분기보고서" in nm or "반기보고서" in nm
                                      or "사업보고서" in nm)},
}

calls = 0


def key() -> str:
    k = os.getenv("DART_API_KEY", "").strip()
    if not k:
        sys.exit("DART_API_KEY is not set. Load it first:  set -a; source .env; set +a")
    return k


def get(**params) -> dict:
    global calls
    time.sleep(SLEEP)
    calls += 1
    r = requests.get(BASE, params={"crtfc_key": key(), **params}, timeout=30)
    r.raise_for_status()
    return r.json()


def month_windows(start: str) -> list[tuple[str, str]]:
    """[(YYYYMMDD, YYYYMMDD)] per calendar month from start to today."""
    y, m = int(start[:4]), int(start[4:6])
    today = dt.date.today()
    out = []
    while (y, m) <= (today.year, today.month):
        first = dt.date(y, m, 1)
        last = (dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1))
        out.append((first.strftime("%Y%m%d"),
                    min(last, today).strftime("%Y%m%d")))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def fetch_window(bgn: str, end: str, kind: str) -> pd.DataFrame:
    spec = KINDS[kind]
    rows, page, use_detail = [], 1, True
    while True:
        params = {"bgn_de": bgn, "end_de": end, "page_no": page,
                  "page_count": PAGE}
        if use_detail:
            params["pblntf_detail_ty"] = spec["detail"]
        else:
            params["pblntf_ty"] = spec["broad"]
        d = get(**params)
        status = d.get("status")
        if status == "013":                      # nothing in this window
            break
        if status in ("100", "101") and use_detail and page == 1:
            use_detail = False                   # detail code rejected: go broad
            continue
        if status != "000":
            raise RuntimeError(f"DART status {status}: {d.get('message')}")
        for it in d.get("list", []):
            nm = (it.get("report_nm") or "").strip()
            if it.get("corp_cls") not in ("Y", "K"):
                continue
            if not spec["match"](nm):
                continue
            rows.append({
                "ticker": (it.get("stock_code") or "").strip(),
                "corp_code": it.get("corp_code", ""),
                "corp_name": it.get("corp_name", ""),
                "report_nm": nm,
                "rcept_no": it.get("rcept_no", ""),
                "rcept_dt": it.get("rcept_dt", ""),
                "kind": kind,
                "is_correction": ("정정" in nm),
            })
        if page >= int(d.get("total_page", 1)):
            break
        page += 1
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--start", default="201801", help="YYYYMM")
    a = ap.parse_args()

    if a.check:
        d = get(bgn_de="20240102", end_de="20240102", page_no=1, page_count=1)
        if d.get("status") in ("000", "013"):
            print("key works.")
            return 0
        sys.exit(f"key problem: status {d.get('status')} {d.get('message')}")

    OUT.mkdir(parents=True, exist_ok=True)
    this_month = dt.date.today().strftime("%Y%m")
    COLS = ["ticker", "corp_code", "corp_name", "report_nm", "rcept_no",
            "rcept_dt", "kind", "is_correction"]
    frames = []
    for bgn, end in month_windows(a.start):
        ym = bgn[:6]
        for kind in KINDS:
            cache = OUT / f"dart_{kind}_{ym}.parquet"
            legacy = OUT / f"dart_list_{ym}.parquet"
            if cache.exists() and ym != this_month:
                frames.append(pd.read_parquet(cache))
                continue
            if (kind == "periodic" and not cache.exists()
                    and legacy.exists() and ym != this_month):
                frames.append(pd.read_parquet(legacy))
                continue
            df = fetch_window(bgn, end, kind)
            if df.empty:
                df = pd.DataFrame(columns=COLS)
            df.to_parquet(cache, index=False)
            frames.append(df)
            print(f"  {ym} {kind}: {len(df):,} kept  (calls so far {calls:,})")

    ev = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    ev = ev[ev["ticker"].str.len() == 6]
    ev = ev.sort_values("rcept_no").drop_duplicates(
        subset=["ticker", "rcept_dt", "kind"], keep="last").reset_index(drop=True)
    dest = OUT / "earnings_events.parquet"
    ev.to_parquet(dest, index=False)

    print("\n  ================ summary ================")
    print(f"  events kept: {len(ev):,}  "
          f"({(~ev['is_correction']).sum():,} original, "
          f"{ev['is_correction'].sum():,} corrections)")
    for kind, n in ev["kind"].value_counts().items():
        print(f"    {kind:<12} {n:,}")
    yr = ev["rcept_dt"].str[:4]
    print("  by year:")
    for y, n in yr.value_counts().sort_index().items():
        print(f"    {y}  {n:,}")
    print(f"\n  wrote {dest.relative_to(ROOT)}")
    print(f"  API calls used: {calls:,} (limit 20,000/day)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
