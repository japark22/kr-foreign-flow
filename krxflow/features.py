"""Feature layer: turn raw ownership snapshots into the flow signal.

The signal, per the project spec:

    normalised foreign flow
      = (Δ foreign shares held − change caused by corporate actions)
        / free float          (free float unavailable -> shares outstanding)

    or / ADV, when market data has been collected.

Everything here reads the immutable raw layer and returns panels indexed by
trade_date with tickers as columns. Nothing here writes to the raw layer.
"""
from __future__ import annotations

import pandas as pd

from . import config, exdiv, rebalance, storage

# Tickers starting with '9' are foreign companies listed in Korea. Their
# foreign ownership sits pinned near 100% by construction and carries no flow
# information, so they are dropped from the research universe.
FOREIGN_LISTING_PREFIX = "9"


PANEL_COLUMNS = ["trade_date", "market", "ticker", "shares_listed",
                 "foreign_shares", "foreign_limit_shares", "foreign_pct"]


def load_panels(start: str | None = None, end: str | None = None) -> dict[str, pd.DataFrame]:
    """Raw ownership store -> wide panels (index=date, columns=ticker).

    Only the columns the signal needs are read, and each panel is released
    from the long frame as it is built, so peak memory stays near one panel
    rather than the whole 10-million-row store plus four copies of it.
    """
    df = storage.read_range("foreign_ownership", start, end, columns=PANEL_COLUMNS)
    if df.empty:
        raise RuntimeError("No foreign-ownership data stored. Run 01_backfill.py first.")

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    market = df.groupby("ticker", observed=True)["market"].last()

    def pivot(col: str) -> pd.DataFrame:
        # float64 throughout, deliberately. Share counts reach ~6e9 and float32
        # only resolves to ~1e3 at that magnitude, which would quantise the very
        # daily changes the signal is made of.
        wide = df.pivot_table(index="trade_date", columns="ticker",
                              values=col, aggfunc="last", observed=True)
        return wide.sort_index().astype("float64")

    panels = {
        "foreign_pct": pivot("foreign_pct"),
        "foreign_shares": pivot("foreign_shares"),
        "shares_listed": pivot("shares_listed"),
        "foreign_limit_shares": pivot("foreign_limit_shares"),
        "market": market,
    }

    del df  # the long frame is the memory hog; the panels are ~90 MB each
    return panels


def universe_mask(panels: dict, min_days: int = 60,
                  drop_foreign_listings: bool = True,
                  max_pinned_pct: float = 99.99) -> pd.DataFrame:
    """Boolean panel: True = ticker is usable on that date.

    Excludes, with reasons:
      - foreign-company listings (ticker starts with '9')
      - names pinned at ~100% foreign ownership (no flow can occur)
      - tickers with too little history to compute a change
      - dates where the ticker has no observation
    """
    pct = panels["foreign_pct"]
    ok = pct.notna()

    if drop_foreign_listings:
        foreign_listed = [c for c in pct.columns if str(c).startswith(FOREIGN_LISTING_PREFIX)]
        ok.loc[:, foreign_listed] = False

    # Pinned at the ceiling for its whole life -> structurally not tradable flow.
    pinned = (pct.min(skipna=True) >= max_pinned_pct)
    ok.loc[:, pinned[pinned].index] = False

    thin = ok.sum() < min_days
    ok.loc[:, thin[thin].index] = False

    return ok


def corporate_action_mask(panels: dict, tol: float = 1e-9) -> pd.DataFrame:
    """True where shares outstanding moved -> the Δ is not a trade.

    This is the crude proxy: rights issues, bonus issues, splits, conversions
    and buyback cancellations all move shares_listed. It is replaced by the
    precise OpenDART event list once DART_API_KEY is available.
    """
    shares = panels["shares_listed"]
    changed = shares.pct_change(fill_method=None).abs() > tol
    return changed.fillna(False).astype(bool)


def build_flow(panels: dict,
               denominator: str = "shares",
               adv: pd.DataFrame | None = None,
               scrub_corporate_actions: bool = True) -> pd.DataFrame:
    """Daily normalised foreign flow.

    Args:
        denominator: "shares" -> Δ held / shares outstanding (unit: fraction
                     of the company bought that day, x100 for pp)
                     "adv"    -> Δ held x close / average daily traded value
        adv: average daily traded value panel, required when denominator="adv".
    """
    held = panels["foreign_shares"]
    shares = panels["shares_listed"]

    d_held = held.diff()

    if scrub_corporate_actions:
        d_held = d_held.mask(corporate_action_mask(panels))

    if denominator == "shares":
        flow = d_held / shares.where(shares > 0)
    elif denominator == "adv":
        if adv is None:
            raise ValueError("denominator='adv' requires the adv panel")
        flow = d_held / adv.where(adv > 0)
    else:
        raise ValueError(f"unknown denominator {denominator!r}")

    return flow


def load_fiscal_months() -> pd.Series | None:
    """Cached 결산월 per ticker from OpenDART, or None if not collected yet."""
    cached = config.DATA_DIR / "dart" / "fiscal_months.parquet"
    if not cached.exists():
        return None
    df = pd.read_parquet(cached)
    if df.empty:
        return None
    return df.set_index("ticker")["acc_mt"]


def apply_filters(flow: pd.DataFrame, panels: dict,
                  drop_rebalance: bool = True,
                  rebalance_window: int = 2,
                  drop_exdiv: bool = True,
                  exdiv_before: int = 3,
                  exdiv_after: int = 2,
                  universe: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Blank out observations we do not trust.

    Returns (filtered_flow, report) where report counts what each filter
    removed — so the cost of every exclusion is visible rather than implied.
    """
    out = flow.copy()
    report: dict[str, int | str] = {}
    start = int(out.notna().to_numpy().sum())
    report["observations_in"] = start

    if universe is None:
        universe = universe_mask(panels)
    out = out.where(universe)
    after_universe = int(out.notna().to_numpy().sum())
    report["removed_universe"] = start - after_universe

    if drop_rebalance:
        mask = rebalance.rebalance_mask(out.index, window=rebalance_window)
        rows = mask.reindex(out.index).fillna(False).to_numpy()
        out.loc[rows, :] = float("nan")
        after_reb = int(out.notna().to_numpy().sum())
        report["removed_rebalance"] = after_universe - after_reb
    else:
        after_reb = after_universe
        report["removed_rebalance"] = 0

    if drop_exdiv:
        fiscal = load_fiscal_months()
        panel = pd.DataFrame(False, index=out.index, columns=out.columns)

        # Market-wide December window: applies to every ticker with no DART
        # record, and to all tickers when DART has not been collected at all.
        dec_rows = (exdiv.year_end_window(out.index, exdiv_before, exdiv_after)
                    .reindex(out.index).fillna(False).to_numpy())

        if fiscal is None:
            panel.loc[dec_rows, :] = True
        else:
            fiscal = fiscal[~fiscal.index.duplicated()]
            known = [t for t in out.columns if t in fiscal.index]
            unknown = [t for t in out.columns if t not in fiscal.index]

            if known:
                windows = exdiv.fiscal_month_windows(
                    out.index, fiscal.loc[known],
                    before=exdiv_before, after=exdiv_after)
                windows = windows.reindex(index=out.index, columns=known)
                panel.loc[:, known] = windows.to_numpy() == True  # noqa: E712
            if unknown:
                panel.loc[dec_rows, unknown] = True

        out = out.mask(panel)
        report["removed_exdiv"] = after_reb - int(out.notna().to_numpy().sum())
        report["exdiv_basis"] = exdiv.coverage_note(fiscal)
    else:
        report["removed_exdiv"] = 0
        report["exdiv_basis"] = "ex-dividend filter disabled"

    report["observations_out"] = int(out.notna().to_numpy().sum())
    return out, report


def cross_sectional_rank(panel: pd.DataFrame, min_names: int = 50) -> pd.DataFrame:
    """Rank each day's cross-section into [-0.5, +0.5].

    Ranking is what makes the signal comparable across days and immune to the
    scale differences between a 300bn-won large cap and a micro cap. Days with
    too few valid names are dropped rather than ranked on noise.
    """
    ranked = panel.rank(axis=1, pct=True) - 0.5
    thin = panel.notna().sum(axis=1) < min_names
    ranked.loc[thin] = pd.NA
    return ranked.astype("float64")


def load_investor_panels(start: str | None = None,
                         end: str | None = None) -> dict[str, pd.DataFrame]:
    """Net-buying panels keyed by investor type.

    Returns {investor: DataFrame(index=date, columns=ticker)} of net traded
    value, plus a "_counts" entry giving tickers per day per investor so that a
    truncated response is impossible to miss.
    """
    df = storage.read_range("investor_flow", start, end,
                            columns=["trade_date", "ticker", "investor", "net_value"])
    if df.empty:
        return {}

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    out: dict[str, pd.DataFrame] = {}

    for investor, group in df.groupby("investor", observed=True):
        out[str(investor)] = (group
                              .pivot_table(index="trade_date", columns="ticker",
                                           values="net_value", aggfunc="last",
                                           observed=True)
                              .sort_index().astype("float64"))

    out["_counts"] = (df.groupby(["investor", "trade_date"], observed=True)
                      .size().unstack(0))
    return out


def forward_returns(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Simple forward return over `horizon` trading days, aligned to date t."""
    return close.shift(-horizon) / close - 1.0
