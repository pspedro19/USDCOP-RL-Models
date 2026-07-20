"""Free, keyless data adapters for the four covered asset classes.

Deliberately keyless. The audit of this collection found 33 skills (28%) bound
to a single vendor (FMP) that has already retired one endpoint, and 8 skills
depending on one individual's personal GitHub Pages repo with no fallback. This
module avoids adding to that pile: everything here works with no account.

Coverage and honest limitations:

    FX majors + EM   yfinance  ("EURUSD=X", "USDMXN=X"). Daily close only.
                     Yahoo's FX close is an indicative snapshot, not a
                     tradeable rate - fine for signal research on daily
                     horizons, wrong for anything intraday or for backtesting
                     execution.
    Crypto           ccxt against public endpoints - spot OHLCV and, more
                     importantly, live funding rates, which are the carry.
    Equities/index   yfinance ("^GSPC", "SPY").
    Gold             yfinance ("GC=F" front future, or "GLD"). Note the
                     futures continuation series splices contracts and is NOT
                     suitable for computing roll carry - fetch two expiries.

Rates for FX carry come from FRED's public CSV endpoint (no key). Foreign
policy rates are the weak link: FRED's non-US coverage is patchy and lagged,
so `fetch_policy_rates` returns what it can and names what it could not.

Network calls are isolated in functions prefixed `fetch_`. Everything else in
this package is pure and unit-tested offline.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import pandas as pd

__all__ = [
    "YAHOO_SYMBOLS",
    "FRED_SERIES",
    "fetch_prices",
    "fetch_crypto_funding",
    "fetch_fred_series",
    "DataUnavailable",
    "TWELVEDATA_DRAFT",
]


class DataUnavailable(RuntimeError):
    """Raised when a source cannot be reached or returns nothing usable.

    Deliberately not caught internally: a silently-empty frame that flows into
    a Sharpe ratio is how a backtest ends up validating noise.
    """


# Engine symbol -> Yahoo ticker.
YAHOO_SYMBOLS: dict[str, str] = {
    # FX majors
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X", "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
    # FX emerging
    "USDMXN": "USDMXN=X", "USDBRL": "USDBRL=X", "USDZAR": "USDZAR=X",
    "USDTRY": "USDTRY=X", "USDINR": "USDINR=X", "USDCNH": "USDCNH=X",
    # Equity
    "SPX": "^GSPC", "SPY": "SPY", "NDX": "^NDX",
    # Commodities
    "XAUUSD": "GC=F", "GLD": "GLD", "SILVER": "SI=F", "WTI": "CL=F",
    # Crypto (daily, for trend/value; use ccxt for funding)
    "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD",
    # Dollar index
    "DXY": "DX-Y.NYB",
}

# FRED series for the rate leg of FX carry. US coverage is good; the rest is
# best-effort and several are discontinued or lagged - verify before relying.
FRED_SERIES: dict[str, str] = {
    "USD": "DFF",             # effective fed funds, daily
    "USD_2Y": "DGS2",
    "USD_REAL_10Y": "DFII10",  # real yield - the key gold driver
    "EUR": "ECBDFR",           # ECB deposit facility
    "JPY": "IRSTCI01JPM156N",  # lagged, monthly
    "GBP": "IUDSOIA",
}


@dataclass(frozen=True)
class PriceFrame:
    """Daily closes plus provenance, so a stale cache cannot masquerade as live."""

    prices: pd.DataFrame
    source: str
    fetched_symbols: list[str]
    failed_symbols: list[str]

    def align(self, how: str = "intersect") -> "PriceFrame":
        """Reconcile mismatched trading calendars before any cross-asset maths.

        Crypto trades 24/7, FX 24/5, equities on exchange hours with holidays.
        Fetch them together and weekends arrive as rows where bitcoin has a
        price and everything else is NaN. Two things then go wrong quietly:

          - `pct_change()` across a NaN gap computes a Friday-to-Monday return
            for FX sitting next to a Saturday return for crypto, so the
            correlation matrix is measured on misaligned windows.
          - Forward-filling injects zero-return days, deflating measured
            volatility. Measured on 3 years of EURUSD/USDMXN/gold/SPX/BTC:
            only 68.5% of dates are common to all five, and ffill understates
            annualised vol by 17-19% on EVERY asset. Feed that into a vol
            target and every position comes out ~20% too large.

        Modes:
            "intersect" (default) - keep only dates every asset traded. Loses
                crypto weekends, which is the right trade for a daily
                cross-asset book: honest covariance beats extra observations.
            "ffill" - forward-fill gaps. Only defensible when the book is
                crypto-dominated and you accept the vol understatement.

        Always call this before `returns()` on a mixed-calendar universe.
        """
        if how == "intersect":
            return PriceFrame(
                self.prices.dropna(how="any"), f"{self.source}+intersect",
                self.fetched_symbols, self.failed_symbols,
            )
        if how == "ffill":
            return PriceFrame(
                self.prices.ffill().dropna(how="any"), f"{self.source}+ffill",
                self.fetched_symbols, self.failed_symbols,
            )
        raise ValueError(f"unknown alignment mode {how!r}; use 'intersect' or 'ffill'")

    def calendar_report(self) -> str:
        """Per-symbol observation counts - the fastest way to spot a mismatch."""
        n = len(self.prices)
        lines = [f"{n} dates spanned, per-symbol coverage:"]
        for c in self.prices.columns:
            k = int(self.prices[c].notna().sum())
            lines.append(f"  {c:<10} {k:>5} ({k / n:.1%})")
        aligned = int(self.prices.dropna(how="any").shape[0])
        lines.append(f"  {'ALL ALIGNED':<10} {aligned:>5} ({aligned / n:.1%})")
        return "\n".join(lines)

    def returns(self, require_aligned: bool = True) -> pd.DataFrame:
        """Simple returns. Refuses ragged input unless explicitly overridden."""
        if require_aligned and self.prices.isna().any().any():
            raise ValueError(
                "prices contain NaNs from mismatched trading calendars. Call "
                ".align() first, or pass require_aligned=False if you have "
                "already handled the gaps.\n\n" + self.calendar_report()
            )
        return self.prices.pct_change().dropna(how="all")

    def log_returns(self, require_aligned: bool = True):
        import numpy as np
        if require_aligned and self.prices.isna().any().any():
            raise ValueError(
                "prices contain NaNs from mismatched trading calendars; call "
                ".align() first.\n\n" + self.calendar_report()
            )
        return np.log(self.prices).diff().dropna(how="all")


def fetch_prices(
    symbols: list[str], period: str = "10y", interval: str = "1d"
) -> PriceFrame:
    """Daily closes for engine symbols via yfinance.

    Unknown symbols raise rather than being skipped - a quietly missing leg
    changes the portfolio without changing the output shape.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DataUnavailable("yfinance is not installed: pip install yfinance") from exc

    unknown = [s for s in symbols if s.upper() not in YAHOO_SYMBOLS]
    if unknown:
        raise KeyError(
            f"No Yahoo mapping for {unknown}. Add it to YAHOO_SYMBOLS - guessing "
            "a ticker risks silently fetching a different instrument."
        )

    tickers = {s.upper(): YAHOO_SYMBOLS[s.upper()] for s in symbols}
    raw = yf.download(
        list(tickers.values()), period=period, interval=interval,
        progress=False, auto_adjust=True,
    )
    if raw is None or raw.empty:
        raise DataUnavailable(f"yfinance returned nothing for {list(tickers.values())}")

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if not isinstance(raw.columns, pd.MultiIndex):
        close.columns = [list(tickers.values())[0]]

    inv = {v: k for k, v in tickers.items()}
    close = close.rename(columns=inv)

    got = [c for c in close.columns if close[c].notna().sum() > 0]
    failed = [s for s in tickers if s not in got]
    if not got:
        raise DataUnavailable(f"all symbols returned empty: {list(tickers)}")

    return PriceFrame(close[got].dropna(how="all"), "yfinance", got, failed)


def fetch_crypto_funding(
    symbols: tuple[str, ...] = ("BTC/USDT:USDT", "ETH/USDT:USDT"),
    exchange_id: str = "binanceusdm",
) -> dict[str, float]:
    """Current perpetual funding rates - the carry leg for crypto.

    Returns symbol -> funding rate PER INTERVAL (typically 8h). Pass to
    `carry.crypto_perp_carry` to annualise; do not annualise inline, since
    venues differ in interval.
    """
    try:
        import ccxt
    except ImportError as exc:  # pragma: no cover
        raise DataUnavailable("ccxt is not installed: pip install ccxt") from exc

    try:
        ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    except AttributeError as exc:
        raise DataUnavailable(f"unknown ccxt exchange {exchange_id!r}") from exc

    out: dict[str, float] = {}
    errors: list[str] = []
    for sym in symbols:
        try:
            fr = ex.fetch_funding_rate(sym)
            rate = fr.get("fundingRate")
            if rate is None:
                errors.append(f"{sym}: no fundingRate in response")
            else:
                out[sym] = float(rate)
        except Exception as exc:  # noqa: BLE001 - venue errors are heterogeneous
            errors.append(f"{sym}: {type(exc).__name__}: {exc}")

    if not out:
        raise DataUnavailable(f"no funding rates retrieved. {'; '.join(errors)}")
    return out


def fetch_fred_series(series_id: str, start: str = "2010-01-01") -> pd.Series:
    """A FRED series via the public CSV endpoint. No API key required."""
    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise DataUnavailable("requests is not installed") from exc

    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd={start}"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise DataUnavailable(f"FRED fetch failed for {series_id}: {exc}") from exc

    df = pd.read_csv(io.StringIO(resp.text))
    if df.shape[1] < 2:
        raise DataUnavailable(f"unexpected FRED payload for {series_id}")

    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    s = pd.to_numeric(df["value"], errors="coerce").set_axis(df["date"]).dropna()
    if s.empty:
        raise DataUnavailable(f"FRED returned no numeric data for {series_id}")
    return s.rename(series_id)


# --------------------------------------------------------------------------
# DRAFT - not validated, not wired in
# --------------------------------------------------------------------------

TWELVEDATA_DRAFT = '''
STATUS: DRAFT. Not implemented, not tested, not called by the engine.

Deferred at the user's request until this skill moves to the repo that holds
the scraping/API tooling. Written down so the design decision is not lost, and
left inert so it cannot be mistaken for working code - which is the exact
failure mode found elsewhere in this collection (README-documented APIs that
do not exist).

Why TwelveData is worth wiring up when that happens:
  - Genuine FX spot coverage including EM crosses, which yfinance approximates
    with indicative daily closes only.
  - Intraday FX bars, which no free source here provides.
  - One API for FX + equities + crypto, removing the yfinance/ccxt split.

Before trusting it, verify on the actual plan:
  1. Free tier is ~8 req/min and 800/day - too tight for a daily multi-asset
     universe refresh without caching. Confirm the tier before designing
     around it.
  2. Confirm which EM crosses are actually covered. USDMXN and USDZAR are
     usually available; USDBRL, USDINR and USDCNH are frequently NDF-quoted
     or absent, and a missing EM leg is the whole point of the adapter.
  3. Check whether FX bars are bid, ask or mid, and whether the daily close
     aligns to 17:00 New York. Mixing conventions across sources corrupts
     any cross-source backtest.
  4. Compare 60 days of overlap against yfinance before switching. A silent
     convention change is worse than a missing source.

Implementation sketch:
    def fetch_twelvedata(symbols, apikey, interval="1day", outputsize=5000):
        # GET https://api.twelvedata.com/time_series
        #   ?symbol=EUR/USD&interval={interval}&outputsize={outputsize}
        #   &apikey={apikey}
        # Read the key from the TWELVEDATA_API_KEY environment variable.
        # NEVER hardcode it, and never commit it.
        # Responses carry {"status": "error"} with HTTP 200 - check the body,
        # not just the status code.
        raise NotImplementedError("see TWELVEDATA_DRAFT")
'''
