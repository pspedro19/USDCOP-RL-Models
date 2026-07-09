"""BTC/USDT derivatives ingestion — Binance USDT-M perp (funding / OI / long-short / basis).

The BTC science stack (`src/btc_strategy/`) is 100% price today. In crypto the edge is mostly
**non-price**: perp funding, open interest, positioning. Migration 052 designed
`crypto_derivatives_daily` but nothing filled it. This is that extractor.

**Honest API reality (drives what is/ isn't backtesteable):**
  * FUNDING rate — `fapi/v1/fundingRate` (8h) has **deep history (~2019-09 -> now)** -> the ONLY
    derivatives signal we can backtest over 2018-2026. Carries `markPrice` per settle.
  * BASIS — derived from funding-history `markPrice` vs Binance **spot** daily close (both deep).
  * OPEN INTEREST / LONG-SHORT / TAKER — `futures/data/*` return **only the last ~30 days** ->
    FORWARD-ONLY: we start accruing them now; they are NOT in the historical backtest yet.
  * LIQUIDATIONS — no public REST (deprecated) -> left NULL (Fase 3: WS `!forceOrder@arr`).

All endpoints are **public, no API key** (keys in `.env` only raise the rate-limit). Writes to
`crypto_derivatives_daily` (PK (date, symbol)) via idempotent UPSERT + a file-driven seed parquet
`seeds/latest/btcusdt_derivatives_daily.parquet`. Degrades gracefully if the DB is down (seed still
written). Anti-leakage: `date` = event day (UTC), `published_at` = availability; feature code
shift(1)s so the backtest never sees future funding.

Usage:
    python scripts/data/ingest_btc_derivatives.py                 # funding deep + forward 30d + DB
    python scripts/data/ingest_btc_derivatives.py --no-db         # seed only
    python scripts/data/ingest_btc_derivatives.py --no-forward    # funding/basis history only
    python scripts/data/ingest_btc_derivatives.py --start 2019-09-01

Contract: CTR-L0-CRYPTO-001 (derivatives variant) · pairs with ingest_btc_ohlcv.py
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[2]
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("btc_deriv")

UTC = ZoneInfo("UTC")
FAPI = "https://fapi.binance.com"
SPOT_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_SYMBOL = "BTCUSDT"
FUNDING_GENESIS_MS = 1567900800000  # 2019-09-08 00:00 UTC (BTCUSDT perp funding start)
SPOT_GENESIS_MS = 1502928000000     # 2017-08-17 (spot; for basis join)
SYMBOL = "BTC/USDT"                  # canonical AssetProfile symbol
SEED_REL = "seeds/latest/btcusdt_derivatives_daily.parquet"
Z_WINDOW = 30                        # rolling window (days) for funding_zscore
RATE_DELAY = 0.35

# crypto_derivatives_daily columns we populate (migration 052).
DERIV_COLS = ["date", "symbol", "funding_rate", "funding_zscore", "open_interest",
              "basis_annualized", "long_short_ratio", "liquidations_usd",
              "source", "published_at"]


# ------------------------------------------------------------------ env / db (mirror ingest_btc_ohlcv)
def _load_env() -> None:
    envf = REPO / ".env"
    if not envf.exists():
        return
    for line in envf.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _headers() -> dict:
    """Public endpoints need no auth; if a key is present we send it (higher rate-limit only)."""
    key = os.environ.get("BINANCE_API_KEY")
    return {"X-MBX-APIKEY": key} if key else {}


def _db_conn():
    import psycopg2
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        dbname=os.environ.get("POSTGRES_DB", "usdcop_trading"),
        user=os.environ.get("POSTGRES_USER", "admin"),
        password=os.environ.get("POSTGRES_PASSWORD", "admin123"),
        connect_timeout=6,
    )


def _get(url: str, params: dict, *, label: str, retries: int = 3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=_headers(), timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning("  [%s] attempt %d failed: %s", label, attempt + 1, e)
            _time.sleep(1.0 + attempt)
    return None


# ------------------------------------------------------------------ FUNDING (deep history)
def fetch_funding(start_ms: int) -> pd.DataFrame:
    """Paginate funding history FORWARD. Returns 8h rows: [fundingTime(UTC), funding_rate, mark]."""
    frames: list[pd.DataFrame] = []
    cur = start_ms
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    for call in range(400):
        rows = _get(f"{FAPI}/fapi/v1/fundingRate",
                    {"symbol": BINANCE_SYMBOL, "startTime": cur, "limit": 1000},
                    label="funding")
        if not rows:
            break
        df = pd.DataFrame([{
            "ts": pd.Timestamp(int(k["fundingTime"]), unit="ms", tz="UTC"),
            "funding_rate": float(k["fundingRate"]),
            "mark": float(k["markPrice"]) if k.get("markPrice") not in (None, "", "0") else np.nan,
        } for k in rows])
        frames.append(df)
        last = int(rows[-1]["fundingTime"])
        log.info("  [funding] call %d: %d rows, latest=%s", call + 1, len(rows), df["ts"].max().date())
        if len(rows) < 1000 or last >= now_ms:
            break
        cur = last + 1
        _time.sleep(RATE_DELAY)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def fetch_spot_daily(start_ms: int) -> pd.DataFrame:
    """Binance spot daily close (for basis). openTime is UTC 00:00 -> the canonical decision day."""
    frames: list[pd.DataFrame] = []
    cur = start_ms
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    for call in range(60):
        rows = _get(SPOT_KLINES,
                    {"symbol": BINANCE_SYMBOL, "interval": "1d", "startTime": cur, "limit": 1000},
                    label="spot")
        if not rows:
            break
        df = pd.DataFrame([{
            "date": pd.Timestamp(int(k[0]), unit="ms", tz="UTC").date(),
            "spot_close": float(k[4]),
        } for k in rows])
        frames.append(df)
        last = int(rows[-1][0])
        if len(rows) < 1000 or last >= now_ms:
            break
        cur = last + 86400000
        _time.sleep(RATE_DELAY)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("date").sort_values("date").reset_index(drop=True)


def daily_from_funding(fund: pd.DataFrame, spot: pd.DataFrame) -> pd.DataFrame:
    """Resample 8h funding to daily UTC and derive funding_zscore + basis_annualized.

    funding_rate(day) = SUM of the day's fundings (~3 x 8h) = the day's realized carry.
    basis_annualized  = (mean daily mark / spot_close - 1) * 365  (perp premium annualized proxy).
    funding_zscore    = rolling-Z_WINDOW z of daily funding_rate.
    """
    if fund.empty:
        return pd.DataFrame()
    f = fund.copy()
    f["date"] = f["ts"].dt.tz_convert(UTC).dt.date
    g = f.groupby("date").agg(funding_rate=("funding_rate", "sum"),
                              mark=("mark", "mean")).reset_index()
    if not spot.empty:
        g = g.merge(spot, on="date", how="left")
        g["basis_annualized"] = (g["mark"] / g["spot_close"] - 1.0) * 365.0
    else:
        g["basis_annualized"] = np.nan
    mu = g["funding_rate"].rolling(Z_WINDOW, min_periods=max(5, Z_WINDOW // 3)).mean()
    sd = g["funding_rate"].rolling(Z_WINDOW, min_periods=max(5, Z_WINDOW // 3)).std(ddof=0)
    g["funding_zscore"] = ((g["funding_rate"] - mu) / sd.replace(0, np.nan))
    return g[["date", "funding_rate", "funding_zscore", "basis_annualized"]]


# ------------------------------------------------------------------ FORWARD-ONLY (last ~30 days)
def _ts_to_date_map(rows, key: str) -> dict:
    out: dict = {}
    for k in rows or []:
        d = pd.Timestamp(int(k["timestamp"]), unit="ms", tz="UTC").date()
        try:
            out[d] = float(k[key])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def fetch_forward_metrics() -> pd.DataFrame:
    """OI (USD) + top-trader long/short ratio for the last ~30 days (endpoint hard cap)."""
    oi = _get(f"{FAPI}/futures/data/openInterestHist",
              {"symbol": BINANCE_SYMBOL, "period": "1d", "limit": 500}, label="oi")
    ls = _get(f"{FAPI}/futures/data/topLongShortAccountRatio",
              {"symbol": BINANCE_SYMBOL, "period": "1d", "limit": 500}, label="lsr")
    oi_map = _ts_to_date_map(oi, "sumOpenInterestValue")
    ls_map = _ts_to_date_map(ls, "longShortRatio")
    dates = sorted(set(oi_map) | set(ls_map))
    if not dates:
        return pd.DataFrame()
    return pd.DataFrame([{"date": d,
                          "open_interest": oi_map.get(d),
                          "long_short_ratio": ls_map.get(d)} for d in dates])


# ------------------------------------------------------------------ persistence
def _write_seed(df: pd.DataFrame) -> None:
    path = REPO / SEED_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("  seed -> %s (%d rows)", SEED_REL, len(df))


def _upsert(conn, df: pd.DataFrame) -> int:
    from psycopg2.extras import execute_values
    if df.empty:
        return 0
    vals = []
    for _, r in df.iterrows():
        def _f(x):
            return None if pd.isna(x) else float(x)
        pub = r["published_at"].to_pydatetime() if pd.notna(r["published_at"]) else None
        vals.append((r["date"], SYMBOL, _f(r["funding_rate"]), _f(r["funding_zscore"]),
                     _f(r["open_interest"]), _f(r["basis_annualized"]), _f(r["long_short_ratio"]),
                     _f(r.get("liquidations_usd")), r.get("source", "binance"), pub))
    cur = conn.cursor()
    try:
        execute_values(cur, """
            INSERT INTO crypto_derivatives_daily
                (date, symbol, funding_rate, funding_zscore, open_interest, basis_annualized,
                 long_short_ratio, liquidations_usd, source, published_at)
            VALUES %s
            ON CONFLICT (date, symbol) DO UPDATE SET
                funding_rate     = COALESCE(EXCLUDED.funding_rate, crypto_derivatives_daily.funding_rate),
                funding_zscore   = COALESCE(EXCLUDED.funding_zscore, crypto_derivatives_daily.funding_zscore),
                open_interest    = COALESCE(EXCLUDED.open_interest, crypto_derivatives_daily.open_interest),
                basis_annualized = COALESCE(EXCLUDED.basis_annualized, crypto_derivatives_daily.basis_annualized),
                long_short_ratio = COALESCE(EXCLUDED.long_short_ratio, crypto_derivatives_daily.long_short_ratio),
                liquidations_usd = COALESCE(EXCLUDED.liquidations_usd, crypto_derivatives_daily.liquidations_usd),
                source           = EXCLUDED.source,
                published_at     = EXCLUDED.published_at,
                updated_at       = NOW()
        """, vals, page_size=1000)
        conn.commit()
        return len(vals)
    except Exception as e:
        conn.rollback()
        log.error("  UPSERT crypto_derivatives_daily failed: %s", e)
        raise
    finally:
        cur.close()


# ------------------------------------------------------------------ quality
def _audit(df: pd.DataFrame) -> dict:
    rep = {"rows": len(df)}
    if df.empty:
        rep["problems"] = ["EMPTY"]
        return rep
    rep.update(
        earliest=str(df["date"].min()), latest=str(df["date"].max()),
        funding_days=int(df["funding_rate"].notna().sum()),
        zscore_days=int(df["funding_zscore"].notna().sum()),
        basis_days=int(df["basis_annualized"].notna().sum()),
        oi_days=int(df["open_interest"].notna().sum()),
        lsr_days=int(df["long_short_ratio"].notna().sum()),
        dup_dates=int(df["date"].duplicated().sum()),
    )
    probs = []
    if rep["dup_dates"]:
        probs.append(f"{rep['dup_dates']} dup dates")
    if rep["funding_days"] < 100:
        probs.append("suspiciously few funding days")
    # funding rate sane band: |daily sum| < 0.02 (2% would be extreme)
    extreme = int((df["funding_rate"].abs() > 0.02).sum())
    if extreme:
        probs.append(f"{extreme} extreme funding days (|f|>2%)")
    rep["problems"] = probs
    return rep


# ------------------------------------------------------------------ main
def run(*, use_db: bool, forward: bool, start: str | None) -> dict:
    _load_env()
    log.info("=" * 70)
    log.info("INGEST %s derivatives (Binance USDT-M perp, public)", SYMBOL)
    log.info("=" * 70)
    start_ms = FUNDING_GENESIS_MS
    if start:
        start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)

    log.info("[1] Funding history (deep, backtesteable)")
    fund = fetch_funding(start_ms)
    log.info("[2] Spot daily close (for basis)")
    spot = fetch_spot_daily(max(SPOT_GENESIS_MS, start_ms))
    daily = daily_from_funding(fund, spot)
    log.info("  daily funding rows=%d", len(daily))

    if forward:
        log.info("[3] Forward-only metrics (OI + long/short, last ~30d)")
        fwd = fetch_forward_metrics()
        if not fwd.empty and not daily.empty:
            daily = daily.merge(fwd, on="date", how="outer").sort_values("date").reset_index(drop=True)
        elif not fwd.empty:
            daily = fwd
        log.info("  forward rows merged=%d", 0 if fwd is None else len(fwd))
    for c in ("open_interest", "long_short_ratio"):
        if c not in daily.columns:
            daily[c] = np.nan
    daily["liquidations_usd"] = np.nan  # Fase 3 (WS !forceOrder@arr)
    daily["source"] = "binance"
    # availability: funding for day D is known by ~16:00 UTC that day -> publish end-of-day D
    daily["published_at"] = daily["date"].apply(
        lambda d: pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=23, minutes=59))
    daily = daily.sort_values("date").reset_index(drop=True)

    rep = _audit(daily)
    log.info("  audit: %s", rep)
    summary = {"asset_id": "btcusdt", "symbol": SYMBOL, "audit": rep}

    if not daily.empty:
        seed_cols = ["date", "funding_rate", "funding_zscore", "open_interest", "basis_annualized",
                     "long_short_ratio", "liquidations_usd", "source", "published_at"]
        _write_seed(daily[seed_cols])

    if use_db and not daily.empty:
        log.info("[4] UPSERT -> crypto_derivatives_daily (idempotent, best-effort)")
        try:
            conn = _db_conn()
            n = _upsert(conn, daily)
            conn.close()
            log.info("  upserted %d rows (symbol=%s)", n, SYMBOL)
            summary["db_upserted"] = n
        except Exception as e:
            log.warning("  DB step skipped/failed (seed still written): %s", e)
            summary["db_error"] = str(e)

    log.info("DONE %s derivatives: %s", SYMBOL, {k: v for k, v in summary.items() if k != "audit"})
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="BTC/USDT derivatives ingestion (Binance public)")
    ap.add_argument("--no-db", action="store_true", help="seed only, skip DB UPSERT")
    ap.add_argument("--no-forward", action="store_true",
                    help="skip forward-only 30d metrics (OI/long-short); funding+basis history only")
    ap.add_argument("--start", type=str, default=None, help="funding history start date (YYYY-MM-DD)")
    a = ap.parse_args()
    run(use_db=not a.no_db, forward=not a.no_forward, start=a.start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
