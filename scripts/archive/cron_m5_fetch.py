#!/usr/bin/env python3
"""
cron_m5_fetch.py — Fetch latest M5 OHLCV bars from TwelveData
==============================================================
Cron replacement for Airflow DAG: l0_ohlcv_realtime

Fetches 5-minute candles for USD/COP, USD/MXN, USD/BRL from TwelveData API.
Inserts into PostgreSQL usdcop_m5_ohlcv table via UPSERT.
All timestamps stored in America/Bogota (COT).

Schedule: Every 5 min, 08:00-13:00 COT (= 13:00-18:00 UTC), Mon-Fri

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0 (cron)
Date: 2026-02-17
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional

import psycopg2
from psycopg2.extras import execute_values
import pytz
import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cron_m5_fetch")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COT_TZ = pytz.timezone("America/Bogota")
UTC_TZ = pytz.UTC

MARKET_START_HOUR = 8   # 08:00 COT
MARKET_END_HOUR   = 13  # 12:55 COT (we allow up to 12:59)

# Symbol config: api_tz controls what we request from TwelveData
# BRL returns bad data when requesting America/Bogota, so we use UTC and convert
SYMBOL_CONFIG = {
    "USD/COP": {"api_tz": "America/Bogota", "needs_tz_convert": False},
    "USD/MXN": {"api_tz": "America/Bogota", "needs_tz_convert": False},
    "USD/BRL": {"api_tz": "UTC",            "needs_tz_convert": True},
}

# TwelveData API keys — rotate through pool
API_KEYS = [
    v for k, v in sorted(os.environ.items())
    if k.startswith("TWELVEDATA_API_KEY_") and v
]
if not API_KEYS:
    # Fallback to single key
    single = os.environ.get("TWELVEDATA_API_KEY_1") or os.environ.get("TWELVEDATA_API_KEY")
    if single:
        API_KEYS = [single]

_key_idx = 0

# Number of bars to fetch per call (covers gaps from missed runs)
OUTPUTSIZE = 12  # ~1 hour of 5min bars

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_next_api_key() -> Optional[str]:
    global _key_idx
    if not API_KEYS:
        return None
    key = API_KEYS[_key_idx % len(API_KEYS)]
    _key_idx += 1
    return key


def get_db_connection():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        database=os.environ.get("POSTGRES_DB", "usdcop_trading"),
        user=os.environ.get("POSTGRES_USER", "admin"),
        password=os.environ.get("POSTGRES_PASSWORD", "admin123"),
    )


def is_market_hours() -> bool:
    """Check if current COT time is within trading session."""
    now = datetime.now(COT_TZ)
    if now.weekday() >= 5:  # Saturday/Sunday
        return False
    if now.hour < MARKET_START_HOUR or now.hour >= MARKET_END_HOUR:
        return False
    return True


def is_bar_in_session(bar_time: datetime) -> bool:
    """Check if bar timestamp is within 08:00-12:55 COT, Mon-Fri."""
    if bar_time.weekday() >= 5:
        return False
    h, m = bar_time.hour, bar_time.minute
    if h < MARKET_START_HOUR:
        return False
    if h >= MARKET_END_HOUR:
        return False
    if h == (MARKET_END_HOUR - 1) and m > 55:
        return False
    return True


# ---------------------------------------------------------------------------
# Main: fetch one symbol
# ---------------------------------------------------------------------------

def fetch_symbol(symbol: str) -> dict:
    """Fetch M5 bars for one symbol from TwelveData and insert into DB."""
    cfg = SYMBOL_CONFIG[symbol]
    api_key = get_next_api_key()
    if not api_key:
        logger.error(f"[{symbol}] No TwelveData API key configured")
        return {"status": "error", "reason": "no_api_key"}

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "5min",
        "format": "JSON",
        "timezone": cfg["api_tz"],
        "apikey": api_key,
        "outputsize": OUTPUTSIZE,
    }

    try:
        logger.info(f"[{symbol}] Fetching {OUTPUTSIZE} bars (tz={cfg['api_tz']})")
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.error(f"[{symbol}] API request failed: {e}")
        return {"status": "error", "reason": str(e)}

    if "values" not in data:
        msg = data.get("message", "unknown error")
        logger.warning(f"[{symbol}] No values returned: {msg}")
        return {"status": "no_data", "message": msg}

    # Parse bars
    records = []
    for bar in data["values"]:
        bar_time = datetime.fromisoformat(bar["datetime"])

        # BRL: convert UTC -> COT
        if cfg["needs_tz_convert"]:
            if bar_time.tzinfo is None:
                bar_time = UTC_TZ.localize(bar_time)
            bar_time = bar_time.astimezone(COT_TZ)
        else:
            if bar_time.tzinfo is None:
                bar_time = COT_TZ.localize(bar_time)

        # Filter to session hours
        if not is_bar_in_session(bar_time):
            continue

        records.append((
            bar_time,
            symbol,
            float(bar["open"]),
            float(bar["high"]),
            float(bar["low"]),
            float(bar["close"]),
            float(bar.get("volume", 0)),
            "twelvedata_cron",
        ))

    if not records:
        logger.info(f"[{symbol}] No in-session bars after filtering")
        return {"status": "empty", "fetched": len(data["values"])}

    # UPSERT into DB
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        execute_values(
            cur,
            """
            INSERT INTO usdcop_m5_ohlcv
                (time, symbol, open, high, low, close, volume, source)
            VALUES %s
            ON CONFLICT (time, symbol) DO UPDATE SET
                volume  = EXCLUDED.volume,
                source  = EXCLUDED.source,
                updated_at = NOW()
            """,
            records,
        )
        conn.commit()
        logger.info(f"[{symbol}] Upserted {len(records)} bars")
        return {"status": "ok", "inserted": len(records)}
    except Exception as e:
        conn.rollback()
        logger.error(f"[{symbol}] DB insert failed: {e}")
        return {"status": "error", "reason": str(e)}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # NOTE: Market-hours gating is handled by the crontab schedule (*/5 8-12 COT).
    # Do NOT add an is_market_hours() guard here — if the system timezone is
    # misconfigured in the crontab, this guard silently drops ALL fetches.
    now_cot = datetime.now(COT_TZ)
    logger.info(f"Current time: {now_cot.strftime('%Y-%m-%d %H:%M %Z')} (weekday={now_cot.weekday()})")

    if not API_KEYS:
        logger.error("No TWELVEDATA_API_KEY_* environment variables found. Aborting.")
        sys.exit(1)

    logger.info(f"Starting M5 fetch — {len(API_KEYS)} API keys available")

    results = {}
    for symbol in SYMBOL_CONFIG:
        results[symbol] = fetch_symbol(symbol)

    # Summary
    ok = sum(1 for r in results.values() if r.get("status") == "ok")
    logger.info(f"Done: {ok}/{len(results)} symbols successful")
    for sym, res in results.items():
        logger.info(f"  {sym}: {res}")


if __name__ == "__main__":
    main()
