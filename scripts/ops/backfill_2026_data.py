"""
Backfill 2026 Data: Daily OHLCV + Macro (DXY/WTI) + 5-min OHLCV
=================================================================

Standalone script (no Airflow/DB) that fetches missing data from APIs
and appends to existing parquet files for 2026 analysis.

Sources:
- TwelveData API: Daily + 5-min OHLCV for USD/COP
- FRED API: DXY (DTWEXBGS) + WTI (DCOILWTICO)

Usage:
    python scripts/backfill_2026_data.py
    python scripts/backfill_2026_data.py --start 2025-12-18 --end 2026-02-14
    python scripts/backfill_2026_data.py --skip-5min   # Skip 5-min fetch (slow)

@version 1.0.0
@date 2026-02-15
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_START = "2025-12-18"
DEFAULT_END = "2026-02-14"

DAILY_OHLCV_PATH = PROJECT_ROOT / "seeds" / "latest" / "usdcop_daily_ohlcv.parquet"
M5_OHLCV_PATH = PROJECT_ROOT / "seeds" / "latest" / "usdcop_m5_ohlcv.parquet"
MACRO_PATH = (
    PROJECT_ROOT / "data" / "pipeline" / "04_cleaning" / "output"
    / "MACRO_DAILY_CLEAN.parquet"
)

TWELVEDATA_BASE = "https://api.twelvedata.com"
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# API key rotation
TD_KEYS = []
for i in range(1, 9):
    key = os.getenv(f"TWELVEDATA_API_KEY_{i}")
    if key:
        TD_KEYS.append(key)
if not TD_KEYS:
    logger.error("No TwelveData API keys found in .env")
    sys.exit(1)

FRED_KEY = os.getenv("FRED_API_KEY", "")
_td_key_idx = 0


def _next_td_key() -> str:
    """Rotate through TwelveData API keys."""
    global _td_key_idx
    key = TD_KEYS[_td_key_idx % len(TD_KEYS)]
    _td_key_idx += 1
    return key


# =============================================================================
# TWELVEDATA: DAILY OHLCV
# =============================================================================

def fetch_daily_ohlcv(start: str, end: str) -> pd.DataFrame:
    """Fetch daily OHLCV for USD/COP from TwelveData."""
    logger.info(f"Fetching daily OHLCV: {start} → {end}")

    params = {
        "symbol": "USD/COP",
        "interval": "1day",
        "start_date": start,
        "end_date": end,
        "timezone": "America/Bogota",
        "outputsize": 5000,
        "apikey": _next_td_key(),
    }

    resp = requests.get(f"{TWELVEDATA_BASE}/time_series", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "values" not in data:
        logger.error(f"TwelveData error: {data.get('message', data)}")
        return pd.DataFrame()

    rows = []
    for v in data["values"]:
        rows.append({
            "time": pd.Timestamp(v["datetime"]),
            "open": float(v["open"]),
            "high": float(v["high"]),
            "low": float(v["low"]),
            "close": float(v["close"]),
            "volume": 0.0,
            "symbol": "USD/COP",
        })

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize("America/Bogota")
    df = df.sort_values("time").reset_index(drop=True)

    # Validate
    valid = (
        (df["close"] >= 3000) & (df["close"] <= 6000)
        & (df["high"] >= df["low"])
        & df[["open", "high", "low", "close"]].notna().all(axis=1)
    )
    n_invalid = (~valid).sum()
    if n_invalid > 0:
        logger.warning(f"  Dropping {n_invalid} invalid rows")
        df = df[valid].reset_index(drop=True)

    logger.info(f"  Fetched {len(df)} daily bars: {df['time'].min()} → {df['time'].max()}")
    return df


def append_daily_ohlcv(df_new: pd.DataFrame) -> None:
    """Append new daily bars to existing parquet."""
    if len(df_new) == 0:
        logger.info("No new daily bars to append")
        return

    if DAILY_OHLCV_PATH.exists():
        df_existing = pd.read_parquet(DAILY_OHLCV_PATH)
        logger.info(f"  Existing daily: {len(df_existing)} rows")

        # Ensure index is 'time'
        if df_existing.index.name == "time":
            df_existing = df_existing.reset_index()

        # Match schema
        if "time" in df_existing.columns:
            df_existing["time"] = pd.to_datetime(df_existing["time"])
            if df_existing["time"].dt.tz is None:
                df_existing["time"] = df_existing["time"].dt.tz_localize("America/Bogota")

        # Deduplicate by date (normalize to date for daily)
        existing_dates = set(df_existing["time"].dt.normalize().dt.date)
        new_dates = df_new["time"].dt.normalize().dt.date
        mask = ~new_dates.isin(existing_dates)
        df_append = df_new[mask].copy()

        if len(df_append) == 0:
            logger.info("  All daily bars already exist, nothing to append")
            return

        # Ensure same columns
        for col in ["volume", "symbol"]:
            if col not in df_existing.columns:
                df_existing[col] = 0.0 if col == "volume" else "USD/COP"

        common_cols = [c for c in df_existing.columns if c in df_append.columns]
        df_combined = pd.concat(
            [df_existing[common_cols], df_append[common_cols]],
            ignore_index=True,
        )
    else:
        df_combined = df_new.copy()

    df_combined = df_combined.sort_values("time").drop_duplicates(
        subset=["time"], keep="last"
    ).reset_index(drop=True)
    df_combined = df_combined.set_index("time")

    df_combined.to_parquet(DAILY_OHLCV_PATH)
    logger.info(f"  Daily OHLCV saved: {len(df_combined)} total rows → {DAILY_OHLCV_PATH.name}")


# =============================================================================
# TWELVEDATA: 5-MIN OHLCV
# =============================================================================

def fetch_5min_ohlcv_batch(start: str, end: str) -> pd.DataFrame:
    """Fetch 5-min OHLCV for USD/COP, batched by week to respect API limits."""
    logger.info(f"Fetching 5-min OHLCV: {start} → {end}")

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    all_dfs = []

    current = start_dt
    batch = 0
    while current < end_dt:
        batch_end = min(current + timedelta(days=7), end_dt)
        batch += 1

        params = {
            "symbol": "USD/COP",
            "interval": "5min",
            "start_date": current.strftime("%Y-%m-%d"),
            "end_date": batch_end.strftime("%Y-%m-%d"),
            "timezone": "America/Bogota",
            "outputsize": 5000,
            "apikey": _next_td_key(),
        }

        logger.info(
            f"  Batch {batch}: {current.date()} → {batch_end.date()} "
            f"(key #{_td_key_idx % len(TD_KEYS)})"
        )

        try:
            resp = requests.get(
                f"{TWELVEDATA_BASE}/time_series", params=params, timeout=60
            )
            resp.raise_for_status()
            data = resp.json()

            if "values" not in data:
                msg = data.get("message", str(data))
                if "429" in msg or "limit" in msg.lower():
                    logger.warning(f"  Rate limited, waiting 60s...")
                    time.sleep(60)
                    continue
                logger.warning(f"  No values: {msg}")
                current = batch_end
                continue

            rows = []
            for v in data["values"]:
                rows.append({
                    "time": pd.Timestamp(v["datetime"]),
                    "open": float(v["open"]),
                    "high": float(v["high"]),
                    "low": float(v["low"]),
                    "close": float(v["close"]),
                    "volume": 0.0,
                    "symbol": "USD/COP",
                })

            if rows:
                df_batch = pd.DataFrame(rows)
                df_batch["time"] = pd.to_datetime(df_batch["time"]).dt.tz_localize(
                    "America/Bogota"
                )
                # Filter session hours 8:00-12:55 COT
                hours = df_batch["time"].dt.hour
                df_batch = df_batch[(hours >= 8) & (hours <= 12)].copy()
                all_dfs.append(df_batch)
                logger.info(f"    Got {len(df_batch)} bars (session-filtered)")

        except requests.exceptions.RequestException as e:
            logger.warning(f"  Request failed: {e}")

        current = batch_end
        time.sleep(8)  # Rate limit: ~8 calls/min per key

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)

    # Validate
    valid = (
        (df["close"] >= 3000) & (df["close"] <= 6000)
        & (df["high"] >= df["low"])
        & df[["open", "high", "low", "close"]].notna().all(axis=1)
    )
    df = df[valid].reset_index(drop=True)

    logger.info(f"  Total 5-min bars: {len(df)}")
    return df


def append_5min_ohlcv(df_new: pd.DataFrame) -> None:
    """Append new 5-min bars to existing parquet."""
    if len(df_new) == 0:
        logger.info("No new 5-min bars to append")
        return

    if M5_OHLCV_PATH.exists():
        df_existing = pd.read_parquet(M5_OHLCV_PATH)
        logger.info(f"  Existing 5-min: {len(df_existing)} rows")

        if df_existing.index.name == "time":
            df_existing = df_existing.reset_index()

        if "time" in df_existing.columns:
            df_existing["time"] = pd.to_datetime(df_existing["time"])
            if df_existing["time"].dt.tz is None:
                df_existing["time"] = df_existing["time"].dt.tz_localize("America/Bogota")

        # Filter existing for USD/COP only to check overlap
        cop_existing = df_existing
        if "symbol" in df_existing.columns:
            cop_existing = df_existing[df_existing["symbol"] == "USD/COP"]

        existing_times = set(cop_existing["time"])
        mask = ~df_new["time"].isin(existing_times)
        df_append = df_new[mask].copy()

        if len(df_append) == 0:
            logger.info("  All 5-min bars already exist")
            return

        common_cols = [c for c in df_existing.columns if c in df_append.columns]
        df_combined = pd.concat(
            [df_existing[common_cols], df_append[common_cols]],
            ignore_index=True,
        )
    else:
        df_combined = df_new.copy()

    df_combined = df_combined.sort_values("time").drop_duplicates(
        subset=["time", "symbol"] if "symbol" in df_combined.columns else ["time"],
        keep="last",
    ).reset_index(drop=True)
    df_combined = df_combined.set_index("time")

    df_combined.to_parquet(M5_OHLCV_PATH)
    logger.info(f"  5-min OHLCV saved: {len(df_combined)} total rows → {M5_OHLCV_PATH.name}")


# =============================================================================
# FRED: DXY + WTI
# =============================================================================

def fetch_fred_series(series_id: str, start: str, end: str) -> pd.DataFrame:
    """Fetch a FRED series as a DataFrame with [date, value]."""
    if not FRED_KEY:
        logger.warning(f"No FRED_API_KEY, skipping {series_id}")
        return pd.DataFrame()

    params = {
        "series_id": series_id,
        "api_key": FRED_KEY,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }

    resp = requests.get(FRED_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for obs in data.get("observations", []):
        if obs["value"] != ".":
            rows.append({
                "date": pd.Timestamp(obs["date"]),
                "value": float(obs["value"]),
            })

    df = pd.DataFrame(rows)
    logger.info(f"  FRED {series_id}: {len(df)} observations")
    return df


def update_macro_data(start: str, end: str) -> None:
    """Fetch WTI from FRED and merge into macro parquet.

    NOTE: DXY is NOT fetched from FRED. The FRED series DTWEXBGS is the
    'Trade Weighted U.S. Dollar Index: Broad' (~96-130 range), which is a
    DIFFERENT index than the ICE DXY (~88-114 range) used by Investing.com.
    Overwriting Investing.com DXY with DTWEXBGS corrupts the macro data and
    causes the model to see a +3.5σ outlier, breaking all predictions.
    DXY should only be updated via the actualizador_hpc_v3.py scraper.
    """
    logger.info(f"Updating macro data: {start} → {end}")

    # Fetch wider range for forward fill
    wider_start = (pd.Timestamp(start) - timedelta(days=30)).strftime("%Y-%m-%d")

    # CRITICAL: Do NOT fetch DXY from FRED — DTWEXBGS is a different index!
    # The DXY column should only be updated by the Investing.com scraper.
    df_dxy = pd.DataFrame()  # Disabled: was fetch_fred_series("DTWEXBGS", ...)
    df_wti = fetch_fred_series("DCOILWTICO", wider_start, end)

    if len(df_dxy) == 0 and len(df_wti) == 0:
        logger.warning("No macro data fetched from FRED")
        return

    if not MACRO_PATH.exists():
        logger.error(f"Macro parquet not found: {MACRO_PATH}")
        return

    df_macro = pd.read_parquet(MACRO_PATH)
    logger.info(f"  Existing macro: {len(df_macro)} rows, index.name={df_macro.index.name}")

    # The macro parquet has a DatetimeIndex (possibly unnamed) with data columns
    # After reset_index, the date becomes a column named 'index' (if unnamed) or its name
    date_col = "__date__"
    df_macro = df_macro.reset_index()
    # The first column after reset_index is the date (was the index)
    actual_first = df_macro.columns[0]
    df_macro = df_macro.rename(columns={actual_first: date_col})
    df_macro[date_col] = pd.to_datetime(df_macro[date_col]).dt.tz_localize(None).dt.normalize()

    DXY_COL = "FXRT_INDEX_DXY_USA_D_DXY"
    WTI_COL = "COMM_OIL_WTI_GLB_D_WTI"

    existing_dates = set(df_macro[date_col])

    new_rows = []
    # Merge DXY
    if len(df_dxy) > 0:
        for _, row in df_dxy.iterrows():
            d = row["date"].normalize()
            if d in existing_dates:
                mask = df_macro[date_col] == d
                if mask.any() and DXY_COL in df_macro.columns:
                    df_macro.loc[mask, DXY_COL] = row["value"]
            else:
                new_rows.append({date_col: d, DXY_COL: row["value"]})

    # Merge WTI
    if len(df_wti) > 0:
        for _, row in df_wti.iterrows():
            d = row["date"].normalize()
            if d in existing_dates:
                mask = df_macro[date_col] == d
                if mask.any() and WTI_COL in df_macro.columns:
                    df_macro.loc[mask, WTI_COL] = row["value"]
            else:
                found = False
                for nr in new_rows:
                    if nr[date_col] == d:
                        nr[WTI_COL] = row["value"]
                        found = True
                        break
                if not found:
                    new_rows.append({date_col: d, WTI_COL: row["value"]})

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_macro = pd.concat([df_macro, df_new], ignore_index=True)
        logger.info(f"  Added {len(new_rows)} new date rows")

    df_macro = df_macro.sort_values(date_col).drop_duplicates(
        subset=[date_col], keep="last"
    ).reset_index(drop=True)

    # Forward fill DXY and WTI
    if DXY_COL in df_macro.columns:
        df_macro[DXY_COL] = df_macro[DXY_COL].ffill()
    if WTI_COL in df_macro.columns:
        df_macro[WTI_COL] = df_macro[WTI_COL].ffill()

    # Restore original index (unnamed DatetimeIndex)
    df_macro = df_macro.set_index(date_col)
    df_macro.index.name = None
    df_macro.to_parquet(MACRO_PATH)
    logger.info(f"  Macro saved: {len(df_macro)} rows → {MACRO_PATH.name}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backfill 2026 data from APIs")
    parser.add_argument("--start", default=DEFAULT_START, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=DEFAULT_END, help="End date (YYYY-MM-DD)")
    parser.add_argument("--skip-5min", action="store_true", help="Skip 5-min OHLCV fetch")
    parser.add_argument("--skip-daily", action="store_true", help="Skip daily OHLCV fetch")
    parser.add_argument("--skip-macro", action="store_true", help="Skip macro (FRED) fetch")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  BACKFILL 2026 DATA")
    logger.info(f"  Range: {args.start} → {args.end}")
    logger.info(f"  API keys: {len(TD_KEYS)} TwelveData, FRED={'yes' if FRED_KEY else 'no'}")
    logger.info("=" * 60)

    t0 = time.time()

    # Step 1: Daily OHLCV
    if not args.skip_daily:
        logger.info("\n--- Step 1: Daily OHLCV ---")
        df_daily = fetch_daily_ohlcv(args.start, args.end)
        append_daily_ohlcv(df_daily)
    else:
        logger.info("\n--- Step 1: Daily OHLCV (SKIPPED) ---")

    # Step 2: Macro (FRED)
    if not args.skip_macro:
        logger.info("\n--- Step 2: Macro (DXY + WTI from FRED) ---")
        update_macro_data(args.start, args.end)
    else:
        logger.info("\n--- Step 2: Macro (SKIPPED) ---")

    # Step 3: 5-min OHLCV
    if not args.skip_5min:
        logger.info("\n--- Step 3: 5-min OHLCV ---")
        df_m5 = fetch_5min_ohlcv_batch(args.start, args.end)
        append_5min_ohlcv(df_m5)
    else:
        logger.info("\n--- Step 3: 5-min OHLCV (SKIPPED) ---")

    # Verification
    logger.info("\n--- Verification ---")
    if DAILY_OHLCV_PATH.exists():
        df = pd.read_parquet(DAILY_OHLCV_PATH)
        if df.index.name == "time":
            logger.info(f"  Daily OHLCV: {len(df)} rows, max date = {df.index.max()}")
        else:
            logger.info(f"  Daily OHLCV: {len(df)} rows")

    if M5_OHLCV_PATH.exists() and not args.skip_5min:
        df = pd.read_parquet(M5_OHLCV_PATH)
        if "symbol" in df.columns:
            cop = df[df["symbol"] == "USD/COP"] if "symbol" in df.columns else df
        else:
            cop = df
        if df.index.name == "time":
            logger.info(f"  5-min OHLCV: {len(cop)} COP bars, max = {cop.index.max()}")
        else:
            logger.info(f"  5-min OHLCV: {len(cop)} COP bars")

    if MACRO_PATH.exists():
        df = pd.read_parquet(MACRO_PATH)
        logger.info(f"  Macro: {len(df)} rows, max date = {df.index.max()}")

    elapsed = time.time() - t0
    logger.info(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
