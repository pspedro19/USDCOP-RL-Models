#!/usr/bin/env python3
"""
cron_macro_update.py — Daily macro variable update via Yahoo Finance
====================================================================
Cron replacement for Airflow DAG: l0_macro_update (daily variables only)

Updates DXY, VIX, WTI, UST10Y, DGS2, USDMXN, FedFunds from Yahoo Finance.
Writes to:
  1. PostgreSQL macro_indicators_daily (WIDE format: fecha + indicator columns)
  2. seeds/latest/macro_indicators_daily.parquet (append + dedupe)

The macro_indicators_daily table uses wide format with column naming convention:
  {category}_{subcategory}_{indicator}_{country}_{freq}_{short}

Key columns used by inference_macro_features view:
  fxrt_index_dxy_usa_d_dxy         -> DXY
  volt_vix_usa_d_vix               -> VIX
  comm_oil_wti_glb_d_wti           -> WTI
  finc_bond_yield10y_usa_d_ust10y  -> UST10Y
  finc_bond_yield2y_usa_d_dgs2     -> DGS2
  fxrt_spot_usdmxn_mex_d_usdmxn   -> USDMXN
  polr_fed_funds_usa_m_fedfunds    -> FedFunds

Schedule: Once daily at 07:00 COT (= 12:00 UTC), Mon-Fri

Author: Pedro @ Lean Tech Solutions
Version: 2.0.0 (wide-format fix)
Date: 2026-02-17
"""

import os
import sys
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cron_macro_update")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = PROJECT_ROOT / "seeds" / "latest" / "macro_indicators_daily.parquet"

# Yahoo Finance tickers -> DB column names (wide-format macro_indicators_daily)
YAHOO_TO_COLUMN = {
    "DX-Y.NYB":    "fxrt_index_dxy_usa_d_dxy",           # US Dollar Index
    "^VIX":        "volt_vix_usa_d_vix",                  # CBOE Volatility Index
    "CL=F":        "comm_oil_wti_glb_d_wti",              # WTI Crude Oil
    "^TNX":        "finc_bond_yield10y_usa_d_ust10y",     # US 10-Year Treasury
    "^IRX":        "finc_bond_yield2y_usa_d_dgs2",        # US 2-Year Treasury (approx)
    "USDMXN=X":   "fxrt_spot_usdmxn_mex_d_usdmxn",      # USD/MXN
}

# Lookback days for Yahoo Finance download (covers weekends + holidays)
LOOKBACK_DAYS = 30

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_db_connection():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        database=os.environ.get("POSTGRES_DB", "usdcop_trading"),
        user=os.environ.get("POSTGRES_USER", "admin"),
        password=os.environ.get("POSTGRES_PASSWORD", "admin123"),
    )


# ---------------------------------------------------------------------------
# Step 1: Fetch from Yahoo Finance
# ---------------------------------------------------------------------------

def fetch_yahoo_data() -> dict:
    """Download recent data for all macro variables from Yahoo Finance.

    Returns dict: db_column_name -> list of (date, close_value) tuples.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)

    all_data = {}

    for ticker, col_name in YAHOO_TO_COLUMN.items():
        try:
            logger.info(f"[Yahoo] Downloading {ticker} -> {col_name}")
            df = yf.download(
                ticker,
                start=str(start_date),
                end=str(end_date + timedelta(days=1)),
                progress=False,
                auto_adjust=True,
            )

            if df.empty:
                logger.warning(f"[Yahoo] No data for {ticker}")
                continue

            # Use Close column; flatten MultiIndex if present
            if hasattr(df.columns, 'levels'):
                df.columns = df.columns.get_level_values(0)

            records = []
            for idx, row in df.iterrows():
                dt = idx.date() if hasattr(idx, 'date') else idx
                val = float(row["Close"])
                records.append((dt, val))

            all_data[col_name] = records
            logger.info(f"[Yahoo] {col_name}: {len(records)} records ({records[0][0]} to {records[-1][0]})")

        except Exception as e:
            logger.error(f"[Yahoo] Failed for {ticker}: {e}")

    return all_data


# ---------------------------------------------------------------------------
# Step 2: Upsert into PostgreSQL (wide-format)
# ---------------------------------------------------------------------------

def upsert_to_db(all_data: dict):
    """Upsert macro data into the wide-format macro_indicators_daily table.

    For each date, we UPDATE the specific column(s) that have new data,
    or INSERT a new row if the date doesn't exist yet.
    """
    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # Collect all unique dates across all variables
        date_map: dict = {}  # date -> {col: value, ...}
        for col_name, records in all_data.items():
            for dt, val in records:
                if dt not in date_map:
                    date_map[dt] = {}
                date_map[dt][col_name] = val

        total_upserted = 0
        for dt, cols in sorted(date_map.items()):
            # Build SET clause for each column with data
            set_parts = []
            values = []
            for col, val in cols.items():
                set_parts.append(f"{col} = %s")
                values.append(val)

            # Add updated_at
            set_parts.append("updated_at = NOW()")

            # Check if row exists
            cur.execute("SELECT 1 FROM macro_indicators_daily WHERE fecha = %s", (dt,))
            exists = cur.fetchone() is not None

            if exists:
                # UPDATE existing row
                sql = f"UPDATE macro_indicators_daily SET {', '.join(set_parts)} WHERE fecha = %s"
                values.append(dt)
                cur.execute(sql, values)
            else:
                # INSERT new row
                col_names = ["fecha"] + list(cols.keys()) + ["updated_at"]
                placeholders = ["%s"] + ["%s"] * len(cols) + ["NOW()"]
                insert_values = [dt] + list(cols.values())
                sql = f"INSERT INTO macro_indicators_daily ({', '.join(col_names)}) VALUES ({', '.join(placeholders)})"
                cur.execute(sql, insert_values)

            total_upserted += 1

        conn.commit()
        logger.info(f"[DB] Upserted {total_upserted} dates across {len(all_data)} variables")

    except Exception as e:
        conn.rollback()
        logger.error(f"[DB] Upsert failed: {e}")
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Step 3: Update parquet file (optional, for offline backup)
# ---------------------------------------------------------------------------

def update_parquet(all_data: dict):
    """Update the wide-format macro parquet file (fecha + indicator columns)."""
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not available, skipping parquet update")
        return

    # Build wide-format rows: fecha -> {col: value, ...}
    date_map: dict = {}
    for col_name, records in all_data.items():
        for dt, val in records:
            ts = pd.Timestamp(dt)
            if ts not in date_map:
                date_map[ts] = {"fecha": ts}
            date_map[ts][col_name] = val

    if not date_map:
        logger.info("[Parquet] No new data to append")
        return

    df_new = pd.DataFrame(list(date_map.values()))

    if PARQUET_PATH.exists():
        existing = pd.read_parquet(PARQUET_PATH)
        existing["fecha"] = pd.to_datetime(existing["fecha"])
        # Update: set_index on fecha, then combine_first (new data wins)
        df_existing = existing.set_index("fecha")
        df_update = df_new.set_index("fecha")
        combined = df_existing.copy()
        combined.update(df_update)
        # Add any new dates not in existing
        new_dates = df_update.index.difference(df_existing.index)
        if len(new_dates) > 0:
            combined = pd.concat([combined, df_update.loc[new_dates]])
        combined = combined.sort_index().reset_index()
    else:
        combined = df_new.sort_values("fecha").reset_index(drop=True)

    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(PARQUET_PATH, index=False)
    logger.info(f"[Parquet] Written {len(combined)} rows to {PARQUET_PATH}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logger.info("Starting daily macro update (wide-format)")
    logger.info(f"Project root: {PROJECT_ROOT}")

    # Fetch
    all_data = fetch_yahoo_data()
    if not all_data:
        logger.error("No data fetched from Yahoo Finance. Aborting.")
        sys.exit(1)

    # Store to DB (wide-format)
    upsert_to_db(all_data)

    # Update parquet backup
    update_parquet(all_data)

    logger.info("Daily macro update complete")


if __name__ == "__main__":
    main()
