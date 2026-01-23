"""
Forecast L0: Daily USDCOP Data Acquisition
==========================================

ARCHITECTURE 10/10: Uses OFFICIAL Investing.com daily values.

Fetches daily USDCOP OHLCV from multiple sources (priority order):
1. Investing.com scraper (PRIMARY - OFFICIAL daily close prices)
2. TwelveData API (secondary)
3. yfinance (fallback)

CRITICAL: Do NOT use 5-min resample for forecasting production.
Use official Investing.com daily values for accuracy.

Schedule: Daily at 7:00 AM Colombia time (after market close)
Depends: None (first DAG in forecasting pipeline)
Triggers: forecast_l1_01_daily_features

Data Lineage:
    Investing.com → bi.dim_daily_usdcop (OFFICIAL)

Author: Trading Team
Date: 2026-01-22
Version: 3.0.0 - Investing.com as PRIMARY source
Contract: CTR-FORECAST-DATA-001
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Import from SSOT
from airflow.dags.contracts.dag_registry import (
    FORECAST_L0_DAILY_DATA,
    get_dag_tags,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS FROM DATA CONTRACT
# =============================================================================

TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY", "")
TWELVEDATA_BASE_URL = "https://api.twelvedata.com"
SYMBOL = "USD/COP"

# Price validation range (COP)
MIN_PRICE = 3000
MAX_PRICE = 6000


# =============================================================================
# DATA FETCHING FUNCTIONS
# =============================================================================

def fetch_from_twelvedata(
    start_date: str,
    end_date: str,
    api_key: str = TWELVEDATA_API_KEY,
) -> Optional[pd.DataFrame]:
    """
    Fetch daily USDCOP data from TwelveData API.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        api_key: TwelveData API key

    Returns:
        DataFrame with OHLCV data or None if failed
    """
    if not api_key:
        logger.warning("TWELVEDATA_API_KEY not set")
        return None

    endpoint = f"{TWELVEDATA_BASE_URL}/time_series"
    params = {
        "symbol": SYMBOL,
        "interval": "1day",
        "start_date": start_date,
        "end_date": end_date,
        "apikey": api_key,
        "format": "JSON",
        "timezone": "America/Bogota",
    }

    try:
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "values" not in data:
            logger.error(f"TwelveData error: {data.get('message', 'Unknown error')}")
            return None

        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.rename(columns={"datetime": "date"})

        # Convert to numeric
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Volume might not exist for forex
        if "volume" not in df.columns:
            df["volume"] = 0
        else:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)

        df["source"] = "twelvedata"

        logger.info(f"TwelveData: Fetched {len(df)} records")
        return df

    except requests.RequestException as e:
        logger.error(f"TwelveData request failed: {e}")
        return None


def fetch_from_investing(
    start_date: str,
    end_date: str,
) -> Optional[pd.DataFrame]:
    """
    PRIMARY: Fetch OHLCV data from Investing.com (OFFICIAL source).

    This is the PRIMARY source for forecasting pipeline.
    Uses official daily close prices from Investing.com.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with OHLCV data or None if failed
    """
    try:
        # Import the dedicated USDCOP scraper
        import sys
        scraper_path = PROJECT_ROOT / "data/pipeline/02_scrapers/02_custom"
        if str(scraper_path) not in sys.path:
            sys.path.insert(0, str(scraper_path))

        from scraper_usdcop_investing import USDCOPInvestingScraper

        scraper = USDCOPInvestingScraper(delay_between_requests=1.0)
        df = scraper.fetch_ohlcv(start_date, end_date)

        if df.empty:
            logger.warning("Investing.com scraper returned no data")
            return None

        df["source"] = "investing"
        logger.info(f"Investing.com: Fetched {len(df)} OFFICIAL records")
        return df[["date", "open", "high", "low", "close", "volume", "source"]]

    except ImportError as e:
        logger.warning(f"Investing.com scraper not available: {e}")
        return None
    except Exception as e:
        logger.error(f"Investing.com fetch failed: {e}")
        return None


def fetch_from_yfinance(
    start_date: str,
    end_date: str,
) -> Optional[pd.DataFrame]:
    """
    FALLBACK: Fetch USDCOP data from yfinance.

    This is a backup source if Investing.com and TwelveData fail.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with OHLCV data or None if failed
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker("USDCOP=X")
        df = ticker.history(start=start_date, end=end_date, interval="1d")

        if df.empty:
            logger.warning("yfinance returned no data")
            return None

        df = df.reset_index()
        df = df.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["volume"] = df["volume"].fillna(0).astype(int)
        df["source"] = "yfinance"

        logger.info(f"yfinance: Fetched {len(df)} records")
        return df[["date", "open", "high", "low", "close", "volume", "source"]]

    except ImportError:
        logger.warning("yfinance not installed, skipping backup source")
        return None
    except Exception as e:
        logger.error(f"yfinance fetch failed: {e}")
        return None


def fetch_from_5min_ohlcv(
    start_date: str,
    end_date: str,
    postgres_conn_id: str = "postgres_default",
) -> Optional[pd.DataFrame]:
    """
    SSOT: Resample daily data from existing 5-min OHLCV table.

    This uses the existing usdcop_m5_ohlcv table (which is updated by RL pipeline)
    as a secondary source. This ensures both pipelines use the same underlying data.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        postgres_conn_id: Airflow Postgres connection ID

    Returns:
        DataFrame with daily OHLCV data or None if failed
    """
    hook = PostgresHook(postgres_conn_id=postgres_conn_id)

    query = """
        SELECT
            DATE(time) as date,
            (array_agg(open ORDER BY time))[1] as open,
            MAX(high) as high,
            MIN(low) as low,
            (array_agg(close ORDER BY time DESC))[1] as close,
            COALESCE(SUM(volume), 0) as volume
        FROM usdcop_m5_ohlcv
        WHERE DATE(time) >= %s
          AND DATE(time) <= %s
        GROUP BY DATE(time)
        ORDER BY date
    """

    try:
        df = hook.get_pandas_df(query, parameters=(start_date, end_date))

        if df.empty:
            logger.warning("No 5-min OHLCV data found for date range")
            return None

        df["source"] = "5min_resample"

        logger.info(f"5-min resample: Generated {len(df)} daily records from SSOT")
        return df[["date", "open", "high", "low", "close", "volume", "source"]]

    except Exception as e:
        logger.error(f"5-min resample failed: {e}")
        return None


def validate_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean OHLCV data.

    Args:
        df: Raw OHLCV DataFrame

    Returns:
        Cleaned DataFrame with invalid records removed
    """
    initial_count = len(df)

    # Remove rows with missing required data
    df = df.dropna(subset=["date", "open", "high", "low", "close"])

    # Validate price range
    df = df[
        (df["close"] >= MIN_PRICE) &
        (df["close"] <= MAX_PRICE)
    ]

    # Validate OHLC consistency
    df = df[
        (df["high"] >= df["low"]) &
        (df["high"] >= df["open"]) &
        (df["high"] >= df["close"]) &
        (df["low"] <= df["open"]) &
        (df["low"] <= df["close"])
    ]

    removed = initial_count - len(df)
    if removed > 0:
        logger.warning(f"Removed {removed} invalid records during validation")

    return df


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

def get_last_date_in_db(postgres_conn_id: str = "postgres_default") -> Optional[str]:
    """Get the most recent date in the database."""
    hook = PostgresHook(postgres_conn_id=postgres_conn_id)

    query = """
    SELECT MAX(date)::text
    FROM bi.dim_daily_usdcop
    """

    try:
        result = hook.get_first(query)
        return result[0] if result and result[0] else None
    except Exception as e:
        logger.warning(f"Could not get last date: {e}")
        return None


def ensure_schema_and_table(postgres_conn_id: str = "postgres_default") -> None:
    """Create schema and table if they don't exist."""
    hook = PostgresHook(postgres_conn_id=postgres_conn_id)

    sql = """
    CREATE SCHEMA IF NOT EXISTS bi;

    CREATE TABLE IF NOT EXISTS bi.dim_daily_usdcop (
        id SERIAL PRIMARY KEY,
        date DATE NOT NULL UNIQUE,
        open DECIMAL(12,4) NOT NULL,
        high DECIMAL(12,4) NOT NULL,
        low DECIMAL(12,4) NOT NULL,
        close DECIMAL(12,4) NOT NULL,
        volume BIGINT DEFAULT 0,
        source VARCHAR(50) DEFAULT 'twelvedata',
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_daily_usdcop_date
        ON bi.dim_daily_usdcop (date DESC);
    """

    hook.run(sql)
    logger.info("Schema and table ensured")


def upsert_daily_data(
    df: pd.DataFrame,
    postgres_conn_id: str = "postgres_default",
) -> Dict[str, int]:
    """
    Upsert daily data into PostgreSQL.

    Returns:
        Dict with inserted/updated counts
    """
    hook = PostgresHook(postgres_conn_id=postgres_conn_id)

    inserted = 0
    updated = 0

    for _, row in df.iterrows():
        sql = """
        INSERT INTO bi.dim_daily_usdcop (date, open, high, low, close, volume, source)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            source = EXCLUDED.source,
            updated_at = NOW()
        RETURNING (xmax = 0) as is_insert
        """

        result = hook.get_first(sql, parameters=(
            row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"]),
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            int(row["volume"]),
            str(row["source"]),
        ))

        if result and result[0]:
            inserted += 1
        else:
            updated += 1

    logger.info(f"Upserted: {inserted} inserted, {updated} updated")
    return {"inserted": inserted, "updated": updated}


# =============================================================================
# AIRFLOW TASKS
# =============================================================================

def task_ensure_schema(**context) -> None:
    """Ensure database schema exists."""
    ensure_schema_and_table()


def task_fetch_daily_data(**context) -> Dict[str, Any]:
    """
    Fetch daily USDCOP data from available sources.

    Source Priority (Architecture 10/10 - v3.0):
    1. Investing.com (PRIMARY - OFFICIAL daily close prices)
    2. TwelveData API (secondary)
    3. yfinance (fallback)

    NOTE: 5-min resample is NOT used for production forecasting.
    Official daily values are more accurate for forecasting.
    """
    # Determine date range
    last_date = get_last_date_in_db()

    if last_date:
        start_date = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        # Initial load: fetch from 2015 for full history
        start_date = "2015-01-01"

    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Check if we need to fetch
    if start_date > end_date:
        logger.info("Data is up to date, no fetch needed")
        return {"status": "up_to_date", "records": 0}

    logger.info(f"Fetching OFFICIAL daily data from {start_date} to {end_date}")

    # Source 1: Investing.com (PRIMARY - OFFICIAL source for forecasting)
    df = fetch_from_investing(start_date, end_date)

    # Source 2: TwelveData API (secondary external source)
    if df is None or df.empty:
        logger.warning("Investing.com failed, trying TwelveData API")
        df = fetch_from_twelvedata(start_date, end_date)

    # Source 3: yfinance (external fallback)
    if df is None or df.empty:
        logger.warning("TwelveData failed, trying yfinance (fallback)")
        df = fetch_from_yfinance(start_date, end_date)

    if df is None or df.empty:
        logger.error("All data sources failed")
        return {"status": "failed", "records": 0}

    # Validate data
    df = validate_ohlcv_data(df)

    # Push to XCom for next task
    context["ti"].xcom_push(key="daily_data", value=df.to_json(orient="records", date_format="iso"))

    return {
        "status": "success",
        "records": len(df),
        "date_range": [df["date"].min().isoformat(), df["date"].max().isoformat()],
        "source": df["source"].iloc[0] if len(df) > 0 else "unknown",
    }


def task_load_to_db(**context) -> Dict[str, Any]:
    """Load fetched data into PostgreSQL."""
    ti = context["ti"]
    data_json = ti.xcom_pull(task_ids="fetch_daily_data", key="daily_data")

    if not data_json:
        logger.warning("No data to load")
        return {"inserted": 0, "updated": 0}

    df = pd.read_json(data_json, orient="records")
    df["date"] = pd.to_datetime(df["date"])

    result = upsert_daily_data(df)

    return result


def task_validate_data(**context) -> Dict[str, Any]:
    """Validate loaded data integrity."""
    hook = PostgresHook(postgres_conn_id="postgres_default")

    # Check for gaps
    gap_check = """
    WITH date_series AS (
        SELECT generate_series(
            (SELECT MIN(date) FROM bi.dim_daily_usdcop),
            (SELECT MAX(date) FROM bi.dim_daily_usdcop),
            '1 day'::interval
        )::date as expected_date
    ),
    weekdays AS (
        SELECT expected_date
        FROM date_series
        WHERE EXTRACT(DOW FROM expected_date) NOT IN (0, 6)  -- Exclude weekends
    )
    SELECT COUNT(*) as missing_days
    FROM weekdays w
    LEFT JOIN bi.dim_daily_usdcop d ON w.expected_date = d.date
    WHERE d.date IS NULL
    """

    result = hook.get_first(gap_check)
    missing_days = result[0] if result else 0

    # Get statistics
    stats_query = """
    SELECT
        COUNT(*) as total_records,
        MIN(date) as first_date,
        MAX(date) as last_date,
        AVG(close) as avg_close
    FROM bi.dim_daily_usdcop
    """

    stats = hook.get_first(stats_query)

    return {
        "total_records": stats[0] if stats else 0,
        "first_date": str(stats[1]) if stats else None,
        "last_date": str(stats[2]) if stats else None,
        "avg_close": float(stats[3]) if stats and stats[3] else None,
        "missing_weekdays": missing_days,
        "validation_passed": missing_days < 10,  # Allow some tolerance
    }


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    "owner": "forecast-pipeline",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id=FORECAST_L0_DAILY_DATA,
    default_args=default_args,
    description="Fetch daily USDCOP prices for forecasting pipeline",
    schedule_interval="0 7 * * 1-5",  # 7 AM on weekdays (Colombia time)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=get_dag_tags(FORECAST_L0_DAILY_DATA),
    doc_md=__doc__,
) as dag:

    ensure_schema = PythonOperator(
        task_id="ensure_schema",
        python_callable=task_ensure_schema,
    )

    fetch_data = PythonOperator(
        task_id="fetch_daily_data",
        python_callable=task_fetch_daily_data,
    )

    load_data = PythonOperator(
        task_id="load_to_db",
        python_callable=task_load_to_db,
    )

    validate = PythonOperator(
        task_id="validate_data",
        python_callable=task_validate_data,
    )

    # Task dependencies
    ensure_schema >> fetch_data >> load_data >> validate
