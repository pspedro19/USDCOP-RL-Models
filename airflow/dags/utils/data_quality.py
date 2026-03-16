"""
Shared data quality gates for Airflow DAGs.
============================================
Reusable validation functions for pre-training data freshness checks
and model artifact staleness detection.

Used by:
    - forecast_h1_l3_weekly_training (pre-training gate)
    - forecast_h5_l3_weekly_training (pre-training gate)
    - forecast_h1_l5_daily_inference (model freshness warning)

Contract: CTR-DQ-001
Version: 1.0.0
Date: 2026-03-12
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def check_table_freshness(conn, table: str, date_col: str, max_age_days: int,
                          label: str, where_clause: str = ""):
    """
    Raise ValueError if the latest row in `table` is older than max_age_days.

    Args:
        conn: psycopg2 connection
        table: Table name
        date_col: Date/timestamp column to check
        max_age_days: Maximum allowed age in days
        label: Human-readable label for error messages
        where_clause: Optional SQL WHERE clause (e.g., "WHERE symbol = 'USD/COP'")

    Returns:
        The latest value of date_col

    Raises:
        ValueError: If table is empty or data is too stale
    """
    cur = conn.cursor()
    query = f"SELECT MAX({date_col}) FROM {table}"
    if where_clause:
        query += f" {where_clause}"
    cur.execute(query)
    latest = cur.fetchone()[0]
    cur.close()

    if latest is None:
        raise ValueError(f"{label}: table {table} is empty")

    # Calculate age depending on whether the value is tz-aware or a date
    if hasattr(latest, 'tzinfo') and latest.tzinfo:
        age = (datetime.now(timezone.utc) - latest).days
    elif hasattr(latest, 'year') and hasattr(latest, 'hour'):
        # Naive datetime — assume UTC
        age = (datetime.now(timezone.utc).replace(tzinfo=None) - latest).days
    elif hasattr(latest, 'year'):
        # date object
        age = (datetime.now().date() - latest).days
    else:
        raise ValueError(f"{label}: unexpected type for {date_col}: {type(latest)}")

    if age > max_age_days:
        raise ValueError(
            f"{label}: latest={latest}, age={age}d, threshold={max_age_days}d"
        )

    logger.info(f"{label}: latest={latest}, age={age}d (threshold={max_age_days}d) — OK")
    return latest


def validate_training_data_freshness(ohlcv_max_age=3, macro_max_age=7):
    """
    Pre-training gate: verify OHLCV and macro data are fresh enough for training.

    Used as a blocking task before H1-L3 and H5-L3 weekly training DAGs.
    If either check fails, the task raises ValueError and training does NOT proceed.

    Args:
        ohlcv_max_age: Max days since last OHLCV row (default 3)
        macro_max_age: Max days since last macro row (default 7)

    Returns:
        dict with ohlcv_latest and macro_latest timestamps
    """
    from utils.dag_common import get_db_connection

    conn = get_db_connection()
    try:
        ohlcv_latest = check_table_freshness(
            conn, "usdcop_m5_ohlcv", "time", ohlcv_max_age,
            "OHLCV freshness", "WHERE symbol = 'USD/COP'"
        )
        macro_latest = check_table_freshness(
            conn, "macro_indicators_daily", "fecha", macro_max_age,
            "Macro freshness"
        )
        return {
            "ohlcv_latest": str(ohlcv_latest),
            "macro_latest": str(macro_latest),
        }
    finally:
        conn.close()


def check_model_freshness(model_dir: str, extension: str = "*.pkl",
                          max_age_days: int = 10) -> dict:
    """
    Soft check: warn if model artifacts are older than max_age_days.
    Does NOT raise — only logs a warning. Used by inference DAGs.

    Args:
        model_dir: Path to the model directory
        extension: Glob pattern for model files
        max_age_days: Threshold for staleness warning

    Returns:
        dict with newest_mtime, age_days, and is_stale flag
    """
    model_path = Path(model_dir)

    if not model_path.exists():
        logger.warning(f"MODEL FRESHNESS: directory not found: {model_dir}")
        return {"exists": False, "is_stale": True, "age_days": None}

    model_files = list(model_path.glob(extension))
    if not model_files:
        logger.warning(f"MODEL FRESHNESS: no {extension} files in {model_dir}")
        return {"exists": True, "is_stale": True, "age_days": None, "file_count": 0}

    newest = max(f.stat().st_mtime for f in model_files)
    age_days = (datetime.now().timestamp() - newest) / 86400

    is_stale = age_days > max_age_days
    if is_stale:
        logger.warning(
            f"MODEL FRESHNESS WARNING: models in {model_dir} are {age_days:.1f} days old "
            f"(threshold: {max_age_days} days). L3 training may have failed."
        )
    else:
        logger.info(
            f"MODEL FRESHNESS: models are {age_days:.1f} days old "
            f"(threshold: {max_age_days} days) — OK"
        )

    return {
        "exists": True,
        "is_stale": is_stale,
        "age_days": round(age_days, 1),
        "file_count": len(model_files),
    }
