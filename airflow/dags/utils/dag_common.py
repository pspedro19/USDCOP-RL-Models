"""
DAG Common Utilities
====================
Shared utilities for all V3 DAGs: DB connections + config loading.
Replaces duplicated code across 5 DAGs.

DRY Refactoring: Eliminates ~113 lines of duplicated code.

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-17
"""

import os
import json
import logging
import psycopg2
from typing import Dict, Any, Optional

# =============================================================================
# DATABASE CONNECTION
# =============================================================================

def get_db_config() -> Dict[str, Any]:
    """
    Get PostgreSQL connection configuration from environment variables.

    SECURITY: All credentials MUST be set via environment variables.
    No default passwords are used (Fail Fast principle).

    Returns:
        Dict with host, port, database, user, password

    Raises:
        ValueError: If required environment variables are not set
    """
    # SECURITY FIX: No default passwords - fail fast if not configured
    password = os.environ.get('POSTGRES_PASSWORD')
    if not password:
        raise ValueError(
            "POSTGRES_PASSWORD environment variable is required. "
            "Set it in .env file or docker-compose.yml"
        )

    return {
        'host': os.environ.get('POSTGRES_HOST', 'timescaledb'),
        'port': int(os.environ.get('POSTGRES_PORT', '5432')),
        'database': os.environ.get('POSTGRES_DB', 'usdcop'),
        'user': os.environ.get('POSTGRES_USER', 'admin'),
        'password': password
    }


def get_db_connection(config: Optional[Dict[str, Any]] = None) -> psycopg2.extensions.connection:
    """
    Get raw psycopg2 database connection.

    Args:
        config: Optional DB config dict. If None, uses get_db_config()

    Returns:
        psycopg2 connection object

    Usage:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM table")
        conn.close()
    """
    if config is None:
        config = get_db_config()

    try:
        return psycopg2.connect(**config)
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        raise


# Backward compatibility: Export DB_CONFIG constant
DB_CONFIG = get_db_config()


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def get_feature_config_path() -> str:
    """
    Get path to feature_config.json - tries multiple locations.

    Returns:
        Absolute path to feature_config.json
    """
    # Try multiple possible locations
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '../config/feature_config.json'),  # dags/config/
        os.path.join(os.path.dirname(__file__), '../../config/feature_config.json'),  # airflow/config/
        '/opt/airflow/config/feature_config.json',  # Docker mounted config
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    # Return first option as default (will fail gracefully in load_feature_config)
    return possible_paths[0]


def get_default_config() -> Dict[str, Any]:
    """Return default configuration when file is not available."""
    return {
        "_meta": {
            "model_id": "ppo_usdcop_v14",
            "version": "14.0.0"
        },
        "sources": {
            "ohlcv": {
                "table": "usdcop_m5_ohlcv",
                "granularity": "5min",
                "lookback_bars_needed": 100
            }
        },
        "trading": {
            "symbol": "USD/COP",
            "market_hours": {
                "timezone": "America/Bogota",
                "local_start": "08:00",
                "local_end": "12:55"
            },
            "bars_per_session": 60,
            "trading_days": [0, 1, 2, 3, 4]
        },
        "observation_space": {
            "order": [
                "log_ret_5m", "log_ret_1h", "log_ret_4h",
                "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
                "brent_change_1d", "rate_spread",
                "rsi_9", "atr_pct", "adx_14", "usdmxn_change_1d"
            ],
            "total_obs_dim": 15
        },
        "cold_start": {
            "min_warmup_bars": 50,
            "max_ohlcv_staleness_minutes": 10,
            "max_macro_staleness_hours": 48
        },
        "holidays_2025_colombia": [
            "2025-01-01", "2025-01-06", "2025-03-24", "2025-04-17", "2025-04-18",
            "2025-05-01", "2025-06-02", "2025-06-23", "2025-06-30", "2025-07-04",
            "2025-07-20", "2025-08-07", "2025-08-18", "2025-10-13", "2025-11-03",
            "2025-11-17", "2025-11-27", "2025-12-08", "2025-12-25"
        ],
        "monitoring": {
            "ohlcv_lag_threshold_seconds": 600,
            "macro_age_threshold_hours": 24,
            "expected_inferences_per_hour": 12,
            "min_ohlcv_rows": 1000,
            "min_macro_rows": 50
        }
    }


def load_feature_config(config_path: Optional[str] = None, raise_on_error: bool = False) -> Dict[str, Any]:
    """
    Load feature_config.json with graceful fallback to defaults.

    Args:
        config_path: Optional path to config file. If None, uses get_feature_config_path()
        raise_on_error: If True, raises exception on load failure. If False, returns default config.

    Returns:
        Config dictionary or default config on failure
    """
    if config_path is None:
        config_path = get_feature_config_path()

    try:
        with open(config_path) as f:
            config = json.load(f)
            logging.info(f"Loaded feature config from {config_path}")
            return config
    except Exception as e:
        error_msg = f"Could not load config from {config_path}: {e}"

        if raise_on_error:
            logging.error(f"CRITICAL: {error_msg}")
            raise
        else:
            logging.warning(f"{error_msg}. Using default config.")
            return get_default_config()


# =============================================================================
# CONSTANTS (for backward compatibility)
# =============================================================================

FEATURE_CONFIG_PATH = get_feature_config_path()
