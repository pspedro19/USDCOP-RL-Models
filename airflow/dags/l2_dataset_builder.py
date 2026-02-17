# -*- coding: utf-8 -*-
"""
L2 Dataset Builder - Unified Feature Calculation + Dataset Generation
======================================================================
Fuses L1 (Feature Calculation) + L2 (Dataset Generation) into a single pipeline
for the TRAINING path.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      L2 DATASET BUILDER                                  │
    │                                                                         │
    │  INPUT:                           OUTPUT:                               │
    │  ├── experiment YAML              ├── {dataset}_train.parquet           │
    │  ├── date_ranges.yaml             ├── {dataset}_val.parquet             │
    │  ├── usdcop_m5_ohlcv              ├── {dataset}_test.parquet            │
    │  └── macro_daily/monthly/quarterly├── norm_stats.json                   │
    │                                   ├── lineage.json                      │
    │                                   └── reports/                          │
    │                                       ├── l2_data_quality_report_*.json │
    │                                       ├── l2_variable_summary_*.csv     │
    │                                       └── l2_data_quality_report_*.html │
    └─────────────────────────────────────────────────────────────────────────┘

Pipeline Steps:
    1. Load experiment config + date ranges (SSOT)
    2. Query OHLCV and Macro from database
    3. Merge OHLCV + Macro (ffill within session only)
    4. Calculate features with CanonicalFeatureBuilder
    5. Apply anti-leakage preprocessing
    6. Compute normalization stats (train period only)
    7. Apply normalization
    8. Split train/val/test
    9. Save .parquet + norm_stats + lineage
    10. Validate output (NaN, Inf, feature order)
    11. Generate Data Quality Report (per-variable statistics)
    12. Push to XCom for L3

Data Quality Report (CTR-L2-QUALITY-REPORT-001):
    Per-variable analysis including:
    - Basic statistics (count, mean, std, min, max, percentiles, skewness, kurtosis)
    - Temporal coverage (date range, frequency, gaps)
    - Missing data analysis (count, %, gap ranges)
    - Anomaly detection (outliers IQR/Z-score, sudden jumps, zeros)
    - Distribution analysis (normality test, type)
    - Trend analysis (direction, stationarity, volatility regime)
    - Anti-leakage verification (T-1 shift, no future data)
    - Top 5 correlations per variable
    - Quality score (0-100%) with levels (excellent/good/acceptable/poor/critical)

Anti-Leakage Guarantees:
    - NO forward-fill across trading sessions
    - Macro data shifted T-1 (use yesterday's data)
    - Normalization stats computed on TRAIN only
    - No future data in feature calculations

Contract: CTR-DATASET-001
Version: 1.1.0
Author: Trading Team
Created: 2026-01-31
Updated: 2026-02-01 (Added Data Quality Report)
"""

import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

# =============================================================================
# PATH SETUP
# =============================================================================

PROJECT_ROOT = Path('/opt/airflow') if Path('/opt/airflow').exists() else Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dag_common import get_db_connection

# Trading Calendar for holiday filtering
try:
    from src.data.calendar import TradingCalendar
    TRADING_CALENDAR_AVAILABLE = True
    _calendar = TradingCalendar()
except ImportError:
    TRADING_CALENDAR_AVAILABLE = False
    _calendar = None
    logging.warning("[L2] TradingCalendar not available - holiday filtering disabled")

# L2 Data Quality Report
try:
    from services.l2_data_quality_report import (
        L2DataQualityReportGenerator,
        generate_l2_report,
    )
    DATA_QUALITY_REPORT_AVAILABLE = True
except ImportError:
    DATA_QUALITY_REPORT_AVAILABLE = False
    logging.warning("[L2] L2DataQualityReportGenerator not available")

# =============================================================================
# SSOT IMPORTS
# =============================================================================

# =============================================================================
# PIPELINE SSOT - Single Source of Truth (NEW v2.0)
# =============================================================================
# Primary SSOT: pipeline_ssot.yaml controls entire pipeline (L2, L3, L4)
# =============================================================================
try:
    from src.config.pipeline_config import load_pipeline_config, PipelineConfig
    from src.data.ssot_dataset_builder import SSOTDatasetBuilder, build_production_dataset
    PIPELINE_SSOT_AVAILABLE = True
    _pipeline_config = load_pipeline_config()
    logging.info(f"[SSOT] Loaded pipeline_ssot.yaml v{_pipeline_config.version}")
except ImportError as e:
    PIPELINE_SSOT_AVAILABLE = False
    _pipeline_config = None
    logging.warning(f"[SSOT] pipeline_config not available: {e}")

# =============================================================================
# EXPERIMENT SSOT - Single Source of Truth for L2 + L3 (Legacy)
# =============================================================================
# IMPORTANTE: Este es el UNICO lugar donde se leen features y configuracion.
# =============================================================================
try:
    from src.config.experiment_loader import (
        load_experiment_config,
        get_feature_order,
        get_observation_dim,
        get_feature_order_hash,
    )
    # Load SSOT configuration
    EXPERIMENT_CONFIG = load_experiment_config()
    FEATURE_ORDER = EXPERIMENT_CONFIG.feature_order
    OBSERVATION_DIM = EXPERIMENT_CONFIG.pipeline.observation_dim
    FEATURE_ORDER_HASH = EXPERIMENT_CONFIG.feature_order_hash
    EXPERIMENT_SSOT_AVAILABLE = True
    logging.info(f"[SSOT] Loaded experiment config v{EXPERIMENT_CONFIG.version}")
except ImportError as e:
    EXPERIMENT_SSOT_AVAILABLE = False
    EXPERIMENT_CONFIG = None
    logging.warning(f"[SSOT] experiment_loader not available: {e}")
    # Fallback: Try old feature contract
    try:
        from src.core.contracts.feature_contract import (
            FEATURE_ORDER,
            OBSERVATION_DIM,
            FEATURE_ORDER_HASH,
        )
        logging.info("[SSOT] Using legacy feature_contract")
    except ImportError:
        # Final fallback: hardcoded v2.0 CLOSE-ONLY
        FEATURE_ORDER = (
            "log_ret_5m", "log_ret_1h", "log_ret_4h",
            "rsi_9", "volatility_pct", "trend_z",
            "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
            "brent_change_1d", "rate_spread", "usdmxn_change_1d",
            "position", "unrealized_pnl",
        )
        OBSERVATION_DIM = 15
        FEATURE_ORDER_HASH = hashlib.md5(str(FEATURE_ORDER).encode()).hexdigest()
        logging.warning("[SSOT] Using hardcoded FEATURE_ORDER fallback")

try:
    from src.feature_store.builders import CanonicalFeatureBuilder
    CANONICAL_BUILDER_AVAILABLE = True
except ImportError:
    CANONICAL_BUILDER_AVAILABLE = False
    logging.warning("[SSOT] CanonicalFeatureBuilder not available")

# SSOT: Technical Indicators (CLOSE-ONLY)
try:
    from src.features.technical_indicators import (
        calculate_rsi_wilders,
        calculate_volatility_pct,
        calculate_trend_z,
        calculate_log_returns,
        calculate_macro_zscore,
    )
    TECHNICAL_INDICATORS_AVAILABLE = True
except ImportError:
    TECHNICAL_INDICATORS_AVAILABLE = False
    logging.warning("[SSOT] technical_indicators not available, using inline fallbacks")

try:
    from contracts.dag_registry import RL_L2_DATASET_BUILD
    DAG_ID = RL_L2_DATASET_BUILD
except ImportError:
    DAG_ID = "rl_l2_01_dataset_build"

# XCom contracts for L3 compatibility
try:
    from airflow.dags.contracts.xcom_contracts import (
        L2Output,
        L2XComKeysEnum,
    )
    XCOM_CONTRACTS_AVAILABLE = True
except ImportError:
    XCOM_CONTRACTS_AVAILABLE = False
    L2Output = None
    logging.warning("[SSOT] XCom contracts not available")

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "data" / "pipeline" / "07_output" / "5min"
NORM_STATS_DIR = PROJECT_ROOT / "config"

# Market features (18 features, without position and unrealized_pnl state features)
# EXP-B-001: Updated to use market_features from SSOT (18 predictor features)
if EXPERIMENT_SSOT_AVAILABLE and EXPERIMENT_CONFIG is not None:
    MARKET_FEATURES = EXPERIMENT_CONFIG.market_features
else:
    # Fallback: v3.1 EXP-B-001 feature order (18 market features)
    MARKET_FEATURES = (
        "log_ret_5m", "log_ret_1h", "log_ret_4h", "log_ret_1d",
        "rsi_9", "rsi_21", "volatility_pct", "trend_z",
        "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
        "brent_change_1d", "rate_spread_z", "rate_spread_change",
        "usdmxn_change_1d", "yield_curve_z", "gold_change_1d",
    )

# Macro columns needed from database
# EXP-B-001: Added ust2y for yield_curve_z, gold for gold_change_1d
MACRO_COLUMNS_NEEDED = [
    "dxy", "vix", "embi", "brent", "ust10y", "ust2y", "col10y", "usdmxn", "gold"
]

# Trading session hours (Colombia Time)
MARKET_OPEN_HOUR = 8
MARKET_CLOSE_HOUR = 13


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DateRanges:
    """Date ranges from SSOT config."""
    training_start: str
    training_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str  # "dynamic" means use current date


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    experiment_name: str
    experiment_version: str
    date_range_start: str
    date_range_end: str
    feature_columns: List[str]
    preprocessing: Dict[str, Any]
    output_name: str = "dataset"


@dataclass
class LineageInfo:
    """Lineage tracking information."""
    dataset_hash: str = ""
    config_hash: str = ""
    feature_order_hash: str = ""
    norm_stats_hash: str = ""
    builder_version: str = ""
    created_at: str = ""
    date_range: Dict[str, str] = field(default_factory=dict)
    num_rows: Dict[str, int] = field(default_factory=dict)
    num_features: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetOutput:
    """Output of dataset builder for XCom."""
    train_path: str
    val_path: str
    test_path: str
    norm_stats_path: str
    lineage_path: str
    lineage: LineageInfo
    quality_report_dir: str = ""
    success: bool = True
    error: Optional[str] = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute hash of configuration dictionary."""
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def load_date_ranges() -> DateRanges:
    """Load date ranges from SSOT config."""
    config_path = CONFIG_DIR / "date_ranges.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"date_ranges.yaml not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Handle dynamic end dates
    test_end = config["test"]["end"]
    if test_end == "dynamic":
        test_end = datetime.now().strftime("%Y-%m-%d")

    return DateRanges(
        training_start=config["training"]["start"],
        training_end=config["training"]["end"],
        validation_start=config["validation"]["start"],
        validation_end=config["validation"]["end"],
        test_start=config["test"]["start"],
        test_end=test_end,
    )


def load_experiment_config(experiment_name: str) -> Optional[Dict[str, Any]]:
    """Load experiment configuration from YAML."""
    config_path = CONFIG_DIR / "experiments" / f"{experiment_name}.yaml"

    if not config_path.exists():
        logger.warning(f"Experiment config not found: {config_path}")
        return None

    with open(config_path) as f:
        return yaml.safe_load(f)


def get_output_paths(dataset_name: str) -> Dict[str, Path]:
    """Get output file paths for dataset."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create reports subdirectory
    reports_dir = OUTPUT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    return {
        "train": OUTPUT_DIR / f"{dataset_name}_train.parquet",
        "val": OUTPUT_DIR / f"{dataset_name}_val.parquet",
        "test": OUTPUT_DIR / f"{dataset_name}_test.parquet",
        "norm_stats": OUTPUT_DIR / f"{dataset_name}_norm_stats.json",
        "lineage": OUTPUT_DIR / f"{dataset_name}_lineage.json",
        "quality_report_dir": reports_dir,
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_ohlcv_data(conn, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load OHLCV data from database.

    Contract: CTR-FEATURES-002 (CLOSE-ONLY)

    IMPORTANTE: Solo usamos CLOSE para feature engineering.
    Razón: CLOSE es el precio más confiable. O/H/L tienen ruido de bid-ask.

    Returns DataFrame with columns: time, close (only close needed)
    """
    # Solo seleccionamos CLOSE - los demás campos no se usan
    query = """
        SELECT time, close
        FROM usdcop_m5_ohlcv
        WHERE symbol = 'USD/COP'
          AND time >= %s
          AND time <= %s
        ORDER BY time ASC
    """

    df = pd.read_sql(query, conn, params=(start_date, end_date))
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')

    logger.info(f"Loaded {len(df)} CLOSE prices from {start_date} to {end_date}")
    return df


def load_macro_data(conn, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load and process macro data from all 3 tables (4-Table Architecture).

    Contract: CTR-L0-4TABLE-001, CTR-L2-MACRO-PROCESS-001

    Process:
    1. Load from 3 separate tables (daily, monthly, quarterly)
    2. Apply bounded FFill per frequency (anti-leakage limits: 5/35/95 days)
    3. Join monthly/quarterly to daily grid by period start
    4. Return daily DataFrame ready for 5-min resample in merge step

    Tables:
    - macro_indicators_daily (18 daily variables, FFill limit: 5 days)
    - macro_indicators_monthly (18 monthly variables, FFill limit: 35 days)
    - macro_indicators_quarterly (4 quarterly variables, FFill limit: 95 days)

    Returns DataFrame indexed by fecha with friendly column names.
    """
    # ==========================================================================
    # COLUMN MAPPINGS (DB names -> friendly names)
    # ==========================================================================
    # EXP-B-001: Added ust2y (yield curve), gold (safe haven)
    DAILY_MAPPING = {
        "fxrt_index_dxy_usa_d_dxy": "dxy",
        "volt_vix_usa_d_vix": "vix",
        "crsk_spread_embi_col_d_embi": "embi",
        "comm_oil_brent_glb_d_brent": "brent",
        "comm_oil_wti_glb_d_wti": "wti",
        "finc_bond_yield10y_usa_d_ust10y": "ust10y",
        "finc_bond_yield2y_usa_d_dgs2": "ust2y",
        "finc_bond_yield10y_col_d_col10y": "col10y",
        "fxrt_spot_usdmxn_mex_d_usdmxn": "usdmxn",
        "polr_policy_rate_col_d_tpm": "tpm_col",
        "eqty_index_colcap_col_d_colcap": "colcap",
        "comm_metal_gold_glb_d_gold": "gold",  # EXP-B-001: Safe haven indicator
    }

    MONTHLY_MAPPING = {
        "polr_fed_funds_usa_m_fedfunds": "fedfunds",
        "labr_unemployment_usa_m_unrate": "unemployment",
        "infl_cpi_core_usa_m_cpilfesl": "core_cpi",
        "sent_consumer_usa_m_umcsent": "umcsent",
    }

    QUARTERLY_MAPPING = {
        "gdpp_real_gdp_usa_q_gdp_q": "gdp_usa",
        "rsbp_current_account_col_q_cacct": "current_account",
    }

    # FFill limits by frequency (days) - Contract: CTR-L0-FFILL-001
    FFILL_LIMITS = {
        'daily': 5,      # Max 5 business days for daily data
        'monthly': 35,   # Max ~1 month for monthly data
        'quarterly': 95, # Max ~1 quarter for quarterly data
    }

    logger.info("[L2-MACRO] Loading from 3 macro tables with anti-leakage FFill...")

    # ==========================================================================
    # STEP 1: Load DAILY macro
    # ==========================================================================
    daily_cols = list(DAILY_MAPPING.keys())
    daily_query = f"""
        SELECT fecha, {', '.join(daily_cols)}
        FROM macro_indicators_daily
        WHERE fecha >= %s AND fecha <= %s
        ORDER BY fecha ASC
    """

    df_daily = pd.read_sql(daily_query, conn, params=(start_date, end_date))
    df_daily['fecha'] = pd.to_datetime(df_daily['fecha'])
    df_daily = df_daily.set_index('fecha')
    df_daily = df_daily.rename(columns=DAILY_MAPPING)

    # Apply bounded FFill (max 5 days)
    df_daily = _apply_bounded_ffill(df_daily, max_days=FFILL_LIMITS['daily'])
    logger.info(f"[L2-MACRO] Daily: {len(df_daily)} rows, {len(df_daily.columns)} cols (FFill limit: 5 days)")

    # ==========================================================================
    # STEP 2: Load MONTHLY macro
    # ==========================================================================
    try:
        monthly_cols = list(MONTHLY_MAPPING.keys())
        monthly_query = f"""
            SELECT fecha, {', '.join(monthly_cols)}
            FROM macro_indicators_monthly
            ORDER BY fecha ASC
        """

        df_monthly = pd.read_sql(monthly_query, conn)
        if len(df_monthly) > 0:
            df_monthly['fecha'] = pd.to_datetime(df_monthly['fecha'])
            df_monthly = df_monthly.set_index('fecha')
            df_monthly = df_monthly.rename(columns=MONTHLY_MAPPING)

            # Apply bounded FFill (max 35 days)
            df_monthly = _apply_bounded_ffill(df_monthly, max_days=FFILL_LIMITS['monthly'])
            logger.info(f"[L2-MACRO] Monthly: {len(df_monthly)} rows, {len(df_monthly.columns)} cols (FFill limit: 35 days)")

            # Join to daily: map each daily date to its month's value
            df_daily['_month_start'] = df_daily.index.to_period('M').to_timestamp()
            for col in df_monthly.columns:
                monthly_values = df_monthly[col].to_dict()
                df_daily[col] = df_daily['_month_start'].map(monthly_values)
            df_daily = df_daily.drop(columns=['_month_start'])
        else:
            logger.warning("[L2-MACRO] Monthly table is empty")

    except Exception as e:
        logger.warning(f"[L2-MACRO] Monthly table not available: {e}")

    # ==========================================================================
    # STEP 3: Load QUARTERLY macro
    # ==========================================================================
    try:
        quarterly_cols = list(QUARTERLY_MAPPING.keys())
        quarterly_query = f"""
            SELECT fecha, {', '.join(quarterly_cols)}
            FROM macro_indicators_quarterly
            ORDER BY fecha ASC
        """

        df_quarterly = pd.read_sql(quarterly_query, conn)
        if len(df_quarterly) > 0:
            df_quarterly['fecha'] = pd.to_datetime(df_quarterly['fecha'])
            df_quarterly = df_quarterly.set_index('fecha')
            df_quarterly = df_quarterly.rename(columns=QUARTERLY_MAPPING)

            # Apply bounded FFill (max 95 days)
            df_quarterly = _apply_bounded_ffill(df_quarterly, max_days=FFILL_LIMITS['quarterly'])
            logger.info(f"[L2-MACRO] Quarterly: {len(df_quarterly)} rows, {len(df_quarterly.columns)} cols (FFill limit: 95 days)")

            # Join to daily: map each daily date to its quarter's value
            df_daily['_quarter_start'] = df_daily.index.to_period('Q').to_timestamp()
            for col in df_quarterly.columns:
                quarterly_values = df_quarterly[col].to_dict()
                df_daily[col] = df_daily['_quarter_start'].map(quarterly_values)
            df_daily = df_daily.drop(columns=['_quarter_start'])
        else:
            logger.warning("[L2-MACRO] Quarterly table is empty")

    except Exception as e:
        logger.warning(f"[L2-MACRO] Quarterly table not available: {e}")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    logger.info(f"[L2-MACRO] Combined macro data: {len(df_daily)} rows, {len(df_daily.columns)} columns")
    logger.info(f"[L2-MACRO] Columns: {list(df_daily.columns)}")

    return df_daily


def _apply_bounded_ffill(df: pd.DataFrame, max_days: int) -> pd.DataFrame:
    """
    Apply forward-fill with a maximum limit (anti-leakage).

    Contract: CTR-L0-FFILL-001

    Args:
        df: DataFrame indexed by date
        max_days: Maximum days to forward-fill

    Returns:
        DataFrame with bounded forward-fill applied
    """
    if len(df) == 0:
        return df

    df = df.copy()

    for col in df.columns:
        # Track days since last valid value
        valid_mask = df[col].notna()
        if not valid_mask.any():
            continue

        # Apply ffill with limit
        df[col] = df[col].ffill(limit=max_days)

        # Additional check: set to NaN if beyond limit
        # This is a safety net for gaps that are larger than expected
        last_valid_idx = None
        for idx in df.index:
            if valid_mask.loc[idx]:
                last_valid_idx = idx
            elif last_valid_idx is not None:
                days_since = (idx - last_valid_idx).days
                if days_since > max_days:
                    df.loc[idx, col] = np.nan

    return df


# =============================================================================
# HOLIDAY FILTERING (US + Colombia)
# =============================================================================

def filter_holidays_and_weekends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out weekends and US/Colombia holidays from the dataset.

    Contract: CTR-L2-HOLIDAYS-001

    Uses TradingCalendar to identify non-trading days:
    - Saturdays and Sundays
    - US market holidays (NYSE closed)
    - Colombian market holidays

    Args:
        df: DataFrame with DatetimeIndex

    Returns:
        DataFrame with non-trading days removed
    """
    if not TRADING_CALENDAR_AVAILABLE or _calendar is None:
        logger.warning("[L2] TradingCalendar not available - skipping holiday filter")
        return df

    if len(df) == 0:
        return df

    df = df.copy()
    initial_rows = len(df)

    # Get dates from index
    if isinstance(df.index, pd.DatetimeIndex):
        dates = df.index
    else:
        logger.warning("[L2] DataFrame index is not DatetimeIndex - skipping holiday filter")
        return df

    # Filter: keep only trading days
    trading_mask = dates.map(lambda dt: _calendar.is_trading_day(dt))
    df = df[trading_mask]

    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        logger.info(f"[L2] Removed {removed_rows} rows from holidays/weekends ({removed_rows/initial_rows*100:.1f}%)")

    return df


# =============================================================================
# ANTI-LEAKAGE PREPROCESSING
# =============================================================================

def merge_ohlcv_macro(ohlcv_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge OHLCV (5-min) with Macro (daily) by resampling macro to 5-min grid.

    Contract: CTR-FEATURES-002, CTR-L2-RESAMPLE-001

    Process:
    1. T-1 Shift: Apply 1-day lag to macro (anti-leakage - use yesterday's data)
    2. Resample: Broadcast daily macro value to all 5-min bars of that day
    3. FFill within session: Only fill gaps within same trading session

    Anti-leakage guarantees:
    - Macro value at time T uses data from T-1 (yesterday's close)
    - No forward-fill across session boundaries
    - No future data leakage

    Args:
        ohlcv_df: 5-min OHLCV data indexed by datetime (contains 'close')
        macro_df: Daily macro data indexed by date

    Returns:
        DataFrame with OHLCV + macro columns at 5-min frequency
    """
    ohlcv_df = ohlcv_df.copy()
    macro_df = macro_df.copy()

    logger.info(f"[L2-MERGE] OHLCV: {len(ohlcv_df)} bars (5-min)")
    logger.info(f"[L2-MERGE] Macro: {len(macro_df)} days")

    # ==========================================================================
    # STEP 1: T-1 SHIFT (Anti-Leakage)
    # ==========================================================================
    # Shift macro by 1 day: for any date D, we use macro from D-1
    # This ensures no same-day lookahead bias
    macro_shifted = macro_df.shift(1)
    macro_shifted.index = pd.to_datetime(macro_shifted.index)

    logger.info("[L2-MERGE] Applied T-1 shift to macro (using yesterday's values)")

    # ==========================================================================
    # STEP 2: RESAMPLE DAILY MACRO TO 5-MIN GRID
    # ==========================================================================
    # Create date column for merge
    ohlcv_df['_merge_date'] = ohlcv_df.index.date

    # Convert macro index to date for merge
    macro_shifted['_merge_date'] = macro_shifted.index.date
    macro_shifted = macro_shifted.reset_index(drop=True)

    # Merge: each 5-min bar gets the macro value from its date (which is T-1 shifted)
    result = ohlcv_df.merge(
        macro_shifted,
        on='_merge_date',
        how='left'
    )

    # Restore datetime index
    result = result.set_index(ohlcv_df.index)
    result = result.drop(columns=['_merge_date'])

    logger.info(f"[L2-MERGE] After resample: {len(result)} rows, {len(result.columns)} columns")

    # ==========================================================================
    # STEP 3: FFILL WITHIN SESSIONS (handle gaps in macro)
    # ==========================================================================
    # For any missing macro values (weekends, holidays), fill from previous session
    # but ONLY within reasonable limits
    result = ffill_within_sessions(result)

    # Log merge statistics
    macro_cols = [c for c in result.columns if c != 'close']
    for col in macro_cols[:5]:  # Log first 5 macro columns
        non_null_pct = result[col].notna().mean() * 100
        logger.info(f"[L2-MERGE] {col}: {non_null_pct:.1f}% coverage")

    logger.info(f"[L2-MERGE] Final merged: {len(result)} rows, {len(result.columns)} columns")
    return result


def ffill_within_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill NaN values ONLY within the same trading session.

    A session is defined as:
    - Same calendar date
    - Within market hours (08:00-13:00 COT)

    This prevents look-ahead bias from filling across session boundaries.
    """
    df = df.copy()

    # Create session identifier
    df['_session_date'] = df.index.date
    df['_session_hour'] = df.index.hour

    # Group by session date and forward-fill within groups
    for col in df.columns:
        if col.startswith('_'):
            continue

        # Group by session and ffill within each group
        df[col] = df.groupby('_session_date')[col].ffill()

    # Drop helper columns
    df = df.drop(columns=['_session_date', '_session_hour'])

    return df


# =============================================================================
# FEATURE CALCULATION
# =============================================================================

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all 18 market features using CLOSE-ONLY data.

    Contract: CTR-FEATURES-002 (CLOSE-ONLY), CTR-EXPERIMENT-001 v3.1 (EXP-B-001)

    Features calculated (v3.1 EXP-B-001 - 18 market features):
    - log_ret_5m, log_ret_1h, log_ret_4h, log_ret_1d (returns from CLOSE)
    - rsi_9, rsi_21 (RSI from CLOSE)
    - volatility_pct (realized vol from returns - REPLACES ATR)
    - trend_z (price vs SMA z-score - REPLACES ADX)
    - dxy_z, dxy_change_1d, vix_z, embi_z (macro z-scores)
    - brent_change_1d, rate_spread_z, rate_spread_change, usdmxn_change_1d (macro derived)
    - yield_curve_z, gold_change_1d (EXP-B-001 new)

    CRITICAL FIX (v2.1): Also adds raw_log_ret_5m for TradingEnv PnL calculation.
    This column is NOT normalized and represents actual returns.

    NOTA: NO usamos O/H/L - solo CLOSE para máxima confiabilidad.

    Uses SSOT functions from src/features/technical_indicators.py when available.
    """
    result = df.copy()

    # ---------------------------------------------------------------------
    # RETURNS (log returns from CLOSE)
    # Uses SSOT calculate_log_returns when available
    # ---------------------------------------------------------------------
    if TECHNICAL_INDICATORS_AVAILABLE:
        result['log_ret_5m'] = calculate_log_returns(result['close'], periods=1)
        result['log_ret_1h'] = calculate_log_returns(result['close'], periods=12)
        result['log_ret_4h'] = calculate_log_returns(result['close'], periods=48)
        result['log_ret_1d'] = calculate_log_returns(result['close'], periods=288)  # EXP-B-001
    else:
        result['log_ret_5m'] = np.log(result['close'] / result['close'].shift(1))
        result['log_ret_1h'] = np.log(result['close'] / result['close'].shift(12))
        result['log_ret_4h'] = np.log(result['close'] / result['close'].shift(48))
        result['log_ret_1d'] = np.log(result['close'] / result['close'].shift(288))  # EXP-B-001

    # ---------------------------------------------------------------------
    # RAW RETURNS FOR PNL (CRITICAL FIX v2.1)
    # ---------------------------------------------------------------------
    # raw_log_ret_5m is NOT normalized and used by TradingEnv for PnL calculation.
    # This preserves actual return magnitudes (~0.0001 per 5-min bar).
    # The normalized log_ret_5m (z-scored) is used as observation feature.
    result['raw_log_ret_5m'] = result['log_ret_5m'].copy()

    # ---------------------------------------------------------------------
    # TECHNICAL INDICATORS (CLOSE-ONLY)
    # Uses SSOT functions from technical_indicators.py
    # ---------------------------------------------------------------------
    if TECHNICAL_INDICATORS_AVAILABLE:
        # SSOT functions for CLOSE-only indicators
        result['rsi_9'] = calculate_rsi_wilders(result['close'], period=9)
        result['rsi_21'] = calculate_rsi_wilders(result['close'], period=21)  # EXP-B-001
        result['volatility_pct'] = calculate_volatility_pct(
            result['close'], period=14, annualize=True, bars_per_day=48
        )
        result['trend_z'] = calculate_trend_z(
            result['close'], sma_period=50, clip_value=3.0
        )
    else:
        # Fallback inline implementations
        result['rsi_9'] = _calculate_rsi_wilders_inline(result['close'], period=9)
        result['rsi_21'] = _calculate_rsi_wilders_inline(result['close'], period=21)  # EXP-B-001

        # VOLATILITY_PCT: Realized volatility (REEMPLAZA ATR)
        log_returns = np.log(result['close'] / result['close'].shift(1))
        annualization_factor = np.sqrt(252 * 48)
        result['volatility_pct'] = log_returns.rolling(window=14).std() * annualization_factor

        # TREND_Z: Price position vs SMA (REEMPLAZA ADX)
        sma_50 = result['close'].rolling(window=50).mean()
        rolling_std_50 = result['close'].rolling(window=50).std()
        result['trend_z'] = ((result['close'] - sma_50) / rolling_std_50.clip(lower=1e-6)).clip(-3, 3)

    # ---------------------------------------------------------------------
    # MACRO Z-SCORES
    # ---------------------------------------------------------------------

    # DXY z-score (rolling 252-day stats)
    # FIX 2026-02-01: Use shift(1) to avoid look-ahead bias (only use T-1 data)
    if 'dxy' in result.columns:
        dxy_shifted = result['dxy'].shift(1)  # T-1 value for z-score calculation
        dxy_mean = dxy_shifted.rolling(252, min_periods=20).mean()
        dxy_std = dxy_shifted.rolling(252, min_periods=20).std()
        result['dxy_z'] = (dxy_shifted - dxy_mean) / dxy_std.clip(lower=0.01)
        result['dxy_change_1d'] = result['dxy'].pct_change(1)  # Change is naturally T-1 based
    else:
        result['dxy_z'] = 0.0
        result['dxy_change_1d'] = 0.0

    # VIX z-score
    # FIX 2026-02-01: Use shift(1) to avoid look-ahead bias
    if 'vix' in result.columns:
        vix_shifted = result['vix'].shift(1)
        vix_mean = vix_shifted.rolling(252, min_periods=20).mean()
        vix_std = vix_shifted.rolling(252, min_periods=20).std()
        result['vix_z'] = (vix_shifted - vix_mean) / vix_std.clip(lower=0.01)
    else:
        result['vix_z'] = 0.0

    # EMBI z-score
    # FIX 2026-02-01: Use shift(1) to avoid look-ahead bias
    if 'embi' in result.columns:
        embi_shifted = result['embi'].shift(1)
        embi_mean = embi_shifted.rolling(252, min_periods=20).mean()
        embi_std = embi_shifted.rolling(252, min_periods=20).std()
        result['embi_z'] = (embi_shifted - embi_mean) / embi_std.clip(lower=0.01)
    else:
        result['embi_z'] = 0.0

    # ---------------------------------------------------------------------
    # MACRO DERIVED
    # ---------------------------------------------------------------------

    # Brent change
    if 'brent' in result.columns:
        result['brent_change_1d'] = result['brent'].pct_change(1)
    else:
        result['brent_change_1d'] = 0.0

    # Rate spread z-score (Colombia 10Y - US 10Y) - EXP-B-001: Now z-score
    if 'col10y' in result.columns and 'ust10y' in result.columns:
        rate_spread = result['col10y'] - result['ust10y']
        rs_shifted = rate_spread.shift(1)
        rs_mean = rs_shifted.rolling(252, min_periods=20).mean()
        rs_std = rs_shifted.rolling(252, min_periods=20).std()
        result['rate_spread_z'] = (rs_shifted - rs_mean) / rs_std.clip(lower=0.01)
        result['rate_spread_z'] = result['rate_spread_z'].clip(-4, 4)
        # EXP-B-001: Rate spread momentum (change in spread)
        result['rate_spread_change'] = rate_spread.pct_change(288).fillna(0)  # 1-day change
    else:
        result['rate_spread_z'] = 0.0
        result['rate_spread_change'] = 0.0

    # USD/MXN change
    if 'usdmxn' in result.columns:
        result['usdmxn_change_1d'] = result['usdmxn'].pct_change(1)
    else:
        result['usdmxn_change_1d'] = 0.0

    # ---------------------------------------------------------------------
    # EXP-B-001 NEW FEATURES
    # ---------------------------------------------------------------------

    # Yield curve z-score (US 10Y - 2Y)
    if 'ust10y' in result.columns and 'ust2y' in result.columns:
        yield_curve = result['ust10y'] - result['ust2y']
        yc_shifted = yield_curve.shift(1)
        yc_mean = yc_shifted.rolling(252, min_periods=20).mean()
        yc_std = yc_shifted.rolling(252, min_periods=20).std()
        result['yield_curve_z'] = (yc_shifted - yc_mean) / yc_std.clip(lower=0.01)
        result['yield_curve_z'] = result['yield_curve_z'].clip(-4, 4)
    else:
        result['yield_curve_z'] = 0.0

    # Gold change (safe haven indicator)
    if 'gold' in result.columns:
        result['gold_change_1d'] = result['gold'].pct_change(288).fillna(0)  # 1-day change in 5-min bars
    else:
        result['gold_change_1d'] = 0.0

    # Select the 18 market features + raw_log_ret_5m for PnL
    feature_columns = list(MARKET_FEATURES) + ['raw_log_ret_5m']
    result = result[feature_columns]

    logger.info(f"Calculated {len(feature_columns)} features (including raw_log_ret_5m for PnL)")
    return result


def _calculate_rsi_wilders_inline(close: pd.Series, period: int = 9) -> pd.Series:
    """
    Fallback RSI calculation using Wilder's smoothing.

    NOTA: Preferir usar calculate_rsi_wilders() de technical_indicators.py (SSOT).
    Esta función solo existe como fallback cuando el módulo SSOT no está disponible.
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Wilder's EMA: alpha = 1/period
    alpha = 1.0 / period

    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    rs = avg_gain / avg_loss.clip(lower=1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def _calculate_atr_wilders_deprecated(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    DEPRECATED: ATR using Wilder's smoothing.

    This function requires H/L data which is unreliable in our dataset.
    Use calculate_volatility_pct() from technical_indicators.py instead.

    Kept for reference/backwards compatibility only.
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's EMA
    alpha = 1.0 / period
    atr = true_range.ewm(alpha=alpha, adjust=False).mean()

    return atr


def _calculate_adx_wilders_deprecated(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    DEPRECATED: ADX using Wilder's smoothing.

    This function requires H/L data which is unreliable in our dataset.
    Use calculate_trend_z() from technical_indicators.py instead.

    Kept for reference/backwards compatibility only.

    FIX v2: Use percentage-based ATR to avoid saturation issues with USDCOP.
    """
    # +DM and -DM as percentage of price
    high_diff = high.diff()
    low_diff = low.diff()

    # DM as percentage of close for consistent scaling
    plus_dm_pct = (high_diff / close).where((high_diff > low_diff.abs()) & (high_diff > 0), 0.0)
    minus_dm_pct = (low_diff.abs() / close).where((low_diff.abs() > high_diff) & (low_diff < 0), 0.0)

    # ATR as percentage for normalization
    atr = _calculate_atr_wilders_deprecated(high, low, close, period)
    atr_pct = (atr / close).clip(lower=1e-6)  # Minimum 0.0001% to avoid div/0

    # Smoothed +DI and -DI
    alpha = 1.0 / period
    plus_di = 100.0 * plus_dm_pct.ewm(alpha=alpha, adjust=False).mean() / atr_pct
    minus_di = 100.0 * minus_dm_pct.ewm(alpha=alpha, adjust=False).mean() / atr_pct

    # Clamp DI values to valid range before DX calculation
    plus_di = plus_di.clip(0, 100)
    minus_di = minus_di.clip(0, 100)

    # DX and ADX
    di_sum = (plus_di + minus_di).clip(lower=1.0)  # Minimum 1 to avoid div/0
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return adx.clip(0, 100)  # Ensure final output is bounded


# =============================================================================
# NORMALIZATION
# =============================================================================

def compute_normalization_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute normalization statistics (mean, std) for each feature.

    Should be computed on TRAINING data only to prevent leakage.
    """
    stats = {}

    for col in df.columns:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            stats[col] = {
                "mean": float(col_data.mean()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "count": int(len(col_data)),
            }
        else:
            stats[col] = {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 0.0, "count": 0}

    return stats


def apply_normalization(df: pd.DataFrame, stats: Dict[str, Dict[str, float]], clip_range: float = 10.0) -> pd.DataFrame:
    """
    Apply z-score normalization using pre-computed statistics.

    Args:
        df: DataFrame to normalize
        stats: Pre-computed mean/std for each column
        clip_range: Clip normalized values to ±clip_range

    CRITICAL: raw_log_ret_5m is EXCLUDED from normalization.
    It preserves actual return magnitudes for TradingEnv PnL calculation.
    """
    # Columns to NEVER normalize (used for PnL calculation)
    SKIP_NORMALIZATION = {'raw_log_ret_5m', 'close'}

    result = df.copy()

    for col in result.columns:
        # Skip columns that should not be normalized
        if col in SKIP_NORMALIZATION:
            logger.info(f"[NORM] Skipping normalization for {col} (used for PnL)")
            continue

        if col in stats:
            mean = stats[col]["mean"]
            std = stats[col]["std"]
            if std > 0:
                result[col] = (result[col] - mean) / std
                result[col] = result[col].clip(-clip_range, clip_range)
            else:
                result[col] = 0.0

    return result


# =============================================================================
# DATASET SPLITTING
# =============================================================================

def split_dataset(df: pd.DataFrame, date_ranges: DateRanges) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/val/test based on date ranges.
    """
    train_mask = (df.index >= date_ranges.training_start) & (df.index <= date_ranges.training_end)
    val_mask = (df.index >= date_ranges.validation_start) & (df.index <= date_ranges.validation_end)
    test_mask = (df.index >= date_ranges.test_start) & (df.index <= date_ranges.test_end)

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df


# =============================================================================
# DESCRIPTIVE STATISTICS
# =============================================================================

def generate_variable_statistics(
    df: pd.DataFrame,
    output_path: Path,
    dataset_name: str = "dataset"
) -> Dict[str, Any]:
    """
    Generate comprehensive descriptive statistics for each variable.

    Contract: CTR-L2-STATS-001

    Statistics generated per variable:
    - count, mean, std, min, max
    - percentiles (1%, 5%, 25%, 50%, 75%, 95%, 99%)
    - skewness, kurtosis
    - missing count and percentage
    - zero count and percentage
    - date range coverage

    Args:
        df: DataFrame to analyze
        output_path: Path to save statistics files
        dataset_name: Name prefix for output files

    Returns:
        Dictionary with statistics summary
    """
    from scipy import stats as scipy_stats

    logger.info("=" * 70)
    logger.info("[L2-STATS] Generating Descriptive Statistics")
    logger.info("=" * 70)

    stats_records = []
    stats_dict = {}

    for col in df.columns:
        col_data = df[col]
        non_null = col_data.dropna()

        # Basic statistics
        record = {
            'variable': col,
            'count': len(non_null),
            'missing': col_data.isna().sum(),
            'missing_pct': col_data.isna().mean() * 100,
            'zeros': (col_data == 0).sum(),
            'zeros_pct': (col_data == 0).mean() * 100 if len(col_data) > 0 else 0,
            'mean': non_null.mean() if len(non_null) > 0 else np.nan,
            'std': non_null.std() if len(non_null) > 0 else np.nan,
            'min': non_null.min() if len(non_null) > 0 else np.nan,
            'max': non_null.max() if len(non_null) > 0 else np.nan,
        }

        # Percentiles
        if len(non_null) > 0:
            percentiles = [1, 5, 25, 50, 75, 95, 99]
            for p in percentiles:
                record[f'p{p}'] = np.percentile(non_null, p)

            # Skewness and Kurtosis
            if len(non_null) > 2:
                try:
                    record['skewness'] = scipy_stats.skew(non_null, nan_policy='omit')
                    record['kurtosis'] = scipy_stats.kurtosis(non_null, nan_policy='omit')
                except Exception:
                    record['skewness'] = np.nan
                    record['kurtosis'] = np.nan
            else:
                record['skewness'] = np.nan
                record['kurtosis'] = np.nan
        else:
            for p in [1, 5, 25, 50, 75, 95, 99]:
                record[f'p{p}'] = np.nan
            record['skewness'] = np.nan
            record['kurtosis'] = np.nan

        stats_records.append(record)
        stats_dict[col] = record

        # Log summary for each variable
        logger.info(
            f"  {col:25s}: count={record['count']:>8,} | "
            f"mean={record['mean']:>10.4f} | std={record['std']:>10.4f} | "
            f"missing={record['missing_pct']:>5.1f}%"
        )

    # Create DataFrame
    stats_df = pd.DataFrame(stats_records)

    # Save to CSV
    csv_path = output_path / f"{dataset_name}_variable_statistics.csv"
    stats_df.to_csv(csv_path, index=False)
    logger.info(f"[L2-STATS] Saved CSV: {csv_path}")

    # Save to JSON
    json_path = output_path / f"{dataset_name}_variable_statistics.json"
    with open(json_path, 'w') as f:
        json.dump(stats_dict, f, indent=2, default=str)
    logger.info(f"[L2-STATS] Saved JSON: {json_path}")

    # Summary
    total_vars = len(df.columns)
    complete_vars = sum(1 for r in stats_records if r['missing_pct'] == 0)
    high_missing = sum(1 for r in stats_records if r['missing_pct'] > 10)

    summary = {
        'total_variables': total_vars,
        'complete_variables': complete_vars,
        'high_missing_variables': high_missing,
        'total_rows': len(df),
        'date_range': {
            'start': str(df.index.min()) if isinstance(df.index, pd.DatetimeIndex) else None,
            'end': str(df.index.max()) if isinstance(df.index, pd.DatetimeIndex) else None,
        },
        'csv_path': str(csv_path),
        'json_path': str(json_path),
    }

    logger.info("=" * 70)
    logger.info(f"[L2-STATS] Summary: {total_vars} variables, {complete_vars} complete, {high_missing} with >10% missing")
    logger.info("=" * 70)

    return summary


# =============================================================================
# SSOT-BASED PIPELINE FUNCTION (NEW v2.0)
# =============================================================================

def build_dataset_ssot(**context) -> Dict[str, Any]:
    """
    Build dataset using pipeline_ssot.yaml configuration.

    This uses the new SSOTDatasetBuilder which reads feature definitions
    dynamically from pipeline_ssot.yaml. Enable via DAG param: use_ssot_builder=True

    Benefits:
    - Dynamic feature calculation from SSOT
    - Single source of truth for L2, L3, L4
    - Automatic training/backtest parity validation
    """
    if not PIPELINE_SSOT_AVAILABLE:
        raise ImportError("SSOT builder not available - missing pipeline_config or ssot_dataset_builder")

    logger.info("=" * 70)
    logger.info("L2 DATASET BUILDER (SSOT v2.0) - Starting")
    logger.info("=" * 70)

    dag_conf = context.get('dag_run').conf if context.get('dag_run') else {}

    # Get dataset prefix from SSOT
    dataset_prefix = dag_conf.get('dataset_name', _pipeline_config.paths.l2_dataset_prefix)

    logger.info(f"SSOT Version: {_pipeline_config.version}")
    logger.info(f"Based on model: {_pipeline_config.based_on_model}")
    logger.info(f"Dataset prefix: {dataset_prefix}")

    # Validate training/backtest parity
    parity_issues = _pipeline_config.validate_training_backtest_parity()
    if parity_issues:
        logger.warning("Training/Backtest parity issues detected:")
        for issue in parity_issues:
            logger.warning(f"  - {issue}")

    # Connect to database
    conn = get_db_connection()

    try:
        # Load raw data from database
        date_ranges = _pipeline_config.date_ranges
        data_start = date_ranges.train_start
        data_end = date_ranges.test_end

        logger.info(f"Loading data from {data_start} to {data_end}")

        # Load OHLCV
        ohlcv_df = load_ohlcv_data(conn, data_start, data_end)

        # Load Macro
        macro_df = load_macro_data(conn, data_start, data_end)

        # Build dataset using SSOT builder
        builder = SSOTDatasetBuilder()
        result = builder.build(
            df_ohlcv=ohlcv_df.reset_index(),  # Reset index to have 'time' as column
            df_macro=macro_df.reset_index(),
            output_dir=OUTPUT_DIR,
            dataset_prefix=dataset_prefix
        )

        # Create lineage record compatible with legacy format
        lineage = LineageInfo(
            dataset_hash=hashlib.sha256(str(result.lineage).encode()).hexdigest()[:16],
            config_hash=_pipeline_config.version,
            feature_order_hash=result.lineage.get('feature_order_hash', ''),
            norm_stats_hash=hashlib.sha256(str(result.norm_stats).encode()).hexdigest()[:16],
            builder_version="SSOTDatasetBuilder_v1.0.0",
            created_at=datetime.now().isoformat(),
            date_range={"start": data_start, "end": data_end},
            num_rows={
                "train": len(result.train_df),
                "val": len(result.val_df),
                "test": len(result.test_df),
                "total": len(result.train_df) + len(result.val_df) + len(result.test_df),
            },
            num_features=len(result.feature_columns),
        )

        paths = get_output_paths(dataset_prefix)

        # Create output for XCom
        output = DatasetOutput(
            train_path=str(paths["train"]),
            val_path=str(paths["val"]),
            test_path=str(paths["test"]),
            norm_stats_path=str(paths["norm_stats"]),
            lineage_path=str(paths["lineage"]),
            lineage=lineage,
            quality_report_dir=str(paths["quality_report_dir"]),
            success=True,
        )

        # Push to XCom
        ti = context.get('ti')
        if ti:
            ti.xcom_push(key='dataset_output', value=asdict(output))
            ti.xcom_push(key='train_path', value=str(paths["train"]))
            ti.xcom_push(key='norm_stats_path', value=str(paths["norm_stats"]))
            ti.xcom_push(key='dataset_hash', value=lineage.dataset_hash)

        logger.info("=" * 70)
        logger.info("L2 DATASET BUILDER (SSOT v2.0) - Complete")
        logger.info(f"  Train: {len(result.train_df)} rows")
        logger.info(f"  Val:   {len(result.val_df)} rows")
        logger.info(f"  Test:  {len(result.test_df)} rows")
        logger.info(f"  Features: {len(result.feature_columns)}")
        logger.info("=" * 70)

        return asdict(output)

    finally:
        conn.close()


# =============================================================================
# MAIN PIPELINE FUNCTION (Legacy)
# =============================================================================

def build_dataset(**context) -> Dict[str, Any]:
    """
    Main pipeline function: Build complete dataset for training.

    Steps:
    1. Load configuration + date ranges (SSOT)
    2. Load OHLCV data from usdcop_m5_ohlcv
    3. Load Macro data from macro_combined_for_l2 (3 tables joined)
    4. Merge OHLCV + Macro with anti-leakage (T-1 shift)
    5. Filter holidays and weekends (US + Colombia)
    6. Calculate features (CLOSE-ONLY v2.0)
    7. Drop NaN rows
    8. Split train/val/test
    9. Compute normalization stats (TRAIN only - anti-leakage)
    10. Apply normalization + Save outputs + Create lineage

    Note: Set use_ssot_builder=True in DAG params to use new SSOT builder instead.
    """
    # Check if user wants SSOT builder
    dag_conf = context.get('dag_run').conf if context.get('dag_run') else {}
    use_ssot = dag_conf.get('use_ssot_builder', False)

    if use_ssot and PIPELINE_SSOT_AVAILABLE:
        logger.info("[L2] Using SSOT builder (use_ssot_builder=True)")
        return build_dataset_ssot(**context)

    logger.info("=" * 70)
    logger.info("L2 DATASET BUILDER - Starting")
    logger.info("=" * 70)

    # Get configuration from dag_run.conf or defaults
    dag_conf = context.get('dag_run').conf if context.get('dag_run') else {}

    # Use SSOT output prefix when available
    if EXPERIMENT_SSOT_AVAILABLE and EXPERIMENT_CONFIG is not None:
        ssot_prefix = EXPERIMENT_CONFIG.pipeline.output_prefix
        experiment_name = dag_conf.get('experiment_name', ssot_prefix.replace('DS_', ''))
        dataset_name = dag_conf.get('dataset_name', ssot_prefix)
    else:
        experiment_name = dag_conf.get('experiment_name', 'default')
        dataset_name = dag_conf.get('dataset_name', f'DS_{experiment_name}')

    # Load experiment config if available
    exp_config = load_experiment_config(experiment_name)

    # Load date ranges (SSOT)
    date_ranges = load_date_ranges()

    # Determine date range for data loading
    if exp_config and 'data' in exp_config:
        data_start = exp_config['data'].get('date_range', {}).get('start', date_ranges.training_start)
        data_end = exp_config['data'].get('date_range', {}).get('end', date_ranges.test_end)
    else:
        data_start = date_ranges.training_start
        data_end = date_ranges.test_end

    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Date range: {data_start} to {data_end}")

    # Connect to database
    conn = get_db_connection()

    try:
        # ---------------------------------------------------------------------
        # STEP 1: Load raw data
        # ---------------------------------------------------------------------
        logger.info("[Step 1/9] Loading OHLCV data...")
        ohlcv_df = load_ohlcv_data(conn, data_start, data_end)

        logger.info("[Step 2/9] Loading Macro data...")
        macro_df = load_macro_data(conn, data_start, data_end)

        # ---------------------------------------------------------------------
        # STEP 2: Merge with anti-leakage
        # ---------------------------------------------------------------------
        logger.info("[Step 3/10] Merging OHLCV + Macro (anti-leakage)...")
        merged_df = merge_ohlcv_macro(ohlcv_df, macro_df)

        # ---------------------------------------------------------------------
        # STEP 3: Filter holidays and weekends (US + Colombia)
        # ---------------------------------------------------------------------
        logger.info("[Step 4/10] Filtering holidays and weekends (US + Colombia)...")
        merged_df = filter_holidays_and_weekends(merged_df)

        # ---------------------------------------------------------------------
        # STEP 4: Calculate features
        # ---------------------------------------------------------------------
        logger.info("[Step 5/10] Calculating features (CanonicalFeatureBuilder logic)...")
        features_df = calculate_features(merged_df)

        # ---------------------------------------------------------------------
        # STEP 5: Drop NaN rows
        # ---------------------------------------------------------------------
        logger.info("[Step 6/10] Dropping NaN rows...")
        initial_rows = len(features_df)
        features_df = features_df.dropna()
        dropped_rows = initial_rows - len(features_df)
        logger.info(f"  Dropped {dropped_rows} rows with NaN ({dropped_rows/initial_rows*100:.1f}%)")

        # ---------------------------------------------------------------------
        # STEP 6: Split into train/val/test
        # ---------------------------------------------------------------------
        logger.info("[Step 7/10] Splitting train/val/test...")
        train_df, val_df, test_df = split_dataset(features_df, date_ranges)

        # ---------------------------------------------------------------------
        # STEP 7: Compute normalization stats (TRAIN ONLY!)
        # ---------------------------------------------------------------------
        logger.info("[Step 8/10] Computing normalization stats (train only)...")
        norm_stats = compute_normalization_stats(train_df)

        # ---------------------------------------------------------------------
        # STEP 8: Apply normalization to all splits
        # ---------------------------------------------------------------------
        logger.info("[Step 9/10] Applying normalization...")
        train_normalized = apply_normalization(train_df, norm_stats)
        val_normalized = apply_normalization(val_df, norm_stats)
        test_normalized = apply_normalization(test_df, norm_stats)

        # ---------------------------------------------------------------------
        # STEP 9: Save outputs
        # ---------------------------------------------------------------------
        logger.info("[Step 10/10] Saving outputs...")
        paths = get_output_paths(dataset_name)

        # Save parquet files
        train_normalized.to_parquet(paths["train"])
        val_normalized.to_parquet(paths["val"])
        test_normalized.to_parquet(paths["test"])

        # Save norm_stats
        with open(paths["norm_stats"], 'w') as f:
            json.dump({
                "features": norm_stats,
                "created_at": datetime.now().isoformat(),
                "train_rows": len(train_df),
                "feature_order": list(MARKET_FEATURES),
            }, f, indent=2)

        # Create lineage record
        lineage = LineageInfo(
            dataset_hash=compute_file_hash(paths["train"]),
            config_hash=compute_config_hash(exp_config or {}),
            feature_order_hash=FEATURE_ORDER_HASH,
            norm_stats_hash=compute_file_hash(paths["norm_stats"]),
            builder_version="L2DatasetBuilder_v1.0.0",
            created_at=datetime.now().isoformat(),
            date_range={"start": data_start, "end": data_end},
            num_rows={
                "train": len(train_normalized),
                "val": len(val_normalized),
                "test": len(test_normalized),
                "total": len(train_normalized) + len(val_normalized) + len(test_normalized),
            },
            num_features=len(MARKET_FEATURES),
        )

        # Save lineage
        with open(paths["lineage"], 'w') as f:
            json.dump(lineage.to_dict(), f, indent=2)

        # ---------------------------------------------------------------------
        # Generate Descriptive Statistics for each variable
        # ---------------------------------------------------------------------
        logger.info("[L2] Generating descriptive statistics for all variables...")

        # Statistics for raw features (before normalization)
        raw_stats_summary = generate_variable_statistics(
            df=features_df,
            output_path=paths["quality_report_dir"],
            dataset_name=f"{dataset_name}_raw"
        )

        # Statistics for normalized train data
        norm_stats_summary = generate_variable_statistics(
            df=train_normalized,
            output_path=paths["quality_report_dir"],
            dataset_name=f"{dataset_name}_normalized_train"
        )

        # Create output for XCom
        output = DatasetOutput(
            train_path=str(paths["train"]),
            val_path=str(paths["val"]),
            test_path=str(paths["test"]),
            norm_stats_path=str(paths["norm_stats"]),
            lineage_path=str(paths["lineage"]),
            lineage=lineage,
            quality_report_dir=str(paths["quality_report_dir"]),
            success=True,
        )

        # Push to XCom using contracts for L3 compatibility
        ti = context.get('ti')
        if ti:
            # Legacy keys for backward compatibility
            ti.xcom_push(key='dataset_output', value=asdict(output))
            ti.xcom_push(key='train_path', value=str(paths["train"]))
            ti.xcom_push(key='norm_stats_path', value=str(paths["norm_stats"]))
            ti.xcom_push(key='dataset_hash', value=lineage.dataset_hash)
            ti.xcom_push(key='config_hash', value=lineage.config_hash)

            # Contract-based XCom push (SSOT for L3 consumption)
            if XCOM_CONTRACTS_AVAILABLE and L2Output is not None:
                l2_output = L2Output(
                    dataset_path=str(paths["train"]),
                    dataset_hash=lineage.dataset_hash,
                    date_range_start=data_start,
                    date_range_end=data_end,
                    feature_order_hash=FEATURE_ORDER_HASH,
                    feature_columns=list(MARKET_FEATURES),
                    row_count=lineage.num_rows.get("train", 0),
                    experiment_name=experiment_name,
                    norm_stats_path=str(paths["norm_stats"]),
                    manifest_path=str(paths["lineage"]),
                )
                l2_output.push_to_xcom(ti)
                logger.info("[SSOT] L2Output pushed to XCom via contracts")

        logger.info("=" * 70)
        logger.info("L2 DATASET BUILDER - Complete")
        logger.info(f"  Train: {len(train_normalized)} rows → {paths['train']}")
        logger.info(f"  Val:   {len(val_normalized)} rows → {paths['val']}")
        logger.info(f"  Test:  {len(test_normalized)} rows → {paths['test']}")
        logger.info(f"  Hash:  {lineage.dataset_hash}")
        logger.info("=" * 70)

        return asdict(output)

    except Exception as e:
        logger.error(f"Dataset build failed: {e}")
        raise

    finally:
        conn.close()


def validate_output(**context) -> Dict[str, Any]:
    """Validate the generated dataset."""
    ti = context.get('ti')
    train_path = ti.xcom_pull(key='train_path')

    if not train_path or not Path(train_path).exists():
        raise ValueError(f"Train file not found: {train_path}")

    # Load and validate
    df = pd.read_parquet(train_path)

    validation_results = {
        "rows": len(df),
        "columns": len(df.columns),
        "null_count": int(df.isnull().sum().sum()),
        "inf_count": int(np.isinf(df.select_dtypes(include=[np.number])).sum().sum()),
        "feature_order_match": list(df.columns) == list(MARKET_FEATURES),
    }

    # Check for issues
    if validation_results["null_count"] > 0:
        logger.warning(f"Dataset has {validation_results['null_count']} null values")

    if validation_results["inf_count"] > 0:
        logger.warning(f"Dataset has {validation_results['inf_count']} infinite values")

    if not validation_results["feature_order_match"]:
        logger.warning("Feature order does not match SSOT contract!")

    logger.info(f"Validation results: {validation_results}")

    return validation_results


def generate_quality_report(**context) -> Dict[str, Any]:
    """
    Generate comprehensive data quality report for L2 dataset.

    Contract: CTR-L2-QUALITY-REPORT-001

    Generates per-variable statistics including:
    - Basic statistics (mean, std, min, max, percentiles, skewness, kurtosis)
    - Temporal coverage (date range, frequency, gaps)
    - Missing data analysis (count, %, gap ranges)
    - Anomaly detection (outliers, sudden jumps, zeros)
    - Distribution analysis (normality test, type)
    - Trend analysis (direction, stationarity)
    - Anti-leakage verification (T-1 shift, no future data)
    - Correlations (top 5 per variable)
    - Quality scores (0-100% with levels)

    Output:
    - JSON report (full structured data)
    - CSV summary (1 row per variable)
    - HTML report (visual with colors)
    """
    if not DATA_QUALITY_REPORT_AVAILABLE:
        logger.warning("[L2] Data quality report skipped - module not available")
        return {"status": "skipped", "reason": "module_not_available"}

    ti = context.get('ti')

    # Get paths from previous task
    dataset_output = ti.xcom_pull(task_ids='build_dataset', key='dataset_output')
    if not dataset_output:
        logger.warning("[L2] No dataset output found in XCom")
        return {"status": "skipped", "reason": "no_dataset_output"}

    train_path = dataset_output.get('train_path')
    val_path = dataset_output.get('val_path')
    test_path = dataset_output.get('test_path')
    norm_stats_path = dataset_output.get('norm_stats_path')
    lineage_path = dataset_output.get('lineage_path')

    if not train_path or not Path(train_path).exists():
        logger.warning(f"[L2] Train file not found: {train_path}")
        return {"status": "failed", "reason": "train_file_not_found"}

    logger.info("=" * 70)
    logger.info("L2 DATA QUALITY REPORT - Generating")
    logger.info("=" * 70)

    # Load datasets
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path) if val_path and Path(val_path).exists() else None
    test_df = pd.read_parquet(test_path) if test_path and Path(test_path).exists() else None

    # Combine all datasets for comprehensive analysis (with split marker)
    train_df['_split'] = 'train'
    all_dfs = [train_df]

    if val_df is not None:
        val_df['_split'] = 'val'
        all_dfs.append(val_df)

    if test_df is not None:
        test_df['_split'] = 'test'
        all_dfs.append(test_df)

    combined_df = pd.concat(all_dfs, axis=0)

    # Restore datetime index if available
    if 'time' in combined_df.columns:
        combined_df = combined_df.set_index('time')
    elif not isinstance(combined_df.index, pd.DatetimeIndex):
        # Create synthetic datetime index from position
        combined_df = combined_df.reset_index(drop=True)

    # Load norm_stats for transformation info
    norm_stats = None
    if norm_stats_path and Path(norm_stats_path).exists():
        with open(norm_stats_path, 'r') as f:
            norm_stats = json.load(f)

    # Determine output directory
    output_dir = Path(train_path).parent / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate report
    generator = L2DataQualityReportGenerator(
        zscore_threshold=3.0,
        iqr_multiplier=1.5,
        jump_threshold_pct=50.0,
    )

    # Use 'time' or index as date column
    date_column = 'time' if 'time' in combined_df.reset_index().columns else None

    # Reset index to have 'time' as column if it's a DatetimeIndex
    if isinstance(combined_df.index, pd.DatetimeIndex):
        combined_df = combined_df.reset_index()
        date_column = combined_df.columns[0]  # First column after reset

    report = generator.generate_report(
        df=combined_df,
        date_column=date_column,
        reference_date=datetime.now(),
        norm_stats=norm_stats,
        expected_frequency='5min',
        exclude_columns=[date_column, '_split', 'symbol', 'source'] if date_column else ['_split', 'symbol', 'source'],
    )

    # Add dataset-specific metadata
    report.dataset_name = Path(train_path).stem.replace('_train', '')

    # Save reports in all formats
    saved_files = generator.save_report(
        report=report,
        output_dir=output_dir,
        formats=['json', 'csv', 'html'],
    )

    # Log summary
    logger.info("=" * 70)
    logger.info("L2 DATA QUALITY REPORT - Complete")
    logger.info("=" * 70)
    logger.info(f"  Total Variables: {report.total_variables}")
    logger.info(f"  Total Records: {report.total_records:,}")
    logger.info(f"  Overall Quality: {report.overall_quality_score*100:.1f}% ({report.overall_quality_level})")
    logger.info(f"  Anti-Leakage Verified: {report.anti_leakage_verified}")
    logger.info("-" * 70)
    logger.info("  Quality Distribution:")
    logger.info(f"    Excellent: {report.variables_excellent}")
    logger.info(f"    Good: {report.variables_good}")
    logger.info(f"    Acceptable: {report.variables_acceptable}")
    logger.info(f"    Poor: {report.variables_poor}")
    logger.info(f"    Critical: {report.variables_critical}")
    logger.info("-" * 70)

    # Log per-variable summary
    logger.info("  Variable Summary:")
    for var_name, var_report in sorted(report.variables.items(), key=lambda x: x[1].quality_score, reverse=True):
        logger.info(
            f"    {var_name:20s}: {var_report.quality_score*100:5.1f}% ({var_report.quality_level:10s}) "
            f"| Missing: {var_report.missing.missing_percentage:5.1f}% "
            f"| Anomalies: {var_report.anomalies.total_anomalies:3d}"
        )

    logger.info("-" * 70)
    logger.info(f"  Reports saved to: {output_dir}")
    for fmt, path in saved_files.items():
        logger.info(f"    {fmt.upper()}: {path.name}")
    logger.info("=" * 70)

    # Push to XCom
    result = {
        "status": "success",
        "overall_quality_score": report.overall_quality_score,
        "overall_quality_level": report.overall_quality_level,
        "total_variables": report.total_variables,
        "total_records": report.total_records,
        "anti_leakage_verified": report.anti_leakage_verified,
        "quality_distribution": {
            "excellent": report.variables_excellent,
            "good": report.variables_good,
            "acceptable": report.variables_acceptable,
            "poor": report.variables_poor,
            "critical": report.variables_critical,
        },
        "recommendations": report.recommendations,
        "warnings": report.warnings,
        "report_paths": {
            fmt: str(path) for fmt, path in saved_files.items()
        },
    }

    ti.xcom_push(key='quality_report', value=result)
    ti.xcom_push(key='quality_report_json_path', value=str(saved_files.get('json', '')))
    ti.xcom_push(key='quality_report_html_path', value=str(saved_files.get('html', '')))

    return result


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'trading-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='L2 Dataset Builder - Unified Feature + Dataset Generation for Training',
    schedule_interval=None,  # Manual trigger
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['l2', 'dataset', 'features', 'training', 'ssot'],
    params={
        'experiment_name': 'default',
        'dataset_name': 'DS_production',
        'use_ssot_builder': True,  # Use new SSOT builder (reads from pipeline_ssot.yaml)
    },
) as dag:

    # Task: Build dataset
    build_task = PythonOperator(
        task_id='build_dataset',
        python_callable=build_dataset,
        provide_context=True,
        doc_md="""
        ### Build Dataset

        Executes the complete dataset building pipeline:
        1. Load OHLCV and Macro data
        2. Merge with anti-leakage guarantees
        3. Calculate 13 market features
        4. Normalize using train-only statistics
        5. Split into train/val/test
        6. Save parquet files with lineage
        """,
    )

    # Task: Validate output
    validate_task = PythonOperator(
        task_id='validate_output',
        python_callable=validate_output,
        provide_context=True,
        doc_md="""
        ### Validate Output

        Validates the generated dataset:
        - Check for null values
        - Check for infinite values
        - Verify feature order matches contract
        """,
    )

    # Task: Generate Quality Report
    quality_report_task = PythonOperator(
        task_id='generate_quality_report',
        python_callable=generate_quality_report,
        provide_context=True,
        doc_md="""
        ### Generate Data Quality Report

        Comprehensive statistical analysis per variable:

        **Statistics:**
        - Basic: count, mean, std, min, max, percentiles, skewness, kurtosis
        - Temporal: date range, frequency, coverage %
        - Missing: count, %, gaps, longest gap

        **Analysis:**
        - Anomalies: outliers (IQR/Z-score), sudden jumps, zeros
        - Distribution: normality test, type, symmetry
        - Trend: direction, stationarity, volatility regime

        **Quality:**
        - Anti-leakage verification (T-1 shift, no future data)
        - Quality score per variable (0-100%)
        - Overall dataset quality level

        **Output:**
        - JSON: Full structured report
        - CSV: Variable summary (1 row per variable)
        - HTML: Visual report with color-coded quality
        """,
    )

    # Task: Complete
    complete_task = EmptyOperator(
        task_id='complete',
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # Define flow: build -> validate -> quality_report -> complete
    build_task >> validate_task >> quality_report_task >> complete_task
