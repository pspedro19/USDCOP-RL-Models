"""
Forecasting Data Contracts (SSOT)
=================================

Single Source of Truth for forecasting data pipeline contracts.
Defines schemas, column names, and data lineage for DAILY forecasting data.

CRITICAL: Forecasting uses DAILY data, NOT 5-minute data like RL pipeline.

Data Lineage:
    TwelveData/Investing.com → bi.dim_daily_usdcop → bi.v_forecasting_features → Models

@version 1.0.0
@contract CTR-FORECAST-DATA-001
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import json
import logging

_logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class DataSource(str, Enum):
    """Source of daily USDCOP data."""
    TWELVEDATA = "twelvedata"
    INVESTING = "investing"
    MANUAL = "manual"


class FeatureCategory(str, Enum):
    """Categories of features for forecasting."""
    PRICE = "price"              # Raw price-based features
    RETURNS = "returns"          # Return calculations
    MOMENTUM = "momentum"        # Momentum indicators
    VOLATILITY = "volatility"    # Volatility measures
    TREND = "trend"              # Trend indicators
    MACRO = "macro"              # Macroeconomic indicators
    CALENDAR = "calendar"        # Calendar-based features
    TARGET = "target"            # Target variables


# =============================================================================
# CONSTANTS - DATA SCHEMA
# =============================================================================

# Table names in PostgreSQL
DAILY_OHLCV_TABLE = "bi.dim_daily_usdcop"
FEATURES_VIEW = "bi.v_forecasting_features"
FORECASTS_TABLE = "bi.fact_forecasts"
CONSENSUS_TABLE = "bi.fact_consensus"
METRICS_TABLE = "bi.fact_model_metrics"

# Required columns for raw daily data
RAW_DAILY_COLUMNS: Tuple[str, ...] = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
)

# Target column (what we're predicting)
TARGET_COLUMN = "close"  # Daily USDCOP close price

# Feature columns for model input (in strict order - SSOT)
# YAML-first with hardcoded fallback
try:
    from src.forecasting.ssot_config import ForecastingSSOTConfig as _SSOTCfg
    _cfg = _SSOTCfg.load()
    FEATURE_COLUMNS: Tuple[str, ...] = _cfg.get_feature_columns()
    TARGET_HORIZONS: Tuple[int, ...] = _cfg.get_horizons()
    _logger.debug("[data_contracts] Loaded %d features from YAML", len(FEATURE_COLUMNS))
except Exception as _e:
    _logger.debug("[data_contracts] YAML load failed (%s), using hardcoded fallback", _e)
    FEATURE_COLUMNS: Tuple[str, ...] = (
        # Price-based
        "close",
        "open",
        "high",
        "low",
        # Returns (momentum)
        "return_1d",
        "return_5d",
        "return_10d",
        "return_20d",
        # Volatility
        "volatility_5d",
        "volatility_10d",
        "volatility_20d",
        # Technical
        "rsi_14d",
        "ma_ratio_20d",
        "ma_ratio_50d",
        # Calendar
        "day_of_week",
        "month",
        "is_month_end",
        # Macro (lagged to avoid lookahead)
        "dxy_close_lag1",
        "oil_close_lag1",
        "vix_close_lag1",
        "embi_close_lag1",
    )
    TARGET_HORIZONS: Tuple[int, ...] = (1, 5, 10, 15, 20, 25, 30)

# Number of features (for validation)
NUM_FEATURES = len(FEATURE_COLUMNS)


# =============================================================================
# DATABASE SCHEMAS (SQL)
# =============================================================================

CREATE_DAILY_OHLCV_SQL = """
-- Daily USDCOP OHLCV data (target for forecasting)
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

-- Index for time-series queries
CREATE INDEX IF NOT EXISTS idx_daily_usdcop_date
    ON bi.dim_daily_usdcop (date DESC);

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION bi.update_daily_usdcop_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_daily_usdcop_updated ON bi.dim_daily_usdcop;
CREATE TRIGGER trg_daily_usdcop_updated
    BEFORE UPDATE ON bi.dim_daily_usdcop
    FOR EACH ROW
    EXECUTE FUNCTION bi.update_daily_usdcop_timestamp();
"""

CREATE_FEATURES_VIEW_SQL = """
-- Forecasting features view (computed from daily OHLCV)
-- This is the SSOT for feature engineering
CREATE OR REPLACE VIEW bi.v_forecasting_features AS
WITH price_data AS (
    SELECT
        date,
        open,
        high,
        low,
        close,
        volume,
        -- Returns
        (close - LAG(close, 1) OVER (ORDER BY date)) / NULLIF(LAG(close, 1) OVER (ORDER BY date), 0) AS return_1d,
        (close - LAG(close, 5) OVER (ORDER BY date)) / NULLIF(LAG(close, 5) OVER (ORDER BY date), 0) AS return_5d,
        (close - LAG(close, 10) OVER (ORDER BY date)) / NULLIF(LAG(close, 10) OVER (ORDER BY date), 0) AS return_10d,
        (close - LAG(close, 20) OVER (ORDER BY date)) / NULLIF(LAG(close, 20) OVER (ORDER BY date), 0) AS return_20d,
        -- Moving averages
        AVG(close) OVER (ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS ma_20d,
        AVG(close) OVER (ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS ma_50d,
        -- Calendar
        EXTRACT(DOW FROM date) AS day_of_week,
        EXTRACT(MONTH FROM date) AS month,
        CASE WHEN date = (date_trunc('month', date) + INTERVAL '1 month' - INTERVAL '1 day')::date
             THEN 1 ELSE 0 END AS is_month_end
    FROM bi.dim_daily_usdcop
),
volatility_data AS (
    SELECT
        date,
        STDDEV(return_1d) OVER (ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS volatility_5d,
        STDDEV(return_1d) OVER (ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS volatility_10d,
        STDDEV(return_1d) OVER (ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS volatility_20d
    FROM price_data
),
rsi_data AS (
    SELECT
        p.date,
        -- RSI calculation (14-day)
        100 - (100 / (1 +
            NULLIF(
                AVG(CASE WHEN p.return_1d > 0 THEN p.return_1d ELSE 0 END) OVER (ORDER BY p.date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW),
                0
            ) /
            NULLIF(
                AVG(CASE WHEN p.return_1d < 0 THEN ABS(p.return_1d) ELSE 0.0001 END) OVER (ORDER BY p.date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW),
                0
            )
        )) AS rsi_14d
    FROM price_data p
)
SELECT
    p.date,
    -- Price
    p.close,
    p.open,
    p.high,
    p.low,
    -- Returns
    p.return_1d,
    p.return_5d,
    p.return_10d,
    p.return_20d,
    -- Volatility
    v.volatility_5d,
    v.volatility_10d,
    v.volatility_20d,
    -- Technical
    r.rsi_14d,
    p.close / NULLIF(p.ma_20d, 0) AS ma_ratio_20d,
    p.close / NULLIF(p.ma_50d, 0) AS ma_ratio_50d,
    -- Calendar
    p.day_of_week::INT,
    p.month::INT,
    p.is_month_end,
    -- Macro (placeholders - joined from macro tables)
    NULL::DECIMAL(12,4) AS dxy_close_lag1,
    NULL::DECIMAL(12,4) AS oil_close_lag1,
    -- Target columns (future returns)
    LEAD(p.close, 1) OVER (ORDER BY p.date) AS target_1d,
    LEAD(p.close, 5) OVER (ORDER BY p.date) AS target_5d,
    LEAD(p.close, 10) OVER (ORDER BY p.date) AS target_10d,
    LEAD(p.close, 15) OVER (ORDER BY p.date) AS target_15d,
    LEAD(p.close, 20) OVER (ORDER BY p.date) AS target_20d,
    LEAD(p.close, 25) OVER (ORDER BY p.date) AS target_25d,
    LEAD(p.close, 30) OVER (ORDER BY p.date) AS target_30d
FROM price_data p
LEFT JOIN volatility_data v ON p.date = v.date
LEFT JOIN rsi_data r ON p.date = r.date
WHERE p.date >= '2015-01-01'  -- Minimum history needed
ORDER BY p.date;
"""


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DailyOHLCV:
    """Single day OHLCV record."""
    date: str  # YYYY-MM-DD
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    source: DataSource = DataSource.TWELVEDATA

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "source": self.source.value,
        }


@dataclass
class DailyFetchRequest:
    """Request to fetch daily USDCOP data."""
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD
    source: DataSource = DataSource.TWELVEDATA
    fill_gaps: bool = True
    validate: bool = True


@dataclass
class DailyFetchResult:
    """Result of daily data fetch."""
    success: bool
    records_fetched: int
    records_inserted: int
    records_updated: int
    date_range: Tuple[str, str]  # (start, end)
    gaps_filled: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class FeatureBuildRequest:
    """Request to build forecasting features."""
    as_of_date: str  # YYYY-MM-DD
    lookback_days: int = 365
    include_targets: bool = True


@dataclass
class FeatureBuildResult:
    """Result of feature building."""
    success: bool
    rows_generated: int
    feature_columns: List[str]
    date_range: Tuple[str, str]
    null_counts: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_daily_record(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a daily OHLCV record."""
    errors = []

    # Required fields
    required = ["date", "open", "high", "low", "close"]
    for field in required:
        if field not in record or record[field] is None:
            errors.append(f"Missing required field: {field}")

    if errors:
        return False, errors

    # Price validation
    o, h, l, c = record["open"], record["high"], record["low"], record["close"]

    if not (3000 <= c <= 6000):  # COP range sanity check
        errors.append(f"Close price {c} outside valid COP range [3000, 6000]")

    if h < l:
        errors.append(f"High ({h}) < Low ({l})")

    if h < max(o, c):
        errors.append(f"High ({h}) < max(Open, Close)")

    if l > min(o, c):
        errors.append(f"Low ({l}) > min(Open, Close)")

    return len(errors) == 0, errors


def validate_feature_row(row: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a feature row has all required columns."""
    errors = []

    for col in FEATURE_COLUMNS:
        if col not in row:
            errors.append(f"Missing feature column: {col}")

    return len(errors) == 0, errors


# =============================================================================
# HASH UTILITIES
# =============================================================================

def compute_data_contract_hash() -> str:
    """Compute hash of this data contract for versioning."""
    contract_data = {
        "raw_columns": RAW_DAILY_COLUMNS,
        "feature_columns": FEATURE_COLUMNS,
        "target_horizons": TARGET_HORIZONS,
        "target_column": TARGET_COLUMN,
    }
    content = json.dumps(contract_data, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# Contract metadata
DATA_CONTRACT_VERSION = "1.0.0"
DATA_CONTRACT_HASH = compute_data_contract_hash()


# =============================================================================
# TWELVEDATA CONFIG
# =============================================================================

TWELVEDATA_CONFIG = {
    "symbol": "USD/COP",
    "interval": "1day",
    "timezone": "America/Bogota",
    "outputsize": 5000,  # Max records per request
    "format": "JSON",
}

# Investing.com scraping config (backup source)
INVESTING_CONFIG = {
    "pair_id": "usd-cop",  # Investing.com pair identifier
    "history_endpoint": "/currencies/usd-cop-historical-data",
    "max_days_per_request": 365,
}


__all__ = [
    # Enums
    "DataSource",
    "FeatureCategory",
    # Constants
    "DAILY_OHLCV_TABLE",
    "FEATURES_VIEW",
    "FORECASTS_TABLE",
    "CONSENSUS_TABLE",
    "METRICS_TABLE",
    "RAW_DAILY_COLUMNS",
    "TARGET_COLUMN",
    "FEATURE_COLUMNS",
    "NUM_FEATURES",
    "TARGET_HORIZONS",
    # SQL
    "CREATE_DAILY_OHLCV_SQL",
    "CREATE_FEATURES_VIEW_SQL",
    # Data classes
    "DailyOHLCV",
    "DailyFetchRequest",
    "DailyFetchResult",
    "FeatureBuildRequest",
    "FeatureBuildResult",
    # Validation
    "validate_daily_record",
    "validate_feature_row",
    # Config
    "TWELVEDATA_CONFIG",
    "INVESTING_CONFIG",
    # Contract metadata
    "DATA_CONTRACT_VERSION",
    "DATA_CONTRACT_HASH",
    "compute_data_contract_hash",
]
