"""
Core Constants for USD/COP Trading System
==========================================

SINGLE SOURCE OF TRUTH (SSOT) for all system-wide constants.

This module consolidates constants that were previously scattered across
the codebase into a centralized location. All components MUST import
constants from here to ensure consistency.

Author: Trading Team
Version: 2.0.0
Date: 2026-01-17

Contract References:
    - CTR-FEATURE-001: Feature contract definitions
    - CTR-MODEL-001: Model contract definitions
    - CTR-TEMPORAL-001: Temporal join contract definitions
"""

from typing import Final

# =============================================================================
# CONTRACT IDENTIFIERS - Single Source of Truth
# =============================================================================
# These identifiers are used throughout the codebase to reference contracts.
# All contract references MUST use these constants.

CTR_FEATURE_001: Final[str] = "CTR-FEATURE-001"
CTR_MODEL_001: Final[str] = "CTR-MODEL-001"
CTR_TEMPORAL_001: Final[str] = "CTR-TEMPORAL-001"


# =============================================================================
# TRADING CONSTANTS - Colombia (Bogota) Market Hours
# =============================================================================
# These constants define trading hours for the Colombian market.
# USD/COP trades during Colombian business hours.

TRADING_TIMEZONE: Final[str] = "America/Bogota"
TRADING_START_HOUR: Final[int] = 8   # 08:00 local time
TRADING_END_HOUR: Final[int] = 17    # 17:00 local time
UTC_OFFSET_BOGOTA: Final[int] = -5   # Colombia is UTC-5 (no DST)


# =============================================================================
# DATA PROCESSING CONSTANTS - Forward Fill Limits
# =============================================================================
# These limits prevent excessive forward-filling of missing data.
# Different limits for different data frequencies.

DEFAULT_FFILL_DAILY_LIMIT: Final[int] = 5        # Max 5 days forward fill
DEFAULT_FFILL_MONTHLY_LIMIT: Final[int] = 35     # Max ~1 month forward fill
DEFAULT_FFILL_QUARTERLY_LIMIT: Final[int] = 95   # Max ~3 months forward fill
DEFAULT_MERGE_TOLERANCE: Final[str] = "3D"       # 3-day tolerance for temporal joins


# =============================================================================
# MODEL CONSTANTS - Observation and Action Space
# =============================================================================
# OBSERVATION_DIM is imported from feature_contract.py which is the SSOT.
# ACTION_COUNT is imported from action_contract.py which is the SSOT.

# =============================================================================
# FEATURE CONTRACTS - SSOT for Production and Experiments
# =============================================================================
# CANONICAL values (FEATURE_ORDER, OBSERVATION_DIM) are for PRODUCTION only.
# For experiments with different features, use get_feature_contract(contract_id).
#
# Usage:
#   # Production code (uses canonical contract v1.0.0)
#   from src.core.constants import FEATURE_ORDER, OBSERVATION_DIM
#
#   # Experiment code (uses specific contract)
#   from src.core.constants import get_feature_contract
#   contract = get_feature_contract("v1.1.0")
#   feature_order = contract.feature_order
#   observation_dim = contract.observation_dim

try:
    # Import canonical (production) values
    from src.core.contracts.feature_contract import (
        OBSERVATION_DIM,
        FEATURE_ORDER,
        FEATURE_ORDER_HASH,
    )
    # Import registry functions for experiments
    from src.core.contracts.feature_contracts_registry import (
        get_contract as get_feature_contract,
        get_production_contract,
        validate_model_contract,
        list_contracts as list_feature_contracts,
        get_norm_stats_path,
        FeatureContract,
        CONTRACTS as FEATURE_CONTRACTS,
        CANONICAL_CONTRACT_ID,
    )
    _FEATURE_REGISTRY_AVAILABLE = True
except ImportError:
    _FEATURE_REGISTRY_AVAILABLE = False
    # Fallback for environments where the contract is not available
    OBSERVATION_DIM: Final[int] = 15  # type: ignore[no-redef]
    FEATURE_ORDER = (  # type: ignore[misc]
        "log_ret_5m", "log_ret_1h", "log_ret_4h",
        "rsi_9", "atr_pct", "adx_14",
        "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
        "brent_change_1d", "rate_spread", "usdmxn_change_1d",
        "position", "time_normalized",
    )
    import hashlib
    FEATURE_ORDER_HASH = hashlib.sha256(",".join(FEATURE_ORDER).encode()).hexdigest()[:16]  # type: ignore[misc]
    # Stubs for registry functions
    get_feature_contract = None  # type: ignore
    get_production_contract = None  # type: ignore
    validate_model_contract = None  # type: ignore
    list_feature_contracts = None  # type: ignore
    get_norm_stats_path = None  # type: ignore
    FeatureContract = None  # type: ignore
    FEATURE_CONTRACTS = {}  # type: ignore
    CANONICAL_CONTRACT_ID = "v1.0.0"  # type: ignore


# =============================================================================
# API CONFIGURATION CONSTANTS
# =============================================================================
# Default URIs and paths for external services.
# These can be overridden by environment variables in production.

MLFLOW_TRACKING_URI: Final[str] = "http://localhost:5000"
FEAST_REPO_PATH: Final[str] = "feature_repo/"


# =============================================================================
# Action Enum - Imported from SSOT (Single Source of Truth)
# =============================================================================
# IMPORTANT: Action enum is defined in src.core.contracts.action_contract
# The CORRECT order is: SELL=0, HOLD=1, BUY=2 (matches PPO model output)
from src.core.contracts.action_contract import (
    Action,
    ACTION_SELL,
    ACTION_HOLD,
    ACTION_BUY,
    ACTION_COUNT,
    ACTION_NAMES,
    VALID_ACTIONS,
    InvalidActionError,
)

# Verify action order matches PPO model output expectations
assert ACTION_SELL == 0, f"ACTION_SELL must be 0, got {ACTION_SELL}"
assert ACTION_HOLD == 1, f"ACTION_HOLD must be 1, got {ACTION_HOLD}"
assert ACTION_BUY == 2, f"ACTION_BUY must be 2, got {ACTION_BUY}"
assert Action.SELL.value == 0, f"Action.SELL must be 0, got {Action.SELL.value}"
assert Action.HOLD.value == 1, f"Action.HOLD must be 1, got {Action.HOLD.value}"
assert Action.BUY.value == 2, f"Action.BUY must be 2, got {Action.BUY.value}"

# Re-export for backwards compatibility
__all_actions__ = [
    "Action",
    "ACTION_SELL",
    "ACTION_HOLD",
    "ACTION_BUY",
    "ACTION_COUNT",
    "ACTION_NAMES",
    "VALID_ACTIONS",
    "InvalidActionError",
]

# =============================================================================
# Technical Indicator Constants
# =============================================================================

# Scaling factor for percentage calculations
PERCENTAGE_SCALE: Final[int] = 100

# Time-based bar calculations (assuming 5-minute bars)
BARS_PER_HOUR: Final[int] = 12      # 60 / 5 = 12 bars per hour
BARS_PER_DAY: Final[int] = 288      # 24 * 12 = 288 bars per day
BARS_PER_WEEK: Final[int] = 2016    # 7 * 288 = 2016 bars per week

# =============================================================================
# PRODUCTION INDICATOR PERIODS - SSOT
# =============================================================================
# These are the CANONICAL periods used in production training.
# MUST match src/training/config.py IndicatorConfig

RSI_PERIOD: Final[int] = 9          # Production RSI period
ATR_PERIOD: Final[int] = 10         # Production ATR period
ADX_PERIOD: Final[int] = 14         # Production ADX period
WARMUP_BARS: Final[int] = 14        # max(RSI, ATR, ADX)

# Default/generic periods (for other use cases)
DEFAULT_RSI_PERIOD: Final[int] = 14
DEFAULT_ATR_PERIOD: Final[int] = 14
DEFAULT_ADX_PERIOD: Final[int] = 14
DEFAULT_MA_PERIOD: Final[int] = 20
DEFAULT_BBANDS_PERIOD: Final[int] = 20
DEFAULT_MACD_FAST: Final[int] = 12
DEFAULT_MACD_SLOW: Final[int] = 26
DEFAULT_MACD_SIGNAL: Final[int] = 9

# Return periods (in bars)
RETURN_PERIOD_1M: Final[int] = 1
RETURN_PERIOD_5M: Final[int] = 1
RETURN_PERIOD_1H: Final[int] = 12
RETURN_PERIOD_4H: Final[int] = 48
RETURN_PERIOD_1D: Final[int] = 288

# Session length (COT trading hours: 8:00 AM - 1:00 PM = 5 hours = 60 bars at 5min)
BARS_PER_SESSION: Final[int] = 60

# Rate spread calculation base (US Treasury 10Y baseline)
RATE_SPREAD_BASE: Final[float] = 10.0


# =============================================================================
# Feature Normalization Constants - SSOT
# =============================================================================
# CRITICAL: These values MUST be consistent across training and inference.
# The production model uses -5.0 / 5.0 for clipping.

# Production clipping bounds (used in ObservationBuilder)
CLIP_MIN: Final[float] = -5.0       # Production clip minimum
CLIP_MAX: Final[float] = 5.0        # Production clip maximum

# Aliases for backwards compatibility
DEFAULT_CLIP_MIN: Final[float] = -5.0
DEFAULT_CLIP_MAX: Final[float] = 5.0


# =============================================================================
# Action Thresholds - SSOT
# =============================================================================
# These thresholds determine when continuous action output maps to discrete actions.
# MUST match config/trading_config.yaml

THRESHOLD_LONG: Final[float] = 0.33     # Action > 0.33 = BUY
THRESHOLD_SHORT: Final[float] = -0.33   # Action < -0.33 = SELL
# HOLD when -0.33 <= action <= 0.33

# Z-score normalization parameters
DEFAULT_ZSCORE_LOOKBACK: Final[int] = 20
DEFAULT_ZSCORE_CENTER: Final[float] = 0.0
DEFAULT_ZSCORE_SCALE: Final[float] = 1.0

# RSI normalization (0-100 scale to -1 to 1)
RSI_SCALE_MIN: Final[float] = 0.0
RSI_SCALE_MAX: Final[float] = 100.0
RSI_NORMALIZED_MIN: Final[float] = -1.0
RSI_NORMALIZED_MAX: Final[float] = 1.0

# Indicator-specific bounds
ADX_MAX: Final[float] = 100.0
ATR_MAX_PCT: Final[float] = 10.0  # Maximum ATR as percentage


# =============================================================================
# Risk Management Constants
# =============================================================================

# Confidence thresholds for trade execution
MIN_CONFIDENCE_THRESHOLD: Final[float] = 0.6
HIGH_CONFIDENCE_THRESHOLD: Final[float] = 0.8
LOW_CONFIDENCE_THRESHOLD: Final[float] = 0.4

# Position sizing
MAX_POSITION_SIZE: Final[float] = 1.0
MIN_POSITION_SIZE: Final[float] = 0.01
DEFAULT_POSITION_SIZE_PCT: Final[float] = 0.1  # 10% of capital

# Stop loss and take profit
DEFAULT_STOP_LOSS_PCT: Final[float] = 0.02      # 2%
DEFAULT_TAKE_PROFIT_PCT: Final[float] = 0.04    # 4%
MAX_STOP_LOSS_PCT: Final[float] = 0.05          # 5%

# Daily limits
MAX_DAILY_LOSS_PCT: Final[float] = 0.05         # 5%
MAX_DAILY_TRADES: Final[int] = 20
MAX_DRAWDOWN_PCT: Final[float] = 0.15           # 15% kill switch

# Circuit breaker
CONSECUTIVE_LOSS_LIMIT: Final[int] = 5
COOLDOWN_BARS: Final[int] = 12
COOLDOWN_MINUTES: Final[int] = 60


# =============================================================================
# Market Hours Constants (Colombia Time - COT)
# =============================================================================

# Trading session hours
MARKET_OPEN_HOUR: Final[int] = 8    # 8:00 AM COT
MARKET_CLOSE_HOUR: Final[int] = 17  # 5:00 PM COT

# Extended hours (if applicable)
PRE_MARKET_OPEN_HOUR: Final[int] = 7
POST_MARKET_CLOSE_HOUR: Final[int] = 18

# Weekend detection
SATURDAY: Final[int] = 5
SUNDAY: Final[int] = 6


# =============================================================================
# Model Inference Constants
# =============================================================================

# Observation dimensions
DEFAULT_OBSERVATION_DIM: Final[int] = 45
MIN_OBSERVATION_DIM: Final[int] = 10
MAX_OBSERVATION_DIM: Final[int] = 200

# Action space (discrete) - REMOVED: Now imported from SSOT
# See: src.core.contracts.action_contract for Action enum
# Correct order: SELL=0, HOLD=1, BUY=2 (matches PPO model output)

# Model warmup
DEFAULT_WARMUP_ITERATIONS: Final[int] = 10

# Inference latency thresholds (milliseconds)
MAX_INFERENCE_LATENCY_MS: Final[float] = 100.0
TARGET_INFERENCE_LATENCY_MS: Final[float] = 10.0


# =============================================================================
# Data Quality Constants
# =============================================================================

# Maximum allowed missing data ratio
MAX_MISSING_RATIO: Final[float] = 0.1  # 10%

# Minimum data points for indicator calculation
MIN_DATA_POINTS: Final[int] = 50
MIN_DATA_POINTS_FOR_NORMALIZATION: Final[int] = 20

# Gap detection thresholds
MAX_GAP_DURATION_MINUTES: Final[int] = 30
MAX_PRICE_CHANGE_PCT: Final[float] = 0.05  # 5% max change per bar


# =============================================================================
# Network and API Constants
# =============================================================================

# API timeouts (seconds)
DEFAULT_API_TIMEOUT: Final[int] = 30
MAX_API_TIMEOUT: Final[int] = 60

# Retry settings
MAX_RETRIES: Final[int] = 3
RETRY_DELAY_SECONDS: Final[float] = 1.0
RETRY_BACKOFF_MULTIPLIER: Final[float] = 2.0

# Rate limiting
MAX_REQUESTS_PER_MINUTE: Final[int] = 60
MAX_REQUESTS_PER_SECOND: Final[int] = 5


# =============================================================================
# Database Constants
# =============================================================================

# Connection pool settings
DB_POOL_MIN_SIZE: Final[int] = 2
DB_POOL_MAX_SIZE: Final[int] = 10

# Query timeouts (seconds)
DB_QUERY_TIMEOUT: Final[int] = 30
DB_STATEMENT_TIMEOUT: Final[int] = 60

# Batch sizes
DB_BATCH_INSERT_SIZE: Final[int] = 100
DB_BATCH_SELECT_SIZE: Final[int] = 1000


# =============================================================================
# Logging and Monitoring Constants
# =============================================================================

# Log rotation
LOG_MAX_BYTES: Final[int] = 10_000_000  # 10 MB
LOG_BACKUP_COUNT: Final[int] = 5

# Metrics collection interval (seconds)
METRICS_COLLECTION_INTERVAL: Final[int] = 60

# Health check interval (seconds)
HEALTH_CHECK_INTERVAL: Final[int] = 30


# =============================================================================
# USD/COP Specific Constants
# =============================================================================

# Currency pair specifics
USDCOP_PIP_VALUE: Final[float] = 1.0   # 1 COP pip
USDCOP_DECIMALS: Final[int] = 2        # Price precision
USDCOP_TICK_SIZE: Final[float] = 0.01  # Minimum price movement

# Typical spread and slippage (COP)
TYPICAL_SPREAD: Final[float] = 5.0
MAX_SLIPPAGE: Final[float] = 10.0

# Volatility expectations
NORMAL_DAILY_VOLATILITY_PCT: Final[float] = 0.5
HIGH_VOLATILITY_THRESHOLD_PCT: Final[float] = 1.0
LOW_VOLATILITY_THRESHOLD_PCT: Final[float] = 0.2


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Contract Identifiers
    "CTR_FEATURE_001",
    "CTR_MODEL_001",
    "CTR_TEMPORAL_001",
    # Trading Constants
    "TRADING_TIMEZONE",
    "TRADING_START_HOUR",
    "TRADING_END_HOUR",
    "UTC_OFFSET_BOGOTA",
    # Data Processing Constants
    "DEFAULT_FFILL_DAILY_LIMIT",
    "DEFAULT_FFILL_MONTHLY_LIMIT",
    "DEFAULT_FFILL_QUARTERLY_LIMIT",
    "DEFAULT_MERGE_TOLERANCE",
    # Model Constants
    "OBSERVATION_DIM",
    "FEATURE_ORDER",
    "FEATURE_ORDER_HASH",
    "ACTION_COUNT",
    # API Configuration
    "MLFLOW_TRACKING_URI",
    "FEAST_REPO_PATH",
    # Action Enum (re-exported from action_contract)
    "Action",
    "ACTION_SELL",
    "ACTION_HOLD",
    "ACTION_BUY",
    "ACTION_NAMES",
    "VALID_ACTIONS",
    "InvalidActionError",
    # Production Indicator Periods (SSOT)
    "RSI_PERIOD",
    "ATR_PERIOD",
    "ADX_PERIOD",
    "WARMUP_BARS",
    # Technical Indicator Constants
    "PERCENTAGE_SCALE",
    "BARS_PER_HOUR",
    "BARS_PER_DAY",
    "BARS_PER_WEEK",
    "DEFAULT_RSI_PERIOD",
    "DEFAULT_ATR_PERIOD",
    "DEFAULT_ADX_PERIOD",
    "DEFAULT_MA_PERIOD",
    "DEFAULT_BBANDS_PERIOD",
    "DEFAULT_MACD_FAST",
    "DEFAULT_MACD_SLOW",
    "DEFAULT_MACD_SIGNAL",
    "RETURN_PERIOD_1M",
    "RETURN_PERIOD_5M",
    "RETURN_PERIOD_1H",
    "RETURN_PERIOD_4H",
    "RETURN_PERIOD_1D",
    # Feature Normalization Constants (SSOT)
    "CLIP_MIN",
    "CLIP_MAX",
    "DEFAULT_CLIP_MIN",
    "DEFAULT_CLIP_MAX",
    # Action Thresholds (SSOT)
    "THRESHOLD_LONG",
    "THRESHOLD_SHORT",
    "DEFAULT_ZSCORE_LOOKBACK",
    "DEFAULT_ZSCORE_CENTER",
    "DEFAULT_ZSCORE_SCALE",
    "RSI_SCALE_MIN",
    "RSI_SCALE_MAX",
    "RSI_NORMALIZED_MIN",
    "RSI_NORMALIZED_MAX",
    "ADX_MAX",
    "ATR_MAX_PCT",
    # Risk Management Constants
    "MIN_CONFIDENCE_THRESHOLD",
    "HIGH_CONFIDENCE_THRESHOLD",
    "LOW_CONFIDENCE_THRESHOLD",
    "MAX_POSITION_SIZE",
    "MIN_POSITION_SIZE",
    "DEFAULT_POSITION_SIZE_PCT",
    "DEFAULT_STOP_LOSS_PCT",
    "DEFAULT_TAKE_PROFIT_PCT",
    "MAX_STOP_LOSS_PCT",
    "MAX_DAILY_LOSS_PCT",
    "MAX_DAILY_TRADES",
    "MAX_DRAWDOWN_PCT",
    "CONSECUTIVE_LOSS_LIMIT",
    "COOLDOWN_BARS",
    "COOLDOWN_MINUTES",
    # Market Hours Constants
    "MARKET_OPEN_HOUR",
    "MARKET_CLOSE_HOUR",
    "PRE_MARKET_OPEN_HOUR",
    "POST_MARKET_CLOSE_HOUR",
    "SATURDAY",
    "SUNDAY",
    # Model Inference Constants
    "DEFAULT_OBSERVATION_DIM",
    "MIN_OBSERVATION_DIM",
    "MAX_OBSERVATION_DIM",
    "DEFAULT_WARMUP_ITERATIONS",
    "MAX_INFERENCE_LATENCY_MS",
    "TARGET_INFERENCE_LATENCY_MS",
    # Data Quality Constants
    "MAX_MISSING_RATIO",
    "MIN_DATA_POINTS",
    "MIN_DATA_POINTS_FOR_NORMALIZATION",
    "MAX_GAP_DURATION_MINUTES",
    "MAX_PRICE_CHANGE_PCT",
    # Network and API Constants
    "DEFAULT_API_TIMEOUT",
    "MAX_API_TIMEOUT",
    "MAX_RETRIES",
    "RETRY_DELAY_SECONDS",
    "RETRY_BACKOFF_MULTIPLIER",
    "MAX_REQUESTS_PER_MINUTE",
    "MAX_REQUESTS_PER_SECOND",
    # Database Constants
    "DB_POOL_MIN_SIZE",
    "DB_POOL_MAX_SIZE",
    "DB_QUERY_TIMEOUT",
    "DB_STATEMENT_TIMEOUT",
    "DB_BATCH_INSERT_SIZE",
    "DB_BATCH_SELECT_SIZE",
    # Logging and Monitoring Constants
    "LOG_MAX_BYTES",
    "LOG_BACKUP_COUNT",
    "METRICS_COLLECTION_INTERVAL",
    "HEALTH_CHECK_INTERVAL",
    # USD/COP Specific Constants
    "USDCOP_PIP_VALUE",
    "USDCOP_DECIMALS",
    "USDCOP_TICK_SIZE",
    "TYPICAL_SPREAD",
    "MAX_SLIPPAGE",
    "NORMAL_DAILY_VOLATILITY_PCT",
    "HIGH_VOLATILITY_THRESHOLD_PCT",
    "LOW_VOLATILITY_THRESHOLD_PCT",
]
