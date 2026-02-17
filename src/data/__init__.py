"""
Unified Data Layer - SSOT for RL and Forecasting pipelines.

Contract: CTR-DATA-001
Version: 3.0.0

ARCHITECTURE 10/10 - DISTINCT DATA PATHS:
==========================================

┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA SOURCES                                    │
├───────────────────────────────┬─────────────────────────────────────────┤
│     TwelveData API            │       Investing.com (Official)          │
│     (5-minute bars)           │       (Daily close values)              │
│             │                 │              │                          │
│             ▼                 │              ▼                          │
│     usdcop_m5_ohlcv           │       bi.dim_daily_usdcop               │
│     (TimescaleDB)             │       (PostgreSQL bi schema)            │
│             │                 │              │                          │
│             ▼                 │              ▼                          │
│       load_5min()             │        load_daily()                     │
│             │                 │              │                          │
│             ▼                 │              ▼                          │
│      RL TRAINING              │       FORECASTING                       │
│  (intraday patterns)          │  (official daily close)                 │
└───────────────────────────────┴─────────────────────────────────────────┘

CRITICAL RULES:
- RL Pipeline: ALWAYS use load_5min() for intraday data
- Forecasting: ALWAYS use load_daily() for official Investing.com values
- NEVER use resample_to_daily() for production forecasting

Usage:
    from src.data import UnifiedOHLCVLoader, UnifiedMacroLoader

    loader = UnifiedOHLCVLoader()
    macro = UnifiedMacroLoader()

    # RL Pipeline (5-min intraday)
    df_5min = loader.load_5min("2024-01-01", "2024-12-31")
    df_macro_5min = macro.load_5min("2024-01-01", "2024-12-31")

    # Forecasting Pipeline (daily official)
    df_daily = loader.load_daily("2024-01-01", "2024-12-31")  # OFFICIAL
    df_macro_daily = macro.load_daily("2024-01-01", "2024-12-31")
"""

# Safe merge operations (original)
from .safe_merge import safe_ffill, safe_merge_macro, validate_no_future_data

# Unified loaders (new)
from .ohlcv_loader import UnifiedOHLCVLoader
from .macro_loader import UnifiedMacroLoader

# Trading calendar
from .calendar import TradingCalendar, is_trading_day, filter_market_hours

# SSOT Dataset Builder (v2.0)
from .ssot_dataset_builder import (
    SSOTDatasetBuilder,
    DatasetBuildResult,
    build_production_dataset,
)

# Contracts
from .contracts import (
    # Column mappings
    MACRO_DB_TO_FRIENDLY,
    MACRO_FRIENDLY_TO_DB,
    # RL pipeline
    RL_MACRO_COLUMNS,
    RL_OHLCV_TABLE,
    RL_MACRO_TABLE,
    # Forecasting pipeline
    FORECASTING_MACRO_COLUMNS,
    FORECASTING_FEATURES,
    FORECASTING_TARGETS,
    FORECASTING_HORIZONS,
    NUM_FORECASTING_FEATURES,
    FORECASTING_OHLCV_TABLE,
    FORECASTING_FEATURES_VIEW,
    # OHLCV schema
    OHLCV_COLUMNS,
    OHLCV_REQUIRED,
    DAILY_OHLCV_COLUMNS,
    DAILY_OHLCV_REQUIRED,
    DAILY_OHLCV_SOURCES,
    # Validation
    validate_forecasting_features,
    validate_ohlcv_columns,
    validate_daily_ohlcv_columns,
    get_data_lineage_info,
)

__all__ = [
    # Safe merge (original)
    "safe_ffill",
    "safe_merge_macro",
    "validate_no_future_data",
    # SSOT Dataset Builder (v2.0)
    "SSOTDatasetBuilder",
    "DatasetBuildResult",
    "build_production_dataset",
    # Loaders
    "UnifiedOHLCVLoader",
    "UnifiedMacroLoader",
    # Calendar
    "TradingCalendar",
    "is_trading_day",
    "filter_market_hours",
    # Column mappings
    "MACRO_DB_TO_FRIENDLY",
    "MACRO_FRIENDLY_TO_DB",
    # RL pipeline
    "RL_MACRO_COLUMNS",
    "RL_OHLCV_TABLE",
    "RL_MACRO_TABLE",
    # Forecasting pipeline
    "FORECASTING_MACRO_COLUMNS",
    "FORECASTING_FEATURES",
    "FORECASTING_TARGETS",
    "FORECASTING_HORIZONS",
    "NUM_FORECASTING_FEATURES",
    "FORECASTING_OHLCV_TABLE",
    "FORECASTING_FEATURES_VIEW",
    # OHLCV schema
    "OHLCV_COLUMNS",
    "OHLCV_REQUIRED",
    "DAILY_OHLCV_COLUMNS",
    "DAILY_OHLCV_REQUIRED",
    "DAILY_OHLCV_SOURCES",
    # Validation
    "validate_forecasting_features",
    "validate_ohlcv_columns",
    "validate_daily_ohlcv_columns",
    "get_data_lineage_info",
]

__version__ = "3.0.0"
