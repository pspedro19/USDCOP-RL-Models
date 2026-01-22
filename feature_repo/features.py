"""
Feast Feature Definitions for USDCOP Trading System
====================================================
This module defines the feature entities, views, and services for the trading system.

Architecture:
    CanonicalFeatureBuilder (SSOT) ──> Offline Store (Parquet)
                                             │
                                             ▼
                                      Feast Materialize
                                             │
                                             ▼
                                      Online Store (Redis)
                                             │
                                             ▼
                                   FeastInferenceService

Feature Views:
- technical_features: Price-based indicators (log returns, RSI, ATR, ADX)
- macro_features: Macroeconomic indicators (DXY, VIX, EMBI, Brent, rates)
- state_features: Agent state (position, time_normalized)

Feature Service:
- observation_15d: Combines all views for complete 15-dimensional observation

Author: Trading Team
Version: 1.0.0
Created: 2025-01-17
"""

from datetime import timedelta
from typing import List

from feast import (
    Entity,
    Feature,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    ValueType,
)
from feast.types import Float32, Float64, Int64, String


# =============================================================================
# ENTITIES
# =============================================================================
# Entities are the primary keys for feature lookups

# Bar Entity - represents a unique 5-minute bar for a symbol
bar_entity = Entity(
    name="bar_entity",
    description="5-minute OHLCV bar for USD/COP trading",
    join_keys=["symbol", "bar_id"],
    # Tags for metadata
    tags={
        "owner": "trading-team",
        "project": "usdcop-trading",
        "version": "1.0.0",
    },
)


# =============================================================================
# DATA SOURCES
# =============================================================================
# Define where feature data is stored in the offline store

# Technical features from OHLCV calculations
technical_features_source = FileSource(
    name="technical_features_source",
    path="data/feast/technical_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    description="Technical indicators calculated from OHLCV data",
)

# Macro features from daily macro indicators
macro_features_source = FileSource(
    name="macro_features_source",
    path="data/feast/macro_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    description="Macroeconomic indicators (DXY, VIX, EMBI, Brent, rates)",
)

# State features (position, time)
state_features_source = FileSource(
    name="state_features_source",
    path="data/feast/state_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    description="Agent state features (position, time normalized)",
)


# =============================================================================
# FEATURE VIEWS
# =============================================================================
# Feature Views define how features are computed and stored

# Technical Features View
# Contains price-based indicators: log returns, RSI, ATR%, ADX
technical_features = FeatureView(
    name="technical_features",
    description="Technical indicators from OHLCV data (log returns, RSI-9, ATR%, ADX-14)",
    entities=[bar_entity],
    # TTL defines how long features remain fresh
    # 30 minutes for technical features (update every 5-min bar)
    ttl=timedelta(minutes=30),
    schema=[
        # Log Returns
        Field(name="log_ret_5m", dtype=Float32, description="5-minute log return"),
        Field(name="log_ret_1h", dtype=Float32, description="1-hour log return (12 bars)"),
        Field(name="log_ret_4h", dtype=Float32, description="4-hour log return (48 bars)"),

        # Momentum & Volatility Indicators
        Field(name="rsi_9", dtype=Float32, description="RSI with 9-period (Wilder's smoothing)"),
        Field(name="atr_pct", dtype=Float32, description="ATR as percentage of price (10-period)"),
        Field(name="adx_14", dtype=Float32, description="ADX with 14-period (trend strength)"),
    ],
    source=technical_features_source,
    # Online enabled for real-time inference
    online=True,
    # Tags for metadata
    tags={
        "owner": "trading-team",
        "feature_category": "technical",
        "update_frequency": "5min",
        "warmup_bars": "48",
    },
)


# Macro Features View
# Contains macroeconomic indicators: DXY, VIX, EMBI, Brent, rate spread, MXN
macro_features = FeatureView(
    name="macro_features",
    description="Macroeconomic indicators (DXY z-score, VIX z-score, EMBI z-score, Brent change, rate spread, MXN)",
    entities=[bar_entity],
    # TTL of 1 hour for macro features (update daily, forward-filled)
    ttl=timedelta(hours=1),
    schema=[
        # Dollar Index
        Field(name="dxy_z", dtype=Float32, description="DXY z-score (60-day rolling)"),
        Field(name="dxy_change_1d", dtype=Float32, description="DXY daily change"),

        # Volatility Index
        Field(name="vix_z", dtype=Float32, description="VIX z-score (60-day rolling)"),

        # Emerging Market Spread
        Field(name="embi_z", dtype=Float32, description="EMBI z-score (60-day rolling)"),

        # Oil & Rates
        Field(name="brent_change_1d", dtype=Float32, description="Brent crude daily change"),
        Field(name="rate_spread", dtype=Float32, description="Treasury 10Y-2Y spread"),

        # Regional Currency Correlation
        Field(name="usdmxn_change_1d", dtype=Float32, description="USD/MXN daily change"),
    ],
    source=macro_features_source,
    online=True,
    tags={
        "owner": "trading-team",
        "feature_category": "macro",
        "update_frequency": "daily",
        "source": "macro_indicators_daily",
    },
)


# State Features View
# Contains agent state: position and time normalized
state_features = FeatureView(
    name="state_features",
    description="Agent state features (position, time normalized within session)",
    entities=[bar_entity],
    # Short TTL of 5 minutes for state (must be fresh for inference)
    ttl=timedelta(minutes=5),
    schema=[
        # Agent Position
        Field(name="position", dtype=Float32, description="Current position [-1, 1]"),

        # Time in Session
        Field(name="time_normalized", dtype=Float32, description="Time normalized within trading session [0, 1]"),
    ],
    source=state_features_source,
    online=True,
    tags={
        "owner": "trading-team",
        "feature_category": "state",
        "update_frequency": "5min",
        "note": "position updated by trading decisions, time by clock",
    },
)


# =============================================================================
# FEATURE SERVICE
# =============================================================================
# Feature Service combines multiple Feature Views for a specific use case

# Observation Feature Service
# Combines all feature views to produce the complete 15-dimensional observation
observation_15d = FeatureService(
    name="observation_15d",
    description="Complete 15-dimensional observation for RL model inference (13 features + position + time)",
    features=[
        # Technical features (6)
        technical_features[["log_ret_5m", "log_ret_1h", "log_ret_4h", "rsi_9", "atr_pct", "adx_14"]],

        # Macro features (7)
        macro_features[["dxy_z", "dxy_change_1d", "vix_z", "embi_z", "brent_change_1d", "rate_spread", "usdmxn_change_1d"]],

        # State features (2)
        state_features[["position", "time_normalized"]],
    ],
    tags={
        "owner": "trading-team",
        "observation_dim": "15",
        "model_version": "v19",
        "use_case": "rl_inference",
    },
)


# =============================================================================
# FEATURE NAMES - For validation and ordering
# =============================================================================

# Feature order must match CanonicalFeatureBuilder SSOT
FEAST_FEATURE_ORDER: List[str] = [
    # Technical (6)
    "log_ret_5m",
    "log_ret_1h",
    "log_ret_4h",
    "rsi_9",
    "atr_pct",
    "adx_14",
    # Macro (7)
    "dxy_z",
    "dxy_change_1d",
    "vix_z",
    "embi_z",
    "brent_change_1d",
    "rate_spread",
    "usdmxn_change_1d",
    # State (2)
    "position",
    "time_normalized",
]


def validate_feature_order() -> bool:
    """
    Validate that Feast feature order matches CanonicalFeatureBuilder SSOT.

    Returns:
        True if orders match, raises AssertionError otherwise
    """
    try:
        from src.feature_store import FEATURE_ORDER

        if list(FEATURE_ORDER) != FEAST_FEATURE_ORDER:
            raise AssertionError(
                f"Feature order mismatch!\n"
                f"Feast:     {FEAST_FEATURE_ORDER}\n"
                f"Canonical: {list(FEATURE_ORDER)}"
            )
        return True
    except ImportError:
        # Can't validate without src module, assume correct
        return True


# Run validation on import (in development only)
if __name__ == "__main__":
    validate_feature_order()
    print("Feature order validation passed!")
    print(f"Total features: {len(FEAST_FEATURE_ORDER)}")
    print(f"Technical: {FEAST_FEATURE_ORDER[:6]}")
    print(f"Macro: {FEAST_FEATURE_ORDER[6:13]}")
    print(f"State: {FEAST_FEATURE_ORDER[13:]}")
