"""
Feature Builder
================
Factory for building features from raw data.

This is the SINGLE ENTRY POINT for feature calculation,
ensuring consistency between training, backtest, and inference.

SOLID Principles:
- Single Responsibility: Builder only constructs features
- Open/Closed: New features via registry, not modification
- Dependency Inversion: Uses FeatureSpec abstractions

Design Patterns:
- Factory Pattern: Creates calculators from specs
- Builder Pattern: Fluent API for configuration
- Template Method: Consistent calculation flow

Author: Trading Team
Version: 1.0.0
Created: 2025-01-12
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .contracts import (
    FeatureVersion,
    FeatureSpec,
    FeatureSetSpec,
    FeatureVector,
    FeatureBatch,
    NormalizationStats,
    NormalizationParams,
    CalculationRequest,
    CalculationResult,
    RawDataInput,
    MacroDataInput,
)
from .registry import FeatureRegistry, get_registry
from .calculators import CalculatorRegistry, FeatureCalculator


logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE CALCULATOR MAPPING
# =============================================================================

# Maps feature names to their calculator types
FEATURE_CALCULATOR_MAP = {
    # Price returns
    "return_5min": "momentum",
    "return_15min": "momentum",
    "return_1h": "momentum",
    # Momentum
    "rsi_9": "rsi",
    "adx_14": "adx",
    "macd_signal": "macd_signal",
    # Volatility
    "atr_pct": "atr_pct",
    "bollinger_width": "bollinger_width",
    "volatility_ratio": "volatility_ratio",
    # Trend
    "ema_distance_20": "ema_distance",
    "price_position": "price_position",
    # Macro
    "dxy_z": "dxy_zscore",
    "vix_z": "vix_zscore",
    "wti_z": "wti_zscore",
    "embi_z": "embi_zscore",
}


# =============================================================================
# FEATURE BUILDER
# =============================================================================

class FeatureBuilder:
    """
    Builder for calculating features from raw data.

    Ensures consistent feature calculation across all pipelines.

    Usage:
        builder = FeatureBuilder(FeatureVersion.CURRENT)
        features = builder.build(ohlcv_df, macro_df)

        # Or with custom normalization stats
        builder = FeatureBuilder(FeatureVersion.CURRENT)
        builder.with_normalization_stats(stats)
        features = builder.build(ohlcv_df)
    """

    def __init__(self, version: FeatureVersion):
        """
        Initialize builder for a specific feature version.

        Args:
            version: Feature set version to build
        """
        self.version = version
        self.registry = get_registry()
        self.feature_set = self.registry.get_feature_set(version)
        self._norm_stats: Optional[NormalizationStats] = None
        self._calculators: Dict[str, FeatureCalculator] = {}
        self._cache_enabled = True

        # Initialize calculators
        self._init_calculators()

    def _init_calculators(self) -> None:
        """Initialize calculators for all features"""
        for feature in self.feature_set.features:
            calc_type = FEATURE_CALCULATOR_MAP.get(feature.name)
            if calc_type and calc_type in CalculatorRegistry._calculators:
                self._calculators[feature.name] = CalculatorRegistry.create(
                    calc_type, feature
                )

    def with_normalization_stats(self, stats: NormalizationStats) -> "FeatureBuilder":
        """Set custom normalization stats (fluent API)"""
        if stats.version != self.version:
            raise ValueError(
                f"Stats version {stats.version} != builder version {self.version}"
            )
        self._norm_stats = stats
        return self

    def disable_cache(self) -> "FeatureBuilder":
        """Disable caching (fluent API)"""
        self._cache_enabled = False
        return self

    def build(
        self,
        ohlcv_data: pd.DataFrame,
        macro_data: Optional[pd.DataFrame] = None,
        normalize: bool = True
    ) -> FeatureBatch:
        """
        Build features from raw data.

        Args:
            ohlcv_data: DataFrame with columns [open, high, low, close, volume]
            macro_data: Optional DataFrame with columns [dxy, vix, wti, embi]
            normalize: Whether to apply normalization

        Returns:
            FeatureBatch with calculated features
        """
        start_time = time.time()

        # Validate input
        self._validate_input(ohlcv_data, macro_data)

        # Merge data if macro provided
        if macro_data is not None:
            data = ohlcv_data.join(macro_data, how="left")
        else:
            data = ohlcv_data.copy()

        # Calculate each feature
        feature_df = pd.DataFrame(index=data.index)

        for feature in self.feature_set.features:
            try:
                values = self._calculate_feature(feature, data)
                if normalize:
                    values = self._normalize_feature(feature, values)
                feature_df[feature.name] = values
            except Exception as e:
                logger.error(f"Failed to calculate {feature.name}: {e}")
                # Fill with default value
                feature_df[feature.name] = 0.0

        # Convert to FeatureBatch
        batch = FeatureBatch.from_dataframe(
            feature_df,
            version=self.version,
            is_normalized=normalize
        )

        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(f"Built {len(batch.vectors)} feature vectors in {elapsed_ms:.1f}ms")

        return batch

    def build_single(
        self,
        ohlcv_window: pd.DataFrame,
        macro_values: Optional[Dict[str, float]] = None,
        normalize: bool = True
    ) -> FeatureVector:
        """
        Build features for a single observation.

        Optimized for real-time inference where we only need
        the latest feature vector.

        Args:
            ohlcv_window: DataFrame with lookback window (minimum 60 rows)
            macro_values: Dict of macro values {dxy, vix, wti, embi}
            normalize: Whether to apply normalization

        Returns:
            Single FeatureVector
        """
        # Add macro columns if provided
        if macro_values:
            data = ohlcv_window.copy()
            for key, value in macro_values.items():
                data[key] = value
        else:
            data = ohlcv_window

        # Calculate features
        values = {}

        for feature in self.feature_set.features:
            try:
                series = self._calculate_feature(feature, data)
                value = series.iloc[-1]  # Get latest value

                if normalize:
                    value = self._normalize_single(feature, value)

                values[feature.name] = float(value)
            except Exception as e:
                logger.warning(f"Failed to calculate {feature.name}: {e}")
                values[feature.name] = 0.0

        return FeatureVector(
            version=self.version,
            timestamp=ohlcv_window.index[-1],
            values=values,
            is_normalized=normalize
        )

    def _validate_input(
        self,
        ohlcv: pd.DataFrame,
        macro: Optional[pd.DataFrame]
    ) -> None:
        """Validate input data"""
        required_ohlcv = {"open", "high", "low", "close"}
        missing = required_ohlcv - set(ohlcv.columns)
        if missing:
            raise ValueError(f"Missing OHLCV columns: {missing}")

        if len(ohlcv) < 60:
            logger.warning(f"Short lookback window: {len(ohlcv)} < 60 rows")

    def _calculate_feature(
        self,
        feature: FeatureSpec,
        data: pd.DataFrame
    ) -> pd.Series:
        """Calculate single feature using appropriate calculator"""
        # Use registered calculator if available
        if feature.name in self._calculators:
            return self._calculators[feature.name].calculate(data)

        # Otherwise use inline calculation
        if feature.name.startswith("return_"):
            return self._calculate_return(feature, data)

        raise ValueError(f"No calculator for feature: {feature.name}")

    def _calculate_return(
        self,
        feature: FeatureSpec,
        data: pd.DataFrame
    ) -> pd.Series:
        """Calculate return features"""
        close = data["close"]
        window = feature.calculation.window

        returns = np.log(close / close.shift(window))
        return returns.fillna(0.0)

    def _normalize_feature(
        self,
        feature: FeatureSpec,
        values: pd.Series
    ) -> pd.Series:
        """Normalize feature values"""
        # Use custom stats if provided
        if self._norm_stats:
            params = self._norm_stats.get_params(feature.name)
        else:
            params = feature.normalization

        # Skip if no normalization
        if params.method.value == "none":
            return values

        # Apply normalization based on method
        if params.method.value == "zscore":
            normalized = (values - params.mean) / params.std
        elif params.method.value == "rolling":
            window = params.rolling_window or 60
            rolling_mean = values.rolling(window, min_periods=1).mean()
            rolling_std = values.rolling(window, min_periods=1).std().replace(0, 1)
            normalized = (values - rolling_mean) / rolling_std
        elif params.method.value == "minmax":
            normalized = (values - params.min_val) / (params.max_val - params.min_val)
        else:
            normalized = values

        # Clip
        if params.clip_range:
            low, high = params.clip_range
            normalized = normalized.clip(lower=low, upper=high)

        return normalized.fillna(0.0)

    def _normalize_single(
        self,
        feature: FeatureSpec,
        value: float
    ) -> float:
        """Normalize single value"""
        if self._norm_stats:
            params = self._norm_stats.get_params(feature.name)
        else:
            params = feature.normalization

        if params.method.value == "none":
            return value

        if params.method.value == "zscore" and params.mean is not None:
            normalized = (value - params.mean) / params.std
        else:
            normalized = value

        # Clip
        if params.clip_range:
            low, high = params.clip_range
            normalized = max(low, min(high, normalized))

        return normalized


# =============================================================================
# FEATURE STORE FACTORY
# =============================================================================

class FeatureStoreFactory:
    """
    Factory for creating feature builders.

    Provides a clean interface for creating builders
    with appropriate configuration.

    Usage:
        factory = FeatureStoreFactory()
        builder = factory.create_builder(FeatureVersion.CURRENT)
        builder = factory.create_inference_builder(FeatureVersion.CURRENT, norm_stats)
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize factory.

        Args:
            config_path: Optional path to config directory
        """
        self.registry = get_registry()
        self.config_path = config_path

    def create_builder(
        self,
        version: FeatureVersion,
        normalize: bool = True
    ) -> FeatureBuilder:
        """
        Create a standard feature builder.

        Args:
            version: Feature version
            normalize: Whether to enable normalization

        Returns:
            Configured FeatureBuilder
        """
        builder = FeatureBuilder(version)
        return builder

    def create_training_builder(
        self,
        version: FeatureVersion
    ) -> FeatureBuilder:
        """
        Create builder for training pipeline.

        Uses default normalization stats from specs.
        """
        return FeatureBuilder(version)

    def create_inference_builder(
        self,
        version: FeatureVersion,
        norm_stats: Optional[NormalizationStats] = None
    ) -> FeatureBuilder:
        """
        Create builder for inference pipeline.

        Uses provided normalization stats or loads from config.
        """
        builder = FeatureBuilder(version)

        if norm_stats:
            builder.with_normalization_stats(norm_stats)

        return builder

    def create_backtest_builder(
        self,
        version: FeatureVersion,
        norm_stats: Optional[NormalizationStats] = None
    ) -> FeatureBuilder:
        """
        Create builder for backtest pipeline.

        Same as inference builder - uses same stats for consistency.
        """
        return self.create_inference_builder(version, norm_stats)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def build_features(
    ohlcv_data: pd.DataFrame,
    version: FeatureVersion = FeatureVersion.CURRENT,
    macro_data: Optional[pd.DataFrame] = None,
    normalize: bool = True
) -> FeatureBatch:
    """
    Convenience function to build features.

    Args:
        ohlcv_data: OHLCV DataFrame
        version: Feature version
        macro_data: Optional macro DataFrame
        normalize: Whether to normalize

    Returns:
        FeatureBatch with calculated features
    """
    builder = FeatureBuilder(version)
    return builder.build(ohlcv_data, macro_data, normalize)


def build_single_observation(
    ohlcv_window: pd.DataFrame,
    version: FeatureVersion = FeatureVersion.CURRENT,
    macro_values: Optional[Dict[str, float]] = None,
    norm_stats: Optional[NormalizationStats] = None
) -> FeatureVector:
    """
    Convenience function for single observation (inference).

    Args:
        ohlcv_window: Lookback window DataFrame
        version: Feature version
        macro_values: Dict of macro values
        norm_stats: Normalization stats

    Returns:
        Single FeatureVector
    """
    builder = FeatureBuilder(version)
    if norm_stats:
        builder.with_normalization_stats(norm_stats)
    return builder.build_single(ohlcv_window, macro_values)
