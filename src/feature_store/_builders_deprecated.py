"""
DEPRECATED - Feature Builders Module
=====================================

*** THIS FILE IS DEPRECATED AND SHOULD NOT BE USED ***

This file was renamed from builders.py to _builders_deprecated.py on 2026-01-18
as part of P0 remediation (DRY violation fix).

REASON: Python's import system prefers packages over modules when both exist
with the same name. Since we have both:
- src/feature_store/builders.py (this file - NOW DEPRECATED)
- src/feature_store/builders/ (package - CANONICAL)

All imports of `from src.feature_store.builders import X` resolve to the
package, making this file unreachable/dead code.

CANONICAL IMPLEMENTATION:
    from src.feature_store.builders import CanonicalFeatureBuilder

The canonical implementation is in:
    src/feature_store/builders/canonical_feature_builder.py

This file is kept for reference only. DO NOT USE.

Author: Trading Team
Version: 1.0.0 (DEPRECATED)
Created: 2026-01-17
Deprecated: 2026-01-18
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

from .core import (
    FEATURE_CONTRACT,
    FEATURE_ORDER,
    OBSERVATION_DIM,
    CalculatorRegistry,
    RSICalculator,
    ATRPercentCalculator,
    ADXCalculator,
    LogReturnCalculator,
    SmoothingMethod,
)

logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class FeatureCalculationError(Exception):
    """Raised when feature calculation fails."""
    pass


class ObservationDimensionError(Exception):
    """Raised when observation dimension doesn't match contract."""
    pass


# =============================================================================
# CONTEXT DATA CLASS
# =============================================================================

@dataclass
class BuilderContext:
    """
    Context for feature building operations.

    Provides metadata and configuration for a feature computation run.
    """
    timestamp: datetime
    builder_version: str
    feature_count: int
    calculation_time_ms: float = 0.0
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


# =============================================================================
# FEATURE BUILDER INTERFACE
# =============================================================================

class IFeatureBuilder(ABC):
    """Interface for feature builders."""

    @abstractmethod
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features from OHLCV data."""
        pass

    @abstractmethod
    def get_feature_vector(self, df: pd.DataFrame) -> np.ndarray:
        """Get feature vector in canonical order."""
        pass

    @property
    @abstractmethod
    def VERSION(self) -> str:
        """Builder version for audit tracking."""
        pass


# =============================================================================
# CANONICAL FEATURE BUILDER - SINGLE SOURCE OF TRUTH
# =============================================================================

class CanonicalFeatureBuilder(IFeatureBuilder):
    """
    Canonical Feature Builder - SINGLE SOURCE OF TRUTH

    This is the authoritative implementation for feature calculation.
    All pipelines (training, backtest, inference, Airflow DAGs) MUST use
    this builder to ensure perfect feature parity.

    Features:
    - Uses Wilder's EMA for RSI, ATR, ADX (matches TA-Lib and training)
    - Produces exactly 15 features in FEATURE_ORDER
    - Validates output against FEATURE_CONTRACT
    - Tracks version for audit/debugging

    Usage:
        builder = CanonicalFeatureBuilder()

        # Compute all features for a DataFrame
        features_df = builder.compute_features(ohlcv_df)

        # Get latest feature vector for inference
        vector = builder.get_feature_vector(ohlcv_df)

        # Access version for logging
        logger.info(f"Using builder v{builder.VERSION}")
    """

    # Version for audit tracking - increment on calculation changes
    VERSION = "2.0.0"

    def __init__(self):
        """Initialize the canonical feature builder."""
        self._registry = CalculatorRegistry.instance()

        # Initialize technical calculators
        self._rsi_calc = RSICalculator(period=FEATURE_CONTRACT.rsi_period)
        self._atr_calc = ATRPercentCalculator(period=FEATURE_CONTRACT.atr_period)
        self._adx_calc = ADXCalculator(period=FEATURE_CONTRACT.adx_period)

        # Log return calculators
        self._ret_5m_calc = LogReturnCalculator("log_ret_5m", periods=1)
        self._ret_1h_calc = LogReturnCalculator("log_ret_1h", periods=12)
        self._ret_4h_calc = LogReturnCalculator("log_ret_4h", periods=48)

        logger.info(f"CanonicalFeatureBuilder v{self.VERSION} initialized")

    def compute_features(
        self,
        df: pd.DataFrame,
        include_state: bool = False,
        position: float = 0.0
    ) -> pd.DataFrame:
        """
        Compute all features from OHLCV data.

        Args:
            df: DataFrame with columns [time/timestamp, open, high, low, close, volume]
                and optionally macro columns [dxy, vix, brent, etc.]
            include_state: If True, include position and time_normalized columns
            position: Current position for state features

        Returns:
            DataFrame with all 13-15 features (depending on include_state)

        Raises:
            FeatureCalculationError: If calculation fails
        """
        import time as time_module
        start_time = time_module.time()

        try:
            # Ensure we have required columns
            self._validate_ohlcv(df)

            # Create result DataFrame
            features = pd.DataFrame(index=df.index)

            # ================================================================
            # LOG RETURNS
            # ================================================================
            features['log_ret_5m'] = self._ret_5m_calc.calculate_batch(df)
            features['log_ret_1h'] = self._ret_1h_calc.calculate_batch(df)
            features['log_ret_4h'] = self._ret_4h_calc.calculate_batch(df)

            # ================================================================
            # TECHNICAL INDICATORS (Wilder's EMA)
            # ================================================================
            features['rsi_9'] = self._rsi_calc.calculate_batch(df)
            features['atr_pct'] = self._atr_calc.calculate_batch(df)
            features['adx_14'] = self._adx_calc.calculate_batch(df)

            # ================================================================
            # MACRO FEATURES (if columns present)
            # ================================================================
            features['dxy_z'] = self._compute_zscore(df, 'dxy')
            features['dxy_change_1d'] = self._compute_daily_change(df, 'dxy')
            features['vix_z'] = self._compute_zscore(df, 'vix')
            features['embi_z'] = self._compute_zscore(df, 'embi')
            features['brent_change_1d'] = self._compute_daily_change(df, 'brent')

            # Rate spread (treasury 10y - 2y)
            if 'treasury_10y' in df.columns and 'treasury_2y' in df.columns:
                features['rate_spread'] = df['treasury_10y'] - df['treasury_2y']
            elif 'rate_spread' in df.columns:
                features['rate_spread'] = df['rate_spread']
            else:
                features['rate_spread'] = 0.0

            # USDMXN change
            if 'usdmxn' in df.columns:
                features['usdmxn_change_1d'] = np.log(
                    df['usdmxn'] / df['usdmxn'].shift(12)
                ).fillna(0.0)
            else:
                features['usdmxn_change_1d'] = 0.0

            # ================================================================
            # STATE FEATURES (optional)
            # ================================================================
            if include_state:
                features['position'] = position
                features['time_normalized'] = self._compute_time_normalized(df)

            # Fill NaN values
            features = features.fillna(0.0)

            elapsed_ms = (time_module.time() - start_time) * 1000
            logger.debug(
                f"Computed {len(features.columns)} features for {len(features)} rows "
                f"in {elapsed_ms:.1f}ms"
            )

            return features

        except Exception as e:
            raise FeatureCalculationError(f"Feature calculation failed: {e}") from e

    def get_feature_vector(
        self,
        df: pd.DataFrame,
        position: float = 0.0,
        timestamp: Optional[datetime] = None
    ) -> np.ndarray:
        """
        Get feature vector in canonical order for the latest bar.

        Args:
            df: DataFrame with OHLCV and optionally macro data
            position: Current position [-1, 1]
            timestamp: Timestamp for time normalization (uses now if not provided)

        Returns:
            np.ndarray of shape (15,) in FEATURE_ORDER

        Raises:
            ObservationDimensionError: If resulting dimension != 15
        """
        # Compute features without state (we'll add them manually)
        features = self.compute_features(df, include_state=False)

        if len(features) == 0:
            raise FeatureCalculationError("No features computed")

        # Get latest row
        latest = features.iloc[-1].to_dict()

        # Add state features
        latest['position'] = float(np.clip(position, -1.0, 1.0))
        latest['time_normalized'] = self._compute_time_normalized_single(timestamp)

        # Build vector in canonical order
        vector = []
        for feature_name in FEATURE_ORDER:
            value = latest.get(feature_name, 0.0)
            if pd.isna(value) or np.isinf(value):
                value = 0.0
            vector.append(float(value))

        result = np.array(vector, dtype=np.float32)

        # Validate dimension
        if len(result) != OBSERVATION_DIM:
            raise ObservationDimensionError(
                f"Expected {OBSERVATION_DIM} features, got {len(result)}"
            )

        return result

    def get_latest_features_dict(
        self,
        df: pd.DataFrame,
        position: float = 0.0,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Get latest features as a dictionary.

        Useful for JSON serialization and debugging.

        Args:
            df: DataFrame with OHLCV and optionally macro data
            position: Current position [-1, 1]
            timestamp: Timestamp for time normalization

        Returns:
            Dict mapping feature names to values
        """
        features = self.compute_features(df, include_state=False)

        if len(features) == 0:
            raise FeatureCalculationError("No features computed")

        latest = features.iloc[-1].to_dict()

        # Add state features
        latest['position'] = float(np.clip(position, -1.0, 1.0))
        latest['time_normalized'] = self._compute_time_normalized_single(timestamp)

        # Ensure all features are present with correct order
        result = {}
        for feature_name in FEATURE_ORDER:
            value = latest.get(feature_name, 0.0)
            if pd.isna(value) or np.isinf(value):
                value = 0.0
            result[feature_name] = float(value)

        return result

    def validate_features(self, features: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Validate feature dictionary against contract.

        Args:
            features: Dict of feature name -> value

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check feature count
        if len(features) != OBSERVATION_DIM:
            errors.append(
                f"Feature count mismatch: got {len(features)}, expected {OBSERVATION_DIM}"
            )

        # Check all required features present
        missing = set(FEATURE_ORDER) - set(features.keys())
        if missing:
            errors.append(f"Missing features: {missing}")

        # Check for NaN/Inf
        for name, value in features.items():
            if pd.isna(value):
                errors.append(f"NaN value for {name}")
            elif np.isinf(value):
                errors.append(f"Inf value for {name}")

        # Validate ranges
        validations = {
            'rsi_9': (0, 100),
            'atr_pct': (0, 1.0),  # 0-100%
            'adx_14': (0, 100),
            'position': (-1, 1),
            'time_normalized': (0, 1),
        }

        for feature, (min_val, max_val) in validations.items():
            if feature in features:
                val = features[feature]
                if not (min_val <= val <= max_val):
                    errors.append(
                        f"{feature}={val:.4f} outside range [{min_val}, {max_val}]"
                    )

        return len(errors) == 0, errors

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    def _validate_ohlcv(self, df: pd.DataFrame) -> None:
        """Validate required OHLCV columns exist."""
        required = {'open', 'high', 'low', 'close'}
        columns = set(df.columns.str.lower())
        missing = required - columns
        if missing:
            raise FeatureCalculationError(f"Missing required columns: {missing}")

    def _compute_zscore(
        self,
        df: pd.DataFrame,
        column: str,
        window: int = 60
    ) -> pd.Series:
        """Compute rolling z-score for a column."""
        if column not in df.columns:
            return pd.Series(0.0, index=df.index)

        values = df[column]
        rolling_mean = values.rolling(window=window, min_periods=1).mean()
        rolling_std = values.rolling(window=window, min_periods=1).std()

        zscore = (values - rolling_mean) / rolling_std.replace(0, np.nan)
        return zscore.clip(lower=-3.0, upper=3.0).fillna(0.0)

    def _compute_daily_change(
        self,
        df: pd.DataFrame,
        column: str,
        periods: int = 1
    ) -> pd.Series:
        """Compute daily percentage change."""
        if column not in df.columns:
            return pd.Series(0.0, index=df.index)

        values = df[column]
        return ((values - values.shift(periods)) / values.shift(periods)).fillna(0.0)

    def _compute_time_normalized(self, df: pd.DataFrame) -> pd.Series:
        """Compute normalized time for each row."""
        if 'time' in df.columns:
            times = pd.to_datetime(df['time'])
        elif 'timestamp' in df.columns:
            times = pd.to_datetime(df['timestamp'])
        else:
            # Use index if datetime
            if isinstance(df.index, pd.DatetimeIndex):
                times = df.index
            else:
                return pd.Series(0.5, index=df.index)

        # Trading hours: 13:00-17:55 UTC (8:00-12:55 COT)
        start_hour = 13
        end_hour = 17
        end_minute = 55

        result = []
        for t in times:
            if pd.isna(t):
                result.append(0.5)
                continue

            current_minutes = t.hour * 60 + t.minute
            start_minutes = start_hour * 60
            end_minutes = end_hour * 60 + end_minute

            if current_minutes < start_minutes:
                result.append(0.0)
            elif current_minutes > end_minutes:
                result.append(1.0)
            else:
                normalized = (current_minutes - start_minutes) / (end_minutes - start_minutes)
                result.append(float(np.clip(normalized, 0.0, 1.0)))

        return pd.Series(result, index=df.index)

    def _compute_time_normalized_single(
        self,
        timestamp: Optional[datetime] = None
    ) -> float:
        """Compute normalized time for a single timestamp."""
        if timestamp is None:
            timestamp = datetime.utcnow()

        # Trading hours: 13:00-17:55 UTC (8:00-12:55 COT)
        start_minutes = 13 * 60
        end_minutes = 17 * 60 + 55

        current_minutes = timestamp.hour * 60 + timestamp.minute

        if current_minutes < start_minutes:
            return 0.0
        elif current_minutes > end_minutes:
            return 1.0
        else:
            return float(
                np.clip(
                    (current_minutes - start_minutes) / (end_minutes - start_minutes),
                    0.0,
                    1.0
                )
            )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def get_canonical_builder() -> CanonicalFeatureBuilder:
    """
    Factory function to get a CanonicalFeatureBuilder instance.

    Returns:
        Configured CanonicalFeatureBuilder
    """
    return CanonicalFeatureBuilder()


# Export VERSION as module-level constant for easy access
BUILDER_VERSION = CanonicalFeatureBuilder.VERSION
