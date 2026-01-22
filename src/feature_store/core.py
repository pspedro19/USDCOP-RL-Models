"""
Feature Store Core - Single Source of Truth
============================================
Central module that unifies all feature calculation, contracts, and validation.

This is the AUTHORITATIVE implementation that ALL other modules should use.
Any legacy modules (src/features/*, src/core/calculators/*) are thin wrappers
that delegate to this module.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     feature_store.core                          │
    │                                                                 │
    │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐│
    │  │   Contracts  │  │  Calculators │  │      Registry          ││
    │  │   (Pydantic) │  │  (Strategy)  │  │   (Version Mgmt)       ││
    │  └──────────────┘  └──────────────┘  └────────────────────────┘│
    │                           │                                     │
    │                           ▼                                     │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │                    FeatureBuilder                         │  │
    │  │  build() → FeatureBatch                                  │  │
    │  │  build_single() → FeatureVector                          │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │                                                                 │
    │  Consumers:                                                     │
    │  - Training Pipeline (L3)                                       │
    │  - Backtest Validation (L4)                                     │
    │  - Inference API (L5)                                           │
    │  - Replay API                                                   │
    └─────────────────────────────────────────────────────────────────┘

SOLID Principles:
- Single Responsibility: Each class has one job
- Open/Closed: Extend via registry, not modification
- Liskov Substitution: All calculators are interchangeable
- Interface Segregation: Minimal interfaces
- Dependency Inversion: Depend on abstractions

Author: Trading Team
Version: 2.0.0
Created: 2025-01-12
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional, Protocol, Tuple, Type

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

# Import canonical SSOT from src.core.contracts (REQUIRED - no fallback)
from src.core.contracts import (
    FEATURE_ORDER as SSOT_FEATURE_ORDER,
    OBSERVATION_DIM as SSOT_OBSERVATION_DIM,
    FEATURE_CONTRACT as SSOT_FEATURE_CONTRACT,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS - Single Definition
# =============================================================================

class FeatureVersion(str, Enum):
    """Supported feature set versions"""
    CURRENT = "current"  # Current 15-dimensional (production)


class FeatureCategory(str, Enum):
    """Feature categories for organization"""
    PRICE = "price"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    TREND = "trend"
    MACRO = "macro"
    STATE = "state"


class SmoothingMethod(str, Enum):
    """Smoothing methods - MUST be consistent across all pipelines"""
    EMA = "ema"           # Standard EMA: alpha = 2/(n+1)
    WILDER = "wilder"     # Wilder's EMA: alpha = 1/n (for RSI, ATR, ADX)
    SMA = "sma"           # Simple Moving Average
    NONE = "none"


class NormalizationMethod(str, Enum):
    """Normalization methods"""
    ZSCORE = "zscore"           # (x - mean) / std
    ROLLING_ZSCORE = "rolling"  # Rolling window z-score
    MINMAX = "minmax"           # (x - min) / (max - min)
    NONE = "none"


# =============================================================================
# FEATURE SPECIFICATION - Frozen, Immutable
# =============================================================================

@dataclass(frozen=True)
class TechnicalPeriods:
    """Technical indicator periods - IMMUTABLE"""
    rsi: int = 9
    atr: int = 10
    adx: int = 14


@dataclass(frozen=True)
class TradingHours:
    """Trading hours UTC - IMMUTABLE"""
    start: str = "13:00"
    end: str = "19:00"


@dataclass(frozen=True)
class FeatureContract:
    """
    Feature Contract - IMMUTABLE SPECIFICATION

    NOTE: The canonical SSOT is in src/core/contracts/feature_contract.py
    This class uses SSOT values exclusively (no fallback).
    """
    version: str = "current"
    observation_dim: int = SSOT_OBSERVATION_DIM  # From SSOT
    feature_order: Tuple[str, ...] = SSOT_FEATURE_ORDER  # From SSOT
    norm_stats_path: str = "config/norm_stats.json"
    clip_range: Tuple[float, float] = (-5.0, 5.0)

    # Trading hours
    trading_hours_start: str = "13:00"
    trading_hours_end: str = "17:55"

    # Technical periods
    rsi_period: int = 9
    atr_period: int = 10
    adx_period: int = 14

    # Warmup
    warmup_bars: int = 14

    # Metadata
    created_at: str = "2025-01-12"

    def get_trading_hours(self) -> Dict[str, str]:
        return {"start": self.trading_hours_start, "end": self.trading_hours_end}

    def get_technical_periods(self) -> Dict[str, int]:
        return {"rsi": self.rsi_period, "atr": self.atr_period, "adx": self.adx_period}


# Singleton instance
FEATURE_CONTRACT: Final = FeatureContract()

# Convenience exports - Always from SSOT (no fallback)
FEATURE_ORDER: Final = SSOT_FEATURE_ORDER
OBSERVATION_DIM: Final = SSOT_OBSERVATION_DIM
NORM_STATS_PATH: Final = FEATURE_CONTRACT.norm_stats_path


# =============================================================================
# CALCULATOR PROTOCOL - Interface for all calculators
# =============================================================================

class IFeatureCalculator(Protocol):
    """Protocol that all feature calculators must implement"""

    def calculate(self, data: pd.DataFrame, bar_idx: int) -> float:
        """Calculate feature value for a specific bar"""
        ...

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """Calculate feature values for all bars"""
        ...

    @property
    def name(self) -> str:
        """Feature name"""
        ...

    @property
    def requires(self) -> List[str]:
        """Required columns"""
        ...


# =============================================================================
# BASE CALCULATOR - Template Method Pattern
# =============================================================================

class BaseCalculator(ABC):
    """
    Abstract base for all feature calculators.

    Implements Template Method pattern:
    - calculate() is the template that handles common logic
    - _calculate_impl() is the hook for specific calculation
    """

    def __init__(
        self,
        name: str,
        requires: List[str],
        smoothing: SmoothingMethod = SmoothingMethod.NONE,
        window: int = 1
    ):
        self._name = name
        self._requires = requires
        self._smoothing = smoothing
        self._window = window

    @property
    def name(self) -> str:
        return self._name

    @property
    def requires(self) -> List[str]:
        return self._requires

    def calculate(self, data: pd.DataFrame, bar_idx: int) -> float:
        """Template method for single bar calculation"""
        self._validate_data(data)
        if bar_idx < self._window:
            return self._get_default_value()
        return self._calculate_impl(data, bar_idx)

    def calculate_batch(self, data: pd.DataFrame) -> pd.Series:
        """Calculate for all bars"""
        self._validate_data(data)
        return self._calculate_batch_impl(data)

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate required columns exist"""
        missing = set(self._requires) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns for {self._name}: {missing}")

    @abstractmethod
    def _calculate_impl(self, data: pd.DataFrame, bar_idx: int) -> float:
        """Implement specific calculation logic"""
        pass

    @abstractmethod
    def _calculate_batch_impl(self, data: pd.DataFrame) -> pd.Series:
        """Implement batch calculation logic"""
        pass

    def _get_default_value(self) -> float:
        """Default value when insufficient data"""
        return 0.0

    def _apply_smoothing(
        self,
        series: pd.Series,
        method: SmoothingMethod,
        window: int
    ) -> pd.Series:
        """
        Apply smoothing - AUTHORITATIVE IMPLEMENTATION

        This is the SINGLE SOURCE OF TRUTH for smoothing.
        All pipelines MUST use this method.
        """
        if method == SmoothingMethod.NONE:
            return series

        if method == SmoothingMethod.SMA:
            return series.rolling(window=window, min_periods=1).mean()

        if method == SmoothingMethod.EMA:
            # Standard EMA: alpha = 2 / (n + 1)
            alpha = 2.0 / (window + 1)
            return series.ewm(alpha=alpha, min_periods=1, adjust=False).mean()

        if method == SmoothingMethod.WILDER:
            # Wilder's smoothing: alpha = 1 / n
            # Used by RSI, ATR, ADX to match TA-Lib and industry standard
            alpha = 1.0 / window
            return series.ewm(alpha=alpha, min_periods=1, adjust=False).mean()

        raise ValueError(f"Unknown smoothing method: {method}")


# =============================================================================
# CONCRETE CALCULATORS - Strategy Pattern
# =============================================================================

class LogReturnCalculator(BaseCalculator):
    """Log return calculator"""

    def __init__(self, name: str, periods: int):
        super().__init__(
            name=name,
            requires=["close"],
            window=periods
        )
        self._periods = periods

    def _calculate_impl(self, data: pd.DataFrame, bar_idx: int) -> float:
        close = data["close"].values
        if bar_idx < self._periods:
            return 0.0
        return float(np.log(close[bar_idx] / close[bar_idx - self._periods]))

    def _calculate_batch_impl(self, data: pd.DataFrame) -> pd.Series:
        return np.log(data["close"] / data["close"].shift(self._periods)).fillna(0.0)


class RSICalculator(BaseCalculator):
    """
    RSI Calculator using Wilder's smoothing

    CRITICAL: Uses alpha = 1/period (Wilder's method)
    This matches TA-Lib and ensures parity with training.
    """

    def __init__(self, period: int = 9):
        super().__init__(
            name=f"rsi_{period}",
            requires=["close"],
            smoothing=SmoothingMethod.WILDER,
            window=period
        )
        self._period = period

    def _calculate_impl(self, data: pd.DataFrame, bar_idx: int) -> float:
        if bar_idx < self._period + 1:
            return 50.0

        close = data["close"].values
        deltas = np.diff(close[bar_idx - self._period:bar_idx + 1])

        gains = np.maximum(deltas, 0)
        losses = np.abs(np.minimum(deltas, 0))

        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _calculate_batch_impl(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        delta = close.diff()

        gains = delta.where(delta > 0, 0.0)
        losses = (-delta).where(delta < 0, 0.0)

        # Wilder's smoothing
        avg_gains = self._apply_smoothing(gains, SmoothingMethod.WILDER, self._period)
        avg_losses = self._apply_smoothing(losses, SmoothingMethod.WILDER, self._period)

        rs = avg_gains / avg_losses.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        return rsi.fillna(50.0)

    def _get_default_value(self) -> float:
        return 50.0


class ATRPercentCalculator(BaseCalculator):
    """
    ATR as percentage of price

    CRITICAL: Uses Wilder's smoothing for ATR calculation
    """

    def __init__(self, period: int = 10):
        super().__init__(
            name="atr_pct",
            requires=["high", "low", "close"],
            smoothing=SmoothingMethod.WILDER,
            window=period
        )
        self._period = period

    def _calculate_impl(self, data: pd.DataFrame, bar_idx: int) -> float:
        if bar_idx < self._period + 1:
            return 0.0

        high = data["high"].values
        low = data["low"].values
        close = data["close"].values

        tr_values = []
        for i in range(bar_idx - self._period, bar_idx + 1):
            if i > 0:
                tr = max(
                    high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1])
                )
                tr_values.append(tr)

        atr = np.mean(tr_values) if tr_values else 0
        current_price = close[bar_idx]

        return (atr / current_price) if current_price > 0 else 0.0

    def _calculate_batch_impl(self, data: pd.DataFrame) -> pd.Series:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Wilder's smoothing for ATR
        atr = self._apply_smoothing(tr, SmoothingMethod.WILDER, self._period)

        atr_pct = atr / close
        return atr_pct.fillna(0.0)


class ADXCalculator(BaseCalculator):
    """
    ADX Calculator using Wilder's smoothing

    CRITICAL: All components (+DI, -DI, DX, ADX) use Wilder's smoothing
    """

    def __init__(self, period: int = 14):
        super().__init__(
            name=f"adx_{period}",
            requires=["high", "low", "close"],
            smoothing=SmoothingMethod.WILDER,
            window=period
        )
        self._period = period

    def _calculate_impl(self, data: pd.DataFrame, bar_idx: int) -> float:
        if bar_idx < self._period + 1:
            return 25.0

        high = data["high"].values
        low = data["low"].values
        close = data["close"].values

        plus_dm = []
        minus_dm = []
        tr_values = []

        for i in range(bar_idx - self._period, bar_idx + 1):
            if i > 0:
                high_diff = high[i] - high[i - 1]
                low_diff = low[i - 1] - low[i]

                plus_dm.append(max(high_diff, 0) if high_diff > low_diff else 0)
                minus_dm.append(max(low_diff, 0) if low_diff > high_diff else 0)

                tr = max(
                    high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1])
                )
                tr_values.append(tr)

        atr = np.mean(tr_values) if tr_values else 1
        plus_di = 100 * (np.mean(plus_dm) / atr) if atr > 0 else 0
        minus_di = 100 * (np.mean(minus_dm) / atr) if atr > 0 else 0

        di_sum = plus_di + minus_di
        dx = 100 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0

        return float(dx)

    def _calculate_batch_impl(self, data: pd.DataFrame) -> pd.Series:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        # Smooth with Wilder's
        atr = self._apply_smoothing(tr, SmoothingMethod.WILDER, self._period)
        plus_dm_smooth = self._apply_smoothing(plus_dm, SmoothingMethod.WILDER, self._period)
        minus_dm_smooth = self._apply_smoothing(minus_dm, SmoothingMethod.WILDER, self._period)

        # +DI and -DI
        plus_di = 100 * plus_dm_smooth / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm_smooth / atr.replace(0, np.nan)

        # DX
        di_sum = plus_di + minus_di
        di_diff = (plus_di - minus_di).abs()
        dx = 100 * di_diff / di_sum.replace(0, np.nan)

        # ADX = smoothed DX
        adx = self._apply_smoothing(dx, SmoothingMethod.WILDER, self._period)

        return adx.fillna(25.0)

    def _get_default_value(self) -> float:
        return 25.0


class MacroZScoreCalculator(BaseCalculator):
    """
    Macro indicator z-score calculator

    Uses rolling window for z-score to prevent look-ahead bias.
    """

    def __init__(self, name: str, column: str, window: int = 60):
        super().__init__(
            name=name,
            requires=[column],
            window=window
        )
        self._column = column
        self._zscore_window = window

    def _calculate_impl(self, data: pd.DataFrame, bar_idx: int) -> float:
        if self._column not in data.columns:
            return 0.0

        values = data[self._column].values
        start_idx = max(0, bar_idx - self._zscore_window + 1)
        window_data = values[start_idx:bar_idx + 1]

        if len(window_data) < 2:
            return 0.0

        mean = np.mean(window_data)
        std = np.std(window_data)

        if std == 0:
            return 0.0

        z = (values[bar_idx] - mean) / std
        return float(np.clip(z, -3.0, 3.0))

    def _calculate_batch_impl(self, data: pd.DataFrame) -> pd.Series:
        if self._column not in data.columns:
            return pd.Series(0.0, index=data.index)

        values = data[self._column]
        rolling_mean = values.rolling(window=self._zscore_window, min_periods=1).mean()
        rolling_std = values.rolling(window=self._zscore_window, min_periods=1).std()

        zscore = (values - rolling_mean) / rolling_std.replace(0, np.nan)
        return zscore.clip(lower=-3.0, upper=3.0).fillna(0.0)


class MacroChangeCalculator(BaseCalculator):
    """Daily change calculator for macro indicators"""

    def __init__(self, name: str, column: str, periods: int = 1):
        super().__init__(
            name=name,
            requires=[column],
            window=periods
        )
        self._column = column
        self._periods = periods

    def _calculate_impl(self, data: pd.DataFrame, bar_idx: int) -> float:
        if self._column not in data.columns:
            return 0.0

        values = data[self._column].values
        if bar_idx < self._periods:
            return 0.0

        current = values[bar_idx]
        prev = values[bar_idx - self._periods]

        if prev == 0:
            return 0.0

        return float((current - prev) / prev)

    def _calculate_batch_impl(self, data: pd.DataFrame) -> pd.Series:
        if self._column not in data.columns:
            return pd.Series(0.0, index=data.index)

        values = data[self._column]
        return ((values - values.shift(self._periods)) / values.shift(self._periods)).fillna(0.0)


# =============================================================================
# CALCULATOR REGISTRY - Factory Pattern
# =============================================================================

class CalculatorRegistry:
    """
    Registry for feature calculators.

    Implements Factory Pattern for calculator creation.
    """

    _instance: Optional["CalculatorRegistry"] = None
    _calculators: Dict[str, BaseCalculator] = {}

    def __new__(cls) -> "CalculatorRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_features()
        return cls._instance

    @classmethod
    def instance(cls) -> "CalculatorRegistry":
        return cls()

    def _initialize_features(self) -> None:
        """Initialize feature calculators based on the contract specification."""
        periods = FEATURE_CONTRACT.get_technical_periods()

        self._calculators = {
            # Returns
            "log_ret_5m": LogReturnCalculator("log_ret_5m", periods=1),
            "log_ret_1h": LogReturnCalculator("log_ret_1h", periods=12),
            "log_ret_4h": LogReturnCalculator("log_ret_4h", periods=48),

            # Technical
            "rsi_9": RSICalculator(period=periods["rsi"]),
            "atr_pct": ATRPercentCalculator(period=periods["atr"]),
            "adx_14": ADXCalculator(period=periods["adx"]),

            # Macro z-scores
            "dxy_z": MacroZScoreCalculator("dxy_z", "dxy", window=60),
            "vix_z": MacroZScoreCalculator("vix_z", "vix", window=60),
            "embi_z": MacroZScoreCalculator("embi_z", "embi", window=60),

            # Macro changes
            "dxy_change_1d": MacroChangeCalculator("dxy_change_1d", "dxy", periods=1),
            "brent_change_1d": MacroChangeCalculator("brent_change_1d", "brent", periods=1),
            "usdmxn_change_1d": MacroChangeCalculator("usdmxn_change_1d", "usdmxn", periods=1),
        }

        logger.info(f"Initialized {len(self._calculators)} feature calculators")

    def get(self, name: str) -> Optional[BaseCalculator]:
        """Get calculator by name"""
        return self._calculators.get(name)

    def register(self, calculator: BaseCalculator) -> None:
        """Register a calculator"""
        self._calculators[calculator.name] = calculator

    def list_calculators(self) -> List[str]:
        """List all registered calculator names"""
        return list(self._calculators.keys())


# =============================================================================
# UNIFIED FEATURE BUILDER - Wrapper to CanonicalFeatureBuilder (SSOT)
# =============================================================================

class UnifiedFeatureBuilder:
    """
    Unified Feature Builder - WRAPPER to CanonicalFeatureBuilder (SSOT)

    NOTE: This class now delegates all calculations to CanonicalFeatureBuilder
    from src.feature_store.builders. It exists for backward compatibility.

    For new code, use CanonicalFeatureBuilder directly:
        from src.feature_store.builders import CanonicalFeatureBuilder
        builder = CanonicalFeatureBuilder.for_inference()

    v3.0.0 CHANGELOG:
    - REMOVED: ~185 lines of duplicate calculation logic
    - DELEGATES: All methods now use CanonicalFeatureBuilder (SSOT)
    - DRY: Zero code duplication
    """

    def __init__(self, version: str = "current"):
        """
        Initialize builder - delegates to CanonicalFeatureBuilder.

        Args:
            version: Contract version (default: "current")
        """
        # Import here to avoid circular imports
        from .builders import CanonicalFeatureBuilder

        self._canonical = CanonicalFeatureBuilder.for_training()
        self.contract = FEATURE_CONTRACT
        self.norm_stats = self._canonical.get_norm_stats()

        logger.info(f"UnifiedFeatureBuilder initialized (delegates to CanonicalFeatureBuilder)")

    def get_observation_dim(self) -> int:
        """Get observation dimension"""
        return self._canonical.get_observation_dim()

    def get_feature_names(self) -> Tuple[str, ...]:
        """Get feature names in order"""
        return tuple(self._canonical.get_feature_order())

    def build_observation(
        self,
        ohlcv: pd.DataFrame,
        macro_df: pd.DataFrame,
        position: float,
        timestamp: pd.Timestamp,
        bar_idx: int
    ) -> np.ndarray:
        """
        Build observation array - DELEGATES to CanonicalFeatureBuilder.

        Args:
            ohlcv: DataFrame with OHLCV columns
            macro_df: DataFrame with macro columns
            position: Current position [-1, 1]
            timestamp: Current timestamp UTC
            bar_idx: Bar index for warmup validation

        Returns:
            np.ndarray of shape (observation_dim,)
        """
        return self._canonical.build_observation(
            ohlcv=ohlcv,
            macro=macro_df,
            position=position,
            bar_idx=bar_idx,
            timestamp=timestamp
        )


# =============================================================================
# CONTRACT ACCESSOR - Factory Pattern
# =============================================================================

def get_contract(version: str = "current") -> FeatureContract:
    """
    Factory for contracts by version.

    Args:
        version: Contract version (default: "current")

    Returns:
        Contract instance
    """
    contracts = {"current": FEATURE_CONTRACT}
    if version not in contracts:
        raise ValueError(f"Unknown version: {version}. Available: {list(contracts.keys())}")
    return contracts[version]


def get_feature_builder(version: str = "current") -> UnifiedFeatureBuilder:
    """
    Factory for feature builders.

    Args:
        version: Contract version (default: "current")

    Returns:
        Configured builder
    """
    return UnifiedFeatureBuilder(version=version)
