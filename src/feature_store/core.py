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

    This is the Single Source of Truth for feature definitions.
    Contains the 15-dimensional observation space used across all pipelines.
    """
    version: str = "current"
    observation_dim: int = 15
    feature_order: Tuple[str, ...] = (
        "log_ret_5m", "log_ret_1h", "log_ret_4h",
        "rsi_9", "atr_pct", "adx_14",
        "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
        "brent_change_1d", "rate_spread", "usdmxn_change_1d",
        "position", "time_normalized"
    )
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

# Convenience exports
FEATURE_ORDER: Final = FEATURE_CONTRACT.feature_order
OBSERVATION_DIM: Final = FEATURE_CONTRACT.observation_dim
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
# UNIFIED FEATURE BUILDER
# =============================================================================

class UnifiedFeatureBuilder:
    """
    Unified Feature Builder - SINGLE SOURCE OF TRUTH

    This builder is used by:
    - Training Pipeline
    - Backtest Validation
    - Inference API
    - Replay API

    All feature calculations go through this class to ensure
    perfect parity across all pipelines.
    """

    def __init__(self, version: str = "current"):
        """
        Initialize builder with contract version.

        Args:
            version: Contract version (default: "current")
        """
        self.contract = self._get_contract(version)
        self.registry = CalculatorRegistry.instance()
        self.norm_stats = self._load_norm_stats()

        logger.info(f"UnifiedFeatureBuilder initialized: {version}")

    def _get_contract(self, version: str) -> FeatureContract:
        """Get contract by version"""
        contracts = {"current": FEATURE_CONTRACT}
        if version not in contracts:
            raise ValueError(f"Unknown version: {version}. Available: {list(contracts.keys())}")
        return contracts[version]

    def _load_norm_stats(self) -> Dict[str, Dict[str, float]]:
        """Load normalization stats from config"""
        path = Path(self.contract.norm_stats_path)

        # Try relative to project root
        if not path.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            path = project_root / self.contract.norm_stats_path

        if not path.exists():
            raise FileNotFoundError(
                f"CRITICAL: Norm stats file not found: {path}. "
                f"Model cannot produce correct predictions without exact training stats."
            )

        with open(path) as f:
            return json.load(f)

    def get_observation_dim(self) -> int:
        """Get observation dimension"""
        return self.contract.observation_dim

    def get_feature_names(self) -> Tuple[str, ...]:
        """Get feature names in order"""
        return self.contract.feature_order

    def build_observation(
        self,
        ohlcv: pd.DataFrame,
        macro_df: pd.DataFrame,
        position: float,
        timestamp: pd.Timestamp,
        bar_idx: int
    ) -> np.ndarray:
        """
        Build observation array following the feature contract.

        Args:
            ohlcv: DataFrame with OHLCV columns
            macro_df: DataFrame with macro columns
            position: Current position [-1, 1]
            timestamp: Current timestamp UTC
            bar_idx: Bar index for warmup validation

        Returns:
            np.ndarray of shape (observation_dim,)
        """
        # Validate warmup
        if bar_idx < self.contract.warmup_bars:
            raise ValueError(
                f"bar_idx ({bar_idx}) < warmup_bars ({self.contract.warmup_bars})"
            )

        # Calculate raw features
        raw = self._calculate_raw_features(ohlcv, macro_df, bar_idx)

        # Normalize
        normalized = self._normalize_features(raw)

        # Add state features
        normalized["position"] = float(np.clip(position, -1.0, 1.0))
        normalized["time_normalized"] = self._compute_time_normalized(timestamp)

        # Assemble in order
        observation = self._assemble_observation(normalized)

        # Final validation
        assert observation.shape == (self.contract.observation_dim,)
        assert not np.isnan(observation).any(), "NaN in observation"
        assert not np.isinf(observation).any(), "Inf in observation"

        return observation

    def _calculate_raw_features(
        self,
        ohlcv: pd.DataFrame,
        macro_df: pd.DataFrame,
        bar_idx: int
    ) -> Dict[str, float]:
        """Calculate raw feature values"""
        features = {}

        # Merge data for calculators
        data = ohlcv.copy()
        if macro_df is not None and len(macro_df) > 0:
            for col in macro_df.columns:
                if col not in data.columns:
                    data[col] = macro_df[col]

        # Use registered calculators
        for name in self.contract.feature_order[:13]:
            calc = self.registry.get(name)
            if calc:
                features[name] = calc.calculate(data, bar_idx)
            elif name == "rate_spread":
                # Special case: rate_spread is raw value
                if "rate_spread" in data.columns:
                    features[name] = float(data["rate_spread"].iloc[bar_idx])
                else:
                    features[name] = 0.0
            else:
                features[name] = 0.0

        return features

    def _normalize_features(self, raw: Dict[str, float]) -> Dict[str, float]:
        """Apply z-score normalization"""
        normalized = {}
        clip_min, clip_max = self.contract.clip_range

        for name, value in raw.items():
            stats = self.norm_stats.get(name)
            if stats and stats.get("std", 0) > 0:
                z = (value - stats["mean"]) / stats["std"]
                normalized[name] = float(np.clip(z, clip_min, clip_max))
            else:
                normalized[name] = float(np.clip(value, clip_min, clip_max))

        return normalized

    def _compute_time_normalized(self, timestamp: pd.Timestamp) -> float:
        """Normalize timestamp to [0, 1] within trading hours"""
        hours = self.contract.get_trading_hours()
        start_hour = int(hours["start"].split(":")[0])
        end_hour = int(hours["end"].split(":")[0])

        current_minutes = timestamp.hour * 60 + timestamp.minute
        start_minutes = start_hour * 60
        end_minutes = end_hour * 60

        if end_minutes <= start_minutes:
            end_minutes += 24 * 60

        if current_minutes < start_minutes:
            current_minutes += 24 * 60

        normalized = (current_minutes - start_minutes) / (end_minutes - start_minutes)
        return float(np.clip(normalized, 0.0, 1.0))

    def _assemble_observation(self, features: Dict[str, float]) -> np.ndarray:
        """Assemble array in contract order"""
        observation = np.zeros(self.contract.observation_dim, dtype=np.float32)

        for idx, name in enumerate(self.contract.feature_order):
            value = features.get(name, 0.0)
            if np.isnan(value) or np.isinf(value):
                value = 0.0
            observation[idx] = value

        return observation


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
