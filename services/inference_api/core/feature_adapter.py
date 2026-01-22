"""
InferenceFeatureAdapter - SSOT Delegation for Inference
========================================================

This adapter connects the inference pipeline to the Single Source of Truth (SSOT)
feature calculators in src/feature_store/core.py.

CRITICAL: This adapter does NOT implement any calculation logic.
All calculations are delegated to src/feature_store/ calculators to ensure
PERFECT PARITY between training and inference.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                  InferenceFeatureAdapter                        │
    │                                                                 │
    │  Delegates to:                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │           src/feature_store/core.py                       │  │
    │  │  - RSICalculator (Wilder's EMA, alpha=1/period)          │  │
    │  │  - ATRPercentCalculator (Wilder's EMA)                   │  │
    │  │  - ADXCalculator (Wilder's EMA)                          │  │
    │  │  - LogReturnCalculator                                    │  │
    │  │  - MacroZScoreCalculator                                  │  │
    │  │  - MacroChangeCalculator                                  │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │                                                                 │
    │  GUARANTEES:                                                    │
    │  - RSI inference = RSI training (rtol=1e-6)                    │
    │  - ATR inference = ATR training (rtol=1e-6)                    │
    │  - ADX inference = ADX training (rtol=1e-6)                    │
    └─────────────────────────────────────────────────────────────────┘

Author: Trading Team
Version: 1.0.0
Created: 2025-01-14
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import from SSOT - these are the AUTHORITATIVE calculators
from src.feature_store.core import (
    ADXCalculator,
    ATRPercentCalculator,
    BaseCalculator,
    CalculatorRegistry,
    FEATURE_CONTRACT,
    FEATURE_ORDER,
    LogReturnCalculator,
    MacroChangeCalculator,
    MacroZScoreCalculator,
    OBSERVATION_DIM,
    RSICalculator,
    SmoothingMethod,
)

logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE CIRCUIT BREAKER - Phase 13 Integration Point
# =============================================================================

class FeatureCircuitBreakerError(Exception):
    """Raised when circuit breaker is triggered due to data quality issues."""

    def __init__(self, nan_ratio: float, affected_features: List[str]):
        self.nan_ratio = nan_ratio
        self.affected_features = affected_features
        super().__init__(
            f"Feature circuit breaker triggered: {nan_ratio:.1%} NaN "
            f"(threshold: 20%). Affected: {affected_features}"
        )


@dataclass
class FeatureCircuitBreakerConfig:
    """Configuration for feature quality circuit breaker."""
    max_nan_ratio: float = 0.20  # 20% threshold
    max_consecutive_failures: int = 5
    cooldown_seconds: float = 900.0  # 15 minutes


# =============================================================================
# GAP HANDLER - Phase 14 Integration Point
# =============================================================================

@dataclass
class GapConfig:
    """Configuration for gap handling - MUST match training."""
    warmup_bars: int = 14
    max_gap_minutes: int = 30
    fill_strategy: str = "forward_then_zero"


# =============================================================================
# INFERENCE FEATURE ADAPTER
# =============================================================================

class InferenceFeatureAdapter:
    """
    Adapter that connects inference to SSOT feature_store calculators.

    IMPORTANT: This class does NOT implement any calculation logic.
    All calculations are delegated to src/feature_store/calculators/.

    This ensures PERFECT PARITY between training and inference by using
    the EXACT SAME calculators with Wilder's EMA smoothing.

    Usage:
        adapter = InferenceFeatureAdapter()
        obs = adapter.build_observation(
            df=df,
            bar_idx=100,
            position=0.0,
            time_normalized=0.5  # SSOT feature name (index 14)
        )
    """

    # Class constants - match SSOT contract
    OBSERVATION_DIM = OBSERVATION_DIM  # 15
    FEATURE_ORDER = FEATURE_ORDER
    CLIP_RANGE = FEATURE_CONTRACT.clip_range  # (-5.0, 5.0)

    def __init__(
        self,
        norm_stats_path: Optional[str] = None,
        circuit_breaker_config: Optional[FeatureCircuitBreakerConfig] = None,
        gap_config: Optional[GapConfig] = None,
    ):
        """
        Initialize the InferenceFeatureAdapter.

        Args:
            norm_stats_path: Path to normalization stats JSON.
                            Defaults to FEATURE_CONTRACT.norm_stats_path.
            circuit_breaker_config: Optional circuit breaker configuration.
            gap_config: Optional gap handling configuration.

        Raises:
            FileNotFoundError: If norm_stats file not found (FAIL-FAST).
        """
        self._norm_stats_path = norm_stats_path or FEATURE_CONTRACT.norm_stats_path
        self._norm_stats = self._load_norm_stats_strict()
        self._calculators = self._init_ssot_calculators()

        # Circuit breaker state
        self._cb_config = circuit_breaker_config or FeatureCircuitBreakerConfig()
        self._feature_failure_counts: Dict[str, int] = {}
        self._last_circuit_break_time: Optional[float] = None

        # Gap handling
        self._gap_config = gap_config or GapConfig()

        logger.info(
            f"InferenceFeatureAdapter initialized with {len(self._calculators)} "
            f"SSOT calculators (Wilder's EMA smoothing)"
        )

    def _load_norm_stats_strict(self) -> Dict[str, Dict[str, float]]:
        """
        Load normalization statistics - FAIL FAST if not found.

        Returns:
            Dict of feature_name -> {mean, std, ...}

        Raises:
            FileNotFoundError: If norm_stats file not found.
        """
        path = Path(self._norm_stats_path)

        # Try relative to project root
        if not path.is_absolute():
            project_root = Path(__file__).parent.parent.parent.parent
            path = project_root / self._norm_stats_path

        if not path.exists():
            raise FileNotFoundError(
                f"CRITICAL: Normalization statistics file NOT FOUND at {path}. "
                f"Model cannot produce correct predictions without exact training stats."
            )

        with open(path, 'r', encoding='utf-8') as f:
            stats = json.load(f)

        logger.info(f"Loaded norm_stats from {path}: {len(stats)} features")
        return stats

    def _init_ssot_calculators(self) -> Dict[str, BaseCalculator]:
        """
        Initialize calculators from SSOT feature_store.

        CRITICAL: These are the SAME calculators used in training.
        They use Wilder's EMA (alpha = 1/period) for RSI, ATR, ADX.

        Returns:
            Dict of feature_name -> Calculator instance
        """
        periods = FEATURE_CONTRACT.get_technical_periods()

        calculators = {
            # Returns - no smoothing needed
            "log_ret_5m": LogReturnCalculator("log_ret_5m", periods=1),
            "log_ret_1h": LogReturnCalculator("log_ret_1h", periods=12),
            "log_ret_4h": LogReturnCalculator("log_ret_4h", periods=48),

            # Technical indicators - USE WILDER'S EMA
            "rsi_9": RSICalculator(period=periods["rsi"]),      # Wilder's
            "atr_pct": ATRPercentCalculator(period=periods["atr"]),  # Wilder's
            "adx_14": ADXCalculator(period=periods["adx"]),    # Wilder's

            # Macro z-scores
            "dxy_z": MacroZScoreCalculator("dxy_z", "dxy", window=60),
            "vix_z": MacroZScoreCalculator("vix_z", "vix", window=60),
            "embi_z": MacroZScoreCalculator("embi_z", "embi", window=60),

            # Macro changes
            "dxy_change_1d": MacroChangeCalculator("dxy_change_1d", "dxy", periods=1),
            "brent_change_1d": MacroChangeCalculator("brent_change_1d", "brent", periods=1),
            "usdmxn_change_1d": MacroChangeCalculator("usdmxn_change_1d", "usdmxn", periods=1),
        }

        logger.debug(f"Initialized {len(calculators)} SSOT calculators")
        return calculators

    def calculate_technical_features(
        self,
        df: pd.DataFrame,
        bar_idx: int
    ) -> Dict[str, float]:
        """
        Calculate technical features using SSOT calculators.

        CRITICAL: Uses Wilder's EMA for RSI, ATR, ADX to match training.

        Args:
            df: DataFrame with OHLCV columns (open, high, low, close)
            bar_idx: Current bar index

        Returns:
            Dict of feature_name -> raw value
        """
        features = {}

        for name in ["log_ret_5m", "log_ret_1h", "log_ret_4h", "rsi_9", "atr_pct", "adx_14"]:
            calc = self._calculators.get(name)
            if calc:
                try:
                    features[name] = calc.calculate(df, bar_idx)
                except Exception as e:
                    logger.warning(f"Feature {name} calculation failed: {e}")
                    features[name] = calc._get_default_value()
            else:
                features[name] = 0.0

        return features

    def calculate_macro_features(
        self,
        df: pd.DataFrame,
        bar_idx: int
    ) -> Dict[str, float]:
        """
        Calculate macro features using SSOT calculators.

        Args:
            df: DataFrame with macro columns (dxy, vix, embi, brent, usdmxn, treasury_10y)
            bar_idx: Current bar index

        Returns:
            Dict of feature_name -> raw value
        """
        features = {}

        # Z-score features
        for name in ["dxy_z", "vix_z", "embi_z"]:
            calc = self._calculators.get(name)
            if calc and calc._column in df.columns:
                try:
                    features[name] = calc.calculate(df, bar_idx)
                except Exception as e:
                    logger.warning(f"Feature {name} calculation failed: {e}")
                    features[name] = 0.0
            else:
                features[name] = 0.0

        # Change features
        for name in ["dxy_change_1d", "brent_change_1d", "usdmxn_change_1d"]:
            calc = self._calculators.get(name)
            if calc and calc._column in df.columns:
                try:
                    features[name] = calc.calculate(df, bar_idx)
                except Exception as e:
                    logger.warning(f"Feature {name} calculation failed: {e}")
                    features[name] = 0.0
            else:
                features[name] = 0.0

        # Rate spread (special case - formula: 10.0 - treasury_10y)
        if "treasury_10y" in df.columns:
            treasury = df["treasury_10y"].iloc[bar_idx]
            features["rate_spread"] = 10.0 - treasury if not pd.isna(treasury) else 0.0
        else:
            features["rate_spread"] = 0.0

        return features

    def normalize_feature(self, value: float, feature_name: str) -> float:
        """
        Z-score normalize a single feature using training stats.

        Args:
            value: Raw feature value
            feature_name: Name of the feature

        Returns:
            Normalized value, clipped to CLIP_RANGE
        """
        if np.isnan(value) or np.isinf(value):
            return 0.0

        if feature_name not in self._norm_stats:
            return float(np.clip(value, self.CLIP_RANGE[0], self.CLIP_RANGE[1]))

        stats = self._norm_stats[feature_name]
        mean = stats.get("mean", 0.0)
        std = stats.get("std", 1.0)

        # Protect against division by zero
        if std < 1e-8:
            std = 1.0

        z_score = (value - mean) / std
        return float(np.clip(z_score, self.CLIP_RANGE[0], self.CLIP_RANGE[1]))

    def check_circuit_breaker(self, features: Dict[str, float]) -> None:
        """
        Check feature quality and raise if circuit breaker triggered.

        Args:
            features: Dict of feature_name -> value

        Raises:
            FeatureCircuitBreakerError: If quality threshold exceeded
        """
        # Check cooldown
        if self._last_circuit_break_time:
            elapsed = time.time() - self._last_circuit_break_time
            if elapsed < self._cb_config.cooldown_seconds:
                raise FeatureCircuitBreakerError(
                    nan_ratio=1.0,
                    affected_features=["cooldown_active"]
                )

        # Count NaN/Inf features
        nan_features = []
        for name, value in features.items():
            if np.isnan(value) or np.isinf(value):
                nan_features.append(name)
                self._feature_failure_counts[name] = \
                    self._feature_failure_counts.get(name, 0) + 1
            else:
                self._feature_failure_counts[name] = 0

        nan_ratio = len(nan_features) / len(features) if features else 0

        # Check thresholds
        if nan_ratio > self._cb_config.max_nan_ratio:
            self._last_circuit_break_time = time.time()
            raise FeatureCircuitBreakerError(nan_ratio, nan_features)

        # Check consecutive failures
        for name, count in self._feature_failure_counts.items():
            if count >= self._cb_config.max_consecutive_failures:
                self._last_circuit_break_time = time.time()
                raise FeatureCircuitBreakerError(
                    nan_ratio=nan_ratio,
                    affected_features=[name]
                )

    def build_observation(
        self,
        df: pd.DataFrame,
        bar_idx: int,
        position: float,
        time_normalized: float = 0.5,
        check_circuit_breaker: bool = True
    ) -> np.ndarray:
        """
        Build complete 15-dimensional observation vector.

        CRITICAL: Uses SSOT calculators with Wilder's EMA smoothing
        to ensure PERFECT PARITY with training.

        Args:
            df: DataFrame with OHLCV and macro data
            bar_idx: Current bar index
            position: Current position (-1 to 1)
            time_normalized: Normalized trading session time (0 to 1)
                            Maps to SSOT feature index 14 'time_normalized'
            check_circuit_breaker: Whether to check feature quality

        Returns:
            numpy array of shape (15,) with dtype float32

        Raises:
            FeatureCircuitBreakerError: If data quality is too low
        """
        # Calculate raw features using SSOT calculators
        tech_features = self.calculate_technical_features(df, bar_idx)
        macro_features = self.calculate_macro_features(df, bar_idx)

        # Merge all features
        all_features = {**tech_features, **macro_features}

        # Check circuit breaker
        if check_circuit_breaker:
            self.check_circuit_breaker(all_features)

        # Build observation vector in correct order
        obs = np.zeros(self.OBSERVATION_DIM, dtype=np.float32)

        # Core features (indices 0-12) - normalized
        for i, name in enumerate(self.FEATURE_ORDER[:13]):
            raw_value = all_features.get(name, 0.0)
            obs[i] = self.normalize_feature(raw_value, name)

        # State features (indices 13-14) - not normalized
        # Index 13: position (-1 to 1)
        # Index 14: time_normalized (0 to 1) - SSOT feature name
        obs[13] = float(np.clip(position, -1.0, 1.0))
        obs[14] = float(np.clip(time_normalized, 0.0, 1.0))

        # Final NaN/Inf check
        obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)

        return obs

    def get_feature_stats(self, feature_name: str) -> Dict[str, float]:
        """Get normalization statistics for a feature."""
        return self._norm_stats.get(feature_name, {"mean": 0.0, "std": 1.0})

    @property
    def calculators(self) -> Dict[str, BaseCalculator]:
        """Access to underlying SSOT calculators for testing."""
        return self._calculators


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_adapter_instance: Optional[InferenceFeatureAdapter] = None


def get_feature_adapter() -> InferenceFeatureAdapter:
    """Get or create global feature adapter instance."""
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = InferenceFeatureAdapter()
    return _adapter_instance


def calculate_rsi_wilder(prices: pd.Series, period: int = 9) -> pd.Series:
    """
    Calculate RSI using Wilder's smoothing - FOR TESTING PARITY.

    This is a convenience function that delegates to the SSOT RSICalculator.
    Use this in tests to verify parity between training and inference.

    Args:
        prices: Close prices
        period: RSI period (default 9)

    Returns:
        RSI series
    """
    calc = RSICalculator(period=period)
    df = pd.DataFrame({"close": prices})
    return calc.calculate_batch(df)


def calculate_atr_wilder(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10
) -> pd.Series:
    """
    Calculate ATR using Wilder's smoothing - FOR TESTING PARITY.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default 10)

    Returns:
        ATR percentage series
    """
    calc = ATRPercentCalculator(period=period)
    df = pd.DataFrame({"high": high, "low": low, "close": close})
    return calc.calculate_batch(df)


def calculate_adx_wilder(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate ADX using Wilder's smoothing - FOR TESTING PARITY.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period (default 14)

    Returns:
        ADX series
    """
    calc = ADXCalculator(period=period)
    df = pd.DataFrame({"high": high, "low": low, "close": close})
    return calc.calculate_batch(df)
