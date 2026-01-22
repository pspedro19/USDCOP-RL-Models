"""
Feature Store Adapters
=======================
Adapters for integrating the unified Feature Store with existing pipelines.

These adapters provide backward-compatible interfaces while using
the unified feature calculators internally.

SOLID Principles:
- Single Responsibility: Each adapter targets one pipeline
- Open/Closed: New adapters via inheritance
- Liskov Substitution: Drop-in replacement for existing builders
- Dependency Inversion: Uses FeatureStore abstractions

Design Pattern: Adapter Pattern

Author: Trading Team
Version: 1.0.0
Created: 2025-01-12
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .contracts import (
    FeatureVersion,
    NormalizationStats,
    NormalizationParams,
    NormalizationMethod,
)
from .registry import FeatureRegistry, get_registry

# Import FEATURE_ORDER from SSOT
try:
    from src.core.contracts import FEATURE_ORDER as SSOT_FEATURE_ORDER, OBSERVATION_DIM
except ImportError:
    SSOT_FEATURE_ORDER = None
    OBSERVATION_DIM = 15
from .builder import FeatureBuilder
from .calculators import (
    RSICalculator,
    ADXCalculator,
    ATRPercentCalculator,
    CalculatorRegistry,
)

logger = logging.getLogger(__name__)


class NormStatsNotFoundError(RuntimeError):
    """Raised when normalization statistics file is not found."""
    pass


class InferenceObservationAdapter:
    """
    Adapter for inference pipeline observation building.

    Provides the same interface as the existing ObservationBuilder
    but uses the unified Feature Store internally.

    This ensures feature parity between training and inference.

    Usage:
        # Drop-in replacement for ObservationBuilder
        adapter = InferenceObservationAdapter(norm_stats_path)
        obs = adapter.build_observation(df, bar_idx, position, time_normalized)
    """

    # Use SSOT FEATURE_ORDER from src.core.contracts (imported at module level)
    # Fallback only if SSOT import failed
    FEATURE_ORDER = SSOT_FEATURE_ORDER if SSOT_FEATURE_ORDER is not None else (
        "log_ret_5m", "log_ret_1h", "log_ret_4h",
        "rsi_9", "atr_pct", "adx_14",
        "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
        "brent_change_1d", "rate_spread", "usdmxn_change_1d",
        "position", "time_normalized",
    )

    def __init__(self, norm_stats_path: Optional[Path] = None):
        """
        Initialize adapter with normalization stats.

        Args:
            norm_stats_path: Path to normalization stats JSON

        Raises:
            NormStatsNotFoundError: If stats file not found
        """
        self.registry = get_registry()
        self.version = FeatureVersion.CURRENT
        self.norm_stats = self._load_norm_stats(norm_stats_path)
        self.clip_range = (-5.0, 5.0)

        # Initialize unified calculators with specs
        self._init_calculators()

        logger.info(f"InferenceObservationAdapter initialized with unified Feature Store")

    def _load_norm_stats(self, path: Optional[Path]) -> Dict[str, Dict[str, float]]:
        """Load normalization stats from JSON file"""
        if path is None:
            raise NormStatsNotFoundError("No norm_stats_path provided")

        if not path.exists():
            raise NormStatsNotFoundError(
                f"CRITICAL: Normalization statistics file NOT FOUND at {path}. "
                f"The model CANNOT produce correct predictions without exact norm_stats."
            )

        with open(path) as f:
            stats = json.load(f)

        logger.info(f"Loaded norm_stats from {path}: {len(stats)} features")
        return stats

    def _init_calculators(self) -> None:
        """Initialize feature calculators from registry"""
        feature_set = self.registry.get_feature_set(self.version)

        # Get RSI calculator spec
        rsi_spec = feature_set.get_feature("rsi_9")
        if rsi_spec:
            self._rsi_calc = RSICalculator(rsi_spec)

        # Get ADX calculator spec
        adx_spec = feature_set.get_feature("adx_14")
        if adx_spec:
            self._adx_calc = ADXCalculator(adx_spec)

        # Get ATR calculator spec
        atr_spec = feature_set.get_feature("atr_pct")
        if atr_spec:
            self._atr_calc = ATRPercentCalculator(atr_spec)

    def normalize_feature(self, value: float, feature_name: str) -> float:
        """Z-score normalize a single feature using loaded stats"""
        if feature_name not in self.norm_stats:
            return value

        stats = self.norm_stats[feature_name]
        mean = stats.get("mean", 0.0)
        std = stats.get("std", 1.0)

        if std == 0:
            return 0.0

        z_score = (value - mean) / std
        return np.clip(z_score, self.clip_range[0], self.clip_range[1])

    def calculate_technical_features(
        self,
        df: pd.DataFrame,
        bar_idx: int,
        lookback: int = 50
    ) -> Dict[str, float]:
        """
        Calculate technical indicators using unified calculators.

        Uses the same Wilder's EMA smoothing as training.
        """
        start_idx = max(0, bar_idx - lookback)
        window = df.iloc[start_idx:bar_idx + 1].copy()

        if len(window) < 2:
            return {
                "log_ret_5m": 0.0,
                "log_ret_1h": 0.0,
                "log_ret_4h": 0.0,
                "rsi_9": 50.0,
                "atr_pct": 0.0,
                "adx_14": 25.0,
            }

        close = window["close"].values

        # Log returns
        log_ret_5m = np.log(close[-1] / close[-2]) if len(close) >= 2 else 0.0
        log_ret_1h = np.log(close[-1] / close[-12]) if len(close) >= 12 else log_ret_5m
        log_ret_4h = np.log(close[-1] / close[-48]) if len(close) >= 48 else log_ret_1h

        # RSI using unified calculator (Wilder's EMA)
        rsi_series = self._rsi_calc.calculate(window)
        rsi_9 = float(rsi_series.iloc[-1]) * 100  # Convert from 0-1 to 0-100

        # ATR percentage using unified calculator (Wilder's EMA)
        atr_series = self._atr_calc.calculate(window)
        atr_pct = float(atr_series.iloc[-1])

        # ADX using unified calculator (Wilder's EMA)
        adx_series = self._adx_calc.calculate(window)
        adx_14 = float(adx_series.iloc[-1]) * 100  # Convert from 0-1 to 0-100

        return {
            "log_ret_5m": log_ret_5m,
            "log_ret_1h": log_ret_1h,
            "log_ret_4h": log_ret_4h,
            "rsi_9": rsi_9,
            "atr_pct": atr_pct,
            "adx_14": adx_14,
        }

    def calculate_macro_features(
        self,
        row: pd.Series,
        prev_row: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Calculate macro indicator features"""
        # Get raw values with defaults
        dxy = row.get("dxy", 100.0) or 100.0
        vix = row.get("vix", 20.0) or 20.0
        embi = row.get("embi", 300.0) or 300.0
        brent = row.get("brent", 80.0) or 80.0
        treasury_10y = row.get("treasury_10y", 4.0) or 4.0
        usdmxn = row.get("usdmxn", 17.0) or 17.0

        # Calculate daily changes
        if prev_row is not None:
            prev_dxy = prev_row.get("dxy", dxy) or dxy
            prev_brent = prev_row.get("brent", brent) or brent
            prev_usdmxn = prev_row.get("usdmxn", usdmxn) or usdmxn

            dxy_change = (dxy - prev_dxy) / prev_dxy if prev_dxy > 0 else 0.0
            brent_change = (brent - prev_brent) / prev_brent if prev_brent > 0 else 0.0
            usdmxn_change = (usdmxn - prev_usdmxn) / prev_usdmxn if prev_usdmxn > 0 else 0.0
        else:
            dxy_change = 0.0
            brent_change = 0.0
            usdmxn_change = 0.0

        rate_spread = 10.0 - treasury_10y

        return {
            "dxy_z": dxy,
            "dxy_change_1d": dxy_change,
            "vix_z": vix,
            "embi_z": embi,
            "brent_change_1d": brent_change,
            "rate_spread": rate_spread,
            "usdmxn_change_1d": usdmxn_change,
        }

    def build_observation(
        self,
        df: pd.DataFrame,
        bar_idx: int,
        position: float,
        time_normalized: float = 0.5
    ) -> np.ndarray:
        """
        Build complete 15-dimensional observation vector.

        Uses unified Feature Store calculators to ensure
        consistency with training.

        Args:
            df: DataFrame with OHLCV and macro data
            bar_idx: Current bar index
            position: Current position (-1 to 1)
            time_normalized: Normalized trading session time (0 to 1)
                            Maps to SSOT feature index 14 'time_normalized'

        Returns:
            numpy array of shape (15,)
        """
        row = df.iloc[bar_idx]
        prev_row = df.iloc[bar_idx - 1] if bar_idx > 0 else None

        # Calculate features using unified calculators
        tech_features = self.calculate_technical_features(df, bar_idx)
        macro_features = self.calculate_macro_features(row, prev_row)

        # Build observation vector
        obs = np.zeros(15, dtype=np.float32)

        # Normalized features
        obs[0] = self.normalize_feature(tech_features["log_ret_5m"], "log_ret_5m")
        obs[1] = self.normalize_feature(tech_features["log_ret_1h"], "log_ret_1h")
        obs[2] = self.normalize_feature(tech_features["log_ret_4h"], "log_ret_4h")
        obs[3] = self.normalize_feature(tech_features["rsi_9"], "rsi_9")
        obs[4] = self.normalize_feature(tech_features["atr_pct"], "atr_pct")
        obs[5] = self.normalize_feature(tech_features["adx_14"], "adx_14")
        obs[6] = self.normalize_feature(macro_features["dxy_z"], "dxy_z")
        obs[7] = self.normalize_feature(macro_features["dxy_change_1d"], "dxy_change_1d")
        obs[8] = self.normalize_feature(macro_features["vix_z"], "vix_z")
        obs[9] = self.normalize_feature(macro_features["embi_z"], "embi_z")
        obs[10] = self.normalize_feature(macro_features["brent_change_1d"], "brent_change_1d")
        obs[11] = self.normalize_feature(macro_features["rate_spread"], "rate_spread")
        obs[12] = self.normalize_feature(macro_features["usdmxn_change_1d"], "usdmxn_change_1d")

        # State features (not normalized)
        # Index 13: position (-1 to 1)
        # Index 14: time_normalized (0 to 1) - SSOT feature name
        obs[13] = np.clip(position, -1.0, 1.0)
        obs[14] = np.clip(time_normalized, 0.0, 1.0)

        # NaN check
        obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)

        return obs


class TrainingFeatureAdapter:
    """
    Adapter for training pipeline feature building.

    Ensures training uses the same calculators as inference.
    """

    def __init__(self, version: FeatureVersion = FeatureVersion.CURRENT):
        """Initialize training adapter"""
        self.version = version
        self.builder = FeatureBuilder(version)

    def build_features(
        self,
        df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Build features for training.

        Args:
            df: OHLCV DataFrame
            macro_df: Optional macro data DataFrame
            normalize: Whether to normalize features

        Returns:
            DataFrame with calculated features
        """
        batch = self.builder.build(df, macro_df, normalize)
        return batch.to_dataframe()

    def compute_normalization_stats(
        self,
        df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None
    ) -> NormalizationStats:
        """
        Compute normalization statistics from training data.

        Args:
            df: OHLCV DataFrame
            macro_df: Optional macro data

        Returns:
            NormalizationStats to save for inference
        """
        # Build raw features without normalization
        batch = self.builder.build(df, macro_df, normalize=False)
        feature_df = batch.to_dataframe()

        # Compute stats for each feature
        stats = {}
        for col in feature_df.columns:
            mean_val = float(feature_df[col].mean())
            std_val = float(feature_df[col].std())
            if std_val == 0:
                std_val = 1.0

            stats[col] = NormalizationParams(
                method=NormalizationMethod.ZSCORE,
                mean=mean_val,
                std=std_val,
                clip_range=(-3.0, 3.0),
            )

        return NormalizationStats(
            version=self.version,
            stats=stats,
            sample_size=len(feature_df),
        )


class BacktestFeatureAdapter:
    """
    Adapter for backtest pipeline feature building.

    Uses the same calculators as inference for consistency.
    """

    def __init__(
        self,
        version: FeatureVersion = FeatureVersion.CURRENT,
        norm_stats: Optional[NormalizationStats] = None
    ):
        """Initialize backtest adapter"""
        self.version = version
        self.builder = FeatureBuilder(version)

        if norm_stats:
            self.builder.with_normalization_stats(norm_stats)

    def build_features(
        self,
        df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Build features for backtest.

        Uses same normalization as inference.
        """
        batch = self.builder.build(df, macro_df, normalize=True)
        return batch.to_dataframe()


# =============================================================================
# FACTORY FOR ADAPTERS
# =============================================================================

class AdapterFactory:
    """Factory for creating pipeline adapters"""

    @staticmethod
    def create_inference_adapter(
        norm_stats_path: Path
    ) -> InferenceObservationAdapter:
        """Create adapter for inference pipeline"""
        return InferenceObservationAdapter(norm_stats_path)

    @staticmethod
    def create_training_adapter(
        version: FeatureVersion = FeatureVersion.CURRENT
    ) -> TrainingFeatureAdapter:
        """Create adapter for training pipeline"""
        return TrainingFeatureAdapter(version)

    @staticmethod
    def create_backtest_adapter(
        version: FeatureVersion = FeatureVersion.CURRENT,
        norm_stats: Optional[NormalizationStats] = None
    ) -> BacktestFeatureAdapter:
        """Create adapter for backtest pipeline"""
        return BacktestFeatureAdapter(version, norm_stats)
