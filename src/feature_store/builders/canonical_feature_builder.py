"""
CanonicalFeatureBuilder - SINGLE SOURCE OF TRUTH
=================================================
All contexts (Training, Inference, Backtest) MUST use this class for feature building.

This module implements the P0-4 remediation plan item: creating a unified feature
builder that guarantees feature parity across all execution contexts.

Design Principles:
- Single Source of Truth: ONE implementation used everywhere
- Immutable Contracts: Feature order, dimensions are frozen
- Deterministic: Same input always produces same output
- Auditable: Export snapshots for reproducibility verification
- Fail-Fast: Validation errors raised immediately

Architecture:
    Training Pipeline ─────┐
                           │
    Inference API ─────────┼───▶ CanonicalFeatureBuilder ───▶ 15-dim Observation
                           │         (SSOT)
    Backtest Engine ───────┘

CRITICAL INVARIANTS:
- Observation shape is ALWAYS (15,)
- Observations NEVER contain NaN or Inf
- Normalized features are in [-5.0, 5.0]
- Hash of norm_stats must match training hash

Author: Trading Team
Version: 1.0.0
Created: 2025-01-16
Contract: CTR-CANONICAL-001
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Protocol, Tuple, Union

import numpy as np
import pandas as pd

# Import from parent feature_store for SSOT calculators
from ..core import (
    FEATURE_CONTRACT,
    FEATURE_ORDER,
    OBSERVATION_DIM,
    NORM_STATS_PATH,
    FeatureContract,
    SmoothingMethod,
    BaseCalculator,
    LogReturnCalculator,
    RSICalculator,
    ATRPercentCalculator,
    ADXCalculator,
    MacroZScoreCalculator,
    MacroChangeCalculator,
    CalculatorRegistry,
)

logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS - Fail Fast Pattern
# =============================================================================

class NormStatsNotFoundError(RuntimeError):
    """
    Raised when normalization statistics file is not found.

    This is a CRITICAL error - the model CANNOT produce correct
    predictions without the exact norm_stats used during training.
    """

    def __init__(self, path: str):
        self.path = path
        super().__init__(
            f"CRITICAL: Normalization statistics file NOT FOUND at {path}. "
            f"Model cannot produce correct predictions without exact training stats."
        )


class ObservationDimensionError(ValueError):
    """Raised when observation dimension doesn't match expected."""

    def __init__(self, expected: int, actual: int):
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Observation dimension mismatch: expected {expected}, got {actual}"
        )


class FeatureCalculationError(RuntimeError):
    """Raised when feature calculation fails."""

    def __init__(self, feature_name: str, reason: str):
        self.feature_name = feature_name
        self.reason = reason
        super().__init__(f"Feature calculation failed for '{feature_name}': {reason}")


class NormStatsHashMismatchError(ValueError):
    """Raised when norm_stats hash doesn't match expected (training vs inference)."""

    def __init__(self, expected_hash: str, actual_hash: str):
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        super().__init__(
            f"Norm stats hash mismatch: expected {expected_hash[:12]}..., "
            f"got {actual_hash[:12]}... This may cause feature drift."
        )


# =============================================================================
# CONTEXT ENUM - Which pipeline is using the builder
# =============================================================================

class BuilderContext(str, Enum):
    """Context in which the builder is operating."""
    TRAINING = "training"
    INFERENCE = "inference"
    BACKTEST = "backtest"


# =============================================================================
# FEATURE BUILDER PROTOCOL - Interface Definition
# =============================================================================

class IFeatureBuilder(Protocol):
    """
    Protocol (Interface) that ALL feature builders must implement.

    This enables dependency injection and testing with mocks.
    """

    def build_observation(
        self,
        ohlcv: pd.DataFrame,
        macro: Optional[pd.DataFrame],
        position: float,
        bar_idx: int,
    ) -> np.ndarray:
        """Build normalized observation vector."""
        ...

    def get_feature_order(self) -> List[str]:
        """Get ordered list of feature names."""
        ...

    def get_observation_dim(self) -> int:
        """Get observation dimension (15)."""
        ...

    def get_norm_stats_hash(self) -> str:
        """Get hash of normalization statistics for validation."""
        ...


# =============================================================================
# CANONICAL FEATURE BUILDER - SINGLE SOURCE OF TRUTH
# =============================================================================

@dataclass(frozen=True)
class FeatureSnapshot:
    """Immutable snapshot of feature calculation for audit."""
    timestamp: str
    bar_idx: int
    context: str
    raw_features: Dict[str, float]
    normalized_features: Dict[str, float]
    observation: List[float]
    norm_stats_hash: str
    calculation_time_ms: float


class CanonicalFeatureBuilder:
    """
    CANONICAL Feature Builder - Single Source of Truth for ALL contexts.

    This class is the ONLY authorized implementation for building
    observation vectors. All other builders should delegate to this class.

    Usage:
        # For training context
        builder = CanonicalFeatureBuilder.for_training(config)

        # For inference context
        builder = CanonicalFeatureBuilder.for_inference(model_contract)

        # For backtest context
        builder = CanonicalFeatureBuilder.for_backtest(config)

        # Build observation
        obs = builder.build_observation(ohlcv_df, macro_df, position=0.0, bar_idx=100)
        assert obs.shape == (15,)
        assert not np.isnan(obs).any()

    Invariants:
        - observation shape is ALWAYS (15,)
        - observation NEVER contains NaN or Inf
        - features are normalized in [-5.0, 5.0]
        - same input produces EXACT same output (deterministic)
    """

    # Class-level constants from SSOT
    FEATURE_ORDER: Final[Tuple[str, ...]] = FEATURE_ORDER
    OBSERVATION_DIM: Final[int] = OBSERVATION_DIM
    CLIP_RANGE: Final[Tuple[float, float]] = FEATURE_CONTRACT.clip_range
    VERSION: Final[str] = "1.0.0"  # Builder version for L1 DAG compatibility

    def __init__(
        self,
        norm_stats: Dict[str, Dict[str, float]],
        context: BuilderContext = BuilderContext.TRAINING,
        expected_hash: Optional[str] = None,
    ):
        """
        Initialize CanonicalFeatureBuilder.

        Args:
            norm_stats: Normalization statistics (mean, std per feature)
            context: Builder context (training, inference, backtest)
            expected_hash: Optional hash to validate norm_stats integrity

        Raises:
            ValueError: If norm_stats is empty or invalid
            NormStatsHashMismatchError: If hash doesn't match expected
        """
        if not norm_stats:
            raise ValueError("norm_stats cannot be empty")

        self._norm_stats = norm_stats
        self._context = context
        self._norm_stats_hash = self._compute_hash(norm_stats)

        # Validate hash if provided
        if expected_hash and self._norm_stats_hash != expected_hash:
            raise NormStatsHashMismatchError(expected_hash, self._norm_stats_hash)

        # Initialize SSOT calculators
        self._calculators = self._init_calculators()

        # Validate norm_stats has required features
        self._validate()

        logger.info(
            f"CanonicalFeatureBuilder initialized: context={context.value}, "
            f"features={len(self._norm_stats)}, hash={self._norm_stats_hash[:12]}..."
        )

    @classmethod
    def for_training(
        cls,
        config: Optional[Dict[str, Any]] = None,
        norm_stats_path: Optional[str] = None,
    ) -> "CanonicalFeatureBuilder":
        """
        Factory method for training context.

        Args:
            config: Training configuration dict (optional)
            norm_stats_path: Path to norm_stats JSON (uses default if None)

        Returns:
            CanonicalFeatureBuilder configured for training

        Raises:
            NormStatsNotFoundError: If norm_stats file not found
        """
        path = norm_stats_path or NORM_STATS_PATH
        norm_stats = cls._load_norm_stats(path)
        return cls(norm_stats, context=BuilderContext.TRAINING)

    @classmethod
    def for_inference(
        cls,
        model_contract: Optional[Dict[str, Any]] = None,
        norm_stats_path: Optional[str] = None,
        expected_hash: Optional[str] = None,
    ) -> "CanonicalFeatureBuilder":
        """
        Factory method for inference context.

        Args:
            model_contract: Model contract containing norm_stats path
            norm_stats_path: Direct path to norm_stats JSON
            expected_hash: Expected hash for validation (from training)

        Returns:
            CanonicalFeatureBuilder configured for inference

        Raises:
            NormStatsNotFoundError: If norm_stats file not found
            NormStatsHashMismatchError: If hash doesn't match expected
        """
        if model_contract and "norm_stats_path" in model_contract:
            path = model_contract["norm_stats_path"]
        else:
            path = norm_stats_path or NORM_STATS_PATH

        norm_stats = cls._load_norm_stats(path)
        return cls(
            norm_stats,
            context=BuilderContext.INFERENCE,
            expected_hash=expected_hash,
        )

    @classmethod
    def for_backtest(
        cls,
        backtest_config: Optional[Dict[str, Any]] = None,
        norm_stats_path: Optional[str] = None,
        expected_hash: Optional[str] = None,
    ) -> "CanonicalFeatureBuilder":
        """
        Factory method for backtest context.

        Args:
            backtest_config: Backtest configuration dict
            norm_stats_path: Path to norm_stats JSON
            expected_hash: Expected hash for validation (from training)

        Returns:
            CanonicalFeatureBuilder configured for backtest

        Raises:
            NormStatsNotFoundError: If norm_stats file not found
            NormStatsHashMismatchError: If hash doesn't match expected
        """
        if backtest_config and "norm_stats_path" in backtest_config:
            path = backtest_config["norm_stats_path"]
        else:
            path = norm_stats_path or NORM_STATS_PATH

        norm_stats = cls._load_norm_stats(path)
        return cls(
            norm_stats,
            context=BuilderContext.BACKTEST,
            expected_hash=expected_hash,
        )

    @staticmethod
    def _load_norm_stats(path: str) -> Dict[str, Dict[str, float]]:
        """
        Load normalization statistics from JSON file.

        Args:
            path: Path to norm_stats JSON file

        Returns:
            Dict of feature_name -> {mean, std, ...}

        Raises:
            NormStatsNotFoundError: If file not found (FAIL FAST)
        """
        p = Path(path)

        # Try relative to project root if not absolute
        if not p.is_absolute():
            project_root = Path(__file__).parent.parent.parent.parent
            p = project_root / path

        if not p.exists():
            raise NormStatsNotFoundError(str(p))

        with open(p, "r", encoding="utf-8") as f:
            stats = json.load(f)

        logger.debug(f"Loaded norm_stats from {p}: {len(stats)} features")
        return stats

    @staticmethod
    def _compute_hash(norm_stats: Dict[str, Dict[str, float]]) -> str:
        """
        Compute deterministic hash of norm_stats.

        The hash is used to validate that inference uses the same
        normalization statistics as training.

        Args:
            norm_stats: Normalization statistics dict

        Returns:
            SHA-256 hash (hex string)
        """
        # Sort for determinism
        sorted_stats = {
            k: {sk: sv for sk, sv in sorted(v.items())}
            for k, v in sorted(norm_stats.items())
        }
        json_str = json.dumps(sorted_stats, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _init_calculators(self) -> Dict[str, BaseCalculator]:
        """
        Initialize SSOT calculators for feature calculation.

        Uses the same calculators as core.py to ensure parity.

        Returns:
            Dict of feature_name -> Calculator
        """
        periods = FEATURE_CONTRACT.get_technical_periods()

        calculators = {
            # Log returns
            "log_ret_5m": LogReturnCalculator("log_ret_5m", periods=1),
            "log_ret_1h": LogReturnCalculator("log_ret_1h", periods=12),
            "log_ret_4h": LogReturnCalculator("log_ret_4h", periods=48),

            # Technical indicators (Wilder's EMA)
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

        return calculators

    def _validate(self) -> None:
        """
        Validate that norm_stats contains all required features.

        Raises:
            ValueError: If required features are missing
        """
        # Features that require normalization (first 13)
        required_features = set(self.FEATURE_ORDER[:13])
        available_features = set(self._norm_stats.keys())
        missing = required_features - available_features

        if missing:
            raise ValueError(
                f"norm_stats missing required features: {missing}. "
                f"Expected: {required_features}"
            )

    # =========================================================================
    # PUBLIC API - IFeatureBuilder Implementation
    # =========================================================================

    def build_observation(
        self,
        ohlcv: pd.DataFrame,
        macro: Optional[pd.DataFrame],
        position: float,
        bar_idx: int,
        time_normalized: Optional[float] = None,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> np.ndarray:
        """
        Build complete 15-dimensional observation vector.

        CRITICAL: This is the SSOT implementation. All pipelines MUST use this.

        Args:
            ohlcv: DataFrame with OHLCV columns (open, high, low, close, volume)
            macro: DataFrame with macro columns (dxy, vix, embi, brent, usdmxn, etc.)
            position: Current position [-1, 1]
            bar_idx: Current bar index (for warmup validation)
            time_normalized: Normalized trading session time [0, 1] (optional)
                            Maps to SSOT feature index 14 'time_normalized'
            timestamp: Current timestamp for time normalization (optional)

        Returns:
            np.ndarray of shape (15,) with dtype float32

        Raises:
            ValueError: If bar_idx < warmup_bars
            ObservationDimensionError: If output dimension mismatch
        """
        start_time = time.perf_counter()

        # Validate warmup
        if bar_idx < FEATURE_CONTRACT.warmup_bars:
            raise ValueError(
                f"bar_idx ({bar_idx}) < warmup_bars ({FEATURE_CONTRACT.warmup_bars})"
            )

        # Merge data for calculators
        data = self._merge_data(ohlcv, macro)

        # Calculate raw features
        raw_features = self._calculate_raw_features(data, bar_idx)

        # Normalize features
        normalized_features = self._normalize_features(raw_features)

        # Add state features
        normalized_features["position"] = float(np.clip(position, -1.0, 1.0))

        # Time normalization (SSOT feature name: time_normalized, index 14)
        if time_normalized is not None:
            normalized_features["time_normalized"] = float(
                np.clip(time_normalized, 0.0, 1.0)
            )
        elif timestamp is not None:
            normalized_features["time_normalized"] = self._compute_time_normalized(
                timestamp
            )
        else:
            # Default to middle of session
            normalized_features["time_normalized"] = 0.5

        # Assemble observation in contract order
        observation = self._assemble_observation(normalized_features)

        # Final validation
        self._validate_observation(observation)

        calc_time_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(f"Observation built in {calc_time_ms:.2f}ms")

        return observation

    def build_batch(
        self,
        ohlcv: pd.DataFrame,
        macro: Optional[pd.DataFrame] = None,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """
        Build features for a batch of data (training/backtest).

        Args:
            ohlcv: DataFrame with OHLCV columns
            macro: Optional DataFrame with macro columns
            normalize: Whether to apply normalization

        Returns:
            DataFrame with all features calculated
        """
        data = self._merge_data(ohlcv, macro)
        result = data.copy()

        # Calculate batch features using SSOT calculators
        for name, calc in self._calculators.items():
            try:
                result[name] = calc.calculate_batch(data)
            except Exception as e:
                logger.warning(f"Batch calculation failed for {name}: {e}")
                result[name] = 0.0

        # Handle rate_spread special case
        if "treasury_10y" in data.columns:
            result["rate_spread"] = 10.0 - data["treasury_10y"]
        elif "rate_spread" not in result.columns:
            result["rate_spread"] = 0.0

        # Apply normalization
        if normalize:
            for name in self.FEATURE_ORDER[:13]:
                if name in result.columns and name in self._norm_stats:
                    stats = self._norm_stats[name]
                    mean = stats.get("mean", 0.0)
                    std = stats.get("std", 1.0)
                    if std > 0:
                        result[name] = (result[name] - mean) / std
                        result[name] = result[name].clip(
                            self.CLIP_RANGE[0], self.CLIP_RANGE[1]
                        )

        return result

    def compute_features(
        self,
        df: pd.DataFrame,
        include_state: bool = True,
    ) -> pd.DataFrame:
        """
        Compute features for a DataFrame (L1 DAG compatibility method).

        This is a wrapper around build_batch() for backward compatibility
        with L1 DAG expectations.

        Args:
            df: DataFrame with OHLCV and optionally macro columns
            include_state: Whether to include state columns (position, time_normalized)

        Returns:
            DataFrame with all features calculated and normalized
        """
        result = self.build_batch(df, normalize=True)

        if not include_state:
            # Remove state columns if not requested
            state_cols = ["position", "time_normalized"]
            for col in state_cols:
                if col in result.columns:
                    result = result.drop(columns=[col])

        return result

    def get_latest_features_dict(
        self,
        df: pd.DataFrame,
        position: float = 0.0,
        time_normalized: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Get latest features as a dictionary (L1 DAG compatibility method).

        Extracts the most recent row of features from a DataFrame and
        returns them as a dict suitable for inference.

        Args:
            df: DataFrame with OHLCV and optionally macro columns
            position: Current position for state feature
            time_normalized: Normalized time (0-1) or None for auto-compute

        Returns:
            Dict mapping feature names to values
        """
        # Compute batch features
        features_df = self.compute_features(df, include_state=False)

        # Get last row
        if len(features_df) == 0:
            raise ValueError("Cannot get latest features from empty DataFrame")

        latest_idx = features_df.index[-1]
        features_dict = {}

        # Extract features in FEATURE_ORDER
        for name in self.FEATURE_ORDER[:13]:  # First 13 are computed features
            if name in features_df.columns:
                val = features_df.loc[latest_idx, name]
                features_dict[name] = float(val) if not pd.isna(val) else 0.0
            else:
                features_dict[name] = 0.0

        # Add state features
        features_dict["position"] = float(np.clip(position, -1.0, 1.0))
        features_dict["time_normalized"] = float(
            np.clip(time_normalized if time_normalized is not None else 0.5, 0.0, 1.0)
        )

        return features_dict

    def validate_features(
        self,
        features: Dict[str, float],
        strict: bool = True,
    ) -> Tuple[bool, List[str]]:
        """
        Validate a features dictionary (L1 DAG compatibility method).

        Checks that all required features are present and within valid ranges.

        Args:
            features: Dict mapping feature names to values
            strict: If True, also check value ranges

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check required features
        required = set(self.FEATURE_ORDER)
        provided = set(features.keys())
        missing = required - provided

        if missing:
            errors.append(f"Missing features: {sorted(missing)}")

        # Check for NaN/Inf
        for name, value in features.items():
            if name in required:
                if np.isnan(value):
                    errors.append(f"Feature '{name}' is NaN")
                elif np.isinf(value):
                    errors.append(f"Feature '{name}' is Inf")

        # Strict mode: check ranges
        if strict and not errors:
            clip_min, clip_max = self.CLIP_RANGE
            for name, value in features.items():
                if name in self.FEATURE_ORDER[:13]:  # Normalized features
                    if value < clip_min or value > clip_max:
                        errors.append(
                            f"Feature '{name}' = {value:.4f} outside [{clip_min}, {clip_max}]"
                        )

            # State features have different ranges
            if "position" in features:
                if features["position"] < -1.0 or features["position"] > 1.0:
                    errors.append(f"position = {features['position']:.4f} outside [-1, 1]")

            if "time_normalized" in features:
                if features["time_normalized"] < 0.0 or features["time_normalized"] > 1.0:
                    errors.append(
                        f"time_normalized = {features['time_normalized']:.4f} outside [0, 1]"
                    )

        return len(errors) == 0, errors

    def get_feature_order(self) -> List[str]:
        """Get ordered list of feature names (15 total)."""
        return list(self.FEATURE_ORDER)

    def get_observation_dim(self) -> int:
        """Get observation dimension (15)."""
        return self.OBSERVATION_DIM

    def get_norm_stats_hash(self) -> str:
        """Get hash of normalization statistics for validation."""
        return self._norm_stats_hash

    def get_norm_stats(self) -> Dict[str, Dict[str, float]]:
        """Get copy of normalization statistics."""
        return dict(self._norm_stats)

    @property
    def context(self) -> BuilderContext:
        """Get builder context."""
        return self._context

    # =========================================================================
    # AUDIT & REPRODUCIBILITY
    # =========================================================================

    def export_snapshot(
        self,
        ohlcv: pd.DataFrame,
        macro: Optional[pd.DataFrame],
        position: float,
        bar_idx: int,
        time_normalized: Optional[float] = None,
    ) -> FeatureSnapshot:
        """
        Export complete feature snapshot for audit and reproducibility.

        Returns a frozen snapshot containing:
        - Raw feature values before normalization
        - Normalized feature values
        - Final observation vector
        - Norm stats hash for validation
        - Calculation timing

        Args:
            ohlcv: OHLCV DataFrame
            macro: Macro DataFrame
            position: Current position
            bar_idx: Bar index
            time_normalized: Normalized trading session time [0, 1] (optional)
                            Maps to SSOT feature index 14 'time_normalized'

        Returns:
            FeatureSnapshot for audit/logging
        """
        start_time = time.perf_counter()

        data = self._merge_data(ohlcv, macro)
        raw_features = self._calculate_raw_features(data, bar_idx)
        normalized_features = self._normalize_features(raw_features)

        normalized_features["position"] = float(np.clip(position, -1.0, 1.0))
        normalized_features["time_normalized"] = float(
            np.clip(time_normalized if time_normalized is not None else 0.5, 0.0, 1.0)
        )

        observation = self._assemble_observation(normalized_features)
        calc_time_ms = (time.perf_counter() - start_time) * 1000

        return FeatureSnapshot(
            timestamp=datetime.utcnow().isoformat(),
            bar_idx=bar_idx,
            context=self._context.value,
            raw_features={k: float(v) for k, v in raw_features.items()},
            normalized_features={k: float(v) for k, v in normalized_features.items()},
            observation=[float(x) for x in observation],
            norm_stats_hash=self._norm_stats_hash,
            calculation_time_ms=calc_time_ms,
        )

    def to_json_contract(self) -> Dict[str, Any]:
        """
        Export builder configuration as JSON-serializable contract.

        Useful for model registry and deployment manifests.

        Returns:
            Dict with builder configuration
        """
        return {
            "version": "1.0.0",
            "context": self._context.value,
            "observation_dim": self.OBSERVATION_DIM,
            "feature_order": list(self.FEATURE_ORDER),
            "clip_range": list(self.CLIP_RANGE),
            "norm_stats_hash": self._norm_stats_hash,
            "warmup_bars": FEATURE_CONTRACT.warmup_bars,
            "technical_periods": FEATURE_CONTRACT.get_technical_periods(),
            "trading_hours": FEATURE_CONTRACT.get_trading_hours(),
            "created_at": datetime.utcnow().isoformat(),
        }

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _merge_data(
        self, ohlcv: pd.DataFrame, macro: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Merge OHLCV and macro data for calculators."""
        data = ohlcv.copy()

        if macro is not None and len(macro) > 0:
            for col in macro.columns:
                if col not in data.columns:
                    # Forward-fill macro data to match OHLCV index
                    if len(macro) == len(data):
                        data[col] = macro[col].values
                    else:
                        # Simple forward fill for different lengths
                        data[col] = macro[col].reindex(data.index, method="ffill")

        return data

    def _calculate_raw_features(
        self, data: pd.DataFrame, bar_idx: int
    ) -> Dict[str, float]:
        """Calculate raw feature values using SSOT calculators."""
        features = {}

        for name in self.FEATURE_ORDER[:13]:
            calc = self._calculators.get(name)

            if calc:
                try:
                    features[name] = calc.calculate(data, bar_idx)
                except Exception as e:
                    logger.warning(f"Feature {name} calculation failed: {e}")
                    features[name] = calc._get_default_value()

            elif name == "rate_spread":
                # Special case: rate_spread formula
                if "treasury_10y" in data.columns:
                    treasury = data["treasury_10y"].iloc[bar_idx]
                    features[name] = 10.0 - treasury if not pd.isna(treasury) else 0.0
                elif "rate_spread" in data.columns:
                    features[name] = float(data["rate_spread"].iloc[bar_idx])
                else:
                    features[name] = 0.0

            else:
                features[name] = 0.0

        return features

    def _normalize_features(
        self, raw: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply z-score normalization using training stats."""
        normalized = {}
        clip_min, clip_max = self.CLIP_RANGE

        for name, value in raw.items():
            if np.isnan(value) or np.isinf(value):
                normalized[name] = 0.0
                continue

            stats = self._norm_stats.get(name)
            if stats:
                mean = stats.get("mean", 0.0)
                std = stats.get("std", 1.0)

                # Protect against division by zero
                if std < 1e-8:
                    std = 1.0

                z_score = (value - mean) / std
                normalized[name] = float(np.clip(z_score, clip_min, clip_max))
            else:
                # Feature not in norm_stats - just clip
                normalized[name] = float(np.clip(value, clip_min, clip_max))

        return normalized

    def _compute_time_normalized(self, timestamp: pd.Timestamp) -> float:
        """Normalize timestamp to [0, 1] within trading hours."""
        hours = FEATURE_CONTRACT.get_trading_hours()
        start_hour = int(hours["start"].split(":")[0])
        end_hour = int(hours["end"].split(":")[0])

        current_minutes = timestamp.hour * 60 + timestamp.minute
        start_minutes = start_hour * 60
        end_minutes = end_hour * 60

        # Handle overnight
        if end_minutes <= start_minutes:
            end_minutes += 24 * 60

        if current_minutes < start_minutes:
            current_minutes += 24 * 60

        duration = end_minutes - start_minutes
        if duration == 0:
            return 0.5

        normalized = (current_minutes - start_minutes) / duration
        return float(np.clip(normalized, 0.0, 1.0))

    def _assemble_observation(self, features: Dict[str, float]) -> np.ndarray:
        """Assemble observation array in contract order."""
        observation = np.zeros(self.OBSERVATION_DIM, dtype=np.float32)

        for idx, name in enumerate(self.FEATURE_ORDER):
            value = features.get(name, 0.0)

            # Final safety check
            if np.isnan(value) or np.isinf(value):
                value = 0.0

            observation[idx] = value

        return observation

    def _validate_observation(self, obs: np.ndarray) -> None:
        """
        Validate observation meets invariants.

        Raises:
            ObservationDimensionError: If dimension mismatch
            ValueError: If contains NaN/Inf
        """
        if obs.shape != (self.OBSERVATION_DIM,):
            raise ObservationDimensionError(
                self.OBSERVATION_DIM, obs.shape[0]
            )

        if np.isnan(obs).any():
            nan_indices = np.where(np.isnan(obs))[0]
            raise ValueError(
                f"Observation contains NaN at indices {nan_indices.tolist()}"
            )

        if np.isinf(obs).any():
            inf_indices = np.where(np.isinf(obs))[0]
            raise ValueError(
                f"Observation contains Inf at indices {inf_indices.tolist()}"
            )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_default_builder: Optional[CanonicalFeatureBuilder] = None


def get_canonical_builder(
    context: BuilderContext = BuilderContext.TRAINING,
    norm_stats_path: Optional[str] = None,
) -> CanonicalFeatureBuilder:
    """
    Get or create a CanonicalFeatureBuilder instance.

    For singleton behavior in production, this caches the builder.
    For testing, create new instances directly.

    Args:
        context: Builder context
        norm_stats_path: Path to norm_stats (uses default if None)

    Returns:
        CanonicalFeatureBuilder instance
    """
    global _default_builder

    if _default_builder is None:
        if context == BuilderContext.TRAINING:
            _default_builder = CanonicalFeatureBuilder.for_training(
                norm_stats_path=norm_stats_path
            )
        elif context == BuilderContext.INFERENCE:
            _default_builder = CanonicalFeatureBuilder.for_inference(
                norm_stats_path=norm_stats_path
            )
        else:
            _default_builder = CanonicalFeatureBuilder.for_backtest(
                norm_stats_path=norm_stats_path
            )

    return _default_builder


def reset_builder_cache() -> None:
    """Reset the cached builder (useful for testing)."""
    global _default_builder
    _default_builder = None
