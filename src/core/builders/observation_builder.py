"""
ObservationBuilder - Observation Builder for USD/COP Trading System
====================================================================

Implements the observation vector construction for the trading model with:
- 13 core market features
- 2 state features (position, time_normalized)
- Total 15 dimensions

Author: Pedro @ Lean Tech Solutions / Claude Code
Version: 1.0.0
Date: 2025-01-07

Configuration:
    - Feature config: config/feature_config.json
    - Normalization stats: config/norm_stats.json
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union, List, Any

# Default paths relative to project root
DEFAULT_CONFIG_PATH = "config/feature_config.json"
DEFAULT_STATS_PATH = "config/norm_stats.json"


class ObservationBuilder:
    """
    Observation builder for USDCOP trading model.

    Constructs 15-dimensional observation vectors with proper normalization
    and clipping for stable RL training and inference.

    Attributes:
        FEATURE_ORDER: Canonical order of all 15 features (13 core + 2 state)
        CORE_FEATURES: The 13 market-derived features
        STATE_FEATURES: The 2 environment state features
        OBS_DIM: Total observation dimension (15)
        CLIP_MIN: Global minimum clip value (-5.0)
        CLIP_MAX: Global maximum clip value (5.0)

    Example:
        >>> builder = ObservationBuilder()
        >>> market_features = {
        ...     "log_ret_5m": 0.001, "log_ret_1h": 0.002, "log_ret_4h": -0.001,
        ...     "rsi_9": 55.0, "atr_pct": 0.05, "adx_14": 25.0,
        ...     "dxy_z": 0.5, "dxy_change_1d": 0.002, "vix_z": -0.3, "embi_z": 0.1,
        ...     "brent_change_1d": 0.01, "rate_spread": 0.2, "usdmxn_change_1d": 0.005
        ... }
        >>> obs = builder.build(market_features, position=0.0, time_normalized=0.5)
        >>> obs.shape
        (15,)
    """

    # Canonical feature order - MUST match training order exactly
    FEATURE_ORDER: List[str] = [
        "log_ret_5m",
        "log_ret_1h",
        "log_ret_4h",
        "rsi_9",
        "atr_pct",
        "adx_14",
        "dxy_z",
        "dxy_change_1d",
        "vix_z",
        "embi_z",
        "brent_change_1d",
        "rate_spread",
        "usdmxn_change_1d",
        "position",
        "time_normalized"
    ]

    CORE_FEATURES: List[str] = FEATURE_ORDER[:13]
    STATE_FEATURES: List[str] = ["position", "time_normalized"]
    OBS_DIM: int = 15
    CLIP_MIN: float = -5.0
    CLIP_MAX: float = 5.0

    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG_PATH,
        stats_path: Optional[str] = None,
        base_path: Optional[str] = None
    ):
        """
        Initialize the ObservationBuilder.

        Args:
            config_path: Path to feature config JSON file (relative or absolute)
            stats_path: Optional path to norm stats JSON (defaults to config value)
            base_path: Optional base path for config resolution

        Raises:
            FileNotFoundError: If config file is not found
            ValueError: If config is invalid
        """
        self._base_path = Path(base_path) if base_path else self._find_project_root()
        self._config_path = self._resolve_path(config_path)
        self._stats_path = self._resolve_path(stats_path or DEFAULT_STATS_PATH)

        # Load configurations
        self._config = self._load_config()
        self._norm_stats = self._load_norm_stats()

        # Validate configuration
        self._validate_config()

    def _find_project_root(self) -> Path:
        """
        Find the project root directory.

        Returns:
            Path to project root
        """
        current = Path(__file__).resolve()
        for parent in [current] + list(current.parents):
            if (parent / "config" / "feature_config.json").exists():
                return parent
            if (parent / "config").exists():
                return parent
        # Fallback to current working directory
        return Path.cwd()

    def _resolve_path(self, path: str) -> Path:
        """
        Resolve a path relative to base path or as absolute.

        Args:
            path: Path string to resolve

        Returns:
            Resolved Path object
        """
        p = Path(path)
        if p.is_absolute():
            return p
        return self._base_path / path

    def _load_config(self) -> Dict[str, Any]:
        """
        Load the feature configuration file.

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file not found
        """
        if not self._config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self._config_path}. "
                f"Searched from base: {self._base_path}"
            )

        with open(self._config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_norm_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Load normalization statistics.

        Returns:
            Dictionary mapping feature names to their mean/std stats

        Raises:
            FileNotFoundError: If stats file not found
        """
        if not self._stats_path.exists():
            raise FileNotFoundError(
                f"Norm stats file not found: {self._stats_path}. "
                f"Searched from base: {self._base_path}"
            )

        with open(self._stats_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _validate_config(self) -> None:
        """
        Validate that configuration matches expected format.

        Raises:
            ValueError: If config is invalid
        """
        obs_space = self._config.get("observation_space", {})

        # Check dimension
        if obs_space.get("dimension") != self.OBS_DIM:
            raise ValueError(
                f"Config dimension {obs_space.get('dimension')} != expected {self.OBS_DIM}"
            )

        # Check feature order matches
        config_order = obs_space.get("order", [])
        if config_order != self.CORE_FEATURES:
            raise ValueError(
                f"Config feature order mismatch. Expected: {self.CORE_FEATURES}, "
                f"Got: {config_order}"
            )

        # Check all core features have norm stats
        for feature in self.CORE_FEATURES:
            if feature not in self._norm_stats:
                raise ValueError(f"Missing norm stats for feature: {feature}")

    def build(
        self,
        market_features: Dict[str, float],
        position: float,
        time_normalized: float
    ) -> np.ndarray:
        """
        Build the observation vector from market features and state.

        Args:
            market_features: Dictionary with 13 core feature values
            position: Current position (-1 to 1, where -1=short, 0=flat, 1=long)
            time_normalized: Normalized time in episode (0 to 1)

        Returns:
            np.ndarray of shape (15,) with dtype float32, clipped to [-5, 5]

        Raises:
            ValueError: If required features are missing

        Note:
            - NaN/None values are replaced with 0.0
            - All values are clipped to [-5, 5] range
            - Features are normalized using z-score from norm_stats.json
        """
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        # Build core features (indices 0-12)
        for i, feature_name in enumerate(self.CORE_FEATURES):
            raw_value = market_features.get(feature_name)

            # Handle missing/NaN values
            if raw_value is None or (isinstance(raw_value, float) and math.isnan(raw_value)):
                obs[i] = 0.0
            else:
                # Apply normalization
                obs[i] = self.normalize(feature_name, float(raw_value))

        # Add state features (indices 13-14)
        obs[13] = self._safe_float(position)
        obs[14] = self._safe_float(time_normalized)

        # Apply global clipping
        obs = np.clip(obs, self.CLIP_MIN, self.CLIP_MAX)

        # Final NaN check
        obs = np.nan_to_num(obs, nan=0.0, posinf=self.CLIP_MAX, neginf=self.CLIP_MIN)

        return obs

    def normalize(self, feature_name: str, value: float) -> float:
        """
        Normalize a single feature value using z-score normalization.

        Args:
            feature_name: Name of the feature
            value: Raw feature value

        Returns:
            Normalized value (z-score), clipped to [-5, 5]

        Note:
            - If feature not in stats, returns value unchanged
            - NaN values return 0.0
            - Division by zero is protected (std < 1e-8 uses 1.0)
        """
        if math.isnan(value) if isinstance(value, float) else False:
            return 0.0

        if feature_name not in self._norm_stats:
            # Return raw value if no stats available
            return np.clip(value, self.CLIP_MIN, self.CLIP_MAX)

        stats = self._norm_stats[feature_name]
        mean = stats.get("mean", 0.0)
        std = stats.get("std", 1.0)

        # Protect against division by zero
        if std < 1e-8:
            std = 1.0

        z_value = (value - mean) / std
        return np.clip(z_value, self.CLIP_MIN, self.CLIP_MAX)

    def denormalize(self, feature_name: str, z_value: float) -> float:
        """
        Denormalize a z-score back to original scale.

        Args:
            feature_name: Name of the feature
            z_value: Normalized (z-score) value

        Returns:
            Original-scale value
        """
        if feature_name not in self._norm_stats:
            return z_value

        stats = self._norm_stats[feature_name]
        mean = stats.get("mean", 0.0)
        std = stats.get("std", 1.0)

        return (z_value * std) + mean

    def _safe_float(self, value: Any) -> float:
        """
        Safely convert value to float, handling None/NaN.

        Args:
            value: Input value

        Returns:
            Float value, 0.0 if None/NaN
        """
        if value is None:
            return 0.0
        try:
            f = float(value)
            if math.isnan(f):
                return 0.0
            return f
        except (TypeError, ValueError):
            return 0.0

    def get_feature_stats(self, feature_name: str) -> Dict[str, float]:
        """
        Get normalization statistics for a feature.

        Args:
            feature_name: Name of the feature

        Returns:
            Dictionary with 'mean', 'std', and other stats
        """
        return self._norm_stats.get(feature_name, {"mean": 0.0, "std": 1.0})

    def get_config(self) -> Dict[str, Any]:
        """
        Get the loaded configuration.

        Returns:
            Full configuration dictionary
        """
        return self._config.copy()

    def validate_features(self, market_features: Dict[str, float]) -> List[str]:
        """
        Validate that all required features are present.

        Args:
            market_features: Dictionary of feature values

        Returns:
            List of missing feature names (empty if all present)
        """
        return [f for f in self.CORE_FEATURES if f not in market_features]

    def __repr__(self) -> str:
        return (
            f"ObservationBuilder(obs_dim={self.OBS_DIM}, "
            f"core_features={len(self.CORE_FEATURES)}, "
            f"clip=[{self.CLIP_MIN}, {self.CLIP_MAX}])"
        )


# Convenience function for quick instantiation
def create_observation_builder(
    config_path: str = DEFAULT_CONFIG_PATH,
    base_path: Optional[str] = None
) -> ObservationBuilder:
    """
    Factory function to create an ObservationBuilder instance.

    Args:
        config_path: Path to feature config
        base_path: Optional base path for config resolution

    Returns:
        Configured ObservationBuilder instance
    """
    return ObservationBuilder(config_path=config_path, base_path=base_path)
