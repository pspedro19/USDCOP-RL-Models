"""
ZScoreNormalizer - Z-Score Normalization with Config Loading
=============================================================

Enhanced z-score normalizer that loads statistics from configuration files
and provides normalization/denormalization for the trading system.

Author: Pedro @ Lean Tech Solutions / Claude Code
Version: 1.0.0
Date: 2025-01-07

Configuration:
    - Normalization stats: config/norm_stats.json
"""

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union, Any

# Default path relative to project root
DEFAULT_STATS_PATH = "config/norm_stats.json"


class ZScoreNormalizer:
    """
    Z-Score normalizer that loads statistics from configuration.

    Provides feature-aware normalization using pre-computed statistics
    from the training dataset (84,671 samples, 2020-03 to 2025-12).

    Attributes:
        CLIP_MIN: Default minimum clip value (-5.0)
        CLIP_MAX: Default maximum clip value (5.0)

    Example:
        >>> normalizer = ZScoreNormalizer()
        >>> normalized = normalizer.normalize("rsi_9", 55.0)
        >>> original = normalizer.denormalize("rsi_9", normalized)
        >>> abs(original - 55.0) < 0.01
        True
    """

    CLIP_MIN: float = -5.0
    CLIP_MAX: float = 5.0

    def __init__(
        self,
        stats_path: str = DEFAULT_STATS_PATH,
        base_path: Optional[str] = None,
        clip_values: bool = True
    ):
        """
        Initialize the ZScoreNormalizer.

        Args:
            stats_path: Path to the normalization statistics JSON file
            base_path: Optional base path for config resolution
            clip_values: Whether to clip normalized values to [-5, 5] (default True)

        Raises:
            FileNotFoundError: If stats file is not found
            ValueError: If stats file is invalid
        """
        self._base_path = Path(base_path) if base_path else self._find_project_root()
        self._stats_path = self._resolve_path(stats_path)
        self._clip_values = clip_values

        # Load statistics
        self._stats = self._load_stats()

        # Cache for quick access
        self._stats_cache: Dict[str, tuple] = {}
        self._build_cache()

    def _find_project_root(self) -> Path:
        """
        Find the project root directory.

        Returns:
            Path to project root
        """
        current = Path(__file__).resolve()
        for parent in [current] + list(current.parents):
            if (parent / "config").exists():
                return parent
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

    def _load_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Load normalization statistics from JSON file.

        Returns:
            Dictionary mapping feature names to statistics

        Raises:
            FileNotFoundError: If stats file not found
            ValueError: If file format is invalid
        """
        if not self._stats_path.exists():
            raise FileNotFoundError(
                f"Normalization stats file not found: {self._stats_path}. "
                f"Searched from base: {self._base_path}"
            )

        with open(self._stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)

        # Validate format
        if not isinstance(stats, dict):
            raise ValueError(f"Stats file must be a JSON object, got {type(stats)}")

        for feature_name, feature_stats in stats.items():
            if not isinstance(feature_stats, dict):
                raise ValueError(f"Stats for {feature_name} must be a dictionary")
            if "mean" not in feature_stats or "std" not in feature_stats:
                raise ValueError(
                    f"Stats for {feature_name} must have 'mean' and 'std' keys"
                )

        return stats

    def _build_cache(self) -> None:
        """
        Build lookup cache for fast normalization.
        """
        for feature_name, feature_stats in self._stats.items():
            mean = float(feature_stats.get("mean", 0.0))
            std = float(feature_stats.get("std", 1.0))
            # Protect against zero std
            if std < 1e-8:
                std = 1.0
            self._stats_cache[feature_name] = (mean, std)

    def normalize(
        self,
        feature_name: str,
        value: Union[float, np.ndarray, pd.Series]
    ) -> Union[float, np.ndarray, pd.Series]:
        """
        Normalize a value using z-score normalization.

        Formula: z = (x - mean) / std

        Args:
            feature_name: Name of the feature to normalize
            value: Raw value(s) to normalize (scalar, array, or Series)

        Returns:
            Normalized z-score value(s), optionally clipped to [-5, 5]

        Note:
            - NaN values are replaced with 0.0
            - If feature not found in stats, uses mean=0, std=1
        """
        mean, std = self._get_stats(feature_name)

        if isinstance(value, pd.Series):
            z = (value - mean) / std
            z = z.fillna(0.0)
            if self._clip_values:
                z = z.clip(self.CLIP_MIN, self.CLIP_MAX)
            return z

        elif isinstance(value, np.ndarray):
            z = (value - mean) / std
            z = np.nan_to_num(z, nan=0.0)
            if self._clip_values:
                z = np.clip(z, self.CLIP_MIN, self.CLIP_MAX)
            return z

        else:
            # Scalar
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return 0.0
            z = (float(value) - mean) / std
            if self._clip_values:
                z = max(self.CLIP_MIN, min(self.CLIP_MAX, z))
            return z

    def denormalize(
        self,
        feature_name: str,
        z_value: Union[float, np.ndarray, pd.Series]
    ) -> Union[float, np.ndarray, pd.Series]:
        """
        Denormalize a z-score back to original scale.

        Formula: x = (z * std) + mean

        Args:
            feature_name: Name of the feature
            z_value: Normalized z-score value(s)

        Returns:
            Original-scale value(s)

        Note:
            - NaN values are replaced with mean
            - If feature not found in stats, uses mean=0, std=1
        """
        mean, std = self._get_stats(feature_name)

        if isinstance(z_value, pd.Series):
            result = (z_value * std) + mean
            result = result.fillna(mean)
            return result

        elif isinstance(z_value, np.ndarray):
            result = (z_value * std) + mean
            result = np.nan_to_num(result, nan=mean)
            return result

        else:
            # Scalar
            if z_value is None or (isinstance(z_value, float) and math.isnan(z_value)):
                return mean
            return (float(z_value) * std) + mean

    def _get_stats(self, feature_name: str) -> tuple:
        """
        Get mean and std for a feature.

        Args:
            feature_name: Name of the feature

        Returns:
            Tuple of (mean, std)
        """
        if feature_name in self._stats_cache:
            return self._stats_cache[feature_name]
        # Default for unknown features
        return (0.0, 1.0)

    def get_stats(self, feature_name: str) -> Dict[str, float]:
        """
        Get full statistics for a feature.

        Args:
            feature_name: Name of the feature

        Returns:
            Dictionary with mean, std, and other available stats
        """
        if feature_name in self._stats:
            return self._stats[feature_name].copy()
        return {"mean": 0.0, "std": 1.0}

    def get_all_features(self) -> list:
        """
        Get list of all features with normalization stats.

        Returns:
            List of feature names
        """
        return list(self._stats.keys())

    def has_feature(self, feature_name: str) -> bool:
        """
        Check if a feature has normalization stats.

        Args:
            feature_name: Name of the feature

        Returns:
            True if feature has stats
        """
        return feature_name in self._stats

    def normalize_dict(
        self,
        features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Normalize a dictionary of feature values.

        Args:
            features: Dictionary mapping feature names to values

        Returns:
            Dictionary with normalized values
        """
        return {
            name: self.normalize(name, value)
            for name, value in features.items()
        }

    def denormalize_dict(
        self,
        features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Denormalize a dictionary of z-score values.

        Args:
            features: Dictionary mapping feature names to z-scores

        Returns:
            Dictionary with original-scale values
        """
        return {
            name: self.denormalize(name, value)
            for name, value in features.items()
        }

    def get_params(self) -> Dict[str, Any]:
        """
        Get normalizer configuration parameters.

        Returns:
            Dictionary with configuration
        """
        return {
            "type": "zscore",
            "stats_path": str(self._stats_path),
            "clip_values": self._clip_values,
            "clip_range": [self.CLIP_MIN, self.CLIP_MAX],
            "n_features": len(self._stats)
        }

    def __repr__(self) -> str:
        return (
            f"ZScoreNormalizer(n_features={len(self._stats)}, "
            f"clip={self._clip_values})"
        )


# Factory function
def create_zscore_normalizer(
    stats_path: str = DEFAULT_STATS_PATH,
    base_path: Optional[str] = None
) -> ZScoreNormalizer:
    """
    Factory function to create a ZScoreNormalizer instance.

    Args:
        stats_path: Path to normalization stats JSON
        base_path: Optional base path for config resolution

    Returns:
        Configured ZScoreNormalizer instance
    """
    return ZScoreNormalizer(stats_path=stats_path, base_path=base_path)
