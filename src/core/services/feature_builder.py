"""
USD/COP Trading System - Feature Builder (Legacy Wrapper)
==========================================================

DEPRECATION NOTICE (v4.0.0):
This module is DEPRECATED. Use CanonicalFeatureBuilder from
src.feature_store.builders instead.

This wrapper maintains backward compatibility by delegating all
calculations to CanonicalFeatureBuilder (the Single Source of Truth).

Migration Guide:
    # OLD (deprecated):
    from src.core.services.feature_builder import FeatureBuilder
    builder = FeatureBuilder()
    obs = builder.build_observation(features, position, bar_number)

    # NEW (recommended):
    from src.feature_store.builders import CanonicalFeatureBuilder
    builder = CanonicalFeatureBuilder.for_inference()
    obs = builder.build_observation(ohlcv, macro, position, bar_idx)

Author: Trading Team
Version: 4.0.0
Date: 2025-01-17

CHANGELOG v4.0.0:
- REMOVED: All legacy implementation (~900 lines of duplicated code)
- DELEGATES: Pure delegation to CanonicalFeatureBuilder (SSOT)
- REDUCED: From 988 lines to ~120 lines (DRY compliance)
- REMOVED: Legacy fallback mode (CanonicalFeatureBuilder is always available)
"""

import warnings
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple

# Import constants from SSOT
from src.core.constants import (
    RSI_PERIOD, ATR_PERIOD, ADX_PERIOD,
    BARS_PER_SESSION, CLIP_MIN, CLIP_MAX,
)

logger = logging.getLogger(__name__)


def _get_canonical_builder():
    """Get CanonicalFeatureBuilder - raises if not available."""
    from src.feature_store.builders import CanonicalFeatureBuilder
    return CanonicalFeatureBuilder


class FeatureBuilder:
    """
    DEPRECATED: Use CanonicalFeatureBuilder instead.

    This class is maintained for backward compatibility only.
    All calculations are delegated to CanonicalFeatureBuilder (SSOT).

    Migration:
        from src.feature_store.builders import CanonicalFeatureBuilder
        builder = CanonicalFeatureBuilder.for_inference()
        obs = builder.build_observation(ohlcv, macro, position, bar_idx)
    """

    # Constants from SSOT (src/core/constants.py) - imported at module level
    RSI_PERIOD = RSI_PERIOD
    ATR_PERIOD = ATR_PERIOD
    ADX_PERIOD = ADX_PERIOD
    BARS_PER_SESSION = BARS_PER_SESSION
    GLOBAL_CLIP_MIN = CLIP_MIN
    GLOBAL_CLIP_MAX = CLIP_MAX

    _deprecation_warned: bool = False

    def __init__(self, config_path: Optional[str] = None, config_loader=None):
        """
        Initialize FeatureBuilder.

        DEPRECATED: Use CanonicalFeatureBuilder.for_training() or
        CanonicalFeatureBuilder.for_inference() instead.
        """
        if not FeatureBuilder._deprecation_warned:
            warnings.warn(
                "src.core.services.feature_builder.FeatureBuilder is deprecated. "
                "Use CanonicalFeatureBuilder from src.feature_store.builders instead.",
                DeprecationWarning,
                stacklevel=2
            )
            FeatureBuilder._deprecation_warned = True

        CanonicalBuilder = _get_canonical_builder()
        self._canonical = CanonicalBuilder.for_training()
        self._feature_order = list(self._canonical.get_feature_order())
        self._obs_dim = self._canonical.get_observation_dim()
        self._episode_length = self.BARS_PER_SESSION

        logger.info("FeatureBuilder initialized (delegates to CanonicalFeatureBuilder)")

    def get_observation_dim(self) -> int:
        """Return observation dimension from SSOT."""
        return self._obs_dim

    def get_feature_order(self) -> List[str]:
        """Return ordered list of feature names."""
        return self._feature_order.copy()

    def get_norm_stats(self, feature_name: str = None) -> Dict:
        """Get normalization statistics."""
        stats = self._canonical.get_norm_stats()
        if feature_name:
            return stats.get(feature_name, {'mean': 0.0, 'std': 1.0})
        return stats

    def build_observation(
        self,
        features_dict: Dict[str, float],
        position: float,
        bar_number: int,
        episode_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Build observation array from pre-computed features.

        Note: This signature differs from CanonicalFeatureBuilder.
        For new code, use CanonicalFeatureBuilder.build_observation() directly.
        """
        if episode_length is None:
            episode_length = self._episode_length

        # Build observation from dict
        feature_values = []
        for feat in self._feature_order[:13]:  # 13 market features
            val = features_dict.get(feat, 0.0)
            if pd.isna(val) or np.isnan(val):
                val = 0.0
            feature_values.append(float(val))

        # Calculate time_normalized
        time_normalized = (bar_number - 1) / episode_length

        # Construct observation
        obs = np.array(
            feature_values + [float(position), time_normalized],
            dtype=np.float32
        )

        # Clip to valid range
        obs = np.clip(obs, self.GLOBAL_CLIP_MIN, self.GLOBAL_CLIP_MAX)

        return obs

    def calculate_time_normalized(self, bar_number: int, episode_length: Optional[int] = None) -> float:
        """Calculate time_normalized value."""
        if episode_length is None:
            episode_length = self._episode_length
        return (bar_number - 1) / episode_length

    def validate_observation(self, obs: np.ndarray, check_range: bool = True) -> bool:
        """Validate observation array."""
        if obs.shape != (self._obs_dim,):
            raise ValueError(f"Expected shape ({self._obs_dim},), got {obs.shape}")
        if np.any(np.isnan(obs)):
            raise ValueError("Observation contains NaN values")
        if np.any(np.isinf(obs)):
            raise ValueError("Observation contains infinite values")
        if check_range and (np.any(obs < -5.0) or np.any(obs > 5.0)):
            raise ValueError(f"Values out of range: min={obs.min():.3f}, max={obs.max():.3f}")
        return True

    @property
    def feature_order(self) -> List[str]:
        """Get ordered list of features."""
        return self._feature_order

    @property
    def obs_dim(self) -> int:
        """Get observation dimension."""
        return self._obs_dim


def create_feature_builder(config_path: Optional[str] = None, config_loader=None) -> FeatureBuilder:
    """
    Factory function - DEPRECATED.
    Use CanonicalFeatureBuilder factory methods instead.
    """
    return FeatureBuilder(config_path=config_path, config_loader=config_loader)
