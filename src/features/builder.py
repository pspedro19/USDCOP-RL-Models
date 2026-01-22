"""
FeatureBuilder - Legacy Wrapper (DEPRECATED)
=============================================

DEPRECATION NOTICE (v3.0.0):
This module is DEPRECATED. Use CanonicalFeatureBuilder from
src.feature_store.builders instead.

This wrapper maintains backward compatibility by delegating all
calculations to CanonicalFeatureBuilder (the Single Source of Truth).

Migration Guide:
    # OLD (deprecated):
    from src.features.builder import FeatureBuilder
    builder = FeatureBuilder(version="current")
    obs = builder.build_observation(ohlcv, macro_df, position, timestamp, bar_idx)

    # NEW (recommended):
    from src.feature_store.builders import CanonicalFeatureBuilder
    builder = CanonicalFeatureBuilder.for_inference()
    obs = builder.build_observation(ohlcv, macro_df, position, bar_idx)

Author: Trading Team
Version: 3.0.0
Date: 2025-01-17

CHANGELOG v3.0.0:
- REMOVED: All legacy implementation (duplicated code)
- DELEGATES: Pure delegation to CanonicalFeatureBuilder (SSOT)
- REDUCED: From 405 lines to ~80 lines (DRY compliance)
"""

from __future__ import annotations

import warnings
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

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
        obs = builder.build_observation(ohlcv, macro_df, position, bar_idx)
    """

    _deprecation_warned: bool = False

    def __init__(self, version: str = "current"):
        """
        Initialize FeatureBuilder.

        DEPRECATED: Use CanonicalFeatureBuilder.for_training() or
        CanonicalFeatureBuilder.for_inference() instead.
        """
        if not FeatureBuilder._deprecation_warned:
            warnings.warn(
                "src.features.builder.FeatureBuilder is deprecated. "
                "Use CanonicalFeatureBuilder from src.feature_store.builders instead.",
                DeprecationWarning,
                stacklevel=2
            )
            FeatureBuilder._deprecation_warned = True

        CanonicalBuilder = _get_canonical_builder()
        self._canonical = CanonicalBuilder.for_training()
        self._version = version
        logger.info("FeatureBuilder initialized (delegates to CanonicalFeatureBuilder)")

    def get_observation_dim(self) -> int:
        """Return observation dimension from SSOT."""
        return self._canonical.get_observation_dim()

    def get_feature_names(self) -> Tuple[str, ...]:
        """Return feature names in contract order."""
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
        """
        return self._canonical.build_observation(
            ohlcv=ohlcv,
            macro=macro_df,
            position=position,
            bar_idx=bar_idx,
            timestamp=timestamp
        )

    @property
    def norm_stats(self) -> Dict[str, Dict[str, float]]:
        """Get normalization stats from SSOT."""
        return self._canonical.get_norm_stats()


def create_feature_builder(version: str = "current") -> FeatureBuilder:
    """
    Factory function - DEPRECATED.
    Use CanonicalFeatureBuilder factory methods instead.
    """
    return FeatureBuilder(version=version)


def get_canonical_builder(context: str = "training"):
    """
    Helper to get CanonicalFeatureBuilder with appropriate context.

    Args:
        context: One of "training", "inference", "backtest"

    Returns:
        CanonicalFeatureBuilder instance
    """
    CanonicalBuilder = _get_canonical_builder()

    if context == "training":
        return CanonicalBuilder.for_training()
    elif context == "inference":
        return CanonicalBuilder.for_inference()
    elif context == "backtest":
        return CanonicalBuilder.for_backtest()
    else:
        raise ValueError(f"Invalid context: {context}")
