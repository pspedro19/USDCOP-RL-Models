"""
Feature Contract - DEPRECATED WRAPPER
======================================

DEPRECATION NOTICE (v4.0.0):
This module is DEPRECATED. Import directly from src.core.contracts instead.

Migration:
    # OLD (deprecated):
    from src.features.contract import FEATURE_ORDER, FEATURE_CONTRACT

    # NEW (recommended):
    from src.core.contracts import FEATURE_ORDER, FEATURE_CONTRACT

Contract ID: CTR-002
SSOT: src.core.contracts.feature_contract
"""

import warnings
from typing import Final

# Emit deprecation warning once per session
warnings.warn(
    "src.features.contract is deprecated. "
    "Import from src.core.contracts instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from SSOT (REQUIRED - no fallback)
from src.core.contracts.feature_contract import (
    FeatureContract,
    FEATURE_CONTRACT,
    FEATURE_ORDER,
    OBSERVATION_DIM,
)

# Additional imports from feature_store for backward compatibility
from src.feature_store.core import (
    TechnicalPeriods,
    TradingHours,
    get_contract,
)

# NORM_STATS_PATH from contract
NORM_STATS_PATH: Final = "config/norm_stats.json"

# Re-export for backward compatibility
__all__ = [
    "FeatureContract",
    "FEATURE_CONTRACT",
    "FEATURE_ORDER",
    "OBSERVATION_DIM",
    "NORM_STATS_PATH",
    "TechnicalPeriods",
    "TradingHours",
    "get_contract",
]
