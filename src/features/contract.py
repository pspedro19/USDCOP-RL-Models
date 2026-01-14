"""
Feature Contract - Single Source of Truth
==========================================
This module provides the canonical feature contract definition.
All feature-related code should import from here.

Usage:
    from features.contract import FEATURE_CONTRACT, get_contract

Contrato ID: CTR-002
"""

import sys
from pathlib import Path
from typing import Dict, Final, Tuple
from dataclasses import dataclass

# Ensure src is in path for feature_store import
_src_dir = Path(__file__).parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

try:
    # Try importing from feature_store (Single Source of Truth)
    from feature_store.core import (
        FeatureContract,
        FEATURE_CONTRACT,
        FEATURE_ORDER,
        OBSERVATION_DIM,
        NORM_STATS_PATH,
        TechnicalPeriods,
        TradingHours,
        get_contract as _get_contract,
    )

    _USE_FEATURE_STORE = True

except ImportError:
    # Fallback: Define contracts locally if feature_store not available
    _USE_FEATURE_STORE = False

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
        """Feature Contract - IMMUTABLE SPECIFICATION"""
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
        trading_hours_start: str = "13:00"
        trading_hours_end: str = "17:55"
        rsi_period: int = 9
        atr_period: int = 10
        adx_period: int = 14
        warmup_bars: int = 14
        created_at: str = "2025-01-12"

        @property
        def technical_periods(self) -> Dict[str, int]:
            """Get technical indicator periods."""
            return {"rsi": self.rsi_period, "atr": self.atr_period, "adx": self.adx_period}

        def get_trading_hours(self) -> Dict[str, str]:
            return {"start": self.trading_hours_start, "end": self.trading_hours_end}

        def get_technical_periods(self) -> Dict[str, int]:
            return self.technical_periods

    # Singleton instance
    FEATURE_CONTRACT: Final = FeatureContract()

    # Convenience exports
    FEATURE_ORDER: Final = FEATURE_CONTRACT.feature_order
    OBSERVATION_DIM: Final = FEATURE_CONTRACT.observation_dim
    NORM_STATS_PATH: Final = FEATURE_CONTRACT.norm_stats_path

    def _get_contract(version: str = "current") -> FeatureContract:
        """Factory to get contracts by version."""
        # All versions now return the same current contract
        return FEATURE_CONTRACT


# Re-export for modules
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


def get_contract(version: str = "current") -> FeatureContract:
    """
    Factory to get contracts by version.

    Args:
        version: Contract version (default: "current")

    Returns:
        FeatureContract instance
    """
    return _get_contract(version)
