"""
Vol-Targeting Module
====================

Position sizing via volatility targeting for the forecasting pipeline.

Used by:
- scripts/vol_target_backtest.py (offline validation)
- airflow/dags/forecast_l5c_vol_targeting.py (production)

@version 1.0.0
@contract FC-SIZE-001
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class VolTargetConfig:
    """Configuration for vol-targeting (immutable)."""
    target_vol: float = 0.15           # Annualized target volatility
    max_leverage: float = 2.0          # Hard ceiling
    min_leverage: float = 0.5          # Hard floor (always some exposure)
    vol_lookback: int = 21             # Days for realized vol calculation
    vol_floor: float = 0.05            # Min vol to prevent extreme leverage
    annualization_factor: float = 252.0  # Trading days per year


@dataclass
class VolTargetSignal:
    """Output of vol-targeting computation."""
    date: str
    forecast_direction: int            # +1 or -1
    forecast_return: float             # Predicted log-return from ensemble
    realized_vol_21d: float            # Annualized realized volatility
    raw_leverage: float                # Before clipping
    clipped_leverage: float            # After clipping to [min, max]
    position_size: float               # direction * clipped_leverage

    @property
    def is_levered(self) -> bool:
        return abs(self.clipped_leverage) > 1.0


def compute_vol_target_signal(
    forecast_direction: int,
    forecast_return: float,
    realized_vol_21d: float,
    config: VolTargetConfig = VolTargetConfig(),
    date: str = "",
) -> VolTargetSignal:
    """
    Compute vol-targeting signal.

    Vol-targeting scales position size inversely to realized volatility:
        leverage = target_vol / realized_vol
    This maintains roughly constant portfolio volatility regardless of
    market conditions.

    Args:
        forecast_direction: +1 (long) or -1 (short) from ensemble
        forecast_return: predicted log-return from ensemble
        realized_vol_21d: annualized 21-day realized volatility
        config: vol-targeting parameters
        date: date string for the signal

    Returns:
        VolTargetSignal with leverage and position size
    """
    safe_vol = max(realized_vol_21d, config.vol_floor)
    raw_leverage = config.target_vol / safe_vol
    clipped_leverage = float(np.clip(raw_leverage, config.min_leverage, config.max_leverage))
    position_size = forecast_direction * clipped_leverage

    return VolTargetSignal(
        date=date,
        forecast_direction=forecast_direction,
        forecast_return=forecast_return,
        realized_vol_21d=realized_vol_21d,
        raw_leverage=raw_leverage,
        clipped_leverage=clipped_leverage,
        position_size=position_size,
    )


def apply_asymmetric_sizing(
    leverage: float,
    direction: int,
    long_mult: float = 0.5,
    short_mult: float = 1.0,
) -> float:
    """
    Apply direction-dependent leverage scaling.

    SHORT positions get full leverage (short_mult=1.0).
    LONG positions get reduced leverage (long_mult=0.5) due to 2023
    fragility analysis showing LONG trades fail when model generates >50% LONGs.

    Args:
        leverage: Vol-targeted leverage (clipped to [min, max]).
        direction: +1 for long, -1 for short.
        long_mult: Multiplier for LONG positions (default 0.5).
        short_mult: Multiplier for SHORT positions (default 1.0).

    Returns:
        Asymmetric leverage: leverage * direction_multiplier.
    """
    assert direction in (1, -1), f"direction must be 1 or -1, got {direction}"
    assert leverage >= 0, f"leverage must be non-negative, got {leverage}"
    multiplier = long_mult if direction == 1 else short_mult
    return leverage * multiplier


def compute_realized_vol(
    returns: np.ndarray,
    lookback: int = 21,
    annualization: float = 252.0,
) -> float:
    """
    Compute annualized realized volatility from daily returns.

    Args:
        returns: Array of daily returns (at least `lookback` values)
        lookback: Number of days for rolling window
        annualization: Trading days per year

    Returns:
        Annualized realized volatility
    """
    if len(returns) < lookback:
        return 0.0
    recent = returns[-lookback:]
    return float(np.std(recent, ddof=1) * np.sqrt(annualization))
