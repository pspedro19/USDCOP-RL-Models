"""
Dynamic leverage adjustment based on rolling performance.

Scales leverage down when the strategy is underperforming (low WR, high DD)
and restores it when performance recovers. This prevents 2x leverage from
amplifying directional errors during drawdowns.

Config lives in smart_simple_v1.yaml under `dynamic_leverage`.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class DynamicLeverageConfig:
    enabled: bool = True
    lookback_weeks: int = 8
    wr_full: float = 60.0       # WR >= 60% -> full leverage
    wr_half: float = 40.0       # WR 40-60% -> 50% leverage
    wr_pause: float = 30.0      # WR < 30% -> min leverage (25%)
    dd_reduction_threshold: float = 6.0  # DD > 6% -> halve leverage


def compute_leverage_adjustment(
    recent_pnls: List[float],
    current_dd_pct: float,
    config: DynamicLeverageConfig,
) -> float:
    """
    Compute a scale factor [0.25, 1.0] to apply to the base leverage.

    Args:
        recent_pnls: List of recent trade PnL percentages (last N trades).
        current_dd_pct: Current drawdown from equity peak (positive number, e.g. 5.0 for -5%).
        config: Dynamic leverage configuration.

    Returns:
        Scale factor in [0.25, 1.0]. Multiply by base leverage.
    """
    if not config.enabled:
        return 1.0

    # Not enough history -> full leverage
    if len(recent_pnls) < 3:
        return 1.0

    # Rolling win rate
    wins = sum(1 for p in recent_pnls if p > 0)
    wr = (wins / len(recent_pnls)) * 100.0

    # WR-based scaling
    if wr >= config.wr_full:
        wr_factor = 1.0
    elif wr >= config.wr_half:
        # Linear interpolation between 0.5 and 1.0
        wr_factor = 0.5 + 0.5 * (wr - config.wr_half) / (config.wr_full - config.wr_half)
    elif wr >= config.wr_pause:
        # Linear interpolation between 0.25 and 0.5
        wr_factor = 0.25 + 0.25 * (wr - config.wr_pause) / (config.wr_half - config.wr_pause)
    else:
        wr_factor = 0.25

    # Drawdown penalty
    dd_factor = 1.0
    if current_dd_pct > config.dd_reduction_threshold:
        dd_factor = 0.5

    return max(0.25, min(1.0, wr_factor * dd_factor))
