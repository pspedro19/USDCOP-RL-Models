"""
Adaptive Stops — Volatility-scaled take-profit and hard-stop levels.
====================================================================

Computes stop levels from 21-day realized volatility:
    hard_stop = clamp(vol_21d * sqrt(5) * multiplier, min_pct, max_pct)
    take_profit = hard_stop * tp_ratio

sqrt(5) converts daily vol to ~5-day (weekly) expected move.
multiplier=1.5 gives 1.5 standard deviations (93% probability of NOT being hit by noise).

Examples (annualized vol → weekly → stops):
    vol_21d_ann = 10% → daily ~0.63% → weekly ~1.41% → HS = min(2.11%, 3%) = 2.11%, TP = 1.06%
    vol_21d_ann = 15% → daily ~0.94% → weekly ~2.11% → HS = min(3.17%, 3%) = 3.00%, TP = 1.50%
    vol_21d_ann = 20% → daily ~1.26% → weekly ~2.81% → HS = min(4.22%, 3%) = 3.00%, TP = 1.50%
    vol_21d_ann =  5% → daily ~0.31% → weekly ~0.70% → HS = max(1.06%, 1%) = 1.06%, TP = 0.53%

@version 1.0.0
@contract FC-H5-STOP-001
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptiveStopsConfig:
    """Configuration for adaptive stop levels."""
    vol_multiplier: float = 1.5        # Number of std devs for hard stop
    hard_stop_min_pct: float = 0.01    # 1% minimum hard stop
    hard_stop_max_pct: float = 0.03    # 3% maximum hard stop
    tp_ratio: float = 0.5             # take_profit = hard_stop * tp_ratio
    annualization_factor: float = 252.0  # Trading days per year


@dataclass
class AdaptiveStops:
    """Computed stop levels for a specific week."""
    hard_stop_pct: float              # e.g., 0.021 = 2.1%
    take_profit_pct: float            # e.g., 0.0105 = 1.05%
    realized_vol_daily: float         # Daily vol (non-annualized)
    realized_vol_weekly: float        # Weekly vol (daily * sqrt(5))


def compute_adaptive_stops(
    realized_vol_annualized: float,
    config: AdaptiveStopsConfig = AdaptiveStopsConfig(),
) -> AdaptiveStops:
    """
    Compute adaptive stop levels from annualized realized volatility.

    Args:
        realized_vol_annualized: Annualized 21-day realized vol (e.g., 0.15 = 15%).
        config: Stop level parameters.

    Returns:
        AdaptiveStops with hard_stop_pct and take_profit_pct.
    """
    # Convert annualized vol to daily
    daily_vol = realized_vol_annualized / math.sqrt(config.annualization_factor)

    # Scale to weekly (5 trading days)
    weekly_vol = daily_vol * math.sqrt(5)

    # Hard stop = weekly_vol * multiplier, clamped
    raw_hard_stop = weekly_vol * config.vol_multiplier
    hard_stop_pct = max(config.hard_stop_min_pct, min(raw_hard_stop, config.hard_stop_max_pct))

    # Take profit = hard_stop * ratio
    take_profit_pct = hard_stop_pct * config.tp_ratio

    return AdaptiveStops(
        hard_stop_pct=hard_stop_pct,
        take_profit_pct=take_profit_pct,
        realized_vol_daily=daily_vol,
        realized_vol_weekly=weekly_vol,
    )


def check_hard_stop(
    direction: int,
    entry_price: float,
    bar_high: float,
    bar_low: float,
    hard_stop_pct: float,
) -> bool:
    """
    Check if hard stop was triggered on this bar.

    For LONG: triggered if bar_low <= entry * (1 - hard_stop_pct)
    For SHORT: triggered if bar_high >= entry * (1 + hard_stop_pct)
    """
    if direction == 1:  # LONG
        stop_level = entry_price * (1 - hard_stop_pct)
        return bar_low <= stop_level
    else:  # SHORT
        stop_level = entry_price * (1 + hard_stop_pct)
        return bar_high >= stop_level


def check_take_profit(
    direction: int,
    entry_price: float,
    bar_high: float,
    bar_low: float,
    take_profit_pct: float,
) -> bool:
    """
    Check if take-profit was triggered on this bar.

    For LONG: triggered if bar_high >= entry * (1 + take_profit_pct)
    For SHORT: triggered if bar_low <= entry * (1 - take_profit_pct)
    """
    if direction == 1:  # LONG
        tp_level = entry_price * (1 + take_profit_pct)
        return bar_high >= tp_level
    else:  # SHORT
        tp_level = entry_price * (1 - take_profit_pct)
        return bar_low <= tp_level


def get_exit_price(
    direction: int,
    entry_price: float,
    reason: str,
    hard_stop_pct: float,
    take_profit_pct: float,
    bar_close: float,
) -> float:
    """
    Compute the exit price for a given exit reason.

    Limit orders: TP and HS are limit orders → exact price.
    Week end: market order at bar_close.
    """
    if reason == "take_profit":
        if direction == 1:
            return entry_price * (1 + take_profit_pct)
        else:
            return entry_price * (1 - take_profit_pct)
    elif reason == "hard_stop":
        if direction == 1:
            return entry_price * (1 - hard_stop_pct)
        else:
            return entry_price * (1 + hard_stop_pct)
    else:  # week_end
        return bar_close
