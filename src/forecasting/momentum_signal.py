"""
Momentum Signal Generator — Smart Simple v3.0

Replaces Ridge/BR/XGBoost with a parameter-free momentum signal.
Direction comes from multi-horizon return confirmation.
Zero parameters optimized on historical data = zero overfitting risk.

The signal adapts its confirmation requirements to the regime:
  - TRENDING: ret_20d direction with ret_10d confirmation
  - TRANSITION: requires ret_10d + ret_20d + ret_50d alignment
  - MEAN_REVERTING: no signal (regime gate blocks)
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class SignalConfidence(Enum):
    CONFIRMED = "confirmed"  # Multi-horizon alignment in trending
    UNCONFIRMED = "unconfirmed"  # Primary signal only, no confirmation
    ALIGNED = "aligned"  # All 3 horizons agree in transition
    SKIP = "skip"  # Mixed signals or regime blocked


@dataclass
class MomentumSignal:
    direction: int | None  # +1 (LONG), -1 (SHORT), None (skip)
    confidence: SignalConfidence
    sizing_multiplier: float  # 1.0 (full), 0.6 (unconfirmed), 0.4 (transition)
    ret_10d: float
    ret_20d: float
    ret_50d: float
    regime: str  # For logging


def compute_momentum_signal(
    close_prices: np.ndarray,
    regime: str,
) -> MomentumSignal:
    """
    Compute adaptive momentum signal based on regime.

    Args:
        close_prices: Array of daily close prices (most recent last).
                      Needs at least 51 values for ret_50d.
        regime: "trending", "indeterminate", or "mean_reverting"
    """
    n = len(close_prices)

    # Compute returns at multiple horizons
    ret_10d = np.log(close_prices[-1] / close_prices[-11]) if n > 11 else 0.0
    ret_20d = np.log(close_prices[-1] / close_prices[-21]) if n > 21 else 0.0
    ret_50d = np.log(close_prices[-1] / close_prices[-51]) if n > 51 else 0.0

    sig_10d = 1 if ret_10d > 0 else -1
    sig_20d = 1 if ret_20d > 0 else -1
    sig_50d = 1 if ret_50d > 0 else -1

    if regime == "mean_reverting":
        return MomentumSignal(
            direction=None,
            confidence=SignalConfidence.SKIP,
            sizing_multiplier=0.0,
            ret_10d=ret_10d,
            ret_20d=ret_20d,
            ret_50d=ret_50d,
            regime=regime,
        )

    if regime == "trending":
        # Primary: ret_20d. Confirmation: ret_10d agrees.
        if sig_20d == sig_10d:
            return MomentumSignal(
                direction=sig_20d,
                confidence=SignalConfidence.CONFIRMED,
                sizing_multiplier=1.0,
                ret_10d=ret_10d,
                ret_20d=ret_20d,
                ret_50d=ret_50d,
                regime=regime,
            )
        else:
            return MomentumSignal(
                direction=sig_20d,
                confidence=SignalConfidence.UNCONFIRMED,
                sizing_multiplier=0.6,
                ret_10d=ret_10d,
                ret_20d=ret_20d,
                ret_50d=ret_50d,
                regime=regime,
            )

    # TRANSITION / INDETERMINATE: all 3 horizons must agree
    if sig_10d == sig_20d == sig_50d:
        return MomentumSignal(
            direction=sig_20d,
            confidence=SignalConfidence.ALIGNED,
            sizing_multiplier=0.4,
            ret_10d=ret_10d,
            ret_20d=ret_20d,
            ret_50d=ret_50d,
            regime=regime,
        )
    else:
        return MomentumSignal(
            direction=None,
            confidence=SignalConfidence.SKIP,
            sizing_multiplier=0.0,
            ret_10d=ret_10d,
            ret_20d=ret_20d,
            ret_50d=ret_50d,
            regime=regime,
        )
