"""
Regime Gate — Binary classifier for market regime detection.

Determines whether the market is trending (trade normally),
indeterminate (reduce sizing), or mean-reverting (DO NOT TRADE).

Uses Hurst exponent (rescaled range method) as the primary signal.
This is a GATE, not a feature — it decides whether the model runs at all.

Config lives in smart_simple_v1.yaml under `regime_gate`.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np


class RegimeState(Enum):
    TRENDING = "trending"
    INDETERMINATE = "indeterminate"
    MEAN_REVERTING = "mean_reverting"


@dataclass
class RegimeGateConfig:
    enabled: bool = True
    hurst_lookback: int = 60          # Days for Hurst calculation
    hurst_trending: float = 0.52      # v3.0: H > 0.52 = trending (captures 2025 H=0.532)
    hurst_mean_rev: float = 0.42      # v3.0: H < 0.42 = mean-reverting (captures 2026 Q1)
    sizing_indeterminate: float = 0.40  # v3.0: 40% sizing when indeterminate
    sizing_mean_rev: float = 0.0       # 0% sizing = skip trade


@dataclass
class RegimeResult:
    state: RegimeState
    hurst: float
    sizing_factor: float  # 0.0 = skip, 0.25 = reduced, 1.0 = full


def compute_hurst_rs(returns: np.ndarray, max_k: int = 20) -> float:
    """
    Compute Hurst exponent using the rescaled range (R/S) method.

    H > 0.55: trending (persistent)
    H ~ 0.50: random walk
    H < 0.45: mean-reverting (anti-persistent)

    Args:
        returns: Array of daily log returns.
        max_k: Maximum window size for R/S calculation.
    """
    N = len(returns)
    if N < 30:
        return 0.5  # Default to random walk if insufficient data

    RS = []
    for k in range(10, min(max_k + 1, N // 2)):
        rs_vals = []
        for start in range(0, N - k, k):
            chunk = returns[start:start + k]
            mean_c = np.mean(chunk)
            Y = np.cumsum(chunk - mean_c)
            R = np.max(Y) - np.min(Y)
            S = np.std(chunk, ddof=1)
            if S > 1e-10:
                rs_vals.append(R / S)
        if rs_vals:
            RS.append((np.log(k), np.log(np.mean(rs_vals))))

    if len(RS) < 3:
        return 0.5

    x = np.array([r[0] for r in RS])
    y = np.array([r[1] for r in RS])
    H = float(np.polyfit(x, y, 1)[0])
    return np.clip(H, 0.0, 1.0)


def classify_regime(
    daily_returns: List[float],
    config: RegimeGateConfig,
) -> RegimeResult:
    """
    Classify current market regime based on recent daily returns.

    Args:
        daily_returns: List of recent daily returns (at least config.hurst_lookback).
        config: Regime gate configuration.

    Returns:
        RegimeResult with state, hurst value, and sizing factor.
    """
    if not config.enabled:
        return RegimeResult(
            state=RegimeState.TRENDING,
            hurst=0.5,
            sizing_factor=1.0,
        )

    rets = np.array(daily_returns[-config.hurst_lookback:])
    if len(rets) < 30:
        return RegimeResult(
            state=RegimeState.INDETERMINATE,
            hurst=0.5,
            sizing_factor=config.sizing_indeterminate,
        )

    hurst = compute_hurst_rs(rets)

    if hurst > config.hurst_trending:
        return RegimeResult(
            state=RegimeState.TRENDING,
            hurst=hurst,
            sizing_factor=1.0,
        )
    elif hurst < config.hurst_mean_rev:
        return RegimeResult(
            state=RegimeState.MEAN_REVERTING,
            hurst=hurst,
            sizing_factor=config.sizing_mean_rev,
        )
    else:
        return RegimeResult(
            state=RegimeState.INDETERMINATE,
            hurst=hurst,
            sizing_factor=config.sizing_indeterminate,
        )
