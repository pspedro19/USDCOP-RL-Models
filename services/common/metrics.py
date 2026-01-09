"""
Financial Metrics Calculations
==============================
Shared financial metric calculations for all services.

DRY: Centralizes Sharpe, Sortino, CAGR, VaR calculations.

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-17
"""

import numpy as np
from typing import Optional, Union, List
import logging

logger = logging.getLogger(__name__)


def calculate_sharpe_ratio(
    returns: Union[np.ndarray, List[float]],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Array of returns (daily, 5-min, etc.)
        risk_free_rate: Annual risk-free rate (default: 0)
        periods_per_year: Number of periods in a year (252 for daily, 60*5*252 for 5-min)

    Returns:
        Annualized Sharpe ratio
    """
    returns = np.asarray(returns)
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)

    if std_excess == 0 or np.isnan(std_excess):
        return 0.0

    sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)
    return float(sharpe)


def calculate_sortino_ratio(
    returns: Union[np.ndarray, List[float]],
    target_return: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sortino ratio.

    Args:
        returns: Array of returns
        target_return: Target return (default: 0)
        periods_per_year: Number of periods in a year

    Returns:
        Annualized Sortino ratio
    """
    returns = np.asarray(returns)
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - (target_return / periods_per_year)
    downside_returns = np.minimum(excess_returns, 0)
    downside_std = np.std(downside_returns, ddof=1)

    if downside_std == 0 or np.isnan(downside_std):
        return 0.0

    mean_excess = np.mean(excess_returns)
    sortino = (mean_excess / downside_std) * np.sqrt(periods_per_year)
    return float(sortino)


def calculate_cagr(
    prices: Union[np.ndarray, List[float]],
    periods_per_year: int = 252
) -> float:
    """
    Calculate Compound Annual Growth Rate.

    Args:
        prices: Array of prices (first and last used)
        periods_per_year: Number of periods in a year

    Returns:
        CAGR as decimal (0.12 = 12%)
    """
    prices = np.asarray(prices)
    if len(prices) < 2 or prices[0] <= 0 or prices[-1] <= 0:
        return 0.0

    total_return = prices[-1] / prices[0]
    n_periods = len(prices) - 1
    years = n_periods / periods_per_year

    if years <= 0:
        return 0.0

    cagr = (total_return ** (1 / years)) - 1
    return float(cagr)


def calculate_max_drawdown(prices: Union[np.ndarray, List[float]]) -> float:
    """
    Calculate maximum drawdown.

    Args:
        prices: Array of prices or equity curve

    Returns:
        Maximum drawdown as positive decimal (0.15 = 15% drawdown)
    """
    prices = np.asarray(prices)
    if len(prices) < 2:
        return 0.0

    peak = np.maximum.accumulate(prices)
    drawdown = (peak - prices) / peak
    max_dd = np.max(drawdown)

    return float(max_dd) if not np.isnan(max_dd) else 0.0


def calculate_calmar_ratio(
    returns: Union[np.ndarray, List[float]],
    prices: Optional[Union[np.ndarray, List[float]]] = None,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (CAGR / Max Drawdown).

    Args:
        returns: Array of returns
        prices: Array of prices (optional, computed from returns if not provided)
        periods_per_year: Number of periods in a year

    Returns:
        Calmar ratio
    """
    returns = np.asarray(returns)

    if prices is None:
        # Compute prices from returns
        prices = np.cumprod(1 + returns)
        prices = np.insert(prices, 0, 1.0)  # Start at 1.0

    prices = np.asarray(prices)

    cagr = calculate_cagr(prices, periods_per_year)
    max_dd = calculate_max_drawdown(prices)

    if max_dd == 0:
        return 0.0

    return float(cagr / max_dd)


def calculate_var(
    returns: Union[np.ndarray, List[float]],
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Value at Risk (VaR) using historical method.

    Args:
        returns: Array of returns
        confidence_level: Confidence level (default: 0.95 = 95%)

    Returns:
        VaR as positive decimal (loss at risk)
    """
    returns = np.asarray(returns)
    if len(returns) < 2:
        return 0.0

    percentile = (1 - confidence_level) * 100
    var = -np.percentile(returns, percentile)

    return float(var) if not np.isnan(var) else 0.0


def calculate_expected_shortfall(
    returns: Union[np.ndarray, List[float]],
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Expected Shortfall (CVaR).

    Args:
        returns: Array of returns
        confidence_level: Confidence level (default: 0.95)

    Returns:
        Expected Shortfall as positive decimal
    """
    returns = np.asarray(returns)
    if len(returns) < 2:
        return 0.0

    var = calculate_var(returns, confidence_level)
    tail_returns = returns[returns <= -var]

    if len(tail_returns) == 0:
        return var

    es = -np.mean(tail_returns)
    return float(es) if not np.isnan(es) else var


def calculate_win_rate(returns: Union[np.ndarray, List[float]]) -> float:
    """
    Calculate win rate (percentage of positive returns).

    Args:
        returns: Array of returns

    Returns:
        Win rate as decimal (0.55 = 55%)
    """
    returns = np.asarray(returns)
    if len(returns) == 0:
        return 0.0

    wins = np.sum(returns > 0)
    total = len(returns)

    return float(wins / total)


def calculate_profit_factor(returns: Union[np.ndarray, List[float]]) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    Args:
        returns: Array of returns

    Returns:
        Profit factor (>1 is profitable)
    """
    returns = np.asarray(returns)
    if len(returns) == 0:
        return 0.0

    gross_profit = np.sum(returns[returns > 0])
    gross_loss = -np.sum(returns[returns < 0])

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0

    return float(gross_profit / gross_loss)


def calculate_all_metrics(
    returns: Union[np.ndarray, List[float]],
    prices: Optional[Union[np.ndarray, List[float]]] = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0
) -> dict:
    """
    Calculate all financial metrics at once.

    Args:
        returns: Array of returns
        prices: Array of prices (optional)
        periods_per_year: Number of periods in a year
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary with all metrics
    """
    returns = np.asarray(returns)

    if prices is None and len(returns) > 0:
        prices = np.cumprod(1 + returns)
        prices = np.insert(prices, 0, 1.0)

    return {
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        'sortino_ratio': calculate_sortino_ratio(returns, 0.0, periods_per_year),
        'cagr': calculate_cagr(prices, periods_per_year) if prices is not None else 0.0,
        'max_drawdown': calculate_max_drawdown(prices) if prices is not None else 0.0,
        'calmar_ratio': calculate_calmar_ratio(returns, prices, periods_per_year),
        'var_95': calculate_var(returns, 0.95),
        'expected_shortfall_95': calculate_expected_shortfall(returns, 0.95),
        'win_rate': calculate_win_rate(returns),
        'profit_factor': calculate_profit_factor(returns),
        'total_return': float(np.prod(1 + returns) - 1) if len(returns) > 0 else 0.0,
        'volatility': float(np.std(returns, ddof=1) * np.sqrt(periods_per_year)) if len(returns) > 1 else 0.0,
        'n_observations': len(returns),
    }
