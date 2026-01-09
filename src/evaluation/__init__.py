"""
Evaluation module for USDCOP Trading System.

Provides benchmark strategies and comparison tools for RL model evaluation.
"""

from .benchmarks import (
    BenchmarkStrategies,
    compare_with_benchmarks,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
)

__all__ = [
    'BenchmarkStrategies',
    'compare_with_benchmarks',
    'calculate_max_drawdown',
    'calculate_sharpe_ratio',
]
