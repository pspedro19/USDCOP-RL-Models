"""
Evaluation module for RL trading models.

Provides unified backtesting and evaluation utilities.
"""

from .backtest_engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    TradeRecord,
    run_backtest,
)

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "TradeRecord",
    "run_backtest",
]
