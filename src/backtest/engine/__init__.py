"""
Backtest Engine Package
=======================

Contains the Unified Backtest Engine - the Single Source of Truth for all
backtesting operations in the USDCOP trading system.

Exports:
- TradeDirection: Enum for position direction (LONG, SHORT, FLAT)
- BacktestConfig: Immutable configuration dataclass
- Trade: Individual trade record
- BacktestMetrics: Performance metrics dataclass
- BacktestResult: Complete backtest output
- UnifiedBacktestEngine: Main backtest engine class
- create_backtest_engine: Factory function

Author: Trading Team
Version: 1.0.0
Created: 2025-01-17
"""

from .unified_backtest_engine import (
    TradeDirection,
    BacktestConfig,
    Trade,
    BacktestMetrics,
    BacktestResult,
    UnifiedBacktestEngine,
    create_backtest_engine,
)

__all__ = [
    "TradeDirection",
    "BacktestConfig",
    "Trade",
    "BacktestMetrics",
    "BacktestResult",
    "UnifiedBacktestEngine",
    "create_backtest_engine",
]
