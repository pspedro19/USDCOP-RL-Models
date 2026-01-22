"""
Backtest Package
================

Unified backtesting infrastructure for the USDCOP trading system.

This package provides:
- UnifiedBacktestEngine: Single Source of Truth for all backtesting
- BacktestConfig: Immutable configuration for backtest runs
- Trade, BacktestMetrics, BacktestResult: Data structures for results

Design Principles:
- Single Source of Truth: ONE backtest engine used everywhere
- No Look-Ahead Bias: Strict bar-by-bar processing
- Realistic Execution: Slippage and transaction costs modeled
- Reproducible: Deterministic results with same config

Architecture:
    Historical Data ─────┐
                         │
    Model Artifacts ─────┼───▶ UnifiedBacktestEngine ───▶ BacktestResult
                         │         (SSOT)
    Config ──────────────┘

Author: Trading Team
Version: 1.0.0
Created: 2025-01-17
"""

from .engine.unified_backtest_engine import (
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
