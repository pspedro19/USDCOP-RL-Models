"""
Trading Factories Module
========================

Factory patterns for creating trading components.

Author: Trading Team
Version: 1.0.0
Date: 2025-01-16
"""

from .trade_executor_factory import (
    ITradeExecutor,
    TradeExecutorFactory,
    TradeResult,
)

__all__ = [
    "ITradeExecutor",
    "TradeExecutorFactory",
    "TradeResult",
]
