"""
USD/COP Trading System - Trading Module
========================================

Paper trading and simulation components for validating trading strategies
without executing real orders.

Components:
- PaperTrader: Simulates trade execution and tracks positions
- PaperTrade: Dataclass representing a simulated trade

Usage:
    from src.trading import PaperTrader, PaperTrade

    # Initialize paper trader with PostgreSQL connection
    trader = PaperTrader(initial_capital=10000.0, db_connection=conn)

    # Execute signals from models
    trade = trader.execute_signal(
        model_id="ppo_agent_v1",
        signal="LONG",
        current_price=4250.50
    )

    # Get statistics
    stats = trader.get_statistics()

Author: USD/COP Trading System
Version: 1.0.0
"""

from .paper_trader import (
    PaperTrade,
    PaperTrader,
    TradeDirection,
)

__version__ = "1.0.0"

__all__ = [
    "PaperTrade",
    "PaperTrader",
    "TradeDirection",
]
