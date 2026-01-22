"""
Demo/Investor Mode for USDCOP Trading System
=============================================

This module provides a completely isolated demo mode that generates
realistic trading results for investor presentations.

Enable with environment variable: INVESTOR_MODE=true

Features:
- Realistic trade generation based on actual price data
- Configurable target metrics (Sharpe, Win Rate, etc.)
- Maintains all animation/replay functionality
- Completely isolated from production code
"""

from .trade_generator import DemoTradeGenerator
from .config import DEMO_CONFIG, is_investor_mode, is_demo_model

__all__ = ['DemoTradeGenerator', 'DEMO_CONFIG', 'is_investor_mode', 'is_demo_model']
