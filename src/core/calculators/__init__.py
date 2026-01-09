"""
Core Calculators for USD/COP Trading System
============================================

Feature calculator implementations following Template Method Pattern.

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Date: 2025-12-17
"""

from .base_calculator import BaseFeatureCalculator
from .rsi_calculator import RSICalculator
from .atr_calculator import ATRCalculator
from .adx_calculator import ADXCalculator
from .returns_calculator import ReturnsCalculator
from .macro_zscore_calculator import MacroZScoreCalculator
from .macro_change_calculator import MacroChangeCalculator

__all__ = [
    'BaseFeatureCalculator',
    'RSICalculator',
    'ATRCalculator',
    'ADXCalculator',
    'ReturnsCalculator',
    'MacroZScoreCalculator',
    'MacroChangeCalculator',
]
