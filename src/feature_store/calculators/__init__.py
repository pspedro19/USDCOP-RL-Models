"""
Feature Store Calculators
==========================
Single source of truth for all feature calculations.

All calculators are registered in the CalculatorRegistry
and can be accessed by name.

Usage:
    from feature_store.calculators import CalculatorRegistry, RSICalculator

    # Create calculator from registry
    spec = FeatureSpec(name="rsi_9", ...)
    calculator = CalculatorRegistry.create("rsi", spec)
    values = calculator.calculate(data)

    # Or use directly
    calculator = RSICalculator(spec)
    values = calculator.calculate(data)

Author: Trading Team
Version: 1.0.0
Created: 2025-01-12
"""

from .base import CalculatorRegistry, FeatureCalculator
from .macro import (
    DXYZScoreCalculator,
    EMBIZScoreCalculator,
    MacroMomentumCalculator,
    MacroZScoreCalculator,
    RollingCorrelationCalculator,
    VIXZScoreCalculator,
    WTIZScoreCalculator,
)

# Import all calculators to register them
from .momentum import (
    ADXCalculator,
    MACDSignalCalculator,
    MomentumCalculator,
    ROCCalculator,
    RSICalculator,
    StochasticKCalculator,
    WilliamsRCalculator,
)
from .trend import (
    CandleDirectionCalculator,
    EMACalculator,
    EMACrossoverCalculator,
    EMADistanceCalculator,
    HigherHighsCountCalculator,
    LinearRegressionSlopeCalculator,
    PricePositionCalculator,
    SMACalculator,
    TrendStrengthCalculator,
)
from .volatility import (
    ATRCalculator,
    ATRPercentCalculator,
    BollingerPositionCalculator,
    BollingerWidthCalculator,
    HistoricalVolatilityCalculator,
    ParkinsonVolatilityCalculator,
    RangePercentCalculator,
    VolatilityRatioCalculator,
)

__all__ = [
    # Base
    "FeatureCalculator",
    "CalculatorRegistry",
    # Momentum
    "RSICalculator",
    "ADXCalculator",
    "MACDSignalCalculator",
    "MomentumCalculator",
    "ROCCalculator",
    "StochasticKCalculator",
    "WilliamsRCalculator",
    # Volatility
    "ATRCalculator",
    "ATRPercentCalculator",
    "BollingerWidthCalculator",
    "BollingerPositionCalculator",
    "VolatilityRatioCalculator",
    "HistoricalVolatilityCalculator",
    "ParkinsonVolatilityCalculator",
    "RangePercentCalculator",
    # Trend
    "EMACalculator",
    "SMACalculator",
    "EMADistanceCalculator",
    "EMACrossoverCalculator",
    "PricePositionCalculator",
    "TrendStrengthCalculator",
    "LinearRegressionSlopeCalculator",
    "HigherHighsCountCalculator",
    "CandleDirectionCalculator",
    # Macro
    "MacroZScoreCalculator",
    "DXYZScoreCalculator",
    "VIXZScoreCalculator",
    "WTIZScoreCalculator",
    "EMBIZScoreCalculator",
    "MacroMomentumCalculator",
    "RollingCorrelationCalculator",
]
