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

from .base import FeatureCalculator, CalculatorRegistry

# Import all calculators to register them
from .momentum import (
    RSICalculator,
    ADXCalculator,
    MACDSignalCalculator,
    MomentumCalculator,
    ROCCalculator,
    StochasticKCalculator,
    WilliamsRCalculator,
)

from .volatility import (
    ATRCalculator,
    ATRPercentCalculator,
    BollingerWidthCalculator,
    BollingerPositionCalculator,
    VolatilityRatioCalculator,
    HistoricalVolatilityCalculator,
    ParkinsonVolatilityCalculator,
    RangePercentCalculator,
)

from .trend import (
    EMACalculator,
    SMACalculator,
    EMADistanceCalculator,
    EMACrossoverCalculator,
    PricePositionCalculator,
    TrendStrengthCalculator,
    LinearRegressionSlopeCalculator,
    HigherHighsCountCalculator,
    CandleDirectionCalculator,
)

from .macro import (
    MacroZScoreCalculator,
    DXYZScoreCalculator,
    VIXZScoreCalculator,
    WTIZScoreCalculator,
    EMBIZScoreCalculator,
    MacroMomentumCalculator,
    RollingCorrelationCalculator,
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
