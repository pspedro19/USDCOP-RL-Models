"""
Strategy Pattern Implementations
================================

Concrete implementations of strategy interfaces.

Author: Trading Team
Version: 1.0.0
Date: 2025-01-14
"""

from .ensemble_strategies import (
    WeightedAverageStrategy,
    MajorityVoteStrategy,
    SoftVoteStrategy,
    ConfidenceWeightedStrategy,
    EnsembleStrategyRegistry,
)

__all__ = [
    'WeightedAverageStrategy',
    'MajorityVoteStrategy',
    'SoftVoteStrategy',
    'ConfidenceWeightedStrategy',
    'EnsembleStrategyRegistry',
]
