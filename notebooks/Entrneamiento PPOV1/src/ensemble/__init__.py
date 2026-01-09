"""
USD/COP RL Trading System - Ensemble Module
============================================

Provides ensemble prediction capabilities combining multiple PPO models.
"""

from .ensemble_predictor import (
    EnsemblePredictor,
    EnsembleConfig,
    MODEL_A_CONFIG,
    MODEL_B_CONFIG,
    DEFAULT_ENSEMBLE_CONFIGS,
)

__all__ = [
    'EnsemblePredictor',
    'EnsembleConfig',
    'MODEL_A_CONFIG',
    'MODEL_B_CONFIG',
    'DEFAULT_ENSEMBLE_CONFIGS',
]
