# USD/COP RL Trading System V19
# ==============================
# Main training package with curriculum learning

__version__ = '19.0.0'
__author__ = 'Claude Code'

# Preprocessing (existing)
from .preprocessing import (
    DataPreprocessor,
    DataQualityConfig,
    ALL_FEATURES_CLEAN,
    diagnose_dataset,
)

# V19 Components
from .environment_v19 import (
    TradingEnvironmentV19,
    SETFXCostModel,
    create_training_env,
    create_validation_env,
)

from .train_v19 import TrainingPipelineV19

__all__ = [
    # Preprocessing
    'DataPreprocessor',
    'DataQualityConfig',
    'ALL_FEATURES_CLEAN',
    'diagnose_dataset',
    # V19
    'TradingEnvironmentV19',
    'SETFXCostModel',
    'create_training_env',
    'create_validation_env',
    'TrainingPipelineV19',
]
