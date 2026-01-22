# usdcop_forecasting_clean/backend/src/core/__init__.py
"""
Core module with configuration and base classes.
"""

from .config import (
    PipelineConfig,
    OptunaConfig,
    ModelConfig,
    HORIZONS,
    ML_MODELS,
    RANDOM_STATE,
    get_model_config,
    get_all_model_names
)
from .base import BaseModel, BaseStatisticalModel, BaseTrainer, BaseEvaluator, ModelResult
from .exceptions import (
    PipelineError,
    DataValidationError,
    ModelTrainingError,
    ForecastingError,
    ConfigurationError,
    FeatureEngineeringError
)

__all__ = [
    # Config
    'PipelineConfig',
    'OptunaConfig',
    'ModelConfig',
    'HORIZONS',
    'ML_MODELS',
    'RANDOM_STATE',
    'get_model_config',
    'get_all_model_names',
    # Base classes
    'BaseModel',
    'BaseStatisticalModel',
    'BaseTrainer',
    'BaseEvaluator',
    'ModelResult',
    # Exceptions
    'PipelineError',
    'DataValidationError',
    'ModelTrainingError',
    'ForecastingError',
    'ConfigurationError',
    'FeatureEngineeringError'
]
