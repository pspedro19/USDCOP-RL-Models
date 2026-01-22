# backend/__init__.py
"""
USD/COP Forecasting Backend Package.

This package provides the ML pipeline components for USD/COP exchange rate forecasting.

Modules:
    - src: Core source code (models, data, features, evaluation, mlops, monitoring)
    - pipelines: Training and inference pipelines
    - scripts: Utility scripts

Usage:
    from backend.src.core import PipelineConfig, HORIZONS
    from backend.src.data import DataLoader
    from backend.src.models import RidgeModel, XGBoostModel
    from backend.src.mlops import MLflowClient, MinioClient
"""

__version__ = "1.0.0"
__author__ = "MLOps Team"

# Expose key components at package level for convenience
from backend.src.core import (
    PipelineConfig,
    HORIZONS,
    ML_MODELS,
    RANDOM_STATE,
)

__all__ = [
    "__version__",
    "PipelineConfig",
    "HORIZONS",
    "ML_MODELS",
    "RANDOM_STATE",
]
