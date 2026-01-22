# backend/src/__init__.py
"""
Source code for USD/COP Forecasting Pipeline.

This package contains all core functionality:

Subpackages:
    - core: Configuration, base classes, and exceptions
    - config: Centralized settings and environment variables
    - data: Data loading, validation, and reconciliation
    - features: Feature engineering and transformation
    - models: ML model implementations
    - evaluation: Backtesting and metrics
    - mlops: MLflow and MinIO clients
    - monitoring: Data quality monitoring

Example:
    from backend.src.core import PipelineConfig
    from backend.src.data import DataLoader, load_data
    from backend.src.models import RidgeModel
    from backend.src.config import get_settings, DATA_DIR
"""

# Core exports
from .core import (
    PipelineConfig,
    OptunaConfig,
    ModelConfig,
    HORIZONS,
    ML_MODELS,
    RANDOM_STATE,
    BaseModel,
    BaseTrainer,
    ModelResult,
    PipelineError,
    DataValidationError,
    ModelTrainingError,
)

# Data exports
from .data import (
    DataLoader,
    load_data,
    DataValidator,
    DataReconciler,
)

# Models exports
from .models import (
    RidgeModel,
    BayesianRidgeModel,
    XGBoostModel,
    LightGBMModel,
    CatBoostModel,
)

# Evaluation exports
from .evaluation import (
    Backtester,
    direction_accuracy,
    PurgedKFold,
)

# Feature exports
from .features import (
    FeatureTransformer,
    prepare_features,
    create_targets,
)

# Database exports
from .database import (
    get_db_connection,
    db_connection,
    get_minio_client,
    upsert_to_postgres,
    save_to_minio,
    load_from_minio,
)

__all__ = [
    # Core
    "PipelineConfig",
    "OptunaConfig",
    "ModelConfig",
    "HORIZONS",
    "ML_MODELS",
    "RANDOM_STATE",
    "BaseModel",
    "BaseTrainer",
    "ModelResult",
    "PipelineError",
    "DataValidationError",
    "ModelTrainingError",
    # Data
    "DataLoader",
    "load_data",
    "DataValidator",
    "DataReconciler",
    # Models
    "RidgeModel",
    "BayesianRidgeModel",
    "XGBoostModel",
    "LightGBMModel",
    "CatBoostModel",
    # Evaluation
    "Backtester",
    "direction_accuracy",
    "PurgedKFold",
    # Features
    "FeatureTransformer",
    "prepare_features",
    "create_targets",
    # Database
    "get_db_connection",
    "db_connection",
    "get_minio_client",
    "upsert_to_postgres",
    "save_to_minio",
    "load_from_minio",
]
