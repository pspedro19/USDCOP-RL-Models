"""
Forecasting Configuration (SSOT)
================================

Single Source of Truth for all forecasting pipeline configuration.
This module consolidates training, inference, and data parameters.

Design Pattern: Centralized Configuration with Dataclasses
- Immutable default values
- Type safety via dataclasses
- Hash-based versioning for reproducibility

Usage:
    from src.forecasting.config import ForecastingConfig, get_config
    config = get_config()
    print(config.training.learning_rate)

@version 1.0.0
@author Trading Team
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import hashlib
import json
import os
from pathlib import Path


# =============================================================================
# ENUMS
# =============================================================================

class ModelType(str, Enum):
    """Forecasting model types."""
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    HYBRID_XGBOOST = "hybrid_xgboost"
    HYBRID_LIGHTGBM = "hybrid_lightgbm"
    HYBRID_CATBOOST = "hybrid_catboost"
    RIDGE = "ridge"
    BAYESIAN_RIDGE = "bayesian_ridge"
    ARD = "ard"


class DataSourceType(str, Enum):
    """Data source types."""
    YFINANCE = "yfinance"
    TWELVEDATA = "twelvedata"
    INVESTING = "investing"
    DATABASE = "database"


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass(frozen=True)
class DataConfig:
    """Data pipeline configuration."""

    # Date ranges
    train_start: str = "2020-01-01"
    train_end: str = "2024-12-31"
    validation_start: str = "2025-01-01"
    validation_end: str = "2025-06-30"
    test_start: str = "2025-07-01"
    test_end: str = "2025-12-31"

    # Lookback for feature calculation
    lookback_days: int = 50  # For MA50

    # Data quality thresholds
    min_records: int = 500
    max_null_pct: float = 5.0
    price_range_min: float = 3000.0  # COP sanity check
    price_range_max: float = 6000.0

    # Sources
    primary_source: DataSourceType = DataSourceType.YFINANCE
    backup_source: DataSourceType = DataSourceType.TWELVEDATA


@dataclass(frozen=True)
class FeatureConfig:
    """Feature engineering configuration."""

    # Feature columns (SSOT - imported from data_contracts)
    feature_columns: Tuple[str, ...] = (
        "close", "open", "high", "low",
        "return_1d", "return_5d", "return_10d", "return_20d",
        "volatility_5d", "volatility_10d", "volatility_20d",
        "rsi_14d", "ma_ratio_20d", "ma_ratio_50d",
        "day_of_week", "month", "is_month_end",
        "dxy_close_lag1", "oil_close_lag1",
    )

    # Target horizons (days)
    target_horizons: Tuple[int, ...] = (1, 5, 10, 15, 20, 25, 30)

    # Target column
    target_column: str = "close"

    # RSI period
    rsi_period: int = 14

    # MA periods
    ma_short_period: int = 20
    ma_long_period: int = 50

    # Volatility windows
    volatility_windows: Tuple[int, ...] = (5, 10, 20)

    # Return windows
    return_windows: Tuple[int, ...] = (1, 5, 10, 20)

    @property
    def num_features(self) -> int:
        """Number of features."""
        return len(self.feature_columns)


@dataclass(frozen=True)
class TrainingConfig:
    """Model training configuration."""

    # Walk-forward validation
    walk_forward_windows: int = 5
    min_train_size: int = 252  # ~1 year of trading days
    expansion_step: int = 63   # ~3 months

    # Model defaults
    default_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "early_stopping_rounds": 50,
    })

    # Hybrid model alpha (linear weight)
    hybrid_alpha: float = 0.3

    # Random seed
    random_seed: int = 42

    # MLflow
    experiment_name: str = "forecasting-training"


@dataclass(frozen=True)
class InferenceConfig:
    """Inference configuration."""

    # Batch settings
    batch_size: int = 1
    max_latency_ms: int = 100

    # Ensemble settings
    generate_ensembles: bool = True
    top_k_models: int = 3

    # Output settings
    persist_to_db: bool = True
    upload_images: bool = True


@dataclass(frozen=True)
class StorageConfig:
    """Storage paths configuration."""

    # MinIO buckets
    models_bucket: str = "forecasting-models"
    forecasts_bucket: str = "forecasts"
    datasets_bucket: str = "forecasting-datasets"

    # Local paths (relative to project root)
    models_dir: str = "outputs/forecasting/models"
    datasets_dir: str = "data/forecasting"
    cache_dir: str = ".cache/forecasting"

    # Database tables
    ohlcv_table: str = "bi.dim_daily_usdcop"
    features_view: str = "bi.v_forecasting_features"
    forecasts_table: str = "bi.fact_forecasts"
    metrics_table: str = "bi.fact_model_metrics"


@dataclass(frozen=True)
class MLflowConfig:
    """MLflow configuration for experiment tracking."""

    # Enable/disable MLflow
    enabled: bool = True

    # Tracking server
    tracking_uri: str = "http://localhost:5000"

    # Experiment name
    experiment_name: str = "forecasting-training"

    # Registry configuration
    registry_enabled: bool = True
    model_name_prefix: str = "forecasting"

    # Logging options
    log_models: bool = True
    log_artifacts: bool = True
    log_params: bool = True
    log_metrics: bool = True

    # Tags
    project_tag: str = "usdcop-forecasting"
    team_tag: str = "algo-trading"
    pipeline_tag: str = "forecasting"


@dataclass(frozen=True)
class ForecastingConfig:
    """
    Master configuration for forecasting pipeline.

    This is the SSOT for all forecasting configuration.
    """

    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)

    # Version
    version: str = "1.0.0"

    def compute_hash(self) -> str:
        """Compute configuration hash for reproducibility."""
        config_dict = {
            "data": {
                "train_start": self.data.train_start,
                "train_end": self.data.train_end,
            },
            "features": {
                "columns": self.features.feature_columns,
                "horizons": self.features.target_horizons,
            },
            "training": {
                "walk_forward_windows": self.training.walk_forward_windows,
                "random_seed": self.training.random_seed,
            },
            "version": self.version,
        }
        content = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "config_hash": self.compute_hash(),
            "data": {
                "train_start": self.data.train_start,
                "train_end": self.data.train_end,
                "validation_start": self.data.validation_start,
                "validation_end": self.data.validation_end,
                "test_start": self.data.test_start,
                "test_end": self.data.test_end,
            },
            "features": {
                "num_features": self.features.num_features,
                "columns": list(self.features.feature_columns),
                "horizons": list(self.features.target_horizons),
            },
            "training": {
                "walk_forward_windows": self.training.walk_forward_windows,
                "random_seed": self.training.random_seed,
                "experiment_name": self.training.experiment_name,
            },
            "storage": {
                "models_bucket": self.storage.models_bucket,
                "forecasts_bucket": self.storage.forecasts_bucket,
            },
            "mlflow": {
                "enabled": self.mlflow.enabled,
                "tracking_uri": self.mlflow.tracking_uri,
                "experiment_name": self.mlflow.experiment_name,
                "registry_enabled": self.mlflow.registry_enabled,
            },
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_config_instance: Optional[ForecastingConfig] = None


def get_config() -> ForecastingConfig:
    """
    Get the forecasting configuration singleton.

    Returns:
        ForecastingConfig instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ForecastingConfig()
    return _config_instance


def load_config_from_yaml(path: str) -> ForecastingConfig:
    """
    Load configuration from YAML file (params.yaml).

    Args:
        path: Path to YAML file

    Returns:
        ForecastingConfig with values from file
    """
    import yaml

    with open(path) as f:
        params = yaml.safe_load(f)

    forecasting_params = params.get("forecasting", {})

    data_params = forecasting_params.get("data", {})
    data_config = DataConfig(
        train_start=data_params.get("train_start", DataConfig.train_start),
        train_end=data_params.get("train_end", DataConfig.train_end),
        validation_start=data_params.get("validation_start", DataConfig.validation_start),
        validation_end=data_params.get("validation_end", DataConfig.validation_end),
        test_start=data_params.get("test_start", DataConfig.test_start),
        test_end=data_params.get("test_end", DataConfig.test_end),
    )

    training_params = forecasting_params.get("training", {})
    training_config = TrainingConfig(
        walk_forward_windows=training_params.get("walk_forward_windows", TrainingConfig.walk_forward_windows),
        random_seed=training_params.get("random_seed", TrainingConfig.random_seed),
        experiment_name=training_params.get("experiment_name", TrainingConfig.experiment_name),
    )

    return ForecastingConfig(
        data=data_config,
        training=training_config,
        version=forecasting_params.get("version", "1.0.0"),
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ModelType",
    "DataSourceType",
    # Config classes
    "DataConfig",
    "FeatureConfig",
    "TrainingConfig",
    "InferenceConfig",
    "StorageConfig",
    "MLflowConfig",
    "ForecastingConfig",
    # Functions
    "get_config",
    "load_config_from_yaml",
]
