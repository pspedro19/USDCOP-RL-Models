"""
Core Schemas Module
===================

Pydantic schemas for configuration validation and serialization.

Available schemas:
- ExperimentConfig: Complete experiment configuration
- DatasetConfig: Dataset configuration
- HyperparametersConfig: PPO hyperparameters
- NetworkConfig: Neural network architecture
- EnvironmentConfig: Trading environment settings
"""

from src.core.schemas.experiment_config import (
    ExperimentConfig,
    ExperimentMetadata,
    DatasetConfig,
    DateRangeConfig,
    FeaturesConfig,
    HyperparametersConfig,
    NetworkConfig,
    EnvironmentConfig,
    TrainingConfig,
    EvaluationConfig,
    PromotionConfig,
    MLflowConfig,
    ExpectedResults,
    load_experiment_config,
    create_experiment_from_baseline,
)

__all__ = [
    "ExperimentConfig",
    "ExperimentMetadata",
    "DatasetConfig",
    "DateRangeConfig",
    "FeaturesConfig",
    "HyperparametersConfig",
    "NetworkConfig",
    "EnvironmentConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "PromotionConfig",
    "MLflowConfig",
    "ExpectedResults",
    "load_experiment_config",
    "create_experiment_from_baseline",
]
