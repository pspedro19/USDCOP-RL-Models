"""Training Module
===============
Professional training infrastructure for USDCOP RL models.

Architecture (Clean Code - DRY):
    config.py (SSOT)
         ↓
    engine.py (UNIFIED TRAINING)
         ↓
    ┌────────────────────────────────────────┐
    │  trainers/ - PPO training              │
    │  environments/ - RL environments       │
    │  utils/ - Reproducibility utilities    │
    └────────────────────────────────────────┘

Components:
- TrainingEngine: Unified training orchestration (SINGLE ENTRY POINT)
- TrainingConfig: SSOT for all training configuration
- PPOHyperparameters: Canonical PPO hyperparameters
- PPOTrainer: Professional PPO training wrapper
- EnvironmentFactory: Environment creation

Usage:
    from src.training import (
        TrainingEngine,
        TrainingRequest,
        run_training,
        PPO_HYPERPARAMETERS,
    )

    # Run training via engine
    result = run_training(
        project_root=Path("."),
        version="v1",
        dataset_path=Path("data/train.csv"),
    )
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# Config SSOT imports (PRIMARY SOURCE)
from .config import (
    DATA_SPLIT_CONFIG,
    ENVIRONMENT_CONFIG,
    INDICATOR_CONFIG,
    MLFLOW_CONFIG,
    NETWORK_CONFIG,
    # Singleton instances (SSOT)
    PPO_HYPERPARAMETERS,
    DataSplitConfig,
    EnvironmentConfig,
    IndicatorConfig,
    MLflowConfig,
    NetworkConfig,
    # Dataclasses
    PPOHyperparameters,
    TrainingConfig,
    get_project_root,
    # Factory functions
    get_training_config,
    load_config_from_yaml,
    # Validation
    validate_config,
)

# Non-config exports stay public but load only when requested.  This keeps pure
# configuration imports dependency-light without hiding errors when callers actually
# request the RL stack.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "TrainingEngine": (".engine", "TrainingEngine"),
    "TrainingRequest": (".engine", "TrainingRequest"),
    "run_training": (".engine", "run_training"),
    "EngineTrainingResult": (".engine", "TrainingResult"),
    "DefaultRewardStrategy": (".environments", "DefaultRewardStrategy"),
    "EnvironmentFactory": (".environments", "EnvironmentFactory"),
    "EnvObservationBuilder": (".environments", "EnvObservationBuilder"),
    "PortfolioState": (".environments", "PortfolioState"),
    "Position": (".environments", "Position"),
    "RewardStrategy": (".environments", "RewardStrategy"),
    "RewardStrategyAdapter": (".environments", "RewardStrategyAdapter"),
    "RewardStrategyRegistry": (".environments", "RewardStrategyRegistry"),
    "StepResult": (".environments", "StepResult"),
    "TradingAction": (".environments", "TradingAction"),
    "TradingEnvConfig": (".environments", "TradingEnvConfig"),
    "TradingEnvironment": (".environments", "TradingEnvironment"),
    "create_training_env": (".environments", "create_training_env"),
    "MultiSeedConfig": (".multi_seed_trainer", "MultiSeedConfig"),
    "MultiSeedResult": (".multi_seed_trainer", "MultiSeedResult"),
    "MultiSeedTrainer": (".multi_seed_trainer", "MultiSeedTrainer"),
    "train_with_multiple_seeds": (".multi_seed_trainer", "train_with_multiple_seeds"),
    "RewardCalculator": (".reward_calculator", "RewardCalculator"),
    "RewardConfig": (".reward_calculator", "RewardConfig"),
    "ActionDistributionCallback": (".trainers", "ActionDistributionCallback"),
    "MetricsCallback": (".trainers", "MetricsCallback"),
    "PPOConfig": (".trainers", "PPOConfig"),
    "PPOTrainer": (".trainers", "PPOTrainer"),
    "ProgressCallback": (".trainers", "ProgressCallback"),
    "TrainingResult": (".trainers", "TrainingResult"),
    "train_ppo": (".trainers", "train_ppo"),
    "compute_file_hash": (".utils", "compute_file_hash"),
    "compute_json_hash": (".utils", "compute_json_hash"),
    "set_reproducible_seeds": (".utils", "set_reproducible_seeds"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

__all__ = [
    # Engine (UNIFIED TRAINING)
    "TrainingEngine",
    "TrainingRequest",
    "run_training",
    # Config SSOT (PRIMARY)
    "PPOHyperparameters",
    "NetworkConfig",
    "EnvironmentConfig",
    "DataSplitConfig",
    "IndicatorConfig",
    "MLflowConfig",
    "TrainingConfig",
    # SSOT Singleton instances
    "PPO_HYPERPARAMETERS",
    "NETWORK_CONFIG",
    "ENVIRONMENT_CONFIG",
    "DATA_SPLIT_CONFIG",
    "INDICATOR_CONFIG",
    "MLFLOW_CONFIG",
    # Factory functions
    "get_training_config",
    "load_config_from_yaml",
    "get_project_root",
    "validate_config",
    # Reproducibility
    "set_reproducible_seeds",
    "compute_file_hash",
    "compute_json_hash",
    # Reward
    "RewardCalculator",
    "RewardConfig",
    # Environment
    "TradingEnvironment",
    "TradingEnvConfig",
    "TradingAction",
    "Position",
    "PortfolioState",
    "StepResult",
    # Reward Strategies
    "DefaultRewardStrategy",
    "RewardStrategy",
    "RewardStrategyAdapter",
    # Environment Factory
    "EnvironmentFactory",
    "RewardStrategyRegistry",
    "create_training_env",
    "EnvObservationBuilder",
    # Trainers
    "PPOTrainer",
    "PPOConfig",
    "TrainingResult",
    # Callbacks
    "ActionDistributionCallback",
    "MetricsCallback",
    "ProgressCallback",
    # Convenience
    "train_ppo",
    # Multi-seed training
    "MultiSeedTrainer",
    "MultiSeedConfig",
    "MultiSeedResult",
    "train_with_multiple_seeds",
]
