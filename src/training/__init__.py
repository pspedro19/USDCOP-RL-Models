"""
Training Module
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

# Engine - UNIFIED TRAINING (replaces train_ssot, training_pipeline)
from .engine import (
    TrainingEngine,
    TrainingRequest,
    run_training,
)
from .engine import (
    TrainingResult as EngineTrainingResult,
)

# Environment imports
from .environments import (
    DefaultRewardStrategy,
    EnvironmentFactory,
    EnvObservationBuilder,
    PortfolioState,
    Position,
    RewardStrategy,
    RewardStrategyAdapter,
    RewardStrategyRegistry,
    StepResult,
    TradingAction,
    TradingEnvConfig,
    TradingEnvironment,
    create_training_env,
)

# Multi-seed training for variance reduction
from .multi_seed_trainer import (
    MultiSeedConfig,
    MultiSeedResult,
    MultiSeedTrainer,
    train_with_multiple_seeds,
)
from .reward_calculator import RewardCalculator, RewardConfig

# Trainer imports
from .trainers import (
    ActionDistributionCallback,
    MetricsCallback,
    PPOConfig,
    PPOTrainer,
    ProgressCallback,
    TrainingResult,
    train_ppo,
)

# Reproducibility utilities
from .utils import (
    compute_file_hash,
    compute_json_hash,
    set_reproducible_seeds,
)

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
