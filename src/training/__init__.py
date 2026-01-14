"""
Training Module
===============
Professional training infrastructure for USDCOP RL models.

Submodules:
- environments: Trading environments for RL training
- trainers: PPO and other training algorithms

Components:
- RewardCalculator: Corrected reward function with proper order of operations
"""

from .reward_calculator import RewardCalculator, RewardConfig

# Environment imports
from .environments import (
    TradingEnvironment,
    TradingEnvConfig,
    TradingAction,
    Position,
    PortfolioState,
    StepResult,
    DefaultRewardStrategy,
    RewardStrategy,
    RewardStrategyAdapter,
    EnvironmentFactory,
    RewardStrategyRegistry,
    create_training_env,
    EnvObservationBuilder,
)

# Trainer imports
from .trainers import (
    PPOTrainer,
    PPOConfig,
    TrainingResult,
    ActionDistributionCallback,
    MetricsCallback,
    ProgressCallback,
    train_ppo,
)

__all__ = [
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
]
