"""
Training Environments Module
============================
Professional trading environments for RL training.
"""

from .env_factory import (
    EnvironmentFactory,
    RewardStrategyRegistry,
    create_training_env,
)
from .trading_env import (
    DefaultRewardStrategy,
    EnvObservationBuilder,
    PortfolioState,
    Position,
    RewardStrategy,
    RewardStrategyAdapter,
    StepResult,
    TradingAction,
    TradingEnvConfig,
    TradingEnvironment,
)

__all__ = [
    # Environment
    "TradingEnvironment",
    "TradingEnvConfig",
    "TradingAction",
    "Position",
    "PortfolioState",
    "StepResult",
    # Reward
    "DefaultRewardStrategy",
    "RewardStrategy",
    "RewardStrategyAdapter",
    # Factory
    "EnvironmentFactory",
    "RewardStrategyRegistry",
    "create_training_env",
    # Builder
    "EnvObservationBuilder",
]
