"""
Training Environments Module
============================
Professional trading environments for RL training.
"""

from .trading_env import (
    TradingEnvironment,
    TradingEnvConfig,
    TradingAction,
    Position,
    PortfolioState,
    StepResult,
    DefaultRewardStrategy,
    RewardStrategy,
    RewardStrategyAdapter,
    EnvObservationBuilder,
)

from .env_factory import (
    EnvironmentFactory,
    RewardStrategyRegistry,
    create_training_env,
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
