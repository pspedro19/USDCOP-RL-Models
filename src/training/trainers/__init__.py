"""
Training Trainers Module
========================
Professional trainer implementations for RL training.
"""

from .ppo_trainer import (
    PPOTrainer,
    PPOConfig,
    TrainingResult,
    ActionDistributionCallback,
    MetricsCallback,
    ProgressCallback,
    train_ppo,
)

__all__ = [
    # Trainer
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
