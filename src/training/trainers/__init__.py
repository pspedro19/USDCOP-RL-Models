"""
Training Trainers Module
========================
Professional trainer implementations for RL training.
"""

from .ppo_trainer import (
    ActionDistributionCallback,
    MetricsCallback,
    PPOConfig,
    PPOTrainer,
    ProgressCallback,
    TrainingResult,
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
