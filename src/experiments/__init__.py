"""
USDCOP Experiment Framework
===========================

Complete A/B experimentation framework for training, evaluating, and comparing
reinforcement learning models for USD/COP trading.

Usage:
    # Load experiment config
    from src.experiments import load_experiment_config
    config = load_experiment_config("config/experiments/my_experiment.yaml")

    # Run experiment
    from src.experiments import ExperimentRunner
    runner = ExperimentRunner(config)
    result = runner.run()

    # Compare experiments
    from src.experiments import compare_experiments
    comparison = compare_experiments("baseline_v1", "new_model_v1")

Author: Trading Team
Date: 2026-01-17
"""

from .experiment_config import (
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    EnvironmentConfig,
    DataConfig,
    EvaluationConfig,
    MLflowConfig,
    CallbacksConfig,
)
from .experiment_loader import (
    load_experiment_config,
    validate_experiment_config,
    list_available_experiments,
)
from .experiment_runner import ExperimentRunner, ExperimentResult
from .experiment_registry import ExperimentRegistry
from .experiment_comparator import compare_experiments, ExperimentComparison

__all__ = [
    # Config classes
    "ExperimentConfig",
    "ModelConfig",
    "TrainingConfig",
    "EnvironmentConfig",
    "DataConfig",
    "EvaluationConfig",
    "MLflowConfig",
    "CallbacksConfig",
    # Loaders
    "load_experiment_config",
    "validate_experiment_config",
    "list_available_experiments",
    # Runner
    "ExperimentRunner",
    "ExperimentResult",
    # Registry
    "ExperimentRegistry",
    # Comparison
    "compare_experiments",
    "ExperimentComparison",
]
