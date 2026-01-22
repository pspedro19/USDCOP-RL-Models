# backend/src/features/__init__.py
"""
Feature engineering module.

Provides feature transformation, scaling utilities, and common feature
preparation functions used across the training pipeline.
"""

from .transformer import FeatureTransformer
from .common import (
    prepare_features,
    create_targets,
    prepare_train_test_split,
    clean_features,
    prepare_features_for_training,
    DEFAULT_HORIZONS,
)

__all__ = [
    # Transformer
    "FeatureTransformer",
    # Common utilities
    "prepare_features",
    "create_targets",
    "prepare_train_test_split",
    "clean_features",
    "prepare_features_for_training",
    "DEFAULT_HORIZONS",
]
