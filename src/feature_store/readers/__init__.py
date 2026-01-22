"""
Feature Store Readers - Read Pre-computed Features
==================================================
This module provides readers for accessing pre-computed features
from the feature store.

The FeatureReader reads features from the inference_features_5m table
populated by the L1 pipeline, allowing L5 to skip feature recalculation.

Usage:
    from src.feature_store.readers import FeatureReader, FeatureResult

    # Initialize reader
    reader = FeatureReader()

    # Get latest features for inference
    result = reader.get_latest_features(
        symbol="USD/COP",
        max_age_minutes=10
    )

    if result and result.is_valid():
        # Use observation for model inference
        action, _ = model.predict(result.observation)
        print(f"Features age: {result.age_minutes:.1f} min")

    # Get historical features for backtesting
    history = reader.get_features_history(
        symbol="USD/COP",
        start_time=start_dt,
        end_time=end_dt,
        limit=1000
    )

Author: Trading Team
Version: 1.0.0
Created: 2026-01-17
"""

from .feature_reader import (
    # Data classes
    FeatureResult,
    # Main class
    FeatureReader,
    # Exceptions
    FeatureReaderError,
    FeatureNotFoundError,
    StaleFeatureError,
    FeatureOrderMismatchError,
)

__all__ = [
    # Data classes
    "FeatureResult",
    # Main class
    "FeatureReader",
    # Exceptions
    "FeatureReaderError",
    "FeatureNotFoundError",
    "StaleFeatureError",
    "FeatureOrderMismatchError",
]

__version__ = "1.0.0"
