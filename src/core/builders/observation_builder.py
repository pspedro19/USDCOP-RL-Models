"""
ObservationBuilder - Builder Pattern for Observation Vectors
=============================================================

Implements Builder Pattern with fluent interface for constructing observations.

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Date: 2025-12-17
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from ..interfaces.observation_builder import IObservationBuilder
from ..interfaces.config_loader import IConfigLoader
from ...shared.exceptions import (
    ObservationDimensionError,
    ValidationError,
    FeatureMissingError
)


class ObservationBuilder(IObservationBuilder):
    """
    Builder for constructing observation vectors with fluent interface.

    Implements Builder Pattern for step-by-step observation construction.

    Example:
        builder = ObservationBuilder(config_loader, feature_order)
        obs = (builder
               .with_features(features_dict)
               .with_position(0.5)
               .with_time_normalized(30, 60)
               .build())
    """

    def __init__(self,
                 feature_order: List[str],
                 obs_dim: int = 15,
                 global_clip_min: float = -5.0,
                 global_clip_max: float = 5.0):
        """
        Initialize observation builder.

        Args:
            feature_order: Ordered list of feature names
            obs_dim: Total observation dimension (default: 15)
            global_clip_min: Global minimum clip value (default: -5.0)
            global_clip_max: Global maximum clip value (default: 5.0)
        """
        self._feature_order = feature_order
        self._obs_dim = obs_dim
        self._global_clip_min = global_clip_min
        self._global_clip_max = global_clip_max

        # Internal state
        self._features: Optional[Dict[str, float]] = None
        self._position: Optional[float] = None
        self._time_normalized: Optional[float] = None

    def with_features(self, features: Dict[str, float]) -> 'ObservationBuilder':
        """
        Set feature values for the observation.

        Args:
            features: Dictionary of feature_name -> value

        Returns:
            Self for method chaining
        """
        self._features = features
        return self

    def with_position(self, position: float) -> 'ObservationBuilder':
        """
        Set position value for the observation.

        Args:
            position: Current position (-1 to 1)

        Returns:
            Self for method chaining
        """
        self._position = position
        return self

    def with_time_normalized(self, bar_number: int, episode_length: int) -> 'ObservationBuilder':
        """
        Set time normalization for the observation.

        CRITICAL FORMULA: time_normalized = (bar_number - 1) / episode_length

        Args:
            bar_number: Current bar number (1-based)
            episode_length: Total episode length in bars

        Returns:
            Self for method chaining
        """
        self._time_normalized = (bar_number - 1) / episode_length
        return self

    def build(self) -> np.ndarray:
        """
        Build the final observation vector.

        Returns:
            np.ndarray of shape (obs_dim,) ready for model

        Raises:
            ObservationDimensionError: If dimension mismatch
            ValidationError: If validation fails
            FeatureMissingError: If required feature is missing
        """
        # Validate that all required data is set
        if self._features is None:
            raise ValidationError("Features must be set before building observation")
        if self._position is None:
            raise ValidationError("Position must be set before building observation")
        if self._time_normalized is None:
            raise ValidationError("Time normalization must be set before building observation")

        # Extract features in correct order
        feature_values = []
        for feat in self._feature_order:
            if feat not in self._features:
                raise FeatureMissingError(feature_name=feat)

            val = self._features.get(feat, 0.0)
            if pd.isna(val) or np.isnan(val):
                val = 0.0
            feature_values.append(float(val))

        # Construct observation: [13 features] + [position] + [time_normalized]
        obs = np.array(
            feature_values + [self._position, self._time_normalized],
            dtype=np.float32
        )

        # Apply global clipping
        obs = np.clip(obs, self._global_clip_min, self._global_clip_max)

        # Validate dimension
        if obs.shape[0] != self._obs_dim:
            raise ObservationDimensionError(
                expected=self._obs_dim,
                actual=obs.shape[0]
            )

        # Validate no NaN/Inf
        if np.any(np.isnan(obs)):
            raise ValidationError("Observation contains NaN values")
        if np.any(np.isinf(obs)):
            raise ValidationError("Observation contains infinite values")

        return obs

    def reset(self) -> 'ObservationBuilder':
        """
        Reset builder to initial state.

        Returns:
            Self for method chaining
        """
        self._features = None
        self._position = None
        self._time_normalized = None
        return self

    def validate(self) -> bool:
        """
        Validate current observation state without building.

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        if self._features is None:
            raise ValidationError("Features not set")
        if self._position is None:
            raise ValidationError("Position not set")
        if self._time_normalized is None:
            raise ValidationError("Time normalization not set")

        # Check all required features present
        for feat in self._feature_order:
            if feat not in self._features:
                raise FeatureMissingError(feature_name=feat)

        return True

    def get_state(self) -> dict:
        """
        Get current builder state (for debugging).

        Returns:
            Dictionary with current state
        """
        return {
            'features': self._features,
            'position': self._position,
            'time_normalized': self._time_normalized,
            'feature_order': self._feature_order,
            'obs_dim': self._obs_dim
        }

    def __repr__(self) -> str:
        features_set = self._features is not None
        position_set = self._position is not None
        time_set = self._time_normalized is not None
        return (f"ObservationBuilder("
                f"features={'set' if features_set else 'unset'}, "
                f"position={'set' if position_set else 'unset'}, "
                f"time={'set' if time_set else 'unset'})")
