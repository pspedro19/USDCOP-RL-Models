"""
IObservationBuilder - Interface for observation builders
=========================================================

Abstract interface for Builder Pattern implementation.

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Date: 2025-12-17
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict


class IObservationBuilder(ABC):
    """
    Abstract interface for building observation vectors.

    Implements Builder Pattern for fluent observation construction.

    Example:
        obs = (ObservationBuilder()
               .with_features(features_dict)
               .with_position(0.5)
               .with_time_normalized(30, 60)
               .build())
    """

    @abstractmethod
    def with_features(self, features: Dict[str, float]) -> 'IObservationBuilder':
        """
        Set feature values for the observation.

        Args:
            features: Dictionary of feature_name -> value

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def with_position(self, position: float) -> 'IObservationBuilder':
        """
        Set position value for the observation.

        Args:
            position: Current position (-1 to 1)

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def with_time_normalized(self, bar_number: int, episode_length: int) -> 'IObservationBuilder':
        """
        Set time normalization for the observation.

        Args:
            bar_number: Current bar number (1-based)
            episode_length: Total episode length in bars

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def build(self) -> np.ndarray:
        """
        Build the final observation vector.

        Returns:
            np.ndarray of shape (obs_dim,) ready for model

        Raises:
            ObservationDimensionError: If dimension mismatch
            ValidationError: If validation fails
        """
        pass

    @abstractmethod
    def reset(self) -> 'IObservationBuilder':
        """
        Reset builder to initial state.

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate current observation state.

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        pass
