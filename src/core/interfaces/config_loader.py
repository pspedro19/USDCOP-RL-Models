"""
IConfigLoader - Interface for configuration loaders
===================================================

Abstract interface for configuration loading.

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Date: 2025-12-17
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class IConfigLoader(ABC):
    """
    Abstract interface for configuration loaders.

    Enables testability and flexibility in configuration sources.

    Example:
        config = JSONConfigLoader('config/feature_config.json')
        features = config.get_feature_order()
    """

    @abstractmethod
    def get_feature_order(self) -> List[str]:
        """
        Get ordered list of features.

        Returns:
            List of feature names in correct order
        """
        pass

    @abstractmethod
    def get_obs_dim(self) -> int:
        """
        Get total observation dimension.

        Returns:
            Total observation dimension (features + state variables)
        """
        pass

    @abstractmethod
    def get_norm_stats(self, feature_name: str) -> Dict[str, float]:
        """
        Get normalization statistics for a feature.

        Args:
            feature_name: Name of the feature

        Returns:
            Dictionary with 'mean' and 'std' keys
        """
        pass

    @abstractmethod
    def get_clip_bounds(self, feature_name: str) -> Optional[tuple]:
        """
        Get clipping bounds for a feature.

        Args:
            feature_name: Name of the feature

        Returns:
            Tuple (min, max) or None
        """
        pass

    @abstractmethod
    def get_technical_period(self, indicator: str) -> Optional[int]:
        """
        Get period for technical indicator.

        Args:
            indicator: Indicator name

        Returns:
            Period value or None
        """
        pass

    @abstractmethod
    def get_trading_params(self) -> Dict[str, Any]:
        """
        Get trading parameters.

        Returns:
            Dictionary with trading configuration
        """
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """
        Get configuration version.

        Returns:
            Version string
        """
        pass
