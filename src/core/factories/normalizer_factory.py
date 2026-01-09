"""
NormalizerFactory - Factory Pattern for Normalizers
====================================================

Creates appropriate normalizer instances based on normalization type.

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Date: 2025-12-17
"""

from typing import Dict, Type
from ..interfaces.normalizer import INormalizer
from ...shared.exceptions import ConfigurationError


class NormalizerFactory:
    """
    Factory for creating normalizer instances.

    Implements Factory Pattern for normalization strategies.

    Usage:
        factory = NormalizerFactory()
        zscore_norm = factory.create('zscore', mean=0.0, std=1.0)
        clip_norm = factory.create('clip', min_val=-4.0, max_val=4.0)
        noop_norm = factory.create('noop')
    """

    _normalizers: Dict[str, Type[INormalizer]] = {}

    @classmethod
    def register(cls, normalizer_type: str, normalizer_class: Type[INormalizer]) -> None:
        """
        Register a normalizer class for a type.

        Args:
            normalizer_type: Normalizer type identifier (e.g., 'zscore', 'clip', 'noop')
            normalizer_class: Normalizer class to register
        """
        cls._normalizers[normalizer_type] = normalizer_class

    @classmethod
    def create(cls, normalizer_type: str, **kwargs) -> INormalizer:
        """
        Create a normalizer instance for the specified type.

        Args:
            normalizer_type: Type of normalizer to create
            **kwargs: Parameters to pass to normalizer constructor

        Returns:
            INormalizer instance

        Raises:
            ConfigurationError: If normalizer type is not registered

        Example:
            >>> factory = NormalizerFactory()
            >>> normalizer = factory.create('zscore', mean=0.0, std=1.0)
        """
        normalizer_class = cls._normalizers.get(normalizer_type)

        if normalizer_class is None:
            raise ConfigurationError(
                f"Unknown normalizer type: '{normalizer_type}'",
                missing_key=normalizer_type
            )

        return normalizer_class(**kwargs)

    @classmethod
    def create_from_config(cls, feature_name: str, norm_stats: dict, clip_bounds: tuple = None) -> INormalizer:
        """
        Create normalizer from configuration stats.

        Automatically selects appropriate normalizer based on available stats.

        Args:
            feature_name: Name of the feature
            norm_stats: Dictionary with 'mean' and 'std'
            clip_bounds: Optional tuple (min, max) for clipping

        Returns:
            INormalizer instance (ZScore + Clip, or NoOp)
        """
        # Import here to avoid circular dependency
        from ..normalizers import ZScoreNormalizer, ClipNormalizer, CompositeNormalizer, NoOpNormalizer

        # If no stats, use NoOp
        if not norm_stats or norm_stats.get('std', 1.0) <= 0:
            return NoOpNormalizer()

        # Create z-score normalizer
        mean = norm_stats.get('mean', 0.0)
        std = norm_stats.get('std', 1.0)
        zscore = ZScoreNormalizer(mean=mean, std=std)

        # Add clipping if bounds provided
        if clip_bounds:
            clip = ClipNormalizer(min_val=clip_bounds[0], max_val=clip_bounds[1])
            return CompositeNormalizer(normalizers=[zscore, clip])

        return zscore

    @classmethod
    def get_registered_types(cls) -> list:
        """
        Get list of all registered normalizer types.

        Returns:
            List of registered normalizer type names
        """
        return list(cls._normalizers.keys())

    @classmethod
    def is_registered(cls, normalizer_type: str) -> bool:
        """
        Check if a normalizer type is registered.

        Args:
            normalizer_type: Normalizer type to check

        Returns:
            True if registered, False otherwise
        """
        return normalizer_type in cls._normalizers
