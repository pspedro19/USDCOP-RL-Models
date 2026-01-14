"""
Feature Normalizers Package.

Provides Strategy Pattern implementation for different normalization methods.

Design Patterns:
- Strategy Pattern: Different normalization algorithms are interchangeable
- Factory Pattern: NormalizerFactory creates appropriate normalizer instances

Usage:
    from features.normalizers import NormalizerFactory, ZScoreNormalizer

    # Using factory
    normalizer = NormalizerFactory.create("zscore", mean=0.0, std=1.0)
    result = normalizer.normalize(value)

    # Direct instantiation
    normalizer = ZScoreNormalizer(mean=0.0, std=1.0, clip=(-5.0, 5.0))
    result = normalizer.normalize(value)

Contrato: CTR-006
"""

from .factory import (
    Normalizer,
    ZScoreNormalizer,
    MinMaxNormalizer,
    ClipNormalizer,
    NoOpNormalizer,
    NormalizerFactory,
)

__all__ = [
    # Protocol
    "Normalizer",
    # Implementations
    "ZScoreNormalizer",
    "MinMaxNormalizer",
    "ClipNormalizer",
    "NoOpNormalizer",
    # Factory
    "NormalizerFactory",
]
