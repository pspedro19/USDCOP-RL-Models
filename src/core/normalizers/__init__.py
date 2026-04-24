"""
Core Normalizers for USD/COP Trading System
============================================

Strategy Pattern implementations for different normalization approaches.

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Date: 2025-01-07
"""

from .clip_normalizer import ClipNormalizer
from .composite_normalizer import CompositeNormalizer
from .noop_normalizer import NoOpNormalizer
from .zscore_normalizer import ZScoreNormalizer, create_zscore_normalizer

__all__ = [
    'ClipNormalizer',
    'CompositeNormalizer',
    'NoOpNormalizer',
    'ZScoreNormalizer',
    'create_zscore_normalizer',
]
