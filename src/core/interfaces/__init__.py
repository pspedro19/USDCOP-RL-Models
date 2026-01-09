"""
Core Interfaces for USD/COP Trading System
==========================================

Defines abstract interfaces for dependency injection and SOLID principles.

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Date: 2025-12-17
"""

from .feature_calculator import IFeatureCalculator
from .normalizer import INormalizer
from .observation_builder import IObservationBuilder
from .config_loader import IConfigLoader

__all__ = [
    'IFeatureCalculator',
    'INormalizer',
    'IObservationBuilder',
    'IConfigLoader',
]
