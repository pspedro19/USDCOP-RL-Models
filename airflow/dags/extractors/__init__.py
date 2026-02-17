# -*- coding: utf-8 -*-
"""
Extractors Layer (SSOT)
=======================
Single Source of Truth for all data extraction logic.

Principles:
- DRY: One extractor per source, reused everywhere
- SOLID: Single responsibility per class
- KISS: Simple, predictable interfaces
"""

from .base import BaseExtractor, ExtractionResult
from .registry import ExtractorRegistry

__all__ = [
    "BaseExtractor",
    "ExtractionResult",
    "ExtractorRegistry",
]
