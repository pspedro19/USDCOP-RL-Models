"""
CompositeNormalizer - Composite Normalization Strategy
=======================================================

Implements Composite Pattern for chaining multiple normalizers.

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Date: 2025-12-17
"""

import pandas as pd
from typing import Union, List
from ..interfaces.normalizer import INormalizer


class CompositeNormalizer(INormalizer):
    """
    Composite normalizer that applies multiple normalizers in sequence.

    Implements Composite Pattern for chaining normalizations.

    Example:
        zscore = ZScoreNormalizer(mean=100.0, std=10.0)
        clip = ClipNormalizer(min_val=-4.0, max_val=4.0)
        composite = CompositeNormalizer(normalizers=[zscore, clip])

        # Applies z-score then clipping
        normalized = composite.normalize(120.0)  # (120-100)/10 = 2.0, clip to [-4,4] = 2.0
    """

    def __init__(self, normalizers: List[INormalizer]):
        """
        Initialize composite normalizer.

        Args:
            normalizers: List of normalizers to apply in sequence
        """
        self.normalizers = normalizers

    def normalize(self, value: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Apply all normalizers in sequence.

        Args:
            value: Input value or series

        Returns:
            Normalized value after applying all normalizers
        """
        result = value
        for normalizer in self.normalizers:
            result = normalizer.normalize(result)
        return result

    def denormalize(self, value: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Apply denormalization in reverse order.

        Args:
            value: Normalized value or series

        Returns:
            Denormalized value
        """
        result = value
        # Apply in reverse order
        for normalizer in reversed(self.normalizers):
            result = normalizer.denormalize(result)
        return result

    def get_params(self) -> dict:
        """
        Get parameters from all normalizers.

        Returns:
            Dictionary with composite type and sub-normalizer params
        """
        return {
            'type': 'composite',
            'normalizers': [n.get_params() for n in self.normalizers]
        }

    def add_normalizer(self, normalizer: INormalizer) -> None:
        """
        Add another normalizer to the chain.

        Args:
            normalizer: Normalizer to add
        """
        self.normalizers.append(normalizer)

    def __repr__(self) -> str:
        normalizer_reprs = [repr(n) for n in self.normalizers]
        return f"CompositeNormalizer([{', '.join(normalizer_reprs)}])"
