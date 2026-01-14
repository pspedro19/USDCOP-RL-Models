"""
Data utilities module.
Safe data operations without look-ahead bias or data leakage.
"""

from .safe_merge import safe_ffill, safe_merge_macro, validate_no_future_data

__all__ = ["safe_ffill", "safe_merge_macro", "validate_no_future_data"]
