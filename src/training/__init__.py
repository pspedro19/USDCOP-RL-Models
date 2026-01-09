"""
Training module for USDCOP RL models.

V20 Components:
- RewardCalculatorV20: Corrected reward function with proper order of operations
"""

from .reward_calculator_v20 import RewardCalculatorV20

__all__ = ['RewardCalculatorV20']
