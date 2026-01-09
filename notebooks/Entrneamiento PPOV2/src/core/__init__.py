# USD/COP RL Trading System - Core Interfaces
# =============================================
# Protocol definitions for dependency injection and testing.

"""
Core module containing Protocol interfaces for the trading system.

This module provides abstract interfaces (using Python's Protocol pattern)
that define contracts for the main system components:

- IRewardFunction: Reward calculation for RL agents
- IRegimeDetector: Market regime detection and classification
- IRiskManager: Risk management and kill switches
- ICostModel: Transaction cost modeling

These interfaces enable:
1. Dependency Injection: Components can be swapped without code changes
2. Testing: Easy mocking and stubbing for unit tests
3. Documentation: Clear contracts for implementers
4. Type Safety: Static analysis via mypy/pyright

Example usage:
    ```python
    from src.core import IRewardFunction, ICostModel

    def train_agent(
        reward_fn: IRewardFunction,
        cost_model: ICostModel
    ) -> Model:
        # reward_fn and cost_model are typed interfaces
        ...
    ```
"""

__version__ = '1.0.0'

from .interfaces import (
    # Main Protocols
    IRewardFunction,
    IRegimeDetector,
    IRiskManager,
    ICostModel,
    # Supporting types
    RewardResult,
    RiskUpdateResult,
)

__all__ = [
    # Main Protocols
    'IRewardFunction',
    'IRegimeDetector',
    'IRiskManager',
    'ICostModel',
    # Supporting types
    'RewardResult',
    'RiskUpdateResult',
]
