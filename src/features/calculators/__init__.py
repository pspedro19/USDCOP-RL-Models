"""
Feature Calculators - Individual calculator modules.
Contrato: CTR-005
CLAUDE-T2 | Plan Item: P1-13 (subtasks)

Each calculator is independently testable and deterministic.
"""

from . import returns
from . import rsi
from . import atr
from . import adx
from . import macro

__all__ = ["returns", "rsi", "atr", "adx", "macro"]
