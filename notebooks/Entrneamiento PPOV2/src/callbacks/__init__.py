"""
USD/COP RL Trading System - Training Callbacks
===============================================

Módulo de callbacks para monitoreo y control del training.
"""

from .sharpe_eval import SharpeEvalCallback
from .action_monitor import ActionDistributionCallback

# Stubs para callbacks no implementados (backward compat)
class EntropySchedulerCallback:
    """Stub - not implemented."""
    def __init__(self, *args, **kwargs): pass

class CostCurriculumCallback:
    """Stub - not implemented."""
    def __init__(self, *args, **kwargs): pass

__all__ = [
    'SharpeEvalCallback',
    'EntropySchedulerCallback',
    'ActionDistributionCallback',
    'CostCurriculumCallback',
]
