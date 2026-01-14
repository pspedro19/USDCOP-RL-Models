"""
USD/COP RL Trading System - Validation Framework
=================================================

Framework de validación robusto para modelos de trading.
"""

from .purged_cv import PurgedKFoldCV, WalkForwardValidator, TimeSeriesSplit
from .metrics import TradingMetrics, calculate_all_metrics
from .stress_testing import StressTester, CrisisPeriodsValidator

# Stubs para módulos no implementados
class MultiSeedTrainer:
    """Stub - not implemented."""
    def __init__(self, *args, **kwargs): pass

class BootstrapConfidenceInterval:
    """Stub - not implemented."""
    def __init__(self, *args, **kwargs): pass

class RobustnessReport:
    """Stub - not implemented."""
    def __init__(self, *args, **kwargs): pass

__all__ = [
    'PurgedKFoldCV',
    'WalkForwardValidator',
    'TimeSeriesSplit',
    'TradingMetrics',
    'calculate_all_metrics',
    'MultiSeedTrainer',
    'BootstrapConfidenceInterval',
    'RobustnessReport',
    'StressTester',
    'CrisisPeriodsValidator',
]
