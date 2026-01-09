"""V19 Configuration Module."""
from .training_config_v19 import (
    TrainingConfigV19,
    RewardConfig,
    EnvironmentConfig,
    CallbacksConfig,
    PPOConfig,
    SACConfig,
    NetworkConfig,
    ValidationConfig,
    AcceptanceCriteria,
    TrainingPhase,
    MarketRegime,
    CrisisPeriod,
    CRISIS_PERIODS,
    get_default_config,
    get_quick_test_config,
    load_config,
)

try:
    from .settings import *
except ImportError:
    pass  # settings.py es opcional
