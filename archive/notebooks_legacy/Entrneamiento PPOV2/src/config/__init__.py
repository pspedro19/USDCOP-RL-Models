"""
USD/COP RL Trading System - Configuration Package
==================================================

Sistema de configuracion centralizado con:
- Defaults sensatos basados en PRODUCTION_CONFIG.json validado
- Dataclasses tipados con validacion
- Soporte para carga desde JSON/YAML
- Compatibilidad con sistema actual

Author: Claude Code
Version: 1.0.0
"""

from .defaults import (
    PRODUCTION_DEFAULTS,
    PPO_DEFAULTS,
    ENVIRONMENT_DEFAULTS,
    REWARD_DEFAULTS,
    VALIDATION_DEFAULTS,
    RISK_DEFAULTS,
    CALLBACK_DEFAULTS,
    DATA_DEFAULTS,
)

from .training_config import (
    # Enums
    TrainingPhase,
    RewardType,
    ProtectionMode,

    # Core configs
    PPOConfig,
    SACConfig,
    NetworkConfig,
    EnvironmentConfig,
    RewardConfig,
    ValidationConfig,
    CallbackConfig,
    RiskConfig,
    AcceptanceConfig,
    DataConfig,

    # Main config
    TrainingConfigV19,

    # Factory functions
    load_config,
    create_production_config,
    create_debug_config,
    create_stress_test_config,
)

__all__ = [
    # Defaults
    'PRODUCTION_DEFAULTS',
    'PPO_DEFAULTS',
    'ENVIRONMENT_DEFAULTS',
    'REWARD_DEFAULTS',
    'VALIDATION_DEFAULTS',
    'RISK_DEFAULTS',
    'CALLBACK_DEFAULTS',
    'DATA_DEFAULTS',

    # Enums
    'TrainingPhase',
    'RewardType',
    'ProtectionMode',

    # Configs
    'PPOConfig',
    'SACConfig',
    'NetworkConfig',
    'EnvironmentConfig',
    'RewardConfig',
    'ValidationConfig',
    'CallbackConfig',
    'RiskConfig',
    'AcceptanceConfig',
    'DataConfig',
    'TrainingConfigV19',

    # Functions
    'load_config',
    'create_production_config',
    'create_debug_config',
    'create_stress_test_config',
]
