"""
Config Module - Single Source of Truth (SSOT)
=============================================
Centralized configuration management for the USDCOP trading system.

SSOT Architecture:
    config/experiment_ssot.yaml  <-- SINGLE SOURCE OF TRUTH
           |
           v
    experiment_loader.py    <-- Loads, validates, caches
           |
           +---> get_gamma()
           +---> get_ppo_hyperparameters()
           +---> get_reward_weights()
           +---> get_environment_config()

Usage:
    from src.config import get_gamma, get_ppo_hyperparameters

    gamma = get_gamma()  # 0.95 from SSOT
    hyperparams = get_ppo_hyperparameters()  # Full dict for SB3

Author: Trading Team
Version: 2.0.0 (Consolidated SSOT)
Date: 2026-02-02
"""

# Experiment SSOT (L2 + L3 unified config) - PRIMARY
from .experiment_loader import (
    # Data classes
    ExperimentConfig,
    FeatureConfig,
    PipelineConfig,
    EnvironmentConfig,
    TrainingConfig,
    RewardConfig,
    AntiLeakageConfig,
    LoggingConfig,
    # FASE 2: Overfitting prevention configs
    LRDecayConfig,
    EarlyStoppingConfig,
    # FASE 3: Rolling window config
    RollingWindowConfig,
    # Main loader
    load_experiment_config,
    # Convenience functions - USE THESE
    get_gamma,
    get_learning_rate,
    get_ent_coef,
    get_batch_size,
    get_ppo_hyperparameters,
    get_reward_weights,
    get_environment_config,
    get_feature_order,
    get_observation_dim,
    get_training_config,
    get_reward_config,
    get_feature_order_hash,
    validate_feature_order,
    # FASE 2: Overfitting prevention
    get_lr_decay_config,
    get_early_stopping_config,
    get_overfitting_prevention_config,
    # FASE 3: Rolling windows
    get_rolling_window_config,
    is_rolling_training_enabled,
    get_rolling_window_months,
)

# Pipeline SSOT (L2 + L3 + L4 unified) - NEW v2.0
from .pipeline_config import (
    PipelineConfig as PipelineSSOTConfig,  # Renamed to avoid conflict
    load_pipeline_config,
    get_feature_order as get_pipeline_feature_order,
    get_observation_dim as get_pipeline_observation_dim,
    validate_parity,
    # Data classes
    FeatureDefinition as PipelineFeatureDefinition,
    PPOConfig,
    EnvironmentConfig as PipelineEnvironmentConfig,
    RewardConfig as PipelineRewardConfig,
    BacktestConfig as PipelineBacktestConfig,
    DateRanges,
    PathsConfig,
)

# Security settings
from .security import SecuritySettings, SecurityError, get_secure_db_url

# Trading flags (kill switch, maintenance mode, etc.) - SSOT
from .trading_flags import (
    # Enums
    TradingMode,
    Environment,
    # Main class (SSOT)
    TradingFlags,
    TradingFlagsEnv,  # Backward compatibility alias
    # Singleton functions
    get_trading_flags,
    reload_trading_flags,
    reset_trading_flags,
    reset_trading_flags_cache,
    # Kill switch functions
    activate_kill_switch,
    deactivate_kill_switch,
    # Convenience functions
    is_live_trading_enabled,
    is_paper_trading_enabled,
    is_kill_switch_active,
    get_current_environment,
    get_current_trading_mode,
    # Backward compatibility
    get_trading_flags_env,
    reload_trading_flags_env,
)

__all__ = [
    # Experiment SSOT (L2 + L3) - PRIMARY
    "ExperimentConfig",
    "FeatureConfig",
    "PipelineConfig",
    "EnvironmentConfig",
    "TrainingConfig",
    "RewardConfig",
    "AntiLeakageConfig",
    "LoggingConfig",
    # FASE 2: Overfitting prevention configs
    "LRDecayConfig",
    "EarlyStoppingConfig",
    # FASE 3: Rolling window config
    "RollingWindowConfig",
    "load_experiment_config",
    # Convenience functions
    "get_gamma",
    "get_learning_rate",
    "get_ent_coef",
    "get_batch_size",
    "get_ppo_hyperparameters",
    "get_reward_weights",
    "get_environment_config",
    "get_feature_order",
    "get_observation_dim",
    "get_training_config",
    "get_reward_config",
    "get_feature_order_hash",
    "validate_feature_order",
    # FASE 2: Overfitting prevention
    "get_lr_decay_config",
    "get_early_stopping_config",
    "get_overfitting_prevention_config",
    # FASE 3: Rolling windows
    "get_rolling_window_config",
    "is_rolling_training_enabled",
    "get_rolling_window_months",
    # Pipeline SSOT (L2 + L3 + L4) - v2.0
    "PipelineSSOTConfig",
    "load_pipeline_config",
    "get_pipeline_feature_order",
    "get_pipeline_observation_dim",
    "validate_parity",
    "PipelineFeatureDefinition",
    "PPOConfig",
    "PipelineEnvironmentConfig",
    "PipelineRewardConfig",
    "PipelineBacktestConfig",
    "DateRanges",
    "PathsConfig",
    # Security
    "SecuritySettings",
    "SecurityError",
    "get_secure_db_url",
    # Trading Flags (SSOT)
    "TradingMode",
    "Environment",
    "TradingFlags",
    "TradingFlagsEnv",
    "get_trading_flags",
    "reload_trading_flags",
    "reset_trading_flags",
    "reset_trading_flags_cache",
    "activate_kill_switch",
    "deactivate_kill_switch",
    "is_live_trading_enabled",
    "is_paper_trading_enabled",
    "is_kill_switch_active",
    "get_current_environment",
    "get_current_trading_mode",
    "get_trading_flags_env",
    "reload_trading_flags_env",
]
