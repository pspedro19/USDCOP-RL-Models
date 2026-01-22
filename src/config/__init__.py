"""
Config Module - Single Source of Truth (SSOT)
=============================================
Centralized configuration management for the USDCOP trading system.

Key Components:
- TradingConfig: Master config for all trading parameters (from config.yaml)
- SecuritySettings: Secure credential handling (CTR-007)

Usage:
    from src.config import load_trading_config, get_trading_config

    # At startup
    config = load_trading_config()

    # Anywhere else
    config = get_trading_config()
    print(config.ppo.gamma)           # 0.90
    print(config.thresholds.long)     # 0.33
"""

# Trading config (SSOT for PPO, thresholds, costs, risk)
from .trading_config import (
    # Main classes
    TradingConfig,
    PPOHyperparameters,
    ThresholdConfig,
    CostConfig,
    RiskConfig,
    RewardConfig,
    FeatureConfig,
    DateConfig,
    # Functions
    load_trading_config,
    get_trading_config,
    reset_trading_config,
    is_config_loaded,
    # Backward compatibility helpers
    get_ppo_hyperparameters,
    get_env_config,
    get_reward_config,
    # Exceptions
    ConfigNotLoadedError,
    ConfigVersionMismatchError,
    ConfigValidationError,
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
    # Trading Config (SSOT)
    "TradingConfig",
    "PPOHyperparameters",
    "ThresholdConfig",
    "CostConfig",
    "RiskConfig",
    "RewardConfig",
    "FeatureConfig",
    "DateConfig",
    "load_trading_config",
    "get_trading_config",
    "reset_trading_config",
    "is_config_loaded",
    "get_ppo_hyperparameters",
    "get_env_config",
    "get_reward_config",
    "ConfigNotLoadedError",
    "ConfigVersionMismatchError",
    "ConfigValidationError",
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
