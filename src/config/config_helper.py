"""
Configuration Helper Functions - Convenient SSOT Access
========================================================
Provides quick access to commonly used configuration values.

Usage:
    from src.config.config_helper import (
        get_gamma,
        get_thresholds,
        get_costs,
        get_all_ppo_hyperparams,
    )

    gamma = get_gamma()  # 0.90
    long_thresh, short_thresh = get_thresholds()  # (0.33, -0.33)
    tx_cost, slippage = get_costs()  # (75.0, 15.0)

Author: Trading Team
Version: 1.0.0
Date: 2026-01-14
"""

from typing import Dict, Tuple, Any
from functools import lru_cache

from src.config.trading_config import (
    get_trading_config,
    load_trading_config,
    ConfigNotLoadedError,
)


# =============================================================================
# INITIALIZATION
# =============================================================================

def ensure_config_loaded() -> None:
    """Ensure config is loaded. Call this at application startup."""
    try:
        get_trading_config()
    except ConfigNotLoadedError:
        load_trading_config("current")


# =============================================================================
# PPO HYPERPARAMETERS
# =============================================================================

def get_gamma() -> float:
    """Get discount factor gamma from SSOT.

    Returns:
        float: Discount factor (default 0.90)
    """
    config = get_trading_config()
    return config.ppo.gamma


def get_learning_rate() -> float:
    """Get learning rate from SSOT.

    Returns:
        float: Learning rate (default 3e-4)
    """
    config = get_trading_config()
    return config.ppo.learning_rate


def get_ent_coef() -> float:
    """Get entropy coefficient from SSOT.

    Returns:
        float: Entropy coefficient (default 0.01)
    """
    config = get_trading_config()
    return config.ppo.ent_coef


def get_all_ppo_hyperparams() -> Dict[str, Any]:
    """Get all PPO hyperparameters as a dictionary.

    Returns:
        Dict with keys: learning_rate, n_steps, batch_size, n_epochs,
                        gamma, gae_lambda, clip_range, ent_coef
    """
    config = get_trading_config()
    ppo = config.ppo
    return {
        "learning_rate": ppo.learning_rate,
        "n_steps": ppo.n_steps,
        "batch_size": ppo.batch_size,
        "n_epochs": ppo.n_epochs,
        "gamma": ppo.gamma,
        "gae_lambda": ppo.gae_lambda,
        "clip_range": ppo.clip_range,
        "ent_coef": ppo.ent_coef,
    }


# =============================================================================
# THRESHOLDS
# =============================================================================

def get_thresholds() -> Tuple[float, float]:
    """Get trading thresholds from SSOT.

    Returns:
        Tuple[float, float]: (threshold_long, threshold_short)
                             Default: (0.33, -0.33)
    """
    config = get_trading_config()
    return config.thresholds.long, config.thresholds.short


def get_threshold_long() -> float:
    """Get long entry threshold from SSOT.

    Returns:
        float: Long threshold (default 0.33)
    """
    config = get_trading_config()
    return config.thresholds.long


def get_threshold_short() -> float:
    """Get short entry threshold from SSOT.

    Returns:
        float: Short threshold (default -0.33)
    """
    config = get_trading_config()
    return config.thresholds.short


# =============================================================================
# COSTS
# =============================================================================

def get_costs() -> Tuple[float, float]:
    """Get transaction costs from SSOT.

    Returns:
        Tuple[float, float]: (transaction_cost_bps, slippage_bps)
                             Default: (75.0, 15.0)
    """
    config = get_trading_config()
    return config.costs.transaction_cost_bps, config.costs.slippage_bps


def get_transaction_cost_bps() -> float:
    """Get transaction cost in basis points from SSOT.

    Returns:
        float: Transaction cost in bps (default 75.0)
    """
    config = get_trading_config()
    return config.costs.transaction_cost_bps


def get_slippage_bps() -> float:
    """Get slippage in basis points from SSOT.

    Returns:
        float: Slippage in bps (default 15.0)
    """
    config = get_trading_config()
    return config.costs.slippage_bps


# =============================================================================
# RISK PARAMETERS
# =============================================================================

def get_risk_params() -> Dict[str, Any]:
    """Get risk management parameters from SSOT.

    Returns:
        Dict with risk configuration values
    """
    config = get_trading_config()
    risk = config.risk
    return {
        "max_drawdown_pct": risk.max_drawdown_pct,
        "max_daily_loss_pct": risk.max_daily_loss_pct,
        "max_consecutive_losses": risk.max_consecutive_losses,
        "cooldown_bars": risk.cooldown_bars,
        "volatility_filter_enabled": risk.volatility_filter_enabled,
        "atr_threshold": risk.atr_threshold,
    }


def get_max_drawdown_pct() -> float:
    """Get maximum drawdown percentage from SSOT.

    Returns:
        float: Max drawdown percentage (default 15.0)
    """
    config = get_trading_config()
    return config.risk.max_drawdown_pct


def get_max_consecutive_losses() -> int:
    """Get maximum consecutive losses before cooldown from SSOT.

    Returns:
        int: Max consecutive losses (default 5)
    """
    config = get_trading_config()
    return config.risk.max_consecutive_losses


# =============================================================================
# REWARD PARAMETERS
# =============================================================================

def get_reward_params() -> Dict[str, Any]:
    """Get reward calculation parameters from SSOT.

    Returns:
        Dict with reward configuration values
    """
    config = get_trading_config()
    reward = config.reward
    return {
        "loss_penalty_multiplier": reward.loss_penalty_multiplier,
        "time_decay_start": reward.time_decay_start,
        "holding_cost_per_bar": reward.holding_cost_per_bar,
        "profit_scaling": reward.profit_scaling,
    }


# =============================================================================
# FEATURES
# =============================================================================

def get_feature_config() -> Dict[str, Any]:
    """Get feature configuration from SSOT.

    Returns:
        Dict with feature configuration values
    """
    config = get_trading_config()
    features = config.features
    return {
        "observation_dim": features.observation_dim,
        "market_feature_count": features.market_feature_count,
        "state_feature_count": features.state_feature_count,
        "clip_value": features.clip_value,
    }


def get_observation_dim() -> int:
    """Get observation dimension from SSOT.

    Returns:
        int: Observation dimension (default 15)
    """
    config = get_trading_config()
    return config.features.observation_dim


# =============================================================================
# FULL CONFIG ACCESS
# =============================================================================

def get_full_config_dict() -> Dict[str, Any]:
    """Get all configuration values as a nested dictionary.

    Returns:
        Dict with all configuration sections
    """
    config = get_trading_config()
    return {
        "ppo": get_all_ppo_hyperparams(),
        "thresholds": {
            "long": config.thresholds.long,
            "short": config.thresholds.short,
        },
        "costs": {
            "transaction_cost_bps": config.costs.transaction_cost_bps,
            "slippage_bps": config.costs.slippage_bps,
        },
        "risk": get_risk_params(),
        "reward": get_reward_params(),
        "features": get_feature_config(),
    }
