"""
Trading Configuration - Single Source of Truth (SSOT)
======================================================
Centralized configuration for all trading parameters across pipelines.

Design Patterns Used:
- Singleton: Single config instance per process (thread-safe)
- Factory: Version-based config creation
- Immutable: Frozen dataclasses prevent runtime changes
- Fail-Fast: Validation at load time catches misconfigurations early

Architecture:
    config/trading_config.yaml  <-- SINGLE SOURCE OF TRUTH
           |
           v
    TradingConfigLoader    <-- Loads, validates, caches
           |
           +---> PPOHyperparameters (gamma, ent_coef, etc.)
           +---> ThresholdConfig (long, short)
           +---> CostConfig (transaction_cost_bps, slippage_bps)
           +---> RiskConfig (circuit breaker, volatility filter)
           +---> RewardConfig (loss_penalty, time_decay, etc.)

Usage:
    from src.config.trading_config import load_trading_config, get_trading_config

    # At application startup (once)
    config = load_trading_config()

    # Anywhere else
    config = get_trading_config()
    print(config.ppo.gamma)           # 0.90
    print(config.thresholds.long)     # 0.33
    print(config.costs.transaction_cost_bps)  # 75.0

Author: Trading Team
Version: 1.0.0
Date: 2026-01-13
"""

import logging
import threading
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class ConfigNotLoadedError(RuntimeError):
    """Raised when trying to access config before loading."""
    pass


class ConfigVersionMismatchError(RuntimeError):
    """Raised when trying to load a different version after initial load."""
    pass


class ConfigValidationError(ValueError):
    """Raised when config fails validation."""
    pass


# =============================================================================
# IMMUTABLE CONFIG DATACLASSES (Frozen = cannot be modified after creation)
# =============================================================================

@dataclass(frozen=True)
class PPOHyperparameters:
    """PPO algorithm hyperparameters - IMMUTABLE."""
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    total_timesteps: int
    eval_freq: int = 25000
    checkpoint_freq: int = 50000

    def __post_init__(self):
        """Validate hyperparameters."""
        if not 0 < self.gamma < 1:
            raise ConfigValidationError(f"gamma must be in (0,1), got {self.gamma}")
        if not 0 < self.ent_coef < 1:
            raise ConfigValidationError(f"ent_coef must be in (0,1), got {self.ent_coef}")
        if self.batch_size <= 0:
            raise ConfigValidationError(f"batch_size must be positive, got {self.batch_size}")


@dataclass(frozen=True)
class ThresholdConfig:
    """Action thresholds - IMMUTABLE."""
    long: float
    short: float
    confidence_min: float = 0.6

    def __post_init__(self):
        """Validate thresholds."""
        if self.long <= 0:
            raise ConfigValidationError(f"threshold_long must be positive, got {self.long}")
        if self.short >= 0:
            raise ConfigValidationError(f"threshold_short must be negative, got {self.short}")
        if abs(self.long) != abs(self.short):
            logger.warning(f"Thresholds are not symmetric: long={self.long}, short={self.short}")


@dataclass(frozen=True)
class CostConfig:
    """Transaction costs - IMMUTABLE."""
    transaction_cost_bps: float
    slippage_bps: float
    initial_capital: float = 10000.0
    max_position_size: float = 1.0

    def __post_init__(self):
        """Validate costs."""
        if self.transaction_cost_bps < 50:
            logger.warning(
                f"transaction_cost_bps={self.transaction_cost_bps} is very low for USDCOP. "
                f"Real spreads are typically 70-100 bps."
            )

    @property
    def total_round_trip_bps(self) -> float:
        """Total cost for a round-trip trade (open + close)."""
        return 2 * (self.transaction_cost_bps + self.slippage_bps)


@dataclass(frozen=True)
class RiskConfig:
    """Risk management parameters - IMMUTABLE."""
    max_drawdown_pct: float
    daily_loss_limit_pct: float
    position_limit: float
    # Circuit breaker
    max_consecutive_losses: int
    cooldown_bars_after_losses: int
    # Volatility filter
    enable_volatility_filter: bool
    max_atr_multiplier: float

    def __post_init__(self):
        """Validate risk parameters."""
        if self.max_consecutive_losses <= 0:
            raise ConfigValidationError(
                f"max_consecutive_losses must be positive, got {self.max_consecutive_losses}"
            )
        if self.cooldown_bars_after_losses <= 0:
            raise ConfigValidationError(
                f"cooldown_bars_after_losses must be positive, got {self.cooldown_bars_after_losses}"
            )


@dataclass(frozen=True)
class RewardConfig:
    """Reward function parameters - IMMUTABLE."""
    # Transaction cost in reward
    transaction_cost_pct: float
    # Loss penalty
    loss_penalty_multiplier: float
    # Hold bonus (can be 0 to disable)
    hold_bonus_per_bar: float
    hold_bonus_requires_profit: bool
    max_hold_bonus_bars: int
    # Consistency bonus
    consecutive_win_bonus: float
    max_consecutive_bonus: int
    # Drawdown penalty
    drawdown_penalty_threshold: float
    drawdown_penalty_multiplier: float
    # Intratrade drawdown
    intratrade_dd_penalty: float
    max_intratrade_dd: float
    # Time decay
    time_decay_start_bars: int
    time_decay_per_bar: float
    time_decay_losing_multiplier: float
    # Clipping
    min_reward: float
    max_reward: float


@dataclass(frozen=True)
class FeatureConfig:
    """Feature configuration - IMMUTABLE."""
    norm_stats_path: str
    clip_range: Tuple[float, float]
    warmup_bars: int
    core_features: Tuple[str, ...]
    state_features: Tuple[str, ...]

    @property
    def observation_dim(self) -> int:
        """Total observation dimension."""
        return len(self.core_features) + len(self.state_features)

    @property
    def feature_order(self) -> Tuple[str, ...]:
        """Complete feature order for observation building."""
        return self.core_features + self.state_features


@dataclass(frozen=True)
class DateConfig:
    """Date ranges for train/val/test - IMMUTABLE."""
    training_start: str
    training_end: str
    validation_start: str
    validation_end: str
    test_start: str


@dataclass(frozen=True)
class TradingConfig:
    """
    Master configuration container - IMMUTABLE.

    This is the Single Source of Truth for all trading parameters.
    All modules should import and use this config instead of hardcoding values.
    """
    version: str
    model_name: str
    observation_dim: int
    action_space: int

    # Sub-configs
    ppo: PPOHyperparameters
    thresholds: ThresholdConfig
    costs: CostConfig
    risk: RiskConfig
    reward: RewardConfig
    features: FeatureConfig
    dates: DateConfig

    def __post_init__(self):
        """Final validation across all sub-configs."""
        # Verify observation_dim matches features
        expected_dim = self.features.observation_dim
        if self.observation_dim != expected_dim:
            raise ConfigValidationError(
                f"observation_dim mismatch: config says {self.observation_dim}, "
                f"but features imply {expected_dim}"
            )
        logger.info(f"TradingConfig v{self.version} loaded and validated successfully")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "version": self.version,
            "model_name": self.model_name,
            "observation_dim": self.observation_dim,
            "ppo": {
                "gamma": self.ppo.gamma,
                "ent_coef": self.ppo.ent_coef,
                "learning_rate": self.ppo.learning_rate,
                "total_timesteps": self.ppo.total_timesteps,
            },
            "thresholds": {
                "long": self.thresholds.long,
                "short": self.thresholds.short,
            },
            "costs": {
                "transaction_cost_bps": self.costs.transaction_cost_bps,
                "slippage_bps": self.costs.slippage_bps,
            },
            "risk": {
                "max_consecutive_losses": self.risk.max_consecutive_losses,
                "cooldown_bars": self.risk.cooldown_bars_after_losses,
                "volatility_filter": self.risk.enable_volatility_filter,
            },
        }


# =============================================================================
# SINGLETON CONFIG LOADER (Thread-Safe)
# =============================================================================

class TradingConfigLoader:
    """
    Singleton loader for trading configuration.

    Thread-safe implementation using double-checked locking pattern.
    Once loaded, the config cannot be changed without explicitly resetting.
    """
    _instance: Optional["TradingConfigLoader"] = None
    _lock = threading.Lock()
    _config: Optional[TradingConfig] = None
    _version: Optional[str] = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def load(cls, version: str = "current", config_dir: Optional[Path] = None) -> TradingConfig:
        """
        Load configuration from YAML file.

        Args:
            version: Config version (e.g., "current", "v1")
            config_dir: Directory containing config files (default: project/config/)

        Returns:
            TradingConfig instance

        Raises:
            ConfigVersionMismatchError: If trying to load different version
            FileNotFoundError: If config file doesn't exist
            ConfigValidationError: If config fails validation
        """
        instance = cls()

        with cls._lock:
            # Check if already loaded with different version
            if cls._config is not None:
                if cls._version != version:
                    raise ConfigVersionMismatchError(
                        f"Config already loaded with version={cls._version}. "
                        f"Cannot change to {version}. Call reset() first or restart process."
                    )
                return cls._config

            # Determine config path
            if config_dir is None:
                config_dir = Path(__file__).parent.parent.parent / "config"

            config_path = config_dir / f"{version}_config.yaml"

            if not config_path.exists():
                raise FileNotFoundError(
                    f"Config file not found: {config_path}. "
                    f"Available versions: {[f.stem for f in config_dir.glob('v*_config.yaml')]}"
                )

            # Load YAML
            logger.info(f"Loading trading config from {config_path}")
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            # Build config from YAML data
            cls._config = cls._build_config(data, version)
            cls._version = version

            return cls._config

    @classmethod
    def _build_config(cls, data: Dict[str, Any], version: str) -> TradingConfig:
        """Build TradingConfig from YAML data."""

        # Build sub-configs
        ppo = PPOHyperparameters(
            learning_rate=data["training"]["learning_rate"],
            n_steps=data["training"]["n_steps"],
            batch_size=data["training"]["batch_size"],
            n_epochs=data["training"]["n_epochs"],
            gamma=data["training"]["gamma"],
            gae_lambda=data["training"]["gae_lambda"],
            clip_range=data["training"]["clip_range"],
            ent_coef=data["training"]["ent_coef"],
            total_timesteps=data["training"]["total_timesteps"],
            eval_freq=data["training"].get("eval_freq", 25000),
            checkpoint_freq=data["training"].get("checkpoint_freq", 50000),
        )

        thresholds = ThresholdConfig(
            long=data["thresholds"]["long"],
            short=data["thresholds"]["short"],
            confidence_min=data["thresholds"].get("confidence_min", 0.6),
        )

        costs = CostConfig(
            transaction_cost_bps=float(data["trading"]["transaction_cost_bps"]),
            slippage_bps=float(data["trading"]["slippage_bps"]),
            initial_capital=float(data["trading"].get("initial_capital", 10000)),
            max_position_size=float(data["trading"].get("max_position_size", 1.0)),
        )

        risk = RiskConfig(
            max_drawdown_pct=data["risk"]["max_drawdown_pct"],
            daily_loss_limit_pct=data["risk"]["daily_loss_limit_pct"],
            position_limit=data["risk"]["position_limit"],
            max_consecutive_losses=data["risk"]["max_consecutive_losses"],
            cooldown_bars_after_losses=data["risk"]["cooldown_bars_after_losses"],
            enable_volatility_filter=data["volatility"]["enable_filter"],
            max_atr_multiplier=data["volatility"]["max_atr_multiplier"],
        )

        reward_data = data["reward"]
        reward = RewardConfig(
            transaction_cost_pct=reward_data["transaction_cost_pct"],
            loss_penalty_multiplier=reward_data["loss_penalty_multiplier"],
            hold_bonus_per_bar=reward_data["hold_bonus_per_bar"],
            hold_bonus_requires_profit=reward_data["hold_bonus_requires_profit"],
            max_hold_bonus_bars=reward_data["max_hold_bonus_bars"],
            consecutive_win_bonus=reward_data["consecutive_win_bonus"],
            max_consecutive_bonus=reward_data["max_consecutive_bonus"],
            drawdown_penalty_threshold=reward_data["drawdown_penalty_threshold"],
            drawdown_penalty_multiplier=reward_data["drawdown_penalty_multiplier"],
            intratrade_dd_penalty=reward_data["intratrade_dd_penalty"],
            max_intratrade_dd=reward_data["max_intratrade_dd"],
            time_decay_start_bars=reward_data["time_decay_start_bars"],
            time_decay_per_bar=reward_data["time_decay_per_bar"],
            time_decay_losing_multiplier=reward_data["time_decay_losing_multiplier"],
            min_reward=reward_data["min_reward"],
            max_reward=reward_data["max_reward"],
        )

        features_data = data["features"]
        features = FeatureConfig(
            norm_stats_path=features_data["norm_stats_path"],
            clip_range=tuple(features_data["clip_range"]),
            warmup_bars=features_data["warmup_bars"],
            core_features=tuple(features_data["core_features"]),
            state_features=tuple(features_data["state_features"]),
        )

        dates = DateConfig(
            training_start=data["dates"]["training_start"],
            training_end=data["dates"]["training_end"],
            validation_start=data["dates"]["validation_start"],
            validation_end=data["dates"]["validation_end"],
            test_start=data["dates"]["test_start"],
        )

        # Build master config
        return TradingConfig(
            version=version,
            model_name=data["model"]["name"],
            observation_dim=data["model"]["observation_dim"],
            action_space=data["model"]["action_space"],
            ppo=ppo,
            thresholds=thresholds,
            costs=costs,
            risk=risk,
            reward=reward,
            features=features,
            dates=dates,
        )

    @classmethod
    def get(cls) -> TradingConfig:
        """
        Get the loaded configuration.

        Returns:
            TradingConfig instance

        Raises:
            ConfigNotLoadedError: If config hasn't been loaded yet
        """
        if cls._config is None:
            raise ConfigNotLoadedError(
                "Trading config not loaded. Call load_trading_config() first."
            )
        return cls._config

    @classmethod
    def is_loaded(cls) -> bool:
        """Check if config has been loaded."""
        return cls._config is not None

    @classmethod
    def get_version(cls) -> Optional[str]:
        """Get loaded config version."""
        return cls._version

    @classmethod
    def reset(cls) -> None:
        """
        Reset the loader to allow loading a different config.

        WARNING: This should only be used in tests or when explicitly
        changing configuration versions. In production, the config
        should be loaded once at startup.
        """
        with cls._lock:
            cls._config = None
            cls._version = None
            logger.warning("TradingConfigLoader reset. Next call will reload config.")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_trading_config(version: str = "current", config_dir: Optional[Path] = None) -> TradingConfig:
    """
    Load trading configuration.

    This is the primary entry point for loading configuration.
    Should be called once at application startup.

    Args:
        version: Config version (e.g., "current")
        config_dir: Optional config directory path

    Returns:
        TradingConfig instance

    Example:
        # At startup
        config = load_trading_config("current")

        # Later
        config = get_trading_config()
    """
    return TradingConfigLoader.load(version, config_dir)


def get_trading_config() -> TradingConfig:
    """
    Get the loaded trading configuration.

    Raises:
        ConfigNotLoadedError: If config hasn't been loaded

    Example:
        config = get_trading_config()
        gamma = config.ppo.gamma
        threshold = config.thresholds.long
    """
    return TradingConfigLoader.get()


def reset_trading_config() -> None:
    """
    Reset the config loader (for testing only).

    WARNING: Do not use in production code.
    """
    TradingConfigLoader.reset()


def is_config_loaded() -> bool:
    """Check if trading config has been loaded."""
    return TradingConfigLoader.is_loaded()


# =============================================================================
# BACKWARD COMPATIBILITY HELPERS
# =============================================================================

def get_ppo_hyperparameters() -> Dict[str, Any]:
    """
    Get PPO hyperparameters as dict (for stable-baselines3).

    Returns:
        Dict compatible with PPO constructor
    """
    config = get_trading_config()
    return {
        "learning_rate": config.ppo.learning_rate,
        "n_steps": config.ppo.n_steps,
        "batch_size": config.ppo.batch_size,
        "n_epochs": config.ppo.n_epochs,
        "gamma": config.ppo.gamma,
        "gae_lambda": config.ppo.gae_lambda,
        "clip_range": config.ppo.clip_range,
        "ent_coef": config.ppo.ent_coef,
    }


def get_env_config() -> Dict[str, Any]:
    """
    Get environment configuration as dict.

    Returns:
        Dict compatible with TradingEnvConfig
    """
    config = get_trading_config()
    return {
        "threshold_long": config.thresholds.long,
        "threshold_short": config.thresholds.short,
        "transaction_cost_bps": config.costs.transaction_cost_bps,
        "slippage_bps": config.costs.slippage_bps,
        "max_consecutive_losses": config.risk.max_consecutive_losses,
        "cooldown_bars_after_losses": config.risk.cooldown_bars_after_losses,
        "enable_volatility_filter": config.risk.enable_volatility_filter,
        "max_atr_multiplier": config.risk.max_atr_multiplier,
    }


def get_reward_config() -> Dict[str, Any]:
    """
    Get reward configuration as dict.

    Returns:
        Dict compatible with RewardConfig
    """
    config = get_trading_config()
    r = config.reward
    return {
        "transaction_cost_pct": r.transaction_cost_pct,
        "loss_penalty_multiplier": r.loss_penalty_multiplier,
        "hold_bonus_per_bar": r.hold_bonus_per_bar,
        "hold_bonus_requires_profit": r.hold_bonus_requires_profit,
        "consecutive_win_bonus": r.consecutive_win_bonus,
        "max_consecutive_bonus": r.max_consecutive_bonus,
        "drawdown_penalty_threshold": r.drawdown_penalty_threshold,
        "drawdown_penalty_multiplier": r.drawdown_penalty_multiplier,
        "intratrade_dd_penalty": r.intratrade_dd_penalty,
        "max_intratrade_dd": r.max_intratrade_dd,
        "time_decay_start_bars": r.time_decay_start_bars,
        "time_decay_per_bar": r.time_decay_per_bar,
        "time_decay_losing_multiplier": r.time_decay_losing_multiplier,
        "min_reward": r.min_reward,
        "max_reward": r.max_reward,
    }
