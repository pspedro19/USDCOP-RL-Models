"""
Training Configuration SSOT (Single Source of Truth)
=====================================================

This module is the CANONICAL source for all training hyperparameters
and configuration. All training scripts, DAGs, and pipelines MUST
import from here to ensure consistency.

Architecture:
    params.yaml (DVC params)
         ↓
    src/training/config.py (THIS FILE - SSOT)
         ↓
    ┌────────────────────────────────────────┐
    │  DAG: l3_model_training.py             │
    │  Script: train_with_mlflow.py          │
    │  Pipeline: training_pipeline.py        │
    │  Module: train_ssot.py                 │
    └────────────────────────────────────────┘

Design Principles:
    - SSOT: Single source for all hyperparameters
    - DRY: No duplication across modules
    - Clean Code: Clear interfaces, type hints, documentation
    - Dependency Injection: Config passed to components, not hardcoded

Usage:
    from src.training.config import (
        PPOHyperparameters,
        TrainingConfig,
        EnvironmentConfig,
        get_training_config,
        load_config_from_yaml,
    )

    # Get default config
    config = get_training_config()

    # Or load from params.yaml
    config = load_config_from_yaml("params.yaml")

    # Access hyperparameters
    print(config.hyperparameters.learning_rate)

Author: Trading Team
Version: 2.0.0
Date: 2026-01-17
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Project root for relative path resolution
_PROJECT_ROOT = Path(__file__).parent.parent.parent


# =============================================================================
# PPO Hyperparameters (SSOT)
# =============================================================================

@dataclass(frozen=True)
class PPOHyperparameters:
    """
    Canonical PPO hyperparameters.

    These values are the SINGLE SOURCE OF TRUTH for all PPO training.
    Any training run MUST use these values unless explicitly overridden
    with documented justification.

    Frozen dataclass ensures immutability after creation.

    Attributes:
        learning_rate: Adam optimizer learning rate
        n_steps: Number of steps per PPO update
        batch_size: Minibatch size for gradient updates
        n_epochs: Number of epochs per PPO update
        gamma: Discount factor for future rewards
        gae_lambda: GAE lambda for advantage estimation
        clip_range: PPO clipping range for policy updates
        ent_coef: Entropy coefficient for exploration bonus
        vf_coef: Value function coefficient in loss
        max_grad_norm: Maximum gradient norm for clipping
        total_timesteps: Total training timesteps
        seed: Random seed for reproducibility
    """
    # Optimization
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10

    # RL-specific
    gamma: float = 0.90  # Shorter horizon for noisy FX 5-min data
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.05  # Higher exploration for FX volatility
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Training duration
    total_timesteps: int = 500_000

    # Reproducibility
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_hash(self) -> str:
        """Compute deterministic hash of hyperparameters."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PPOHyperparameters:
        """Create from dictionary, filtering invalid fields."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# Default instance - THE canonical values
PPO_HYPERPARAMETERS = PPOHyperparameters()


# =============================================================================
# Network Architecture (SSOT)
# =============================================================================

@dataclass(frozen=True)
class NetworkConfig:
    """
    Neural network architecture configuration.

    Defines the policy and value function network architectures
    used by the PPO agent.
    """
    policy_layers: Tuple[int, ...] = (256, 256)
    value_layers: Tuple[int, ...] = (256, 256)
    activation: str = "Tanh"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_layers": list(self.policy_layers),
            "value_layers": list(self.value_layers),
            "activation": self.activation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NetworkConfig:
        return cls(
            policy_layers=tuple(data.get("policy_layers", data.get("pi", [256, 256]))),
            value_layers=tuple(data.get("value_layers", data.get("vf", [256, 256]))),
            activation=data.get("activation", "Tanh"),
        )


# Default network architecture
NETWORK_CONFIG = NetworkConfig()


# =============================================================================
# Environment Configuration (SSOT)
# =============================================================================

@dataclass(frozen=True)
class EnvironmentConfig:
    """
    Trading environment configuration.

    These values define the simulation environment for RL training.
    Must be consistent between training and inference.
    """
    # Capital and costs
    initial_capital: float = 10_000.0
    transaction_cost_bps: float = 75.0  # From config/trading_config.yaml
    slippage_bps: float = 15.0

    # Episode settings
    max_episode_steps: int = 2000
    random_episode_start: bool = True

    # Risk limits
    max_drawdown_pct: float = 15.0
    max_position: float = 1.0

    # Trading hours (UTC) - Colombia market hours
    trading_start_hour: int = 13  # 8:00 Bogota
    trading_end_hour: int = 17
    trading_end_minute: int = 55  # 12:55 Bogota

    # Action thresholds
    threshold_long: float = 0.33
    threshold_short: float = -0.33

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EnvironmentConfig:
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# Default environment config
ENVIRONMENT_CONFIG = EnvironmentConfig()


# =============================================================================
# Data Split Configuration (SSOT)
# =============================================================================

@dataclass(frozen=True)
class DataSplitConfig:
    """
    Train/validation/test split configuration.

    Ensures consistent data splits across all training runs.
    """
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    def __post_init__(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Default data split
DATA_SPLIT_CONFIG = DataSplitConfig()


# =============================================================================
# Technical Indicators Configuration (SSOT)
# =============================================================================

@dataclass(frozen=True)
class IndicatorConfig:
    """
    Technical indicator periods.

    These periods must match between feature generation and model expectations.
    """
    rsi_period: int = 9
    atr_period: int = 10
    adx_period: int = 14

    @property
    def warmup_bars(self) -> int:
        """Minimum bars needed before indicators are valid."""
        return max(self.rsi_period, self.atr_period, self.adx_period)

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "warmup_bars": self.warmup_bars}


# Default indicator config
INDICATOR_CONFIG = IndicatorConfig()


# =============================================================================
# Reward Configuration (SSOT)
# =============================================================================

@dataclass(frozen=True)
class DSRConfig:
    """Differential Sharpe Ratio configuration."""
    eta: float = 0.01
    min_samples: int = 10
    scale: float = 1.0
    weight: float = 0.3

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SortinoConfig:
    """Sortino Ratio configuration."""
    window_size: int = 20
    target_return: float = 0.0
    min_samples: int = 5
    scale: float = 1.0
    weight: float = 0.2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegimeConfig:
    """Market regime detection configuration."""
    low_vol_percentile: int = 25
    high_vol_percentile: int = 75
    crisis_multiplier: float = 1.5
    min_stability: int = 3
    history_window: int = 500
    smoothing_window: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MarketImpactConfig:
    """Almgren-Chriss market impact configuration."""
    permanent_impact_coef: float = 0.1
    temporary_impact_coef: float = 0.3
    volatility_impact_coef: float = 0.15
    adv_base_usd: float = 50_000_000.0
    typical_order_fraction: float = 0.001
    default_spread_bps: int = 100

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HoldingDecayConfig:
    """Holding time decay configuration."""
    half_life_bars: int = 48
    max_penalty: float = 0.3
    flat_threshold: int = 0
    enable_overnight_boost: bool = True
    overnight_multiplier: float = 1.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AntiGamingConfig:
    """Anti-gaming penalties configuration."""
    # Inactivity
    inactivity_grace_period: int = 12
    inactivity_max_penalty: float = 0.2
    inactivity_growth_rate: float = 0.01

    # Churn
    churn_window_size: int = 20
    churn_max_trades: int = 10
    churn_base_penalty: float = 0.1
    churn_excess_penalty: float = 0.02

    # Bias
    bias_imbalance_threshold: float = 0.75
    bias_penalty: float = 0.1
    bias_min_samples: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NormalizerConfig:
    """Reward normalizer configuration (FinRL-Meta style)."""
    decay: float = 0.99
    epsilon: float = 1e-8
    clip_range: float = 10.0
    warmup_steps: int = 1000
    per_episode_reset: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BanrepDetectorConfig:
    """Banrep intervention detector configuration."""
    volatility_spike_zscore: float = 3.0
    volatility_baseline_window: int = 100
    intervention_penalty: float = 0.5
    cooldown_bars: int = 24
    reversal_threshold: float = 0.02
    min_history: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OilCorrelationConfig:
    """Oil correlation tracker configuration."""
    window_size: int = 20
    strong_threshold: float = -0.3
    weak_threshold: float = -0.1
    breakdown_penalty: float = 0.1
    min_samples: int = 10

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PnLTransformConfig:
    """PnL transformation configuration."""
    transform_type: str = "zscore"  # log, asymmetric, clipped, rank, zscore
    clip_min: float = -0.1
    clip_max: float = 0.1
    zscore_window: int = 100
    zscore_clip: float = 3.0
    asymmetric_win_mult: float = 1.0
    asymmetric_loss_mult: float = 1.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CurriculumConfig:
    """Curriculum learning configuration."""
    enabled: bool = True
    phase_1_steps: int = 100_000  # NORMAL regime only
    phase_2_steps: int = 200_000  # NORMAL + HIGH_VOL
    phase_3_steps: int = 300_000  # All regimes including CRISIS
    phase_1_cost_mult: float = 0.5
    phase_2_cost_mult: float = 0.75
    phase_3_cost_mult: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RewardConfig:
    """
    Complete reward system configuration (SSOT).

    Aggregates all reward component configurations.
    Use this for training to ensure consistent reward calculation.
    """
    # Component configs
    dsr: DSRConfig = field(default_factory=DSRConfig)
    sortino: SortinoConfig = field(default_factory=SortinoConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    market_impact: MarketImpactConfig = field(default_factory=MarketImpactConfig)
    holding_decay: HoldingDecayConfig = field(default_factory=HoldingDecayConfig)
    anti_gaming: AntiGamingConfig = field(default_factory=AntiGamingConfig)
    normalizer: NormalizerConfig = field(default_factory=NormalizerConfig)
    banrep: BanrepDetectorConfig = field(default_factory=BanrepDetectorConfig)
    oil_correlation: OilCorrelationConfig = field(default_factory=OilCorrelationConfig)
    pnl_transform: PnLTransformConfig = field(default_factory=PnLTransformConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    # Component weights (for weighted sum)
    weight_pnl: float = 0.5
    weight_dsr: float = 0.3
    weight_sortino: float = 0.2
    weight_regime_penalty: float = 1.0
    weight_holding_decay: float = 1.0
    weight_anti_gaming: float = 1.0

    # Global settings
    enable_normalization: bool = True
    enable_curriculum: bool = True
    enable_banrep_detection: bool = True
    enable_oil_tracking: bool = False  # Off by default - requires oil data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dsr": self.dsr.to_dict(),
            "sortino": self.sortino.to_dict(),
            "regime": self.regime.to_dict(),
            "market_impact": self.market_impact.to_dict(),
            "holding_decay": self.holding_decay.to_dict(),
            "anti_gaming": self.anti_gaming.to_dict(),
            "normalizer": self.normalizer.to_dict(),
            "banrep": self.banrep.to_dict(),
            "oil_correlation": self.oil_correlation.to_dict(),
            "pnl_transform": self.pnl_transform.to_dict(),
            "curriculum": self.curriculum.to_dict(),
            "weight_pnl": self.weight_pnl,
            "weight_dsr": self.weight_dsr,
            "weight_sortino": self.weight_sortino,
            "weight_regime_penalty": self.weight_regime_penalty,
            "weight_holding_decay": self.weight_holding_decay,
            "weight_anti_gaming": self.weight_anti_gaming,
            "enable_normalization": self.enable_normalization,
            "enable_curriculum": self.enable_curriculum,
            "enable_banrep_detection": self.enable_banrep_detection,
            "enable_oil_tracking": self.enable_oil_tracking,
        }

    def to_hash(self) -> str:
        """Compute deterministic hash of reward configuration."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RewardConfig":
        """Create from dictionary."""
        return cls(
            dsr=DSRConfig(**data.get("dsr", {})) if data.get("dsr") else DSRConfig(),
            sortino=SortinoConfig(**data.get("sortino", {})) if data.get("sortino") else SortinoConfig(),
            regime=RegimeConfig(**data.get("regime", {})) if data.get("regime") else RegimeConfig(),
            market_impact=MarketImpactConfig(**data.get("market_impact", {})) if data.get("market_impact") else MarketImpactConfig(),
            holding_decay=HoldingDecayConfig(**data.get("holding_decay", {})) if data.get("holding_decay") else HoldingDecayConfig(),
            anti_gaming=AntiGamingConfig(**data.get("anti_gaming", {})) if data.get("anti_gaming") else AntiGamingConfig(),
            normalizer=NormalizerConfig(**data.get("normalizer", {})) if data.get("normalizer") else NormalizerConfig(),
            banrep=BanrepDetectorConfig(**data.get("banrep", {})) if data.get("banrep") else BanrepDetectorConfig(),
            oil_correlation=OilCorrelationConfig(**data.get("oil_correlation", {})) if data.get("oil_correlation") else OilCorrelationConfig(),
            pnl_transform=PnLTransformConfig(**data.get("pnl_transform", {})) if data.get("pnl_transform") else PnLTransformConfig(),
            curriculum=CurriculumConfig(**data.get("curriculum", {})) if data.get("curriculum") else CurriculumConfig(),
            weight_pnl=data.get("weight_pnl", 0.5),
            weight_dsr=data.get("weight_dsr", 0.3),
            weight_sortino=data.get("weight_sortino", 0.2),
            weight_regime_penalty=data.get("weight_regime_penalty", 1.0),
            weight_holding_decay=data.get("weight_holding_decay", 1.0),
            weight_anti_gaming=data.get("weight_anti_gaming", 1.0),
            enable_normalization=data.get("enable_normalization", True),
            enable_curriculum=data.get("enable_curriculum", True),
            enable_banrep_detection=data.get("enable_banrep_detection", True),
            enable_oil_tracking=data.get("enable_oil_tracking", False),
        )


# Default reward config
REWARD_CONFIG = RewardConfig()


# =============================================================================
# MLflow Configuration (SSOT)
# =============================================================================

@dataclass
class MLflowConfig:
    """
    MLflow tracking configuration.

    Centralized configuration for experiment tracking.
    """
    tracking_uri: str = field(
        default_factory=lambda: os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    experiment_name: str = "usdcop-rl-training"
    model_name: str = "usdcop-ppo-model"
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Default MLflow config
MLFLOW_CONFIG = MLflowConfig()


# =============================================================================
# Complete Training Configuration (SSOT)
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Complete training configuration.

    This is the top-level configuration that aggregates all sub-configs.
    Use this class for training runs to ensure consistency.

    Example:
        config = TrainingConfig(version="v3")
        print(config.hyperparameters.learning_rate)  # 3e-4
        print(config.environment.transaction_cost_bps)  # 75.0
    """
    # Version identifier
    version: str = "current"
    experiment_name: str = ""

    # Sub-configurations (use defaults from SSOT)
    hyperparameters: PPOHyperparameters = field(default_factory=lambda: PPO_HYPERPARAMETERS)
    network: NetworkConfig = field(default_factory=lambda: NETWORK_CONFIG)
    environment: EnvironmentConfig = field(default_factory=lambda: ENVIRONMENT_CONFIG)
    data_split: DataSplitConfig = field(default_factory=lambda: DATA_SPLIT_CONFIG)
    indicators: IndicatorConfig = field(default_factory=lambda: INDICATOR_CONFIG)
    reward: RewardConfig = field(default_factory=lambda: REWARD_CONFIG)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)

    # Dataset configuration
    dataset_path: Optional[Path] = None
    dataset_name: str = "RL_DS3_MACRO_CORE.csv"
    dataset_dir: str = "5min"

    # Output paths
    model_output_dir: Optional[Path] = None
    norm_stats_output_path: Optional[Path] = None
    contract_output_path: Optional[Path] = None

    # Database
    db_connection_string: Optional[str] = None

    # Options
    auto_register: bool = True
    run_backtest_validation: bool = False
    backtest_start_date: Optional[str] = None
    backtest_end_date: Optional[str] = None

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        """Set defaults based on version."""
        if not self.experiment_name:
            self.experiment_name = f"ppo_{self.version}_{datetime.now().strftime('%Y%m%d')}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "experiment_name": self.experiment_name,
            "hyperparameters": self.hyperparameters.to_dict(),
            "network": self.network.to_dict(),
            "environment": self.environment.to_dict(),
            "data_split": self.data_split.to_dict(),
            "indicators": self.indicators.to_dict(),
            "reward": self.reward.to_dict(),
            "mlflow": self.mlflow.to_dict(),
            "dataset_path": str(self.dataset_path) if self.dataset_path else None,
            "dataset_name": self.dataset_name,
            "dataset_dir": self.dataset_dir,
            "auto_register": self.auto_register,
            "created_at": self.created_at,
        }

    def to_hash(self) -> str:
        """Compute deterministic hash of training-affecting configuration."""
        hash_dict = {
            "hyperparameters": self.hyperparameters.to_dict(),
            "network": self.network.to_dict(),
            "environment": self.environment.to_dict(),
            "indicators": self.indicators.to_dict(),
            "reward": self.reward.to_dict(),
        }
        config_str = json.dumps(hash_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrainingConfig:
        """Create from dictionary."""
        hp_data = data.get("hyperparameters", data.get("train", {}))
        network_data = data.get("network", {})
        env_data = data.get("environment", data.get("backtest", {}))
        split_data = data.get("data_split", data.get("prepare", {}))
        indicator_data = data.get("indicators", data.get("prepare", {}))

        return cls(
            version=data.get("version", "current"),
            experiment_name=data.get("experiment_name", ""),
            hyperparameters=PPOHyperparameters.from_dict(hp_data),
            network=NetworkConfig.from_dict(network_data),
            environment=EnvironmentConfig.from_dict(env_data),
            data_split=DataSplitConfig(
                train_ratio=split_data.get("train_ratio", 0.7),
                val_ratio=split_data.get("val_ratio", 0.15),
                test_ratio=split_data.get("test_ratio", 0.15),
            ),
            indicators=IndicatorConfig(
                rsi_period=indicator_data.get("rsi_period", 9),
                atr_period=indicator_data.get("atr_period", 10),
                adx_period=indicator_data.get("adx_period", 14),
            ),
            dataset_path=Path(data["dataset_path"]) if data.get("dataset_path") else None,
            auto_register=data.get("auto_register", True),
        )


# =============================================================================
# Factory Functions
# =============================================================================

def get_training_config(version: str = "current", **overrides) -> TrainingConfig:
    """
    Get training configuration with optional overrides.

    This is the preferred way to get a TrainingConfig instance.
    Uses SSOT defaults with optional overrides.

    Args:
        version: Model version identifier
        **overrides: Any TrainingConfig field to override

    Returns:
        TrainingConfig with SSOT defaults and overrides

    Example:
        # Default config
        config = get_training_config()

        # Custom version
        config = get_training_config(version="v3")

        # Override hyperparameters
        config = get_training_config(
            version="v3",
            hyperparameters=PPOHyperparameters(total_timesteps=1_000_000)
        )
    """
    return TrainingConfig(version=version, **overrides)


def load_config_from_yaml(yaml_path: Union[str, Path]) -> TrainingConfig:
    """
    Load training configuration from params.yaml.

    Parses the DVC params.yaml file and creates a TrainingConfig
    with values from the file.

    Args:
        yaml_path: Path to params.yaml

    Returns:
        TrainingConfig populated from YAML

    Example:
        config = load_config_from_yaml("params.yaml")
    """
    import yaml

    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        logger.warning(f"params.yaml not found at {yaml_path}, using defaults")
        return get_training_config()

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # Extract hyperparameters from 'train' section
    train_section = data.get("train", {})
    hp = PPOHyperparameters(
        learning_rate=train_section.get("learning_rate", 3e-4),
        n_steps=train_section.get("n_steps", 2048),
        batch_size=train_section.get("batch_size", 64),
        n_epochs=train_section.get("n_epochs", 10),
        gamma=train_section.get("gamma", 0.90),
        gae_lambda=train_section.get("gae_lambda", 0.95),
        clip_range=train_section.get("clip_range", 0.2),
        ent_coef=train_section.get("ent_coef", 0.05),
        vf_coef=train_section.get("vf_coef", 0.5),
        max_grad_norm=train_section.get("max_grad_norm", 0.5),
        total_timesteps=train_section.get("total_timesteps", 500_000),
        seed=data.get("prepare", {}).get("random_seed", 42),
    )

    # Extract network config
    network_section = train_section.get("network", {})
    network = NetworkConfig(
        policy_layers=tuple(network_section.get("pi", [256, 256])),
        value_layers=tuple(network_section.get("vf", [256, 256])),
        activation=network_section.get("activation", "Tanh"),
    )

    # Extract environment config from backtest section
    backtest_section = data.get("backtest", {})
    env = EnvironmentConfig(
        initial_capital=backtest_section.get("initial_balance", 10_000.0),
        transaction_cost_bps=backtest_section.get("transaction_cost_bps", 75.0),
        slippage_bps=backtest_section.get("slippage_bps", 15.0),
        threshold_long=backtest_section.get("long_threshold", 0.33),
        threshold_short=backtest_section.get("short_threshold", -0.33),
        trading_start_hour=backtest_section.get("trading_start_hour", 13),
        trading_end_hour=backtest_section.get("trading_end_hour", 17),
        trading_end_minute=backtest_section.get("trading_end_minute", 55),
    )

    # Extract data split config
    prepare_section = data.get("prepare", {})
    split = DataSplitConfig(
        train_ratio=prepare_section.get("train_ratio", 0.7),
        val_ratio=prepare_section.get("val_ratio", 0.15),
        test_ratio=1 - prepare_section.get("train_ratio", 0.7) - prepare_section.get("val_ratio", 0.15),
    )

    # Extract MLflow config
    mlflow_section = data.get("mlflow", {})
    mlflow = MLflowConfig(
        tracking_uri=mlflow_section.get("tracking_uri", os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")),
        experiment_name=train_section.get("experiment_name", "usdcop-rl-training"),
        model_name=train_section.get("model_name", "usdcop-ppo-model"),
    )

    return TrainingConfig(
        version="current",
        hyperparameters=hp,
        network=network,
        environment=env,
        data_split=split,
        mlflow=mlflow,
    )


def get_project_root() -> Path:
    """Get project root directory."""
    return _PROJECT_ROOT


# =============================================================================
# Validation Functions
# =============================================================================

def validate_config(config: TrainingConfig) -> List[str]:
    """
    Validate training configuration.

    Args:
        config: Configuration to validate

    Returns:
        List of validation warnings (empty if all valid)

    Raises:
        ValueError: If configuration has critical errors
    """
    warnings = []
    hp = config.hyperparameters

    # Critical validations (raise)
    if hp.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive: {hp.learning_rate}")

    if hp.total_timesteps <= 0:
        raise ValueError(f"total_timesteps must be positive: {hp.total_timesteps}")

    if not 0 < hp.gamma <= 1:
        raise ValueError(f"gamma must be in (0, 1]: {hp.gamma}")

    if not 0 < hp.clip_range <= 1:
        raise ValueError(f"clip_range must be in (0, 1]: {hp.clip_range}")

    # Warnings (non-fatal)
    if hp.learning_rate > 1e-3:
        warnings.append(f"learning_rate={hp.learning_rate} is unusually high for PPO")

    if hp.ent_coef > 0.1:
        warnings.append(f"ent_coef={hp.ent_coef} is very high, may cause too much exploration")

    if hp.total_timesteps < 100_000:
        warnings.append(f"total_timesteps={hp.total_timesteps} is low for meaningful training")

    logger.debug(f"Config validated with {len(warnings)} warnings")
    return warnings


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Dataclasses
    "PPOHyperparameters",
    "NetworkConfig",
    "EnvironmentConfig",
    "DataSplitConfig",
    "IndicatorConfig",
    "MLflowConfig",
    "TrainingConfig",
    # Reward configuration dataclasses
    "DSRConfig",
    "SortinoConfig",
    "RegimeConfig",
    "MarketImpactConfig",
    "HoldingDecayConfig",
    "AntiGamingConfig",
    "NormalizerConfig",
    "BanrepDetectorConfig",
    "OilCorrelationConfig",
    "PnLTransformConfig",
    "CurriculumConfig",
    "RewardConfig",
    # Singleton instances (SSOT)
    "PPO_HYPERPARAMETERS",
    "NETWORK_CONFIG",
    "ENVIRONMENT_CONFIG",
    "DATA_SPLIT_CONFIG",
    "INDICATOR_CONFIG",
    "REWARD_CONFIG",
    "MLFLOW_CONFIG",
    # Factory functions
    "get_training_config",
    "load_config_from_yaml",
    "get_project_root",
    # Validation
    "validate_config",
]
