# -*- coding: utf-8 -*-
"""
Experiment SSOT Loader
======================
Single Source of Truth loader for L2 Dataset Builder and L3 Training Pipeline.

Both L2 and L3 MUST use this module to read configuration.
DO NOT read experiment_ssot.yaml directly - use this loader.

Contract: CTR-EXPERIMENT-001
Version: 2.0.0
Date: 2026-02-01
"""

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# =============================================================================
# PATHS
# =============================================================================

# Find config directory
_CURRENT_FILE = Path(__file__)
_PROJECT_ROOT = _CURRENT_FILE.parent.parent.parent  # src/config -> src -> project_root
CONFIG_DIR = _PROJECT_ROOT / "config"
EXPERIMENT_SSOT_PATH = CONFIG_DIR / "experiment_ssot.yaml"

# Fallback for Airflow environment
if not EXPERIMENT_SSOT_PATH.exists():
    _AIRFLOW_ROOT = Path("/opt/airflow")
    if _AIRFLOW_ROOT.exists():
        CONFIG_DIR = _AIRFLOW_ROOT / "config"
        EXPERIMENT_SSOT_PATH = CONFIG_DIR / "experiment_ssot.yaml"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FeatureConfig:
    """Configuration for a single feature."""
    name: str
    order: int
    category: str
    description: str
    source: str
    input_column: Optional[str] = None
    input_columns: Optional[List[str]] = None
    formula: Optional[str] = None
    calculator: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    normalization: Dict[str, Any] = field(default_factory=dict)
    is_predictor: bool = True
    is_state: bool = False
    is_target: bool = False
    importance: Optional[str] = None
    note: Optional[str] = None
    replaces: Optional[str] = None


@dataclass
class LRDecayConfig:
    """FASE 2: Learning rate decay configuration."""
    enabled: bool = True
    initial_lr: float = 0.0003
    final_lr: float = 0.00003
    schedule: str = "linear"  # "linear", "exponential", "cosine"


@dataclass
class EarlyStoppingConfig:
    """FASE 2: Early stopping configuration."""
    enabled: bool = True
    patience: int = 5
    min_improvement: float = 0.01
    monitor: str = "mean_reward"


@dataclass
class RollingWindowConfig:
    """FASE 3: Rolling training window configuration for distribution shift."""
    enabled: bool = False
    window_months: int = 18
    retrain_frequency: str = "weekly"  # "weekly", "monthly"
    min_train_rows: int = 50000
    validation_months: int = 6


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    type: str  # "rl" or "forecasting"
    observation_dim: int
    action_space: int
    output_path: str
    output_format: str
    output_prefix: str
    train_split: float
    validation_split: float
    test_split: float
    # FASE 3: Rolling windows
    rolling: RollingWindowConfig = field(default_factory=RollingWindowConfig)


@dataclass
class ThresholdConfig:
    """Action thresholds for LONG/HOLD/SHORT mapping."""
    long: float = 0.50    # EXP-B-001: 0.60→0.50 (more trades)
    short: float = -0.50  # EXP-B-001: -0.60→-0.50 (balanced)


@dataclass
class TrailingStopConfig:
    """EXP-B-001: Trailing stop configuration for dynamic profit locking."""
    enabled: bool = True
    activation_pct: float = 0.015      # Activate at +1.5% unrealized
    trail_factor: float = 0.5          # Trail at 50% of peak
    min_trail_pct: float = 0.005       # Minimum 0.5% trail
    bonus: float = 0.25                # Bonus for trailing stop exit


@dataclass
class EnvironmentConfig:
    """TradingEnv configuration."""
    max_episode_steps: int
    max_drawdown: float
    max_position_holding: int
    clip_range: Tuple[float, float]
    skip_z_suffix: bool
    # PHASE1 FIX: Action thresholds from SSOT
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    # Transaction costs - OPTIMIZED FOR MEXC/BINANCE USDT/COP
    # MEXC: 0% maker, 0.05% taker = 5bps round-trip (2.5bps per side)
    transaction_cost_bps: float = 2.5   # MEXC: 0.025% per side
    slippage_bps: float = 2.5           # Minimal slippage on liquid pair
    # EXP-B-001: Trailing Stop
    trailing_stop: TrailingStopConfig = field(default_factory=TrailingStopConfig)


@dataclass
class TrainingConfig:
    """PPO training hyperparameters."""
    algorithm: str
    total_timesteps: int
    eval_freq: int
    n_eval_episodes: int
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    patience: int
    policy_kwargs: Dict[str, Any]
    # FASE 2: Overfitting prevention
    lr_decay: LRDecayConfig = field(default_factory=LRDecayConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass
class HoldingDecayConfig:
    """Holding decay configuration from SSOT."""
    half_life_bars: int = 24
    max_penalty: float = 0.8
    flat_threshold: int = 4
    enable_overnight_boost: bool = True
    overnight_multiplier: float = 2.0


@dataclass
class AntiGamingConfig:
    """Anti-gaming configuration from SSOT - PHASE 1 FIX."""
    # Inactivity - DISABLED by default to allow HOLD action
    inactivity_grace_period: int = 999   # FIX: Effectively disabled
    inactivity_max_penalty: float = 0.0  # FIX: Disabled
    inactivity_growth_rate: float = 0.0  # FIX: Disabled
    # Churn
    churn_window_size: int = 20
    churn_max_trades: int = 10
    churn_base_penalty: float = 0.1
    churn_excess_penalty: float = 0.02
    # Bias
    bias_imbalance_threshold: float = 0.80
    bias_penalty: float = 0.1
    bias_min_samples: int = 100


@dataclass
class FlatRewardConfig:
    """PHASE 2 FIX v2: Flat reward configuration (direction-neutral)."""
    enabled: bool = True           # Enable counterfactual HOLD reward
    scale: float = 50.0            # Scale to match PnL reward magnitude
    min_move_threshold: float = 0.0001  # Minimum market move (1 pip)
    loss_avoidance_mult: float = 1.0    # Reward for avoiding losses (any direction)


@dataclass
class RewardConfig:
    """Reward function configuration."""
    pnl_weight: float
    dsr_weight: float
    sortino_weight: float
    regime_penalty: float
    holding_decay: float
    anti_gaming: float
    dsr_eta: float
    sortino_window: int
    # SSOT: holding_decay_config from experiment_ssot.yaml
    holding_decay_config: HoldingDecayConfig = field(default_factory=HoldingDecayConfig)
    # SSOT: anti_gaming_config from experiment_ssot.yaml - PHASE 1 FIX
    anti_gaming_config: AntiGamingConfig = field(default_factory=AntiGamingConfig)
    # SSOT: flat_reward_config from experiment_ssot.yaml - PHASE 2 FIX
    flat_reward_config: FlatRewardConfig = field(default_factory=FlatRewardConfig)
    flat_reward_weight: float = 0.3  # Weight for counterfactual HOLD reward


@dataclass
class AntiLeakageConfig:
    """Anti-leakage configuration."""
    macro_shift_days: int
    ffill_daily: int
    ffill_monthly: int
    ffill_quarterly: int
    normalization_source: str
    trading_session_start: int
    trading_session_end: int
    no_ffill_across_sessions: bool


@dataclass
class LoggingConfig:
    """Logging and tracking configuration."""
    mlflow_tracking_uri: str
    experiment_name: str
    tensorboard_log: str
    checkpoint_freq: int
    keep_checkpoints: int


@dataclass
class ExperimentConfig:
    """Complete experiment configuration - SSOT for L2 and L3."""
    version: str
    contract: str
    pipeline: PipelineConfig
    environment: EnvironmentConfig
    training: TrainingConfig
    reward: RewardConfig
    features: List[FeatureConfig]
    anti_leakage: AntiLeakageConfig
    logging: LoggingConfig

    # Derived fields
    feature_order: Tuple[str, ...]
    market_features: Tuple[str, ...]
    state_features: Tuple[str, ...]
    feature_order_hash: str

    def get_feature_by_name(self, name: str) -> Optional[FeatureConfig]:
        """Get feature config by name."""
        for f in self.features:
            if f.name == name:
                return f
        return None

    def get_features_by_category(self, category: str) -> List[FeatureConfig]:
        """Get all features in a category."""
        return [f for f in self.features if f.category == category]

    def get_predictor_features(self) -> List[FeatureConfig]:
        """Get all predictor features (is_predictor=True)."""
        return [f for f in self.features if f.is_predictor]

    def get_normalization_config(self) -> Dict[str, Dict[str, Any]]:
        """Get normalization config for all features."""
        return {f.name: f.normalization for f in self.features}


# =============================================================================
# LOADER FUNCTION
# =============================================================================

_cached_config: Optional[ExperimentConfig] = None


def load_experiment_config(
    config_path: Optional[Path] = None,
    force_reload: bool = False
) -> ExperimentConfig:
    """
    Load experiment configuration from SSOT YAML.

    This is the ONLY function that should be used to read experiment config.
    Both L2 and L3 MUST use this function.

    Args:
        config_path: Optional path to config file. Defaults to experiment_ssot.yaml
        force_reload: If True, bypass cache and reload from disk

    Returns:
        ExperimentConfig: Complete experiment configuration

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If config is invalid
    """
    global _cached_config

    if _cached_config is not None and not force_reload:
        return _cached_config

    path = config_path or EXPERIMENT_SSOT_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Experiment SSOT not found: {path}\n"
            f"Please ensure config/experiment_ssot.yaml exists."
        )

    logger.info(f"Loading experiment SSOT from: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    # Parse meta
    meta = raw.get('_meta', {})
    version = meta.get('version', '0.0.0')
    contract = meta.get('contract', 'UNKNOWN')

    # Parse pipeline
    pipeline_raw = raw.get('pipeline', {})
    output_raw = pipeline_raw.get('output', {})
    splits = output_raw.get('splits', {})

    # FASE 3: Parse rolling window config
    rolling_raw = pipeline_raw.get('rolling', {})
    rolling = RollingWindowConfig(
        enabled=rolling_raw.get('enabled', False),
        window_months=rolling_raw.get('window_months', 18),
        retrain_frequency=rolling_raw.get('retrain_frequency', 'weekly'),
        min_train_rows=rolling_raw.get('min_train_rows', 50000),
        validation_months=rolling_raw.get('validation_months', 6),
    )

    pipeline = PipelineConfig(
        type=pipeline_raw.get('type', 'rl'),
        observation_dim=pipeline_raw.get('observation_dim', 15),
        action_space=pipeline_raw.get('action_space', 3),
        output_path=output_raw.get('path', 'data/pipeline/07_output/5min'),
        output_format=output_raw.get('format', 'parquet'),
        output_prefix=output_raw.get('prefix', 'DS'),
        train_split=splits.get('train', 0.8),
        validation_split=splits.get('validation', 0.1),
        test_split=splits.get('test', 0.1),
        rolling=rolling,
    )

    # Parse environment
    env_raw = raw.get('environment', {})
    clip_range = env_raw.get('clip_range', [-5, 5])

    # PHASE1 FIX: Parse thresholds from SSOT (EXP-B-001: ±0.50)
    thresholds_raw = env_raw.get('thresholds', {})
    thresholds = ThresholdConfig(
        long=thresholds_raw.get('long', 0.50),
        short=thresholds_raw.get('short', -0.50),
    )

    # EXP-B-001: Parse trailing_stop config
    trailing_stop_raw = env_raw.get('trailing_stop', {})
    trailing_stop = TrailingStopConfig(
        enabled=trailing_stop_raw.get('enabled', True),
        activation_pct=trailing_stop_raw.get('activation_pct', 0.015),
        trail_factor=trailing_stop_raw.get('trail_factor', 0.5),
        min_trail_pct=trailing_stop_raw.get('min_trail_pct', 0.005),
        bonus=trailing_stop_raw.get('bonus', 0.25),
    )

    environment = EnvironmentConfig(
        max_episode_steps=env_raw.get('max_episode_steps', 1200),
        max_drawdown=env_raw.get('max_drawdown', 0.25),
        max_position_holding=env_raw.get('max_position_holding', 288),
        clip_range=tuple(clip_range),
        skip_z_suffix=env_raw.get('skip_z_suffix', True),
        thresholds=thresholds,
        # Transaction costs from SSOT - OPTIMIZED FOR MEXC/BINANCE USDT/COP
        transaction_cost_bps=env_raw.get('transaction_cost_bps', 2.5),
        slippage_bps=env_raw.get('slippage_bps', 2.5),
        # EXP-B-001: Trailing Stop
        trailing_stop=trailing_stop,
    )

    # Parse training
    train_raw = raw.get('training', {})
    policy_kwargs = train_raw.get('policy_kwargs', {})

    # FASE 2: Parse LR decay config
    lr_decay_raw = train_raw.get('lr_decay', {})
    lr_decay = LRDecayConfig(
        enabled=lr_decay_raw.get('enabled', True),
        initial_lr=lr_decay_raw.get('initial_lr', train_raw.get('learning_rate', 0.0003)),
        final_lr=lr_decay_raw.get('final_lr', 0.00003),
        schedule=lr_decay_raw.get('schedule', 'linear'),
    )

    # FASE 2: Parse early stopping config
    early_stop_raw = train_raw.get('early_stopping', {})
    early_stopping = EarlyStoppingConfig(
        enabled=early_stop_raw.get('enabled', True),
        patience=early_stop_raw.get('patience', 5),
        min_improvement=early_stop_raw.get('min_improvement', 0.01),
        monitor=early_stop_raw.get('monitor', 'mean_reward'),
    )

    training = TrainingConfig(
        algorithm=train_raw.get('algorithm', 'PPO'),
        total_timesteps=train_raw.get('total_timesteps', 500000),
        eval_freq=train_raw.get('eval_freq', 25000),
        n_eval_episodes=train_raw.get('n_eval_episodes', 5),
        learning_rate=train_raw.get('learning_rate', 0.0003),
        n_steps=train_raw.get('n_steps', 2048),
        batch_size=train_raw.get('batch_size', 256),  # SSOT: 64→256
        n_epochs=train_raw.get('n_epochs', 10),
        gamma=train_raw.get('gamma', 0.95),  # SSOT default
        gae_lambda=train_raw.get('gae_lambda', 0.95),
        clip_range=train_raw.get('clip_range', 0.2),
        ent_coef=train_raw.get('ent_coef', 0.08),  # SSOT: 0.10→0.08
        vf_coef=train_raw.get('vf_coef', 0.5),
        max_grad_norm=train_raw.get('max_grad_norm', 0.5),
        patience=train_raw.get('patience', 10),
        policy_kwargs=policy_kwargs,
        # FASE 2
        lr_decay=lr_decay,
        early_stopping=early_stopping,
    )

    # Parse reward
    reward_raw = raw.get('reward', {})

    # Parse holding_decay_config from SSOT (DRY principle)
    hd_raw = reward_raw.get('holding_decay_config', {})
    holding_decay_cfg = HoldingDecayConfig(
        half_life_bars=hd_raw.get('half_life_bars', 24),
        max_penalty=hd_raw.get('max_penalty', 0.8),
        flat_threshold=hd_raw.get('flat_threshold', 4),
        enable_overnight_boost=hd_raw.get('enable_overnight_boost', True),
        overnight_multiplier=hd_raw.get('overnight_multiplier', 2.0),
    )

    # PHASE 1 FIX: Parse anti_gaming_config from SSOT
    ag_raw = reward_raw.get('anti_gaming_config', {})
    anti_gaming_cfg = AntiGamingConfig(
        inactivity_grace_period=ag_raw.get('inactivity_grace_period', 999),  # DISABLED
        inactivity_max_penalty=ag_raw.get('inactivity_max_penalty', 0.0),    # DISABLED
        inactivity_growth_rate=ag_raw.get('inactivity_growth_rate', 0.0),    # DISABLED
        churn_window_size=ag_raw.get('churn_window_size', 20),
        churn_max_trades=ag_raw.get('churn_max_trades', 10),
        churn_base_penalty=ag_raw.get('churn_base_penalty', 0.1),
        churn_excess_penalty=ag_raw.get('churn_excess_penalty', 0.02),
        bias_imbalance_threshold=ag_raw.get('bias_imbalance_threshold', 0.80),
        bias_penalty=ag_raw.get('bias_penalty', 0.1),
        bias_min_samples=ag_raw.get('bias_min_samples', 100),
    )

    # PHASE 2 FIX: Parse flat_reward_config from SSOT (counterfactual HOLD reward)
    fr_raw = reward_raw.get('flat_reward_config', {})
    flat_reward_cfg = FlatRewardConfig(
        enabled=fr_raw.get('enabled', True),
        scale=fr_raw.get('scale', 50.0),
        min_move_threshold=fr_raw.get('min_move_threshold', 0.0001),
        loss_avoidance_mult=fr_raw.get('loss_avoidance_mult', 1.0),
    )

    reward = RewardConfig(
        pnl_weight=reward_raw.get('pnl_weight', 0.6),
        dsr_weight=reward_raw.get('dsr_weight', 0.15),
        sortino_weight=reward_raw.get('sortino_weight', 0.05),
        regime_penalty=reward_raw.get('regime_penalty', 0.3),
        holding_decay=reward_raw.get('holding_decay', 1.0),
        anti_gaming=reward_raw.get('anti_gaming', 0.3),
        dsr_eta=reward_raw.get('dsr_eta', 0.01),
        sortino_window=reward_raw.get('sortino_window', 240),
        holding_decay_config=holding_decay_cfg,
        anti_gaming_config=anti_gaming_cfg,
        # PHASE 2 FIX: Flat reward for HOLD incentive
        flat_reward_config=flat_reward_cfg,
        flat_reward_weight=reward_raw.get('flat_reward_weight', 0.3),
    )

    # Parse features
    features_raw = raw.get('features', [])
    features = []
    for f in features_raw:
        source = f.get('source', 'ohlcv')
        if isinstance(source, list):
            source = ','.join(source)  # Convert list to string

        feature = FeatureConfig(
            name=f.get('name'),
            order=f.get('order', 0),
            category=f.get('category', 'unknown'),
            description=f.get('description', ''),
            source=source,
            input_column=f.get('input_column'),
            input_columns=f.get('input_columns'),
            formula=f.get('formula'),
            calculator=f.get('calculator'),
            params=f.get('params', {}),
            normalization=f.get('normalization', {}),
            is_predictor=f.get('is_predictor', True),
            is_state=f.get('is_state', False),
            is_target=f.get('is_target', False),
            importance=f.get('importance'),
            note=f.get('note'),
            replaces=f.get('replaces'),
        )
        features.append(feature)

    # Sort features by order
    features.sort(key=lambda x: x.order)

    # Parse anti-leakage
    al_raw = raw.get('anti_leakage', {})
    ffill = al_raw.get('ffill_limits', {})
    session = al_raw.get('trading_session', {})
    anti_leakage = AntiLeakageConfig(
        macro_shift_days=al_raw.get('macro_shift_days', 1),
        ffill_daily=ffill.get('daily', 5),
        ffill_monthly=ffill.get('monthly', 35),
        ffill_quarterly=ffill.get('quarterly', 95),
        normalization_source=al_raw.get('normalization_source', 'train_only'),
        trading_session_start=session.get('start_hour', 8),
        trading_session_end=session.get('end_hour', 13),
        no_ffill_across_sessions=session.get('no_ffill_across_sessions', True),
    )

    # Parse logging
    log_raw = raw.get('logging', {})
    logging_config = LoggingConfig(
        mlflow_tracking_uri=log_raw.get('mlflow_tracking_uri', 'http://mlflow:5000'),
        experiment_name=log_raw.get('experiment_name', 'usdcop_ppo'),
        tensorboard_log=log_raw.get('tensorboard_log', 'models/tensorboard/'),
        checkpoint_freq=log_raw.get('checkpoint_freq', 50000),
        keep_checkpoints=log_raw.get('keep_checkpoints', 3),
    )

    # Compute derived fields
    feature_order = tuple(f.name for f in features)
    market_features = tuple(f.name for f in features if f.is_predictor and not f.is_state)
    state_features = tuple(f.name for f in features if f.is_state)
    feature_order_hash = hashlib.md5(str(feature_order).encode()).hexdigest()

    # Validation
    if len(features) != pipeline.observation_dim:
        logger.warning(
            f"Feature count ({len(features)}) != observation_dim ({pipeline.observation_dim})"
        )

    config = ExperimentConfig(
        version=version,
        contract=contract,
        pipeline=pipeline,
        environment=environment,
        training=training,
        reward=reward,
        features=features,
        anti_leakage=anti_leakage,
        logging=logging_config,
        feature_order=feature_order,
        market_features=market_features,
        state_features=state_features,
        feature_order_hash=feature_order_hash,
    )

    _cached_config = config
    logger.info(
        f"Loaded experiment SSOT v{version}: "
        f"{len(market_features)} market + {len(state_features)} state = "
        f"{len(feature_order)} total features"
    )

    return config


def get_feature_order() -> Tuple[str, ...]:
    """Get feature order from SSOT. Convenience function."""
    config = load_experiment_config()
    return config.feature_order


def get_observation_dim() -> int:
    """Get observation dimension from SSOT. Convenience function."""
    config = load_experiment_config()
    return config.pipeline.observation_dim


def get_training_config() -> TrainingConfig:
    """Get training hyperparameters from SSOT. Convenience function."""
    config = load_experiment_config()
    return config.training


def get_reward_config() -> RewardConfig:
    """Get reward configuration from SSOT. Convenience function."""
    config = load_experiment_config()
    return config.reward


# =============================================================================
# CONVENIENCE FUNCTIONS - Use these instead of hardcoding values
# =============================================================================

def get_gamma() -> float:
    """
    Get gamma (discount factor) from SSOT.

    USE THIS FUNCTION instead of hardcoding gamma in your code.
    This ensures all components use the same value.

    Returns:
        float: Gamma value from experiment_ssot.yaml (default: 0.95)
    """
    config = load_experiment_config()
    return config.training.gamma


def get_learning_rate() -> float:
    """Get learning rate from SSOT."""
    config = load_experiment_config()
    return config.training.learning_rate


def get_ent_coef() -> float:
    """Get entropy coefficient from SSOT."""
    config = load_experiment_config()
    return config.training.ent_coef


def get_batch_size() -> int:
    """Get batch size from SSOT."""
    config = load_experiment_config()
    return config.training.batch_size


def get_ppo_hyperparameters() -> Dict[str, Any]:
    """
    Get all PPO hyperparameters as a dict for Stable-Baselines3.

    USE THIS FUNCTION when creating PPO models instead of hardcoding values.

    Returns:
        Dict with keys: learning_rate, n_steps, batch_size, n_epochs,
        gamma, gae_lambda, clip_range, ent_coef, vf_coef, max_grad_norm
    """
    config = load_experiment_config()
    t = config.training
    return {
        "learning_rate": t.learning_rate,
        "n_steps": t.n_steps,
        "batch_size": t.batch_size,
        "n_epochs": t.n_epochs,
        "gamma": t.gamma,
        "gae_lambda": t.gae_lambda,
        "clip_range": t.clip_range,
        "ent_coef": t.ent_coef,
        "vf_coef": t.vf_coef,
        "max_grad_norm": t.max_grad_norm,
    }


def get_reward_weights() -> Dict[str, float]:
    """
    Get all reward weights as a dict.

    USE THIS FUNCTION instead of hardcoding reward weights.

    Returns:
        Dict with keys: pnl, dsr, sortino, regime_penalty, holding_decay, anti_gaming, flat_reward
    """
    config = load_experiment_config()
    r = config.reward
    return {
        "pnl": r.pnl_weight,
        "dsr": r.dsr_weight,
        "sortino": r.sortino_weight,
        "regime_penalty": r.regime_penalty,
        "holding_decay": r.holding_decay,
        "anti_gaming": r.anti_gaming,
        "flat_reward": r.flat_reward_weight,  # PHASE2
    }


def get_flat_reward_config() -> FlatRewardConfig:
    """
    Get FlatRewardConfig from SSOT.

    PHASE 2 FIX: Counterfactual reward for HOLD action to prevent collapse.

    Returns:
        FlatRewardConfig with enabled, scale, min_move_threshold, etc.
    """
    config = load_experiment_config()
    return config.reward.flat_reward_config


def is_flat_reward_enabled() -> bool:
    """Check if flat reward (counterfactual HOLD) is enabled."""
    config = load_experiment_config()
    return config.reward.flat_reward_config.enabled


def get_holding_decay_config() -> HoldingDecayConfig:
    """
    Get HoldingDecayConfig from SSOT.

    SSOT: Use this instead of hardcoding holding decay parameters.

    Returns:
        HoldingDecayConfig with half_life_bars, max_penalty, flat_threshold, etc.
    """
    config = load_experiment_config()
    return config.reward.holding_decay_config


def get_holding_decay_params() -> Dict[str, Any]:
    """
    Get holding decay parameters as a dict for HoldingDecay component.

    SSOT: Use this for initializing the HoldingDecay reward component.

    Returns:
        Dict with half_life_bars, max_penalty, flat_threshold, etc.
    """
    hd = get_holding_decay_config()
    return {
        "half_life_bars": hd.half_life_bars,
        "max_penalty": hd.max_penalty,
        "flat_threshold": hd.flat_threshold,
        "enable_overnight_boost": hd.enable_overnight_boost,
        "overnight_multiplier": hd.overnight_multiplier,
    }


def get_environment_config() -> Dict[str, Any]:
    """
    Get environment configuration for TradingEnv.

    Returns:
        Dict with max_episode_steps, max_drawdown, max_position_holding, clip_range, thresholds, trailing_stop
    """
    config = load_experiment_config()
    e = config.environment
    return {
        "max_episode_steps": e.max_episode_steps,
        "max_drawdown": e.max_drawdown,
        "max_position_holding": e.max_position_holding,
        "clip_range": e.clip_range,
        "skip_z_suffix": e.skip_z_suffix,
        # PHASE1 FIX: Include thresholds from SSOT
        "threshold_long": e.thresholds.long,
        "threshold_short": e.thresholds.short,
        # EXP-B-001: Trailing stop
        "trailing_stop_enabled": e.trailing_stop.enabled,
        "trailing_stop_activation_pct": e.trailing_stop.activation_pct,
        "trailing_stop_trail_factor": e.trailing_stop.trail_factor,
        "trailing_stop_min_trail_pct": e.trailing_stop.min_trail_pct,
        "trailing_stop_bonus": e.trailing_stop.bonus,
    }


def get_thresholds() -> ThresholdConfig:
    """
    Get action thresholds from SSOT.

    PHASE1 FIX: Use this to ensure consistent thresholds across training and backtest.

    Returns:
        ThresholdConfig with long and short thresholds
    """
    config = load_experiment_config()
    return config.environment.thresholds


def get_threshold_long() -> float:
    """Get LONG threshold from SSOT (default: 0.60)."""
    config = load_experiment_config()
    return config.environment.thresholds.long


def get_threshold_short() -> float:
    """Get SHORT threshold from SSOT (default: -0.50)."""
    config = load_experiment_config()
    return config.environment.thresholds.short


# =============================================================================
# EXP-B-001: Trailing Stop Config
# =============================================================================

def get_trailing_stop_config() -> TrailingStopConfig:
    """
    Get trailing stop configuration from SSOT.

    EXP-B-001: Trailing stop for dynamic profit locking.

    Returns:
        TrailingStopConfig with enabled, activation_pct, trail_factor, etc.
    """
    config = load_experiment_config()
    return config.environment.trailing_stop


def is_trailing_stop_enabled() -> bool:
    """Check if trailing stop is enabled."""
    config = load_experiment_config()
    return config.environment.trailing_stop.enabled


def get_trailing_stop_params() -> Dict[str, Any]:
    """
    Get trailing stop parameters as a dict.

    EXP-B-001: Use this for initializing TradingEnvConfig.

    Returns:
        Dict with enabled, activation_pct, trail_factor, min_trail_pct, bonus
    """
    ts = get_trailing_stop_config()
    return {
        "trailing_stop_enabled": ts.enabled,
        "trailing_stop_activation_pct": ts.activation_pct,
        "trailing_stop_trail_factor": ts.trail_factor,
        "trailing_stop_min_trail_pct": ts.min_trail_pct,
        "trailing_stop_bonus": ts.bonus,
    }


# =============================================================================
# FASE 2: Overfitting Prevention Config
# =============================================================================

def get_lr_decay_config() -> LRDecayConfig:
    """
    Get learning rate decay configuration from SSOT.

    FASE 2: Prevents late-stage overfitting by reducing LR over training.

    Returns:
        LRDecayConfig with enabled, initial_lr, final_lr, schedule
    """
    config = load_experiment_config()
    return config.training.lr_decay


def get_early_stopping_config() -> EarlyStoppingConfig:
    """
    Get early stopping configuration from SSOT.

    FASE 2: Stops training when validation performance plateaus.

    Returns:
        EarlyStoppingConfig with enabled, patience, min_improvement, monitor
    """
    config = load_experiment_config()
    return config.training.early_stopping


def get_overfitting_prevention_config() -> Dict[str, Any]:
    """
    Get complete overfitting prevention configuration from SSOT.

    FASE 2: Returns all settings needed to configure LR decay and early stopping.

    Returns:
        Dict with lr_decay and early_stopping configurations
    """
    config = load_experiment_config()
    t = config.training
    return {
        "lr_decay": {
            "enabled": t.lr_decay.enabled,
            "initial_lr": t.lr_decay.initial_lr,
            "final_lr": t.lr_decay.final_lr,
            "schedule": t.lr_decay.schedule,
        },
        "early_stopping": {
            "enabled": t.early_stopping.enabled,
            "patience": t.early_stopping.patience,
            "min_improvement": t.early_stopping.min_improvement,
            "monitor": t.early_stopping.monitor,
        },
    }


# =============================================================================
# FASE 3: Rolling Window Config
# =============================================================================

def get_rolling_window_config() -> RollingWindowConfig:
    """
    Get rolling training window configuration from SSOT.

    FASE 3: Addresses distribution shift by training on recent data windows.

    Returns:
        RollingWindowConfig with enabled, window_months, retrain_frequency, etc.
    """
    config = load_experiment_config()
    return config.pipeline.rolling


def is_rolling_training_enabled() -> bool:
    """Check if rolling training windows are enabled."""
    config = load_experiment_config()
    return config.pipeline.rolling.enabled


def get_rolling_window_months() -> int:
    """Get the rolling window size in months."""
    config = load_experiment_config()
    return config.pipeline.rolling.window_months


# =============================================================================
# VALIDATION
# =============================================================================

def validate_feature_order(features: List[str]) -> bool:
    """
    Validate that a list of features matches the SSOT order.

    Args:
        features: List of feature names to validate

    Returns:
        bool: True if matches, raises ValueError otherwise
    """
    config = load_experiment_config()
    expected = config.feature_order

    if tuple(features) != expected:
        raise ValueError(
            f"Feature order mismatch!\n"
            f"Expected: {expected}\n"
            f"Got: {tuple(features)}"
        )

    return True


def get_feature_order_hash() -> str:
    """Get hash of feature order for contract validation."""
    config = load_experiment_config()
    return config.feature_order_hash


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)

    config = load_experiment_config()

    print("\n" + "=" * 60)
    print("EXPERIMENT SSOT SUMMARY")
    print("=" * 60)
    print(f"Version: {config.version}")
    print(f"Contract: {config.contract}")
    print(f"Pipeline Type: {config.pipeline.type}")
    print(f"Observation Dim: {config.pipeline.observation_dim}")
    print(f"\nFeature Order ({len(config.feature_order)}):")
    for i, name in enumerate(config.feature_order):
        feat = config.get_feature_by_name(name)
        state_marker = " [STATE]" if feat.is_state else ""
        print(f"  {i:2d}. {name}{state_marker}")

    print(f"\nFeature Order Hash: {config.feature_order_hash}")
    print(f"\nTraining: {config.training.algorithm}")
    print(f"  - learning_rate: {config.training.learning_rate}")
    print(f"  - ent_coef: {config.training.ent_coef}")
    print(f"  - total_timesteps: {config.training.total_timesteps}")
    print("=" * 60)
