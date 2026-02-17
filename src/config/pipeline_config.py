"""
Pipeline Configuration Loader - Single Source of Truth
=======================================================
Loads and provides typed access to pipeline_ssot.yaml for all layers.

Usage:
    from src.config.pipeline_config import load_pipeline_config, PipelineConfig

    config = load_pipeline_config()

    # Access any section
    features = config.get_feature_definitions()
    ppo_config = config.training.ppo
    backtest_config = config.backtest

Contract: CTR-PIPELINE-SSOT-001
Version: 2.0.0
"""

import logging
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from functools import lru_cache

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES - Typed configuration objects
# =============================================================================

@dataclass
class NormalizationConfig:
    """Normalization configuration for a feature."""
    method: str = "zscore"  # zscore, minmax, none, clip
    clip: Tuple[float, float] = (-5, 5)
    compute_on: str = "train_only"
    input_range: Optional[Tuple[float, float]] = None
    output_range: Optional[Tuple[float, float]] = None


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration for a feature."""
    dropna: bool = True
    ffill_limit: Optional[int] = None
    shift: int = 0
    clip_outliers: Optional[Tuple[float, float]] = None
    floor: Optional[float] = None


@dataclass
class FeatureDefinition:
    """Complete definition of a feature from SSOT."""
    name: str
    order: int
    category: str
    description: str
    source: str
    input_columns: List[str]
    calculator: Optional[str]
    params: Dict[str, Any]
    preprocessing: PreprocessingConfig
    normalization: NormalizationConfig
    importance: str = "MEDIUM"
    is_state: bool = False
    custom_formula: Optional[str] = None
    note: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureDefinition":
        """Create FeatureDefinition from SSOT dictionary."""
        preprocessing_data = data.get("preprocessing", {})
        normalization_data = data.get("normalization", {})

        preprocessing = PreprocessingConfig(
            dropna=preprocessing_data.get("dropna", True),
            ffill_limit=preprocessing_data.get("ffill_limit"),
            shift=preprocessing_data.get("shift", 0),
            clip_outliers=tuple(preprocessing_data["clip_outliers"]) if preprocessing_data.get("clip_outliers") else None,
            floor=preprocessing_data.get("floor"),
        )

        normalization = NormalizationConfig(
            method=normalization_data.get("method", "zscore"),
            clip=tuple(normalization_data.get("clip", [-5, 5])),
            compute_on=normalization_data.get("compute_on", "train_only"),
            input_range=tuple(normalization_data["input_range"]) if normalization_data.get("input_range") else None,
            output_range=tuple(normalization_data["output_range"]) if normalization_data.get("output_range") else None,
        )

        return cls(
            name=data["name"],
            order=data["order"],
            category=data.get("category", "unknown"),
            description=data.get("description", ""),
            source=data.get("source", "ohlcv"),
            input_columns=data.get("input_columns", []),
            calculator=data.get("calculator"),
            params=data.get("params", {}),
            preprocessing=preprocessing,
            normalization=normalization,
            importance=data.get("importance", "MEDIUM"),
            is_state=data.get("source") == "runtime",
            custom_formula=data.get("custom_formula"),
            note=data.get("note"),
        )


@dataclass
class PPOConfig:
    """PPO hyperparameters (V21: gamma 0.98, ent_coef 0.01, n_steps 4096, batch 128)."""
    learning_rate: float = 0.0003
    n_steps: int = 4096     # V21: Was 2048
    batch_size: int = 128   # V21: Was 256
    n_epochs: int = 10
    gamma: float = 0.98     # V21: Was 0.95
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    ent_coef: float = 0.01  # V21: Was 0.02/0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True


@dataclass
class SACConfig:
    """SAC hyperparameters for off-policy training."""
    learning_rate: float = 1e-4
    buffer_size: int = 200_000
    learning_starts: int = 10_000
    batch_size: int = 512
    tau: float = 0.005
    gamma: float = 0.99
    ent_coef: str = "auto"         # "auto" or float
    target_entropy: str = "auto"   # "auto" or float
    train_freq: int = 4
    gradient_steps: int = 4


@dataclass
class EnvironmentConfig:
    """Trading environment configuration (V21: symmetric stops, trailing disabled)."""
    episode_length: int = 2400          # V21: was 1200
    warmup_bars: int = 14
    initial_balance: float = 10000.0
    transaction_cost_bps: float = 2.5
    slippage_bps: float = 2.5
    max_position: float = 1.0
    max_drawdown_pct: float = 15.0
    max_position_duration: int = 864   # V21: Was 576
    stop_loss_pct: float = -0.04       # V21: Was -0.025
    stop_loss_penalty: float = 0.3
    take_profit_pct: float = 0.04      # V21: Was 0.03
    take_profit_bonus: float = 0.5
    exit_bonus: float = 0.2
    threshold_long: float = 0.35       # V21: Matches SSOT
    threshold_short: float = -0.35     # V21: Matches SSOT
    trailing_stop_enabled: bool = False # V21: Was True
    trailing_stop_activation_pct: float = 0.015
    trailing_stop_trail_factor: float = 0.5
    trailing_stop_min_trail_pct: float = 0.007  # V21: match SSOT (was 0.005)
    trailing_stop_bonus: float = 0.25
    reward_interval: int = 25          # V21: Deliver reward every 25 bars (~2 hours)
    min_hold_bars: int = 25            # V21.1: Min bars before voluntary exit (~2 hours)
    decision_interval: int = 1         # EXP-SWING-001: Bars between agent decisions (1=every bar, 59=daily)


@dataclass
class RewardConfig:
    """Reward function configuration (V21: PnL-dominant, phantom rewards disabled)."""
    weight_pnl: float = 0.90            # V21: near-pure profit signal (was 0.80)
    weight_dsr: float = 0.00            # V21: DISABLED (was 0.10)
    weight_sortino: float = 0.00        # V21: DISABLED (was 0.05)
    weight_regime_penalty: float = 0.05  # V21: was 0.10
    weight_holding_decay: float = 0.05   # V21: was 0.20
    weight_anti_gaming: float = 0.00     # V21: DISABLED (was 0.10)
    weight_flat_reward: float = 0.0      # V21: DISABLED (was 0.05)
    pnl_clip_range: Tuple[float, float] = (-0.1, 0.1)
    pnl_zscore_window: int = 100
    asymmetric_loss_mult: float = 1.0    # V21: symmetric (was 1.5)
    holding_decay_half_life: int = 576   # V21: was 144
    holding_decay_max_penalty: float = 0.15  # V21: was 0.3


@dataclass
class BacktestConfig:
    """Backtest configuration - MUST match training (V21: symmetric stops)."""
    transaction_cost_bps: float = 2.5
    slippage_bps: float = 2.5
    threshold_long: float = 0.35         # V21: was 0.50
    threshold_short: float = -0.35       # V21: was -0.50
    stop_loss_pct: float = -0.04         # V21: was -0.025
    take_profit_pct: float = 0.04        # V21: was 0.03
    trailing_stop_enabled: bool = False   # V21: was True
    trailing_stop_activation_pct: float = 0.015
    trailing_stop_trail_factor: float = 0.5
    max_position_duration: int = 864     # V21: was 576
    initial_capital: float = 10000.0

    # Validation gates (V21: match pipeline_ssot.yaml)
    min_return_pct: float = 0.0          # V21: break-even (was -10.0)
    min_sharpe_ratio: float = 0.0        # V21: was 0.3
    max_drawdown_pct: float = 20.0       # V21: was 25.0
    min_trades: int = 30                 # V21: was 20
    min_win_rate: float = 0.35           # V21: was 0.30
    max_action_imbalance: float = 0.85
    decision_interval: int = 1         # EXP-SWING-001: Must match training


@dataclass
class LSTMConfig:
    """V22 P3: LSTM configuration for RecurrentPPO."""
    enabled: bool = True
    hidden_size: int = 128
    n_layers: int = 1
    sequence_length: int = 64


@dataclass
class PositionSizingConfig:
    """V22 P1: Kelly criterion position sizing."""
    method: str = "half_kelly"
    base_fraction: float = 0.063
    min_fraction: float = 0.02
    max_fraction: float = 0.15
    consensus_scaling: bool = True


@dataclass
class EnsembleConfig:
    """V22 P1: Ensemble voting configuration."""
    n_models: int = 5
    min_consensus: int = 3
    voting: str = "majority"
    confidence_weighted: bool = True


@dataclass
class TemporalFeaturesConfig:
    """V22 P1: Temporal feature configuration."""
    enabled: bool = True
    features: List[str] = field(default_factory=lambda: ["hour_sin", "hour_cos", "dow_sin", "dow_cos"])


@dataclass
class CloseShapingConfig:
    """V22 P2: Close reason reward shaping configuration."""
    enabled: bool = True
    stop_loss_mult: float = 1.5
    take_profit_mult: float = 1.2
    agent_close_win: float = 1.1
    agent_close_loss: float = 0.7
    timeout_mult: float = 0.8


@dataclass
class WalkForwardConfig:
    """V22 P4: Walk-forward validation configuration."""
    enabled: bool = True
    train_window: str = "expanding"
    retrain_frequency: str = "quarterly"
    min_train_bars: int = 50000
    val_window_bars: int = 4680
    test_window_bars: int = 4680
    n_splits: int = 4


@dataclass
class DateRanges:
    """Date ranges for train/val/test splits."""
    data_start: str = "2020-01-01"
    data_end: str = "2026-02-03"
    train_start: str = "2020-01-01"
    train_end: str = "2024-06-30"
    val_start: str = "2024-07-01"
    val_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    test_end: str = "2025-12-31"
    use_fixed_dates: bool = True
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class PathsConfig:
    """All pipeline paths."""
    project_root: str
    local_root: str

    # L2 paths
    l2_output_dir: str
    l2_dataset_prefix: str
    train_file: str
    val_file: str
    test_file: str
    norm_stats_file: str

    # L3 paths
    models_dir: str
    model_prefix: str


# =============================================================================
# MAIN CONFIG CLASS
# =============================================================================

class PipelineConfig:
    """
    Main configuration class that provides typed access to pipeline_ssot.yaml.

    This is the SINGLE interface for all layers to access configuration.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Load configuration from SSOT file."""
        if config_path is None:
            # Find config relative to this file
            config_path = Path(__file__).parent.parent.parent / "config" / "pipeline_ssot.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Pipeline SSOT not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self._raw = yaml.safe_load(f)

        self._config_path = config_path
        self._parse_config()

        logger.info(f"Loaded pipeline config v{self._raw['_meta']['version']} from {config_path}")

    def _parse_config(self) -> None:
        """Parse raw YAML into typed objects."""
        # Parse paths
        paths_data = self._raw.get("paths", {})
        l2_output = paths_data.get("l2_output", {})
        l3_output = paths_data.get("l3_output", {})

        self.paths = PathsConfig(
            project_root=paths_data.get("project_root", "/opt/airflow"),
            local_root=paths_data.get("local_root", ""),
            l2_output_dir=l2_output.get("base_dir", "data/pipeline/07_output/5min"),
            l2_dataset_prefix=l2_output.get("dataset_prefix", "DS_production"),
            train_file=l2_output.get("train_file", ""),
            val_file=l2_output.get("val_file", ""),
            test_file=l2_output.get("test_file", ""),
            norm_stats_file=l2_output.get("norm_stats_file", ""),
            models_dir=l3_output.get("models_dir", "models"),
            model_prefix=l3_output.get("model_prefix", "ppo_production"),
        )

        # Parse date ranges
        dates = self._raw.get("date_ranges", {})
        self.date_ranges = DateRanges(
            data_start=dates.get("data_start", "2020-01-01"),
            data_end=dates.get("data_end", "2026-02-03"),
            train_start=dates.get("train_start", "2020-01-01"),
            train_end=dates.get("train_end", "2024-06-30"),
            val_start=dates.get("val_start", "2024-07-01"),
            val_end=dates.get("val_end", "2024-12-31"),
            test_start=dates.get("test_start", "2025-01-01"),
            test_end=dates.get("test_end", "2025-12-31"),
            use_fixed_dates=dates.get("use_fixed_dates", True),
            train_ratio=dates.get("train_ratio", 0.70),
            val_ratio=dates.get("val_ratio", 0.15),
            test_ratio=dates.get("test_ratio", 0.15),
        )

        # Parse training config
        training = self._raw.get("training", {})
        ppo_data = training.get("ppo", {})
        env_data = training.get("environment", {})
        reward_data = training.get("reward", {})

        # V21: ALL fallback defaults MUST match pipeline_ssot.yaml (SSOT)
        # If YAML is missing a key, the fallback must produce correct V21 behavior
        self.ppo = PPOConfig(
            learning_rate=ppo_data.get("learning_rate", 0.0003),
            n_steps=ppo_data.get("n_steps", 4096),       # V21: was 2048
            batch_size=ppo_data.get("batch_size", 128),   # V21: was 256
            n_epochs=ppo_data.get("n_epochs", 10),
            gamma=ppo_data.get("gamma", 0.98),            # V21: was 0.95
            gae_lambda=ppo_data.get("gae_lambda", 0.95),
            clip_range=ppo_data.get("clip_range", 0.2),
            ent_coef=ppo_data.get("ent_coef", 0.01),      # V21: was 0.02
            vf_coef=ppo_data.get("vf_coef", 0.5),
            max_grad_norm=ppo_data.get("max_grad_norm", 0.5),
        )

        trailing = env_data.get("trailing_stop", {})
        self.environment = EnvironmentConfig(
            episode_length=env_data.get("episode_length", 2400),        # V21: was 1200
            warmup_bars=env_data.get("warmup_bars", 14),
            initial_balance=env_data.get("initial_balance", 10000.0),
            transaction_cost_bps=env_data.get("transaction_cost_bps", 2.5),
            slippage_bps=env_data.get("slippage_bps", 2.5),
            max_drawdown_pct=env_data.get("max_drawdown_pct", 15.0),
            max_position_duration=env_data.get("max_position_duration", 864),   # V21: was 576
            stop_loss_pct=env_data.get("stop_loss_pct", -0.04),                 # V21: was -0.025
            take_profit_pct=env_data.get("take_profit_pct", 0.04),              # V21: was 0.03
            threshold_long=env_data.get("thresholds", {}).get("long", 0.35),    # V21: was 0.50
            threshold_short=env_data.get("thresholds", {}).get("short", -0.35), # V21: was -0.50
            trailing_stop_enabled=trailing.get("enabled", False),               # V21: was True
            trailing_stop_activation_pct=trailing.get("activation_pct", 0.015),
            trailing_stop_trail_factor=trailing.get("trail_factor", 0.5),
            trailing_stop_min_trail_pct=trailing.get("min_trail_pct", 0.007),
            trailing_stop_bonus=trailing.get("bonus", 0.25),
            reward_interval=training.get("reward_interval", 25),
            min_hold_bars=training.get("min_hold_bars", 25),  # V21.1
            decision_interval=env_data.get("decision_interval", 1),  # EXP-SWING-001
        )

        weights = reward_data.get("weights", {})
        pnl_transform = reward_data.get("pnl_transform", {})
        holding = reward_data.get("holding_decay", {})
        flat = reward_data.get("flat_reward", {})

        # V21: Reward weights match SSOT (PnL-dominant, phantom rewards disabled)
        self.reward = RewardConfig(
            weight_pnl=weights.get("pnl", 0.90),                    # V21: was 0.70
            weight_dsr=weights.get("dsr", 0.00),                    # V21: DISABLED (was 0.15)
            weight_sortino=weights.get("sortino", 0.00),            # V21: DISABLED (was 0.05)
            weight_regime_penalty=weights.get("regime_penalty", 0.05),  # V21: was 0.15
            weight_holding_decay=weights.get("holding_decay", 0.05),    # V21: was 0.20
            weight_anti_gaming=weights.get("anti_gaming", 0.00),        # V21: DISABLED (was 0.15)
            weight_flat_reward=flat.get("weight", 0.0),                 # V21: DISABLED (was 0.05)
            pnl_clip_range=tuple(pnl_transform.get("clip_range", [-0.1, 0.1])),
            asymmetric_loss_mult=pnl_transform.get("asymmetric", {}).get("loss_multiplier", 1.0),  # V21: symmetric (was 1.5)
            holding_decay_half_life=holding.get("half_life_bars", 576),     # V21: was 144
            holding_decay_max_penalty=holding.get("max_penalty", 0.15),     # V21: was 0.3
        )

        # Parse backtest config
        backtest = self._raw.get("backtest", {})
        costs = backtest.get("costs", {})
        thresholds = backtest.get("thresholds", {})
        risk = backtest.get("risk_management", {})
        gates = backtest.get("gates", {})
        trailing_bt = risk.get("trailing_stop", {})

        # V21: Backtest fallbacks MUST match training for parity
        self.backtest = BacktestConfig(
            transaction_cost_bps=costs.get("transaction_cost_bps", 2.5),
            slippage_bps=costs.get("slippage_bps", 2.5),
            threshold_long=thresholds.get("long", 0.35),            # V21: was 0.50
            threshold_short=thresholds.get("short", -0.35),          # V21: was -0.50
            stop_loss_pct=risk.get("stop_loss_pct", -0.04),          # V21: was -0.025
            take_profit_pct=risk.get("take_profit_pct", 0.04),       # V21: was 0.03
            trailing_stop_enabled=trailing_bt.get("enabled", False), # V21: was True
            trailing_stop_activation_pct=trailing_bt.get("activation_pct", 0.015),
            trailing_stop_trail_factor=trailing_bt.get("trail_factor", 0.5),
            max_position_duration=risk.get("max_position_duration", 864),  # V21: was 576
            initial_capital=backtest.get("initial_capital", 10000.0),
            min_return_pct=gates.get("min_return_pct", 0.0),         # V21: break-even (was -10.0)
            min_sharpe_ratio=gates.get("min_sharpe_ratio", 0.0),     # V21: was 0.3
            max_drawdown_pct=gates.get("max_drawdown_pct", 20.0),    # V21: was 25.0
            min_trades=gates.get("min_trades", 30),                  # V21: was 20
            min_win_rate=gates.get("min_win_rate", 0.35),            # V21: was 0.30
            max_action_imbalance=gates.get("max_action_imbalance", 0.85),
            decision_interval=backtest.get("decision_interval", 1),  # EXP-SWING-001
        )

        # V22: Parse new config sections
        self._parse_v22_config(training)

        # Parse features
        self._features = self._parse_features()

    def _parse_v22_config(self, training: Dict) -> None:
        """V22: Parse ensemble, LSTM, position sizing, close shaping, walk-forward, SAC."""
        # Algorithm name (backward compat: derive from lstm.enabled if absent)
        self._algorithm_name = training.get("algorithm", None)

        # SAC config (only parsed, used when algorithm="sac")
        sac_data = training.get("sac", {})
        ent_coef_raw = sac_data.get("ent_coef", "auto")
        target_entropy_raw = sac_data.get("target_entropy", "auto")
        self.sac = SACConfig(
            learning_rate=sac_data.get("learning_rate", 1e-4),
            buffer_size=sac_data.get("buffer_size", 200_000),
            learning_starts=sac_data.get("learning_starts", 10_000),
            batch_size=sac_data.get("batch_size", 512),
            tau=sac_data.get("tau", 0.005),
            gamma=sac_data.get("gamma", 0.99),
            ent_coef=str(ent_coef_raw),
            target_entropy=str(target_entropy_raw),
            train_freq=sac_data.get("train_freq", 4),
            gradient_steps=sac_data.get("gradient_steps", 4),
        )

        # Stop mode config
        env_data = training.get("environment", {})
        self.stop_mode = env_data.get("stop_mode", "fixed_pct")
        self.atr_stop = env_data.get("atr_stop", {})

        # Action interpretation config
        self.action_interpretation = env_data.get("action_interpretation", "threshold_3")
        self.zone_5_config = env_data.get("zone_5", {})

        # LSTM
        lstm_data = training.get("lstm", {})
        self.lstm = LSTMConfig(
            enabled=lstm_data.get("enabled", False),
            hidden_size=lstm_data.get("hidden_size", 128),
            n_layers=lstm_data.get("n_layers", 1),
            sequence_length=lstm_data.get("sequence_length", 64),
        )

        # Action type
        self.action_type = training.get("action_type", "continuous")
        self.n_actions = training.get("n_actions", 4)

        # Ensemble
        ens_data = training.get("ensemble", {})
        self.ensemble = EnsembleConfig(
            n_models=ens_data.get("n_models", 5),
            min_consensus=ens_data.get("min_consensus", 3),
            voting=ens_data.get("voting", "majority"),
            confidence_weighted=ens_data.get("confidence_weighted", True),
        )

        # Position sizing
        ps_data = training.get("position_sizing", {})
        self.position_sizing = PositionSizingConfig(
            method=ps_data.get("method", "half_kelly"),
            base_fraction=ps_data.get("base_fraction", 0.063),
            min_fraction=ps_data.get("min_fraction", 0.02),
            max_fraction=ps_data.get("max_fraction", 0.15),
            consensus_scaling=ps_data.get("consensus_scaling", True),
        )

        # Temporal features
        tf_data = training.get("temporal_features", {})
        self.temporal_features = TemporalFeaturesConfig(
            enabled=tf_data.get("enabled", True),
            features=tf_data.get("features", ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]),
        )

        # Close shaping
        cs_data = training.get("close_shaping", {})
        self.close_shaping = CloseShapingConfig(
            enabled=cs_data.get("enabled", False),
            stop_loss_mult=cs_data.get("stop_loss_mult", 1.5),
            take_profit_mult=cs_data.get("take_profit_mult", 1.2),
            agent_close_win=cs_data.get("agent_close_win", 1.1),
            agent_close_loss=cs_data.get("agent_close_loss", 0.7),
            timeout_mult=cs_data.get("timeout_mult", 0.8),
        )

        # Walk-forward
        wf_data = training.get("walk_forward", self._raw.get("training", {}).get("walk_forward", {}))
        self.walk_forward = WalkForwardConfig(
            enabled=wf_data.get("enabled", False),
            train_window=wf_data.get("train_window", "expanding"),
            retrain_frequency=wf_data.get("retrain_frequency", "quarterly"),
            min_train_bars=wf_data.get("min_train_bars", 50000),
            val_window_bars=wf_data.get("val_window_bars", 4680),
            test_window_bars=wf_data.get("test_window_bars", 4680),
            n_splits=wf_data.get("n_splits", 4),
        )

    def _parse_features(self) -> List[FeatureDefinition]:
        """Parse feature definitions from SSOT."""
        features = []

        features_section = self._raw.get("features", {})

        # Market features
        for f_data in features_section.get("market_features", []):
            features.append(FeatureDefinition.from_dict(f_data))

        # State features
        for f_data in features_section.get("state_features", []):
            f_data["source"] = "runtime"
            features.append(FeatureDefinition.from_dict(f_data))

        # Sort by order
        features.sort(key=lambda f: f.order)

        return features

    # =========================================================================
    # PUBLIC API - Methods for each layer
    # =========================================================================

    def get_feature_definitions(self) -> List[FeatureDefinition]:
        """Get all feature definitions sorted by order."""
        return self._features

    def get_market_features(self) -> List[FeatureDefinition]:
        """Get only market features (not state)."""
        return [f for f in self._features if not f.is_state]

    def get_state_features(self) -> List[FeatureDefinition]:
        """Get only state features."""
        return [f for f in self._features if f.is_state]

    def get_feature_order(self) -> Tuple[str, ...]:
        """Get feature names in correct order."""
        return tuple(f.name for f in self._features)

    def get_feature_by_name(self, name: str) -> Optional[FeatureDefinition]:
        """Get a specific feature by name."""
        for f in self._features:
            if f.name == name:
                return f
        return None

    def get_observation_dim(self) -> int:
        """Get total observation dimension."""
        return self._raw.get("features", {}).get("observation_dim", 20)

    def get_calculators(self) -> Dict[str, Dict[str, str]]:
        """Get calculator registry definition."""
        return self._raw.get("calculators", {})

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self._raw.get("preprocessing", {})

    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation gates configuration."""
        return self._raw.get("validation", {})

    def get_curriculum_config(self) -> Dict[str, Any]:
        """Get curriculum learning configuration."""
        return self._raw.get("training", {}).get("curriculum", {})

    def get_training_schedule(self) -> Dict[str, Any]:
        """Get training schedule (timesteps, checkpoints, etc.)."""
        return self._raw.get("training", {}).get("schedule", {})

    def get_trading_schedule(self) -> Dict[str, Any]:
        """Get trading schedule constants (bars_per_day, bars_per_year, etc.)."""
        return self._raw.get("trading_schedule", {})

    def get_macro_column_map(self) -> Dict[str, str]:
        """Get macro column mapping (source DB names → SSOT feature names)."""
        return self._raw.get("paths", {}).get("sources", {}).get("macro_column_map", {})

    def get_state_feature_names(self) -> List[str]:
        """Get state feature names list from SSOT."""
        return [f.name for f in self.get_state_features()]

    @property
    def algorithm(self) -> str:
        """Get algorithm name with backward compat.

        Priority:
        1. Explicit training.algorithm field
        2. lstm.enabled=true -> "recurrent_ppo"
        3. Default -> "ppo"
        """
        if self._algorithm_name:
            return self._algorithm_name
        if self.lstm.enabled:
            return "recurrent_ppo"
        return "ppo"

    def get_algorithm_config(self) -> Dict[str, Any]:
        """Get algorithm-specific hyperparameter dict for the factory."""
        from src.training.algorithm_factory import get_algorithm_kwargs
        return get_algorithm_kwargs(self)

    @property
    def version(self) -> str:
        """Get SSOT version."""
        return self._raw.get("_meta", {}).get("version", "unknown")

    @property
    def based_on_model(self) -> str:
        """Get the model this config is based on."""
        return self._raw.get("_meta", {}).get("based_on_model", "unknown")

    def validate_training_backtest_parity(self) -> List[str]:
        """
        Validate that training and backtest configs are aligned.
        Returns list of mismatches (empty if OK).
        """
        issues = []

        # Check costs
        if self.environment.transaction_cost_bps != self.backtest.transaction_cost_bps:
            issues.append(
                f"Transaction cost mismatch: training={self.environment.transaction_cost_bps} "
                f"vs backtest={self.backtest.transaction_cost_bps}"
            )

        if self.environment.slippage_bps != self.backtest.slippage_bps:
            issues.append(
                f"Slippage mismatch: training={self.environment.slippage_bps} "
                f"vs backtest={self.backtest.slippage_bps}"
            )

        # Check thresholds
        if self.environment.threshold_long != self.backtest.threshold_long:
            issues.append(
                f"Long threshold mismatch: training={self.environment.threshold_long} "
                f"vs backtest={self.backtest.threshold_long}"
            )

        if self.environment.threshold_short != self.backtest.threshold_short:
            issues.append(
                f"Short threshold mismatch: training={self.environment.threshold_short} "
                f"vs backtest={self.backtest.threshold_short}"
            )

        # Check risk management
        if self.environment.stop_loss_pct != self.backtest.stop_loss_pct:
            issues.append(
                f"Stop loss mismatch: training={self.environment.stop_loss_pct} "
                f"vs backtest={self.backtest.stop_loss_pct}"
            )

        if self.environment.take_profit_pct != self.backtest.take_profit_pct:
            issues.append(
                f"Take profit mismatch: training={self.environment.take_profit_pct} "
                f"vs backtest={self.backtest.take_profit_pct}"
            )

        if self.environment.trailing_stop_enabled != self.backtest.trailing_stop_enabled:
            issues.append(
                f"Trailing stop enabled mismatch: training={self.environment.trailing_stop_enabled} "
                f"vs backtest={self.backtest.trailing_stop_enabled}"
            )

        # EXP-SWING-001: Decision interval must match
        if self.environment.decision_interval != self.backtest.decision_interval:
            issues.append(
                f"Decision interval mismatch: training={self.environment.decision_interval} "
                f"vs backtest={self.backtest.decision_interval}"
            )

        return issues

    def to_dict(self) -> Dict[str, Any]:
        """Export full config as dictionary."""
        return self._raw


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_cached_pipeline_config: Optional[PipelineConfig] = None

def load_pipeline_config(config_path: Optional[str] = None) -> PipelineConfig:
    """
    Load pipeline configuration (cached).

    Resolution order:
        1. Explicit config_path argument
        2. PIPELINE_SSOT_PATH environment variable
        3. Default: config/pipeline_ssot.yaml

    Args:
        config_path: Optional path to pipeline_ssot.yaml

    Returns:
        PipelineConfig instance
    """
    global _cached_pipeline_config

    import os
    resolved_path = config_path or os.environ.get("PIPELINE_SSOT_PATH")
    path = Path(resolved_path) if resolved_path else None

    # Return cached if same path (or both None)
    if _cached_pipeline_config is not None:
        cached_path = str(_cached_pipeline_config._config_path)
        requested_path = str(path) if path else str(Path(__file__).parent.parent.parent / "config" / "pipeline_ssot.yaml")
        if cached_path == requested_path:
            return _cached_pipeline_config

    _cached_pipeline_config = PipelineConfig(path)
    return _cached_pipeline_config


def get_feature_order() -> Tuple[str, ...]:
    """Quick access to feature order."""
    return load_pipeline_config().get_feature_order()


def get_observation_dim() -> int:
    """Quick access to observation dimension."""
    return load_pipeline_config().get_observation_dim()


def validate_parity() -> bool:
    """
    Validate training/backtest parity.
    Raises ValueError if mismatched.
    """
    config = load_pipeline_config()
    issues = config.validate_training_backtest_parity()

    if issues:
        error_msg = "Training/Backtest parity violations:\n" + "\n".join(f"  - {i}" for i in issues)
        raise ValueError(error_msg)

    logger.info("Training/Backtest parity validated successfully")
    return True


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    try:
        config = load_pipeline_config()

        print(f"\n{'='*60}")
        print(f"Pipeline SSOT v{config.version}")
        print(f"Based on model: {config.based_on_model}")
        print(f"{'='*60}")

        print(f"\n[Features] Total: {len(config.get_feature_definitions())}")
        print(f"   Market: {len(config.get_market_features())}")
        print(f"   State: {len(config.get_state_features())}")
        print(f"   Observation dim: {config.get_observation_dim()}")

        print(f"\n[Training Config]")
        print(f"   PPO LR: {config.ppo.learning_rate}")
        print(f"   Timesteps: {config.get_training_schedule().get('total_timesteps')}")
        print(f"   Transaction cost: {config.environment.transaction_cost_bps} bps")
        print(f"   Thresholds: [{config.environment.threshold_short}, {config.environment.threshold_long}]")

        print(f"\n[Backtest Config]")
        print(f"   Transaction cost: {config.backtest.transaction_cost_bps} bps")
        print(f"   Thresholds: [{config.backtest.threshold_short}, {config.backtest.threshold_long}]")
        print(f"   Stop loss: {config.backtest.stop_loss_pct*100}%")
        print(f"   Take profit: {config.backtest.take_profit_pct*100}%")

        print(f"\n[Parity Check]")
        issues = config.validate_training_backtest_parity()
        if issues:
            for issue in issues:
                print(f"   [X] {issue}")
        else:
            print(f"   [OK] Training and Backtest configs are aligned")

        print(f"\n[Date Ranges]")
        print(f"   Train: {config.date_ranges.train_start} -> {config.date_ranges.train_end}")
        print(f"   Val:   {config.date_ranges.val_start} -> {config.date_ranges.val_end}")
        print(f"   Test:  {config.date_ranges.test_start} -> {config.date_ranges.test_end}")

        print(f"\n{'='*60}\n")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
