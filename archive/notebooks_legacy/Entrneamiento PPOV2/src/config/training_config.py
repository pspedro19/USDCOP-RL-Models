"""
USD/COP RL Trading System - Training Configuration V19
========================================================

Sistema de configuracion centralizado con:
- Dataclasses tipados con validacion
- Defaults sensatos de produccion
- Soporte JSON/YAML
- Compatibilidad con codigo existente

ESTRUCTURA:
- TrainingConfigV19: Configuracion principal
  - PPOConfig / SACConfig: Hyperparameters
  - NetworkConfig: Arquitectura de red
  - EnvironmentConfig: Parametros del environment
  - RewardConfig: Reward function
  - ValidationConfig: Cross-validation
  - CallbackConfig: Callbacks de SB3
  - RiskConfig: Risk management
  - AcceptanceConfig: Criterios de aceptacion
  - DataConfig: Preprocesamiento

Author: Claude Code
Version: 1.0.0
"""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum, IntEnum
import warnings

# Importar defaults
from .defaults import (
    PRODUCTION_DEFAULTS,
    PPO_DEFAULTS,
    SAC_DEFAULTS,
    NETWORK_DEFAULTS,
    ENVIRONMENT_DEFAULTS,
    REWARD_DEFAULTS,
    VALIDATION_DEFAULTS,
    CALLBACK_DEFAULTS,
    RISK_DEFAULTS,
    ACCEPTANCE_DEFAULTS,
    DATA_DEFAULTS,
    COST_MODEL_DEFAULTS,
    VOL_SCALER_DEFAULTS,
    REGIME_DETECTOR_DEFAULTS,
)


# =============================================================================
# ENUMS
# =============================================================================

class TrainingPhase(IntEnum):
    """Fases del curriculum learning."""
    EXPLORATION = 0   # Costos 0, exploracion libre
    TRANSITION = 1    # Costos graduales
    REALISTIC = 2     # Costos completos


class RewardType(str, Enum):
    """Tipos de reward function."""
    SYMMETRIC = "symmetric"
    ALPHA = "alpha"
    ALPHA_V2 = "alpha_v2"
    SORTINO = "sortino"
    SIMPLE = "simple"


class ProtectionMode(str, Enum):
    """Modos de proteccion de posicion."""
    MIN = "min"          # Usar minimo de vol_scaler y regime_detector
    MULTIPLY = "multiply"  # Multiplicar ambos
    VOL_ONLY = "vol_only"  # Solo volatility scaler
    REGIME_ONLY = "regime_only"  # Solo regime detector


# =============================================================================
# PPO CONFIGURATION
# =============================================================================

@dataclass
class PPOConfig:
    """Configuracion de hyperparameters PPO."""

    learning_rate: float = PPO_DEFAULTS["learning_rate"]
    n_steps: int = PPO_DEFAULTS["n_steps"]
    batch_size: int = PPO_DEFAULTS["batch_size"]
    n_epochs: int = PPO_DEFAULTS["n_epochs"]
    gamma: float = PPO_DEFAULTS["gamma"]
    gae_lambda: float = PPO_DEFAULTS["gae_lambda"]
    clip_range: float = PPO_DEFAULTS["clip_range"]
    ent_coef: float = PPO_DEFAULTS["ent_coef"]
    vf_coef: float = PPO_DEFAULTS["vf_coef"]
    max_grad_norm: float = PPO_DEFAULTS["max_grad_norm"]
    normalize_advantage: bool = PPO_DEFAULTS["normalize_advantage"]

    def __post_init__(self):
        """Validar configuracion."""
        if not 1e-6 < self.learning_rate < 1e-2:
            warnings.warn(
                f"Learning rate {self.learning_rate} outside typical range [1e-6, 1e-2]"
            )

        if self.batch_size > self.n_steps:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be <= n_steps ({self.n_steps})"
            )

        if not 0 <= self.gamma <= 1:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")

        if not 0 <= self.clip_range <= 1:
            raise ValueError(f"clip_range must be in [0, 1], got {self.clip_range}")


# =============================================================================
# SAC CONFIGURATION
# =============================================================================

@dataclass
class SACConfig:
    """Configuracion de hyperparameters SAC."""

    learning_rate: float = SAC_DEFAULTS["learning_rate"]
    buffer_size: int = SAC_DEFAULTS["buffer_size"]
    learning_starts: int = SAC_DEFAULTS["learning_starts"]
    batch_size: int = SAC_DEFAULTS["batch_size"]
    tau: float = SAC_DEFAULTS["tau"]
    gamma: float = SAC_DEFAULTS["gamma"]
    ent_coef: Union[str, float] = SAC_DEFAULTS["ent_coef"]
    train_freq: int = SAC_DEFAULTS["train_freq"]
    gradient_steps: int = SAC_DEFAULTS["gradient_steps"]

    def __post_init__(self):
        if not 0 <= self.gamma <= 1:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")

        if not 0 < self.tau <= 1:
            raise ValueError(f"tau must be in (0, 1], got {self.tau}")


# =============================================================================
# NETWORK CONFIGURATION
# =============================================================================

@dataclass
class NetworkConfig:
    """Configuracion de arquitectura de red."""

    net_arch: List[int] = field(
        default_factory=lambda: NETWORK_DEFAULTS["net_arch"].copy()
    )
    activation_fn: str = NETWORK_DEFAULTS["activation_fn"]
    separate_networks: bool = NETWORK_DEFAULTS["separate_networks"]

    def __post_init__(self):
        if not self.net_arch:
            raise ValueError("net_arch cannot be empty")

        valid_activations = ["Tanh", "ReLU", "LeakyReLU", "ELU", "GELU"]
        if self.activation_fn not in valid_activations:
            warnings.warn(
                f"activation_fn '{self.activation_fn}' not in common list: {valid_activations}"
            )


# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

@dataclass
class EnvironmentConfig:
    """Configuracion del trading environment."""

    initial_balance: float = ENVIRONMENT_DEFAULTS["initial_balance"]
    max_position: float = ENVIRONMENT_DEFAULTS["max_position"]
    episode_length: int = ENVIRONMENT_DEFAULTS["episode_length"]
    max_drawdown_pct: float = ENVIRONMENT_DEFAULTS["max_drawdown_pct"]
    timeframe: str = ENVIRONMENT_DEFAULTS["timeframe"]
    bars_per_day: int = ENVIRONMENT_DEFAULTS["bars_per_day"]

    # Position protection
    use_vol_scaling: bool = ENVIRONMENT_DEFAULTS["use_vol_scaling"]
    use_regime_detection: bool = ENVIRONMENT_DEFAULTS["use_regime_detection"]
    protection_mode: str = ENVIRONMENT_DEFAULTS["protection_mode"]

    # Column names
    volatility_column: str = ENVIRONMENT_DEFAULTS["volatility_column"]
    return_column: str = ENVIRONMENT_DEFAULTS["return_column"]
    vol_feature_column: str = ENVIRONMENT_DEFAULTS["vol_feature_column"]
    vix_column: str = ENVIRONMENT_DEFAULTS["vix_column"]
    embi_column: str = ENVIRONMENT_DEFAULTS["embi_column"]

    def __post_init__(self):
        if self.initial_balance <= 0:
            raise ValueError(f"initial_balance must be > 0, got {self.initial_balance}")

        if not 0 < self.max_position <= 1:
            raise ValueError(f"max_position must be in (0, 1], got {self.max_position}")

        if self.episode_length < 10:
            raise ValueError(f"episode_length too small: {self.episode_length}")

        valid_modes = ["min", "multiply", "vol_only", "regime_only"]
        if self.protection_mode not in valid_modes:
            raise ValueError(f"protection_mode must be one of {valid_modes}")


# =============================================================================
# REWARD CONFIGURATION
# =============================================================================

@dataclass
class RewardConfig:
    """Configuracion de la reward function."""

    # Tipo de reward
    reward_type: str = REWARD_DEFAULTS["reward_type"]

    # Curriculum phases
    phase_boundaries: Tuple[float, float] = field(
        default_factory=lambda: REWARD_DEFAULTS["phase_boundaries"]
    )

    # Costos por fase
    transition_target_cost_bps: float = REWARD_DEFAULTS["transition_target_cost_bps"]
    realistic_min_cost_bps: float = REWARD_DEFAULTS["realistic_min_cost_bps"]
    realistic_max_cost_bps: float = REWARD_DEFAULTS["realistic_max_cost_bps"]

    # Symmetry
    symmetry_window: int = REWARD_DEFAULTS["symmetry_window"]
    max_directional_bias: float = REWARD_DEFAULTS["max_directional_bias"]
    symmetry_penalty_scale: float = REWARD_DEFAULTS["symmetry_penalty_scale"]

    # Pathological behavior detection
    max_trades_per_bar: float = REWARD_DEFAULTS["max_trades_per_bar"]
    overtrading_lookback: int = REWARD_DEFAULTS["overtrading_lookback"]
    overtrading_penalty: float = REWARD_DEFAULTS["overtrading_penalty"]
    max_hold_duration: int = REWARD_DEFAULTS["max_hold_duration"]
    inactivity_penalty: float = REWARD_DEFAULTS["inactivity_penalty"]
    reversal_threshold: int = REWARD_DEFAULTS["reversal_threshold"]
    churning_penalty: float = REWARD_DEFAULTS["churning_penalty"]

    # Sortino config
    sortino_window: int = REWARD_DEFAULTS["sortino_window"]
    sortino_mar: float = REWARD_DEFAULTS["sortino_mar"]

    # Scaling
    reward_scale: float = REWARD_DEFAULTS["reward_scale"]
    clip_range: Tuple[float, float] = field(
        default_factory=lambda: REWARD_DEFAULTS["clip_range"]
    )

    def __post_init__(self):
        if len(self.phase_boundaries) != 2:
            raise ValueError("phase_boundaries must have 2 elements")

        if not 0 <= self.phase_boundaries[0] < self.phase_boundaries[1] <= 1:
            raise ValueError(
                f"phase_boundaries must be (0 <= p1 < p2 <= 1), got {self.phase_boundaries}"
            )

        valid_types = ["symmetric", "alpha", "alpha_v2", "sortino", "simple"]
        if self.reward_type not in valid_types:
            raise ValueError(f"reward_type must be one of {valid_types}")


# =============================================================================
# VALIDATION CONFIGURATION
# =============================================================================

@dataclass
class ValidationConfig:
    """Configuracion de validation y cross-validation."""

    # Data split
    train_pct: float = VALIDATION_DEFAULTS["train_pct"]
    val_pct: float = VALIDATION_DEFAULTS["val_pct"]
    test_pct: float = VALIDATION_DEFAULTS["test_pct"]

    # Embargo
    embargo_bars: int = VALIDATION_DEFAULTS["embargo_bars"]
    embargo_days: int = VALIDATION_DEFAULTS["embargo_days"]

    # K-Fold
    n_splits: int = VALIDATION_DEFAULTS["n_splits"]
    purge_bars: int = VALIDATION_DEFAULTS["purge_bars"]

    # Walk-forward
    wf_train_size: Optional[int] = VALIDATION_DEFAULTS["wf_train_size"]
    wf_test_size: Optional[int] = VALIDATION_DEFAULTS["wf_test_size"]
    wf_expanding: bool = VALIDATION_DEFAULTS["wf_expanding"]

    # Minimums
    min_train_size: int = VALIDATION_DEFAULTS["min_train_size"]
    min_test_size: int = VALIDATION_DEFAULTS["min_test_size"]

    def __post_init__(self):
        total = self.train_pct + self.val_pct + self.test_pct
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"train_pct + val_pct + test_pct must equal 1.0, got {total}"
            )

        if self.n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {self.n_splits}")


# =============================================================================
# CALLBACK CONFIGURATION
# =============================================================================

@dataclass
class CallbackConfig:
    """Configuracion de callbacks de SB3."""

    # Evaluation
    eval_freq: int = CALLBACK_DEFAULTS["eval_freq"]
    n_eval_episodes: int = CALLBACK_DEFAULTS["n_eval_episodes"]

    # Early stopping
    patience: int = CALLBACK_DEFAULTS["patience"]
    min_delta: float = CALLBACK_DEFAULTS["min_delta"]
    min_sharpe: float = CALLBACK_DEFAULTS["min_sharpe"]

    # Entropy scheduling
    init_entropy: float = CALLBACK_DEFAULTS["init_entropy"]
    final_entropy: float = CALLBACK_DEFAULTS["final_entropy"]
    entropy_schedule: str = CALLBACK_DEFAULTS["entropy_schedule"]
    warmup_fraction: float = CALLBACK_DEFAULTS["warmup_fraction"]

    # Action monitoring
    action_log_freq: int = CALLBACK_DEFAULTS["action_log_freq"]
    collapse_threshold: float = CALLBACK_DEFAULTS["collapse_threshold"]
    hold_warning_threshold: float = CALLBACK_DEFAULTS["hold_warning_threshold"]

    # Cost curriculum
    cost_warmup_steps: int = CALLBACK_DEFAULTS["cost_warmup_steps"]
    cost_rampup_steps: int = CALLBACK_DEFAULTS["cost_rampup_steps"]
    cost_final: float = CALLBACK_DEFAULTS["cost_final"]
    cost_crisis_multiplier: float = CALLBACK_DEFAULTS["cost_crisis_multiplier"]

    def __post_init__(self):
        if self.eval_freq < 100:
            warnings.warn(f"eval_freq {self.eval_freq} is very low")

        if not 0 < self.init_entropy < 1:
            raise ValueError(f"init_entropy must be in (0, 1), got {self.init_entropy}")

        if not 0 < self.final_entropy < self.init_entropy:
            raise ValueError(
                f"final_entropy must be in (0, init_entropy), "
                f"got {self.final_entropy} vs {self.init_entropy}"
            )


# =============================================================================
# RISK CONFIGURATION
# =============================================================================

@dataclass
class RiskConfig:
    """Configuracion de risk management."""

    # Drawdown limits
    max_drawdown_warning: float = RISK_DEFAULTS["max_drawdown_warning"]
    max_drawdown_reduce: float = RISK_DEFAULTS["max_drawdown_reduce"]
    max_drawdown_stop: float = RISK_DEFAULTS["max_drawdown_stop"]

    # Performance limits
    min_sharpe_30d: float = RISK_DEFAULTS["min_sharpe_30d"]
    max_hold_pct: float = RISK_DEFAULTS["max_hold_pct"]

    # Loss limits
    max_consecutive_losses: int = RISK_DEFAULTS["max_consecutive_losses"]
    max_daily_loss: float = RISK_DEFAULTS["max_daily_loss"]

    # Recovery
    auto_resume_after_recovery: bool = RISK_DEFAULTS["auto_resume_after_recovery"]
    recovery_sharpe_threshold: float = RISK_DEFAULTS["recovery_sharpe_threshold"]

    # Rolling window
    rolling_window_days: int = RISK_DEFAULTS["rolling_window_days"]

    def __post_init__(self):
        if not (self.max_drawdown_warning <
                self.max_drawdown_reduce <
                self.max_drawdown_stop):
            raise ValueError(
                "Drawdown thresholds must be in order: warning < reduce < stop"
            )


# =============================================================================
# ACCEPTANCE CRITERIA
# =============================================================================

@dataclass
class AcceptanceConfig:
    """Criterios de aceptacion para modelos."""

    min_sharpe: float = ACCEPTANCE_DEFAULTS["min_sharpe"]
    max_drawdown: float = ACCEPTANCE_DEFAULTS["max_drawdown"]
    min_calmar: float = ACCEPTANCE_DEFAULTS["min_calmar"]
    min_profit_factor: float = ACCEPTANCE_DEFAULTS["min_profit_factor"]
    min_win_rate: float = ACCEPTANCE_DEFAULTS["min_win_rate"]
    max_hold_pct: float = ACCEPTANCE_DEFAULTS["max_hold_pct"]
    min_stress_periods_passed: int = ACCEPTANCE_DEFAULTS["min_stress_periods_passed"]
    stress_pass_rate: float = ACCEPTANCE_DEFAULTS["stress_pass_rate"]

    def __post_init__(self):
        if self.min_sharpe < 0:
            warnings.warn(
                f"min_sharpe {self.min_sharpe} is negative. "
                "Consider using a positive threshold."
            )


# =============================================================================
# DATA CONFIGURATION
# =============================================================================

@dataclass
class DataConfig:
    """Configuracion de preprocesamiento de datos."""

    warmup_days: int = DATA_DEFAULTS["warmup_days"]
    features_to_drop: List[str] = field(
        default_factory=lambda: DATA_DEFAULTS["features_to_drop"].copy()
    )
    features_to_normalize: List[str] = field(
        default_factory=lambda: DATA_DEFAULTS["features_to_normalize"].copy()
    )
    return_columns: List[str] = field(
        default_factory=lambda: DATA_DEFAULTS["return_columns"].copy()
    )
    winsorize_percentile: float = DATA_DEFAULTS["winsorize_percentile"]
    zero_threshold: float = DATA_DEFAULTS["zero_threshold"]

    def __post_init__(self):
        if self.warmup_days < 0:
            raise ValueError(f"warmup_days must be >= 0, got {self.warmup_days}")

        if not 0 < self.winsorize_percentile < 0.5:
            raise ValueError(
                f"winsorize_percentile must be in (0, 0.5), got {self.winsorize_percentile}"
            )


# =============================================================================
# MAIN TRAINING CONFIG
# =============================================================================

@dataclass
class TrainingConfigV19:
    """
    Configuracion principal de training V19.

    Agrupa todas las sub-configuraciones en una estructura coherente.

    Usage:
        # Default config
        config = TrainingConfigV19()

        # From JSON
        config = load_config("path/to/config.json")

        # Custom
        config = TrainingConfigV19(
            algorithm="PPO",
            total_timesteps=100_000,
            ppo=PPOConfig(learning_rate=3e-4),
        )
    """

    # Metadatos
    model_name: str = PRODUCTION_DEFAULTS["model_name"]
    algorithm: str = PRODUCTION_DEFAULTS["algorithm"]
    version: str = PRODUCTION_DEFAULTS["version"]
    total_timesteps: int = PRODUCTION_DEFAULTS["total_timesteps"]
    seed: int = PRODUCTION_DEFAULTS["seed"]
    device: str = PRODUCTION_DEFAULTS["device"]

    # Sub-configs
    ppo: PPOConfig = field(default_factory=PPOConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    callbacks: CallbackConfig = field(default_factory=CallbackConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    acceptance: AcceptanceConfig = field(default_factory=AcceptanceConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self):
        """Validar configuracion completa."""
        if self.algorithm not in ["PPO", "SAC"]:
            raise ValueError(f"algorithm must be 'PPO' or 'SAC', got {self.algorithm}")

        if self.total_timesteps < 1000:
            warnings.warn(f"total_timesteps {self.total_timesteps} is very low")

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return asdict(self)

    def to_json(self, path: str, indent: int = 2):
        """Guardar como JSON."""
        data = self.to_dict()

        # Convertir tuples a lists para JSON
        def convert_tuples(obj):
            if isinstance(obj, dict):
                return {k: convert_tuples(v) for k, v in obj.items()}
            elif isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, list):
                return [convert_tuples(v) for v in obj]
            return obj

        data = convert_tuples(data)

        with open(path, 'w') as f:
            json.dump(data, f, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingConfigV19':
        """Crear desde diccionario."""
        # Extraer sub-configs
        ppo_data = data.pop('ppo', {})
        sac_data = data.pop('sac', {})
        network_data = data.pop('network', {})
        environment_data = data.pop('environment', {})
        reward_data = data.pop('reward', {})
        validation_data = data.pop('validation', {})
        callbacks_data = data.pop('callbacks', {})
        risk_data = data.pop('risk', {})
        acceptance_data = data.pop('acceptance', {})
        data_config = data.pop('data', {})

        # Convertir listas a tuples donde corresponda
        if 'phase_boundaries' in reward_data and isinstance(reward_data['phase_boundaries'], list):
            reward_data['phase_boundaries'] = tuple(reward_data['phase_boundaries'])
        if 'clip_range' in reward_data and isinstance(reward_data['clip_range'], list):
            reward_data['clip_range'] = tuple(reward_data['clip_range'])

        return cls(
            ppo=PPOConfig(**ppo_data) if ppo_data else PPOConfig(),
            sac=SACConfig(**sac_data) if sac_data else SACConfig(),
            network=NetworkConfig(**network_data) if network_data else NetworkConfig(),
            environment=EnvironmentConfig(**environment_data) if environment_data else EnvironmentConfig(),
            reward=RewardConfig(**reward_data) if reward_data else RewardConfig(),
            validation=ValidationConfig(**validation_data) if validation_data else ValidationConfig(),
            callbacks=CallbackConfig(**callbacks_data) if callbacks_data else CallbackConfig(),
            risk=RiskConfig(**risk_data) if risk_data else RiskConfig(),
            acceptance=AcceptanceConfig(**acceptance_data) if acceptance_data else AcceptanceConfig(),
            data=DataConfig(**data_config) if data_config else DataConfig(),
            **data
        )

    def get_ppo_kwargs(self) -> Dict[str, Any]:
        """Obtener kwargs para PPO de SB3."""
        return {
            "learning_rate": self.ppo.learning_rate,
            "n_steps": self.ppo.n_steps,
            "batch_size": self.ppo.batch_size,
            "n_epochs": self.ppo.n_epochs,
            "gamma": self.ppo.gamma,
            "gae_lambda": self.ppo.gae_lambda,
            "clip_range": self.ppo.clip_range,
            "ent_coef": self.ppo.ent_coef,
            "vf_coef": self.ppo.vf_coef,
            "max_grad_norm": self.ppo.max_grad_norm,
            "normalize_advantage": self.ppo.normalize_advantage,
        }

    def get_sac_kwargs(self) -> Dict[str, Any]:
        """Obtener kwargs para SAC de SB3."""
        return {
            "learning_rate": self.sac.learning_rate,
            "buffer_size": self.sac.buffer_size,
            "learning_starts": self.sac.learning_starts,
            "batch_size": self.sac.batch_size,
            "tau": self.sac.tau,
            "gamma": self.sac.gamma,
            "ent_coef": self.sac.ent_coef,
            "train_freq": self.sac.train_freq,
            "gradient_steps": self.sac.gradient_steps,
        }

    def get_policy_kwargs(self) -> Dict[str, Any]:
        """Obtener policy_kwargs para SB3."""
        net_arch = self.network.net_arch

        if self.algorithm == "PPO" and self.network.separate_networks:
            return {
                "net_arch": dict(pi=net_arch, vf=net_arch),
            }
        else:
            return {
                "net_arch": net_arch,
            }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def load_config(path: str) -> TrainingConfigV19:
    """
    Cargar configuracion desde archivo JSON o YAML.

    Args:
        path: Path al archivo de configuracion

    Returns:
        TrainingConfigV19 configurado
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
    elif path.suffix in ['.yaml', '.yml']:
        try:
            import yaml
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML required for YAML configs: pip install pyyaml")
    else:
        raise ValueError(f"Unknown config format: {path.suffix}")

    return TrainingConfigV19.from_dict(data)


def create_production_config() -> TrainingConfigV19:
    """
    Crear configuracion de produccion validada.

    Basada en PRODUCTION_CONFIG.json con Sharpe 2.21.

    Returns:
        TrainingConfigV19 optimizado para produccion
    """
    return TrainingConfigV19(
        model_name="Model B (Aggressive) - Production",
        algorithm="PPO",
        total_timesteps=80_000,
        seed=42,
        ppo=PPOConfig(
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.05,
            clip_range=0.2,
        ),
        network=NetworkConfig(
            net_arch=[256, 256],
            activation_fn="Tanh",
        ),
        environment=EnvironmentConfig(
            initial_balance=10000,
            max_position=1.0,
            episode_length=400,
            use_vol_scaling=True,
            use_regime_detection=True,
            bars_per_day=56,
        ),
        reward=RewardConfig(
            reward_type="symmetric",
            phase_boundaries=(0.30, 0.60),
        ),
    )


def create_debug_config() -> TrainingConfigV19:
    """
    Crear configuracion para debugging rapido.

    Timesteps reducidos, episodios cortos.

    Returns:
        TrainingConfigV19 para debug
    """
    config = TrainingConfigV19(
        model_name="Debug Config",
        algorithm="PPO",
        total_timesteps=10_000,
        seed=42,
        ppo=PPOConfig(
            n_steps=512,
            batch_size=64,
            n_epochs=4,
        ),
        environment=EnvironmentConfig(
            episode_length=100,
        ),
        callbacks=CallbackConfig(
            eval_freq=2_000,
            patience=3,
        ),
    )

    return config


def create_stress_test_config() -> TrainingConfigV19:
    """
    Crear configuracion para stress testing.

    Episodios largos, validacion exhaustiva.

    Returns:
        TrainingConfigV19 para stress testing
    """
    config = create_production_config()

    config.model_name = "Stress Test Config"
    config.environment.episode_length = 1200  # Episodios largos
    config.validation.n_splits = 7  # Mas folds
    config.acceptance.min_stress_periods_passed = 4
    config.acceptance.stress_pass_rate = 0.60

    return config


# =============================================================================
# COMPATIBILITY LAYER
# =============================================================================

def get_env_kwargs_from_config(config: TrainingConfigV19) -> Dict[str, Any]:
    """
    Extraer kwargs para TradingEnvironmentV19.

    Mantiene compatibilidad con codigo existente.

    Args:
        config: TrainingConfigV19

    Returns:
        Dict con kwargs para el environment
    """
    env_cfg = config.environment

    return {
        "initial_balance": env_cfg.initial_balance,
        "max_position": env_cfg.max_position,
        "episode_length": env_cfg.episode_length,
        "max_drawdown_pct": env_cfg.max_drawdown_pct,
        "use_vol_scaling": env_cfg.use_vol_scaling,
        "use_regime_detection": env_cfg.use_regime_detection,
        "protection_mode": env_cfg.protection_mode,
        "volatility_column": env_cfg.volatility_column,
        "return_column": env_cfg.return_column,
        "vol_feature_column": env_cfg.vol_feature_column,
        "vix_column": env_cfg.vix_column,
        "embi_column": env_cfg.embi_column,
    }


def get_reward_kwargs_from_config(config: TrainingConfigV19) -> Dict[str, Any]:
    """
    Extraer kwargs para reward function.

    Args:
        config: TrainingConfigV19

    Returns:
        Dict con kwargs para la reward function
    """
    reward_cfg = config.reward

    return {
        "reward_type": reward_cfg.reward_type,
        "phase_boundaries": reward_cfg.phase_boundaries,
        "total_timesteps": config.total_timesteps,
        "symmetry_window": reward_cfg.symmetry_window,
        "max_directional_bias": reward_cfg.max_directional_bias,
        "sortino_window": reward_cfg.sortino_window,
        "reward_scale": reward_cfg.reward_scale,
        "clip_range": reward_cfg.clip_range,
    }


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("USD/COP RL Trading System - Configuration Test")
    print("=" * 70)

    # Test default config
    print("\n1. Default Config:")
    config = TrainingConfigV19()
    print(f"   Algorithm: {config.algorithm}")
    print(f"   Timesteps: {config.total_timesteps:,}")
    print(f"   Learning rate: {config.ppo.learning_rate}")
    print(f"   Network: {config.network.net_arch}")

    # Test production config
    print("\n2. Production Config:")
    prod_config = create_production_config()
    print(f"   Model: {prod_config.model_name}")
    print(f"   Episode length: {prod_config.environment.episode_length}")
    print(f"   Reward type: {prod_config.reward.reward_type}")

    # Test debug config
    print("\n3. Debug Config:")
    debug_config = create_debug_config()
    print(f"   Timesteps: {debug_config.total_timesteps:,}")
    print(f"   Eval freq: {debug_config.callbacks.eval_freq:,}")

    # Test validation
    print("\n4. Validation Tests:")
    try:
        bad_config = PPOConfig(learning_rate=-1)
    except Exception as e:
        print(f"   PPOConfig validation: PASS (caught: {type(e).__name__})")

    try:
        bad_env = EnvironmentConfig(max_position=2.0)
    except Exception as e:
        print(f"   EnvironmentConfig validation: PASS (caught: {type(e).__name__})")

    # Test serialization
    print("\n5. Serialization:")
    config_dict = config.to_dict()
    print(f"   to_dict keys: {len(config_dict)}")

    restored = TrainingConfigV19.from_dict(config_dict)
    print(f"   from_dict: {restored.algorithm}")

    print("\n" + "=" * 70)
    print("Configuration system ready!")
    print("=" * 70)
