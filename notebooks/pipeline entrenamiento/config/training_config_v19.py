"""
USD/COP RL Trading System - Training Configuration V19
=======================================================

Configuración centralizada con todas las mejoras:
- Curriculum learning para costos y entropy
- Hiperparámetros optimizados
- Criterios de aceptación definidos
- Períodos de crisis para stress testing

Author: Claude Code
Version: 19.0.0
Date: 2024-12-24
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from enum import Enum
import torch.nn as nn


# =============================================================================
# ENUMS
# =============================================================================

class TrainingPhase(Enum):
    """Fases del curriculum learning."""
    EXPLORATION = 0      # Sin costos, alta entropy
    TRANSITION = 1       # Costos parciales, entropy media
    REALISTIC = 2        # Costos reales, entropy baja


class MarketRegime(Enum):
    """Regímenes de mercado."""
    TRENDING = 0
    MEAN_REVERTING = 1
    CRISIS = 2


# =============================================================================
# CONFIGURACIÓN DE REWARD
# =============================================================================

@dataclass
class RewardConfig:
    """Configuración de la reward function."""

    # Curriculum phases (% del training)
    phase_boundaries: Tuple[float, float] = (0.30, 0.60)

    # Cost schedule
    transition_target_cost_bps: float = 10.0
    realistic_min_cost_bps: float = 25.0
    realistic_max_cost_bps: float = 36.0

    # Symmetry parameters
    symmetry_window: int = 60
    max_directional_bias: float = 0.30
    symmetry_penalty_scale: float = 2.0

    # Anti-pathological
    max_trades_per_bar: float = 0.05
    overtrading_lookback: int = 120
    overtrading_penalty: float = 0.5
    max_hold_duration: int = 36
    inactivity_penalty: float = 0.3
    reversal_threshold: int = 5
    churning_penalty: float = 0.4

    # Sortino parameters
    sortino_window: int = 60
    sortino_mar: float = 0.0

    # Scaling
    reward_scale: float = 100.0
    clip_range: Tuple[float, float] = (-5.0, 5.0)


# =============================================================================
# CONFIGURACIÓN DE ENVIRONMENT
# =============================================================================

@dataclass
class EnvironmentConfig:
    """Configuración del environment V19."""

    # Timeframe configuration
    timeframe: str = '15min'  # '5min' o '15min' - NUEVO
    bars_per_day: int = 20    # 20 para 15min, 60 para 5min

    # Observation dimensions
    n_market_features: int = 23
    n_state_features: int = 12  # Mejorado de 2 a 12

    # Termination thresholds
    max_drawdown_threshold: float = 0.15  # 15% (era 50%)
    max_drawdown_pct: float = 15.0  # Alias para compatibilidad (%)
    min_portfolio_threshold: float = 0.85

    # Episode
    episode_length: int = 400   # 20 días @ 20 bars/día (15min)
    warmup_bars: int = 20       # Ajustado para 15min

    # Portfolio
    initial_balance: float = 10000.0
    max_position: float = 1.0  # Máxima posición permitida

    # Cost model
    use_dynamic_costs: bool = True
    base_cost_bps: float = 25.0

    # Normalization
    clip_observations: float = 5.0
    pnl_scale: float = 100.0

    # Trading session (SET-FX Colombia)
    session_start_hour: int = 8
    session_end_hour: int = 13
    timezone: str = 'America/Bogota'

    def __post_init__(self):
        """Ajustar configuración según timeframe."""
        if self.timeframe == '5min':
            self.bars_per_day = 60
            self.episode_length = 1200  # 20 días @ 60 bars/día
            self.warmup_bars = 60
        elif self.timeframe == '15min':
            self.bars_per_day = 20
            self.episode_length = 400   # 20 días @ 20 bars/día
            self.warmup_bars = 20
        elif self.timeframe == '1h':
            self.bars_per_day = 5
            self.episode_length = 100   # 20 días @ 5 bars/día
            self.warmup_bars = 5


# =============================================================================
# CONFIGURACIÓN DE CALLBACKS
# =============================================================================

@dataclass
class CallbacksConfig:
    """Configuración de callbacks de training."""

    # SharpeEvalCallback
    eval_freq: int = 5_000
    n_eval_episodes: int = 5
    patience: int = 10
    min_delta: float = 0.05
    min_sharpe: float = 0.3

    # ActionDistributionCallback
    action_log_freq: int = 1_000
    action_window_size: int = 5_000
    collapse_threshold: float = 0.05
    hold_warning_threshold: float = 70.0
    bias_warning_threshold: float = 0.4

    # EntropySchedulerCallback
    init_entropy: float = 0.01
    final_entropy: float = 0.01
    entropy_schedule: str = 'warmup_cosine'
    entropy_warmup_fraction: float = 0.2

    # CostCurriculumCallback
    cost_warmup_steps: int = 30_000
    cost_rampup_steps: int = 70_000
    cost_final: float = 0.0025
    cost_crisis_multiplier: float = 2.4


# =============================================================================
# CONFIGURACIÓN DE MODELOS
# =============================================================================

@dataclass
class PPOConfig:
    """Configuración de PPO optimizada."""
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 5  # Reducido de 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    target_kl: float = 0.03
    max_grad_norm: float = 0.5
    vf_coef: float = 0.5
    normalize_advantage: bool = True


@dataclass
class SACConfig:
    """Configuración de SAC optimizada."""
    learning_rate: float = 3e-4
    buffer_size: int = 100_000
    learning_starts: int = 5_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = 'auto'


@dataclass
class NetworkConfig:
    """Configuración de la red neuronal."""
    net_arch: List[int] = field(default_factory=lambda: [64, 32])
    activation_fn: str = 'LeakyReLU'
    ortho_init: bool = True


# =============================================================================
# CONFIGURACIÓN DE VALIDACIÓN
# =============================================================================

@dataclass
class ValidationConfig:
    """Configuración del framework de validación."""

    # Walk-Forward
    embargo_bars: int = 360  # 6 días (era 30 días = 1800 bars)
    purge_bars: int = 60     # 1 día adicional
    n_folds: int = 7
    train_pct: float = 0.70
    val_pct: float = 0.15
    test_pct: float = 0.15
    min_train_bars: int = 10_000
    expanding_window: bool = True

    # Robustness
    n_seeds: int = 5
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95

    # Feature lookbacks (para calcular embargo mínimo)
    feature_lookbacks: Dict[str, int] = field(default_factory=lambda: {
        'usdcop_ret_5d': 300,   # 5 días * 60 bars
        'log_ret_4h': 48,       # 4 horas * 12 bars
        'volatility_1h': 12,
        'log_ret_1h': 12,
        'usdcop_ret_1d': 60,
        'usdcop_volatility': 60,
    })


@dataclass
class AcceptanceCriteria:
    """
    Criterios de aceptación para el modelo.

    AJUSTADOS POR TIMEFRAME (basado en investigación 167 papers):
    - 5min: SNR ~1:100, requiere Sharpe > 1.0 para ser significativo
    - 15min: SNR ~1:33, Sharpe > 0.8 es aceptable
    - 1h: SNR ~1:10, Sharpe > 0.5 es suficiente
    """
    # Criterios base (para 15min - RECOMENDADO)
    min_sharpe: float = 0.80
    max_drawdown: float = 0.20
    min_wfe: float = 0.50
    min_win_rate: float = 0.48
    min_sortino: float = 0.80
    min_calmar: float = 0.40
    min_profitable_folds: int = 4
    max_crisis_drawdown: float = 0.30
    min_sharpe_ci_lower: float = 0.30
    max_hold_pct: float = 80.0
    min_profit_factor: float = 1.2

    @classmethod
    def for_timeframe(cls, timeframe: str) -> 'AcceptanceCriteria':
        """Obtener criterios ajustados por timeframe."""
        if timeframe == '5min':
            # 5min requiere criterios más estrictos por bajo SNR
            return cls(
                min_sharpe=1.00,        # Más alto por ruido
                max_drawdown=0.15,      # Más estricto
                min_wfe=0.50,
                min_win_rate=0.48,
                min_sortino=1.00,
                min_calmar=0.50,
                min_profitable_folds=4,
                max_crisis_drawdown=0.25,
                min_sharpe_ci_lower=0.50,
                max_hold_pct=80.0,
                min_profit_factor=1.3,
            )
        elif timeframe == '1h':
            # 1h puede ser más relajado
            return cls(
                min_sharpe=0.50,
                max_drawdown=0.25,
                min_wfe=0.45,
                min_win_rate=0.47,
                min_sortino=0.60,
                min_calmar=0.30,
                min_profitable_folds=3,
                max_crisis_drawdown=0.35,
                min_sharpe_ci_lower=0.20,
                max_hold_pct=85.0,
                min_profit_factor=1.1,
            )
        else:
            # 15min - default recomendado
            return cls()


# =============================================================================
# STRESS TESTING
# =============================================================================

@dataclass
class CrisisPeriod:
    """Definición de período de crisis."""
    name: str
    start_date: str
    end_date: str
    description: str
    max_acceptable_drawdown: float
    usdcop_move_pct: Optional[float] = None


CRISIS_PERIODS = [
    CrisisPeriod(
        name="COVID-19 Panic",
        start_date="2020-03-01",
        end_date="2020-06-30",
        description="Pandemia COVID-19, flight to quality masivo",
        max_acceptable_drawdown=0.35,
        usdcop_move_pct=25.0,
    ),
    CrisisPeriod(
        name="Fed Rate Hikes 2022",
        start_date="2022-03-01",
        end_date="2022-12-31",
        description="Ciclo agresivo de subidas de tasas Fed",
        max_acceptable_drawdown=0.30,
        usdcop_move_pct=15.0,
    ),
    CrisisPeriod(
        name="Peso Crisis 2022",
        start_date="2022-09-01",
        end_date="2022-11-30",
        description="COP a máximos históricos vs USD",
        max_acceptable_drawdown=0.30,
        usdcop_move_pct=12.0,
    ),
    CrisisPeriod(
        name="2023 Banking Crisis",
        start_date="2023-03-01",
        end_date="2023-05-31",
        description="SVB y contagio bancario global",
        max_acceptable_drawdown=0.25,
        usdcop_move_pct=8.0,
    ),
    CrisisPeriod(
        name="BanRep Rate Peak",
        start_date="2023-06-01",
        end_date="2023-09-30",
        description="Tasas BanRep en máximo histórico (13.25%)",
        max_acceptable_drawdown=0.25,
        usdcop_move_pct=5.0,
    ),
]


# =============================================================================
# CONFIGURACIÓN PRINCIPAL V19
# =============================================================================

@dataclass
class TrainingConfigV19:
    """Configuración principal de training V19."""

    # Sub-configs
    reward: RewardConfig = field(default_factory=RewardConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    callbacks: CallbacksConfig = field(default_factory=CallbacksConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    acceptance: AcceptanceCriteria = field(default_factory=AcceptanceCriteria)

    # Training
    total_timesteps: int = 200_000
    algorithm: str = 'SAC'  # 'PPO' o 'SAC'
    reward_type: str = 'alpha'  # 'symmetric', 'alpha', o 'alpha_v2'
    device: str = 'auto'
    seed: int = 42

    # Paths
    dataset_path: str = 'data/pipeline/07_output/datasets_5min/RL_DS12_FINAL.csv'
    output_dir: Path = field(default_factory=lambda: Path('outputs'))
    models_dir: Path = field(default_factory=lambda: Path('models'))
    logs_dir: Path = field(default_factory=lambda: Path('logs'))

    # Features
    features_5min: List[str] = field(default_factory=lambda: [
        'log_ret_5m', 'log_ret_15m', 'log_ret_30m',
        'rsi_9', 'atr_pct', 'bb_position', 'adx_14',
        'ema_cross', 'session_progress'
    ])

    features_hourly: List[str] = field(default_factory=lambda: [
        'log_ret_1h', 'log_ret_4h', 'volatility_1h'
    ])

    features_daily: List[str] = field(default_factory=lambda: [
        'usdcop_ret_1d', 'usdcop_ret_5d', 'usdcop_volatility',
        'vix_z', 'embi_z', 'dxy_z', 'brent_change_1d',
        'rate_spread', 'usdmxn_change_1d', 'usdclp_ret_1d',
        'banrep_intervention_proximity'
    ])

    @property
    def all_features(self) -> List[str]:
        """Todas las features concatenadas."""
        return self.features_5min + self.features_hourly + self.features_daily

    @property
    def obs_dim(self) -> int:
        """Dimensión total del observation space."""
        return len(self.all_features) + self.environment.n_state_features

    def validate(self) -> bool:
        """Validar configuración."""
        # Verificar embargo suficiente
        max_lookback = max(self.validation.feature_lookbacks.values())
        min_embargo = max_lookback * 2

        if self.validation.embargo_bars < min_embargo:
            raise ValueError(
                f"Embargo insuficiente: {self.validation.embargo_bars} < {min_embargo}"
            )

        # Verificar splits suman 1.0
        total = (self.validation.train_pct +
                 self.validation.val_pct +
                 self.validation.test_pct)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Splits no suman 1.0: {total}")

        print("✓ Configuración V19 validada correctamente")
        return True

    def to_dict(self) -> Dict:
        """Convertir a diccionario para serialización."""
        import dataclasses
        return dataclasses.asdict(self)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_default_config() -> TrainingConfigV19:
    """Obtener configuración por defecto."""
    config = TrainingConfigV19()
    config.validate()
    return config


def get_quick_test_config() -> TrainingConfigV19:
    """Configuración para pruebas rápidas."""
    config = TrainingConfigV19(
        total_timesteps=10_000,
        validation=ValidationConfig(n_folds=2, n_seeds=1),
    )
    config.callbacks.eval_freq = 2_000
    config.callbacks.patience = 3
    return config


def load_config(config_path: str) -> TrainingConfigV19:
    """
    Cargar configuración desde archivo YAML o JSON.

    Args:
        config_path: Path al archivo de configuración

    Returns:
        TrainingConfigV19 configurado
    """
    import json
    from pathlib import Path

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if path.suffix in ('.yaml', '.yml'):
        try:
            import yaml
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML required for YAML config files. Install with: pip install pyyaml")
    elif path.suffix == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}. Use .yaml, .yml, or .json")

    # Crear config base
    config = TrainingConfigV19()

    # Aplicar valores del archivo
    for key, value in data.items():
        if hasattr(config, key):
            if isinstance(value, dict):
                # Es un sub-config (reward, environment, etc.)
                sub_config = getattr(config, key)
                for sub_key, sub_value in value.items():
                    if hasattr(sub_config, sub_key):
                        setattr(sub_config, sub_key, sub_value)
            else:
                setattr(config, key, value)

    return config


if __name__ == '__main__':
    # Test
    config = get_default_config()
    print(f"Observation dim: {config.obs_dim}")
    print(f"Features: {len(config.all_features)}")
    print(f"Embargo: {config.validation.embargo_bars} bars")
