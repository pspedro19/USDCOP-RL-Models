"""
USD/COP RL Trading System - Default Configuration Values
==========================================================

Valores por defecto sensatos derivados de:
1. PRODUCTION_CONFIG.json validado (5-fold CV, stress tests)
2. Mejores practicas de RL trading
3. Especificaciones del mercado SET-FX Colombia

VALIDACION DE PRODUCCION:
- Model B (Aggressive): Sharpe 2.21, 4/5 folds positivos
- Stress Tests: 3/5 periodos pasados (60%)
- Max Drawdown validado: 0.2%

Author: Claude Code
Version: 1.0.0
"""

from typing import Dict, List, Tuple, Any

# =============================================================================
# PRODUCTION DEFAULTS - From validated PRODUCTION_CONFIG.json
# =============================================================================

PRODUCTION_DEFAULTS: Dict[str, Any] = {
    "model_name": "Model B (Aggressive)",
    "algorithm": "PPO",
    "framework": "stable-baselines3",
    "version": "V19",

    # Hyperparameters validados
    "learning_rate": 1e-4,
    "n_steps": 2048,
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.99,
    "ent_coef": 0.05,
    "clip_range": 0.2,
    "gae_lambda": 0.95,
    "max_grad_norm": 0.5,
    "vf_coef": 0.5,

    # Network
    "net_arch": [256, 256],
    "activation_fn": "Tanh",

    # Environment
    "initial_balance": 10000,
    "max_position": 1.0,
    "episode_length": 400,
    "use_vol_scaling": True,
    "use_regime_detection": True,
    "bars_per_day": 56,

    # Training
    "total_timesteps": 80000,
    "seed": 42,
    "device": "cpu",
}


# =============================================================================
# PPO HYPERPARAMETERS
# =============================================================================

PPO_DEFAULTS: Dict[str, Any] = {
    # Learning rate
    # Range validado: 1e-5 a 3e-4
    # Valor optimo para USDCOP: 1e-4 (estable, buen convergencia)
    "learning_rate": 1e-4,

    # Rollout steps
    # 2048: Balance entre bias/variance
    # Mayor = menos bias, mas memoria
    # Menor = mas sesgo hacia experiencias recientes
    "n_steps": 2048,

    # Batch size
    # 128: Bueno para datasets ~80k timesteps
    # Debe ser divisor de n_steps * n_envs
    "batch_size": 128,

    # Epochs por update
    # 10: Numero de pasadas sobre cada batch
    # Mas epochs = mejor uso de datos, riesgo de overfitting
    "n_epochs": 10,

    # Discount factor
    # 0.99: Estandar para trading (horizontes medios)
    # Range comun: 0.95-0.999
    "gamma": 0.99,

    # GAE lambda
    # 0.95: Balance entre bias/variance en advantage estimation
    "gae_lambda": 0.95,

    # Clipping
    # 0.2: Valor estandar PPO
    # Menor = updates mas conservadores
    "clip_range": 0.2,

    # Entropy coefficient
    # 0.05 (agresivo) vs 0.10 (conservador)
    # Mayor entropy = mas exploracion, menor performance pico
    "ent_coef": 0.05,

    # Value function coefficient
    # 0.5: Peso de la perdida del critic
    "vf_coef": 0.5,

    # Gradient clipping
    # 0.5: Previene explosion de gradientes
    "max_grad_norm": 0.5,

    # Normalization
    "normalize_advantage": True,
}


# =============================================================================
# SAC HYPERPARAMETERS (alternativa para comparacion)
# =============================================================================

SAC_DEFAULTS: Dict[str, Any] = {
    "learning_rate": 3e-4,
    "buffer_size": 100_000,
    "learning_starts": 1000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "ent_coef": "auto",  # Ajuste automatico
    "train_freq": 1,
    "gradient_steps": 1,
}


# =============================================================================
# NETWORK ARCHITECTURE
# =============================================================================

NETWORK_DEFAULTS: Dict[str, Any] = {
    # Arquitectura de red
    # [256, 256]: Buena capacidad para 33 features + 12 state
    # Alternativas probadas:
    # - [48, 32]: Mas eficiente pero menor capacidad
    # - [128, 128]: Balance
    # - [256, 256, 128]: Mayor capacidad, mas lento
    "net_arch": [256, 256],

    # Funcion de activacion
    # Tanh: Standard para RL, outputs en [-1, 1]
    # Alternativas: ReLU, LeakyReLU
    "activation_fn": "Tanh",

    # Arquitectura separada para actor/critic
    # True: pi y vf tienen redes separadas
    # False: Comparten capas (mas eficiente)
    "separate_networks": True,
}


# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

ENVIRONMENT_DEFAULTS: Dict[str, Any] = {
    # Capital inicial
    "initial_balance": 10_000,

    # Posicion maxima
    # 1.0 = 100% del capital
    "max_position": 1.0,

    # Longitud del episodio
    # 400 barras = ~7 dias de trading (56 barras/dia)
    # Range: 200-1200 dependiendo del objetivo
    "episode_length": 400,

    # Drawdown maximo antes de terminar episodio
    # 15% = terminar si portfolio cae 15%
    "max_drawdown_pct": 15.0,

    # Timeframe
    "timeframe": "5min",

    # Barras por dia de trading (Colombia: 8am-4pm = 8h = 96 barras de 5min)
    # Pero tipicamente 56 debido a gaps y filtrado
    "bars_per_day": 56,

    # Volatility scaling - reducir posicion en alta volatilidad
    "use_vol_scaling": True,

    # Regime detection - usar VIX/EMBI para ajustar comportamiento
    "use_regime_detection": True,

    # Modo de proteccion: como combinar vol_scaler y regime_detector
    # 'min': usar el multiplicador mas conservador (RECOMENDADO)
    # 'multiply': multiplicar ambos (muy conservador)
    # 'vol_only': solo usar VolatilityScaler
    # 'regime_only': solo usar RegimeDetector
    "protection_mode": "min",

    # Columnas del dataset
    "volatility_column": "volatility_pct",
    "return_column": "close_return",
    "vol_feature_column": "atr_pct",
    "vix_column": "vix_z",
    "embi_column": "embi_z",
}


# =============================================================================
# VOLATILITY SCALER CONFIGURATION
# =============================================================================

VOL_SCALER_DEFAULTS: Dict[str, Any] = {
    # Ventana de lookback para calcular percentiles
    "lookback_window": 60,

    # Quantiles para buckets de volatilidad
    "quantiles": [0.5, 0.75, 0.9],

    # Factores de escala por bucket
    # [normal, medio, alto, crisis]
    "scale_factors": [1.0, 0.75, 0.5, 0.25],
}


# =============================================================================
# REGIME DETECTOR CONFIGURATION
# =============================================================================

REGIME_DETECTOR_DEFAULTS: Dict[str, Any] = {
    # VIX z-score thresholds
    "vix_crisis_threshold": 2.0,
    "vix_volatile_threshold": 1.0,

    # EMBI z-score thresholds
    "embi_crisis_threshold": 2.0,
    "embi_volatile_threshold": 1.0,

    # Volatility percentile thresholds
    "vol_crisis_percentile": 95.0,
    "vol_volatile_percentile": 75.0,

    # Position multipliers por regimen
    "crisis_multiplier": 0.0,    # No trading en crisis
    "volatile_multiplier": 0.5,  # 50% posicion en volatil
    "normal_multiplier": 1.0,    # 100% en normal

    # Pesos para probabilidades
    "vix_weight": 0.4,
    "embi_weight": 0.3,
    "vol_weight": 0.3,
}


# =============================================================================
# REWARD FUNCTION CONFIGURATION
# =============================================================================

REWARD_DEFAULTS: Dict[str, Any] = {
    # Tipo de reward
    # 'symmetric': curriculum con simetria LONG/SHORT (RECOMENDADO)
    # 'alpha': basado en alpha sobre mercado
    # 'sortino': optimiza Sortino ratio
    "reward_type": "symmetric",

    # Fases del curriculum (como fraccion del training)
    # Fase 1: 0-30% - Exploracion libre (costos 0)
    # Fase 2: 30-60% - Transicion (costos graduales)
    # Fase 3: 60-100% - Realista (costos completos)
    "phase_boundaries": (0.30, 0.60),

    # Costos objetivo por fase
    "transition_target_cost_bps": 10.0,    # Fase 2
    "realistic_min_cost_bps": 25.0,        # Fase 3 minimo
    "realistic_max_cost_bps": 36.0,        # Fase 3 en alta volatilidad

    # Configuracion de simetria
    "symmetry_window": 60,
    "max_directional_bias": 0.30,
    "symmetry_penalty_scale": 2.0,

    # Deteccion de comportamientos patologicos
    "max_trades_per_bar": 0.05,
    "overtrading_lookback": 120,
    "overtrading_penalty": 0.5,
    "max_hold_duration": 36,
    "inactivity_penalty": 0.3,
    "reversal_threshold": 5,
    "churning_penalty": 0.4,

    # Configuracion Sortino
    "sortino_window": 60,
    "sortino_mar": 0.0,  # Minimum Acceptable Return

    # Scaling y clipping
    "reward_scale": 100.0,
    "clip_range": (-5.0, 5.0),
}


# =============================================================================
# SET-FX COST MODEL (Colombia)
# =============================================================================

COST_MODEL_DEFAULTS: Dict[str, Any] = {
    # Spread base en condiciones normales
    "base_spread_bps": 14.0,

    # Spread en alta volatilidad
    "high_vol_spread_bps": 28.0,

    # Spread en crisis
    "crisis_spread_bps": 45.0,

    # Slippage promedio
    "slippage_bps": 3.0,

    # Thresholds de volatilidad para cambiar spread
    "volatility_threshold_high": 0.7,
    "volatility_threshold_crisis": 0.9,
}


# =============================================================================
# VALIDATION CONFIGURATION
# =============================================================================

VALIDATION_DEFAULTS: Dict[str, Any] = {
    # Split de datos
    "train_pct": 0.70,
    "val_pct": 0.15,
    "test_pct": 0.15,

    # Embargo entre splits (evita data leakage)
    "embargo_bars": 180,  # 3 dias * 60 barras/dia
    "embargo_days": 3,

    # Purged K-Fold CV
    "n_splits": 5,
    "purge_bars": 60,  # 1 dia de purge

    # Walk-forward validation
    "wf_train_size": None,  # Auto-calculado
    "wf_test_size": None,
    "wf_expanding": False,  # Sliding window por defecto

    # Minimos
    "min_train_size": 10_000,
    "min_test_size": 2_000,
}


# =============================================================================
# CALLBACK CONFIGURATION
# =============================================================================

CALLBACK_DEFAULTS: Dict[str, Any] = {
    # Evaluacion
    "eval_freq": 5_000,         # Evaluar cada N steps
    "n_eval_episodes": 5,       # Episodios por evaluacion

    # Early stopping
    "patience": 10,             # Evals sin mejora antes de parar
    "min_delta": 0.05,          # Mejora minima para contar
    "min_sharpe": 0.3,          # Sharpe minimo para guardar modelo

    # Entropy scheduling
    "init_entropy": 0.10,       # Entropy inicial (alta exploracion)
    "final_entropy": 0.02,      # Entropy final (explotacion)
    "entropy_schedule": "warmup_cosine",
    "warmup_fraction": 0.2,

    # Action monitoring
    "action_log_freq": 2_000,
    "collapse_threshold": 0.15,
    "hold_warning_threshold": 70.0,

    # Cost curriculum
    "cost_warmup_steps": 0,     # Steps sin costos
    "cost_rampup_steps": 30_000,  # Steps para llegar a costos completos
    "cost_final": 0.0025,       # 25 bps
    "cost_crisis_multiplier": 2.0,
}


# =============================================================================
# RISK MANAGEMENT CONFIGURATION
# =============================================================================

RISK_DEFAULTS: Dict[str, Any] = {
    # Drawdown limits
    "max_drawdown_warning": 0.03,   # 3% -> Warning
    "max_drawdown_reduce": 0.05,    # 5% -> Reducir sizing 50%
    "max_drawdown_stop": 0.10,      # 10% -> Pausar trading

    # Performance limits
    "min_sharpe_30d": 0.0,          # Sharpe rolling < 0 -> Review
    "max_hold_pct": 0.95,           # 95% HOLD por 5 dias -> Alert

    # Loss limits
    "max_consecutive_losses": 10,
    "max_daily_loss": 0.02,         # 2% perdida diaria -> Warning

    # Recovery
    "auto_resume_after_recovery": True,
    "recovery_sharpe_threshold": 0.5,

    # Rolling window para metricas
    "rolling_window_days": 30,
}


# =============================================================================
# ACCEPTANCE CRITERIA
# =============================================================================

ACCEPTANCE_DEFAULTS: Dict[str, Any] = {
    # Sharpe ratio minimo anualizado
    "min_sharpe": 0.8,

    # Maximum drawdown permitido
    "max_drawdown": 15.0,  # %

    # Calmar ratio minimo
    "min_calmar": 0.5,

    # Profit factor minimo
    "min_profit_factor": 1.2,

    # Win rate minimo
    "min_win_rate": 0.45,  # 45%

    # Maximo HOLD permitido
    "max_hold_pct": 80.0,  # %

    # Stress test requirements
    "min_stress_periods_passed": 3,
    "stress_pass_rate": 0.50,  # 50% de periodos
}


# =============================================================================
# DATA QUALITY CONFIGURATION
# =============================================================================

DATA_DEFAULTS: Dict[str, Any] = {
    # Warmup period (dias de datos iniciales a descartar)
    "warmup_days": 60,

    # Features a eliminar (redundantes o problematicas)
    "features_to_drop": [
        "hour_sin",           # Redundante con session_progress
        "hour_cos",           # Redundante con session_progress
        "oil_above_60_flag",  # Redundante con rate_spread
        "vix_zscore",         # 68% zeros (bug)
        "high_low_range",     # 93.5% zeros
    ],

    # Features a normalizar por fold
    "features_to_normalize": [
        "vix_z",
        "embi_z",
        "dxy_z",
        "usdcop_volatility",
        "rate_spread",
    ],

    # Columnas de retornos para winsorizar
    "return_columns": [
        "log_ret_5m",
        "log_ret_15m",
        "log_ret_30m",
        "log_ret_1h",
        "log_ret_4h",
        "usdcop_ret_1d",
        "usdcop_ret_5d",
        "brent_change_1d",
        "usdmxn_change_1d",
        "usdclp_ret_1d",
    ],

    # Percentil de winsorization
    "winsorize_percentile": 0.01,

    # Threshold para validar warmup
    "zero_threshold": 0.05,
}


# =============================================================================
# CRISIS PERIODS FOR STRESS TESTING
# =============================================================================

CRISIS_PERIODS_DEFAULTS: List[Dict[str, Any]] = [
    {
        "name": "COVID_Crash",
        "start_date": "2020-02-20",
        "end_date": "2020-04-30",
        "description": "COVID-19 market crash, extreme volatility",
        "expected_volatility": "extreme",
        "max_acceptable_dd": 30.0,
        "min_acceptable_sharpe": -2.0,
    },
    {
        "name": "Fed_Hikes_2022",
        "start_date": "2022-03-01",
        "end_date": "2022-12-31",
        "description": "Fed aggressive rate hikes, USD strength",
        "expected_volatility": "high",
        "max_acceptable_dd": 25.0,
        "min_acceptable_sharpe": -1.0,
    },
    {
        "name": "Petro_Election",
        "start_date": "2022-05-15",
        "end_date": "2022-08-15",
        "description": "Colombian presidential election uncertainty",
        "expected_volatility": "high",
        "max_acceptable_dd": 20.0,
        "min_acceptable_sharpe": -0.5,
    },
    {
        "name": "LatAm_Selloff",
        "start_date": "2022-09-01",
        "end_date": "2022-11-30",
        "description": "LatAm regional selloff, risk-off",
        "expected_volatility": "high",
        "max_acceptable_dd": 20.0,
        "min_acceptable_sharpe": -0.5,
    },
    {
        "name": "Banking_Crisis_2023",
        "start_date": "2023-03-01",
        "end_date": "2023-04-30",
        "description": "SVB collapse, banking sector stress",
        "expected_volatility": "high",
        "max_acceptable_dd": 20.0,
        "min_acceptable_sharpe": -1.0,
    },
]


# =============================================================================
# ENSEMBLE CONFIGURATION
# =============================================================================

ENSEMBLE_DEFAULTS: Dict[str, Any] = {
    # Model A: Conservador
    "model_a": {
        "name": "A_conservative",
        "weight": 0.70,
        "ent_coef": 0.10,  # Mayor exploracion = mas estable
        "net_arch": [48, 32],
    },

    # Model B: Agresivo
    "model_b": {
        "name": "B_aggressive",
        "weight": 0.30,
        "ent_coef": 0.05,  # Menor exploracion = mayor performance
        "net_arch": [48, 32],
    },

    # Modo de combinacion
    "action_mode": "weighted_mean",  # o 'weighted_vote'
}


# =============================================================================
# FEATURE LISTS
# =============================================================================

# Features de 5 minutos (limpias, sin redundantes)
FEATURES_5MIN: List[str] = [
    "log_ret_5m",
    "log_ret_15m",
    "log_ret_30m",
    "rsi_9",
    "atr_pct",
    "bb_position",
    "adx_14",
    "ema_cross",
    "session_progress",
]

# Features horarias
FEATURES_HOURLY: List[str] = [
    "log_ret_1h",
    "log_ret_4h",
    "volatility_1h",
]

# Features diarias
FEATURES_DAILY: List[str] = [
    "usdcop_ret_1d",
    "usdcop_ret_5d",
    "usdcop_volatility",
    "vix_z",
    "embi_z",
    "dxy_z",
    "brent_change_1d",
    "rate_spread",
    "usdmxn_change_1d",
    "usdclp_ret_1d",
    "banrep_intervention_proximity",
]

# Todas las features limpias
ALL_FEATURES: List[str] = FEATURES_5MIN + FEATURES_HOURLY + FEATURES_DAILY


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_bars_per_day(timeframe: str) -> int:
    """
    Obtener barras por dia para un timeframe.

    Args:
        timeframe: '5min', '15min', '1h', etc.

    Returns:
        Barras por dia de trading (8 horas)
    """
    timeframe_bars = {
        "1min": 480,
        "5min": 96,
        "15min": 32,
        "30min": 16,
        "1h": 8,
        "4h": 2,
        "1d": 1,
    }

    # Valor real para USDCOP (ajustado por gaps/filtrado)
    if timeframe == "5min":
        return 56  # Valor empirico del dataset

    return timeframe_bars.get(timeframe, 60)


def get_embargo_bars(bars_per_day: int, embargo_days: int = 3) -> int:
    """
    Calcular barras de embargo para cross-validation.

    Args:
        bars_per_day: Barras por dia
        embargo_days: Dias de embargo

    Returns:
        Numero de barras de embargo
    """
    return bars_per_day * embargo_days


def get_cost_bps(volatility_percentile: float) -> float:
    """
    Obtener costo estimado en basis points para un nivel de volatilidad.

    Args:
        volatility_percentile: Percentil de volatilidad [0, 1]

    Returns:
        Costo estimado en bps
    """
    if volatility_percentile >= 0.9:
        return COST_MODEL_DEFAULTS["crisis_spread_bps"]
    elif volatility_percentile >= 0.7:
        return COST_MODEL_DEFAULTS["high_vol_spread_bps"]
    else:
        return COST_MODEL_DEFAULTS["base_spread_bps"]
