"""
USD/COP RL Trading System - Environment Factory
================================================

Factory pattern para simplificar la creacion de environments de trading.

OBJETIVOS:
1. Simplificar la creacion de environments (TradingEnvironmentV19 y Enhanced)
2. Configuracion centralizada mediante dataclasses
3. NO cambiar el comportamiento actual de los environments
4. Ser 100% backward-compatible con codigo existente

PATRONES DE USO:

1. Uso Simple (defaults):
   >>> factory = EnvironmentFactory(df)
   >>> env = factory.create_training_env()

2. Uso con Configuracion:
   >>> config = EnvironmentConfig(
   ...     initial_balance=50000,
   ...     episode_length=2400,
   ...     use_enhanced=True,
   ... )
   >>> factory = EnvironmentFactory(df, config)
   >>> env = factory.create_training_env()

3. Uso con Presets:
   >>> factory = EnvironmentFactory(df)
   >>> env = factory.create_from_preset('aggressive')

4. Backward-compatible:
   >>> env = EnvironmentFactory.create_v19(df, initial_balance=10000)

Author: Claude Code
Version: 1.0.0
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union, Callable, Literal, TYPE_CHECKING
from enum import Enum

import numpy as np
import pandas as pd

# Lazy imports to avoid circular dependencies
if TYPE_CHECKING:
    from ..environment_v19 import TradingEnvironmentV19, SETFXCostModel, VolatilityScaler
    from ..environment_v19_enhanced import TradingEnvironmentV19Enhanced
    from ..regime_detector import RegimeDetector, RegimeConfig
    from ..risk_manager import RiskManager, RiskLimits
    from ..feedback_tracker import FeedbackTracker, RegimeFeatureGenerator


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class CostModelConfig:
    """
    Configuracion del modelo de costos SET-FX.

    Valores por defecto basados en costos reales de SET-FX Colombia:
    - Spread normal: 14-18 bps
    - Spread alta volatilidad: 25-36 bps
    - Spread crisis: 40-50 bps
    - Slippage: 2-5 bps
    """
    base_spread_bps: float = 14.0
    high_vol_spread_bps: float = 28.0
    crisis_spread_bps: float = 45.0
    slippage_bps: float = 3.0
    volatility_threshold_high: float = 0.7
    volatility_threshold_crisis: float = 0.9

    def to_cost_model(self) -> "SETFXCostModel":
        """Crear instancia de SETFXCostModel desde config."""
        from ..environment_v19 import SETFXCostModel
        return SETFXCostModel(
            base_spread_bps=self.base_spread_bps,
            high_vol_spread_bps=self.high_vol_spread_bps,
            crisis_spread_bps=self.crisis_spread_bps,
            slippage_bps=self.slippage_bps,
            volatility_threshold_high=self.volatility_threshold_high,
            volatility_threshold_crisis=self.volatility_threshold_crisis,
        )


@dataclass
class VolatilityScalingConfig:
    """
    Configuracion del position sizing dinamico basado en volatilidad.

    - Reduce posicion en alta volatilidad para proteger contra drawdowns
    - Usa percentiles historicos de volatilidad
    """
    enabled: bool = True
    lookback_window: int = 60  # 60 barras = 1 dia en 15min
    quantiles: List[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])
    scale_factors: List[float] = field(default_factory=lambda: [1.0, 0.75, 0.5, 0.25])
    feature_column: str = 'atr_pct'

    def __post_init__(self):
        """Validar configuracion."""
        if len(self.scale_factors) != len(self.quantiles) + 1:
            raise ValueError(
                f"scale_factors ({len(self.scale_factors)}) debe tener "
                f"len(quantiles) + 1 ({len(self.quantiles) + 1}) elementos"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a dict para pasar a VolatilityScaler."""
        return {
            'lookback_window': self.lookback_window,
            'quantiles': self.quantiles,
            'scale_factors': self.scale_factors,
        }


@dataclass
class RegimeDetectionConfig:
    """
    Configuracion de deteccion de regimen de mercado.

    Detecta 3 regimenes: NORMAL, VOLATILE, CRISIS
    Basado en VIX, EMBI, y volatilidad realizada.
    """
    enabled: bool = False
    vix_column: str = 'vix_z'
    embi_column: str = 'embi_z'

    # Thresholds
    vix_crisis_threshold: float = 2.0
    vix_volatile_threshold: float = 1.0
    embi_crisis_threshold: float = 2.0
    embi_volatile_threshold: float = 1.0
    vol_crisis_percentile: float = 95.0
    vol_volatile_percentile: float = 75.0

    # Position multipliers
    crisis_multiplier: float = 0.0
    volatile_multiplier: float = 0.5
    normal_multiplier: float = 1.0

    # Protection mode: how to combine vol_scaler and regime_detector
    protection_mode: Literal['min', 'multiply', 'vol_only', 'regime_only'] = 'min'

    def to_regime_config(self) -> "RegimeConfig":
        """Crear RegimeConfig desde esta configuracion."""
        from ..regime_detector import RegimeConfig
        return RegimeConfig(
            vix_crisis_threshold=self.vix_crisis_threshold,
            vix_volatile_threshold=self.vix_volatile_threshold,
            embi_crisis_threshold=self.embi_crisis_threshold,
            embi_volatile_threshold=self.embi_volatile_threshold,
            vol_crisis_percentile=self.vol_crisis_percentile,
            vol_volatile_percentile=self.vol_volatile_percentile,
            crisis_multiplier=self.crisis_multiplier,
            volatile_multiplier=self.volatile_multiplier,
            normal_multiplier=self.normal_multiplier,
        )

    def create_detector(self) -> Optional["RegimeDetector"]:
        """Crear instancia de RegimeDetector si esta habilitado."""
        if not self.enabled:
            return None
        from ..regime_detector import RegimeDetector
        return RegimeDetector(config=self.to_regime_config())


@dataclass
class EnhancedFeaturesConfig:
    """
    Configuracion de features mejoradas (V19 Enhanced).

    Añade 9 features adicionales al observation:
    - 6 regime features (is_crisis, is_volatile, is_normal, confidence, vix_trend, days)
    - 3 feedback features (accuracy, trend, consecutive_wrong)
    """
    use_regime_features: bool = True
    use_feedback_features: bool = True

    # Feedback tracker config
    feedback_window_size: int = 20
    feedback_action_threshold: float = 0.1

    # Regime feature generator config
    vix_lookback: int = 20


@dataclass
class RiskManagerConfig:
    """
    Configuracion del Risk Manager con kill switches.

    Proporciona proteccion automatica contra:
    - Drawdowns extremos
    - Sharpe negativo sostenido
    - HOLD excesivo
    - Rachas perdedoras
    """
    enabled: bool = False

    # Drawdown limits
    max_drawdown_warning: float = 0.03   # 3%
    max_drawdown_reduce: float = 0.05    # 5%
    max_drawdown_stop: float = 0.10      # 10%

    # Performance limits
    min_sharpe_30d: float = 0.0
    max_hold_pct: float = 0.95

    # Loss limits
    max_consecutive_losses: int = 10
    max_daily_loss: float = 0.02

    # Recovery
    auto_resume_after_recovery: bool = True
    recovery_sharpe_threshold: float = 0.5

    # Operational
    rolling_window: int = 30
    bars_per_day: int = 56

    def to_risk_limits(self) -> "RiskLimits":
        """Crear RiskLimits desde config."""
        from ..risk_manager import RiskLimits
        return RiskLimits(
            max_drawdown_warning=self.max_drawdown_warning,
            max_drawdown_reduce=self.max_drawdown_reduce,
            max_drawdown_stop=self.max_drawdown_stop,
            min_sharpe_30d=self.min_sharpe_30d,
            max_hold_pct=self.max_hold_pct,
            max_consecutive_losses=self.max_consecutive_losses,
            max_daily_loss=self.max_daily_loss,
            auto_resume_after_recovery=self.auto_resume_after_recovery,
            recovery_sharpe_threshold=self.recovery_sharpe_threshold,
        )

    def create_manager(self) -> Optional["RiskManager"]:
        """Crear instancia de RiskManager si esta habilitado."""
        if not self.enabled:
            return None
        from ..risk_manager import RiskManager
        return RiskManager(
            limits=self.to_risk_limits(),
            rolling_window=self.rolling_window,
            bars_per_day=self.bars_per_day,
        )


@dataclass
class EnvironmentConfig:
    """
    Configuracion centralizada para creacion de environments.

    Esta clase agrupa TODAS las configuraciones necesarias para crear
    cualquier variante de environment (V19 base o Enhanced).

    Ejemplo:
        >>> config = EnvironmentConfig(
        ...     initial_balance=50000,
        ...     episode_length=2400,
        ...     use_enhanced=True,
        ...     enhanced_features=EnhancedFeaturesConfig(
        ...         use_regime_features=True,
        ...         use_feedback_features=True,
        ...     ),
        ... )
    """
    # === CORE SETTINGS ===
    initial_balance: float = 10_000
    max_position: float = 1.0
    episode_length: int = 1200
    max_drawdown_pct: float = 15.0

    # === COST SETTINGS ===
    use_curriculum_costs: bool = True
    cost_model: Optional[CostModelConfig] = None

    # === FEATURE COLUMNS ===
    feature_columns: Optional[List[str]] = None
    volatility_column: str = 'volatility_pct'
    return_column: str = 'close_return'

    # === VOLATILITY SCALING ===
    vol_scaling: VolatilityScalingConfig = field(default_factory=VolatilityScalingConfig)

    # === REGIME DETECTION ===
    regime_detection: RegimeDetectionConfig = field(default_factory=RegimeDetectionConfig)

    # === ENHANCED ENVIRONMENT (V19 Enhanced) ===
    use_enhanced: bool = False
    enhanced_features: EnhancedFeaturesConfig = field(default_factory=EnhancedFeaturesConfig)

    # === RISK MANAGER ===
    risk_manager: RiskManagerConfig = field(default_factory=RiskManagerConfig)

    # === REWARD FUNCTION ===
    reward_function: Optional[Any] = None  # Funcion externa de reward

    # === MISC ===
    verbose: int = 0

    def __post_init__(self):
        """Inicializar configuraciones por defecto si son None."""
        if self.cost_model is None:
            self.cost_model = CostModelConfig()

    @classmethod
    def for_training(cls, **kwargs) -> "EnvironmentConfig":
        """Factory method para configuracion de training."""
        defaults = {
            'use_curriculum_costs': True,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def for_validation(cls, **kwargs) -> "EnvironmentConfig":
        """Factory method para configuracion de validacion."""
        defaults = {
            'use_curriculum_costs': False,  # Costos completos
        }
        defaults.update(kwargs)

        # En validacion, habilitar risk manager por defecto
        if 'risk_manager' not in kwargs:
            defaults['risk_manager'] = RiskManagerConfig(enabled=True)

        return cls(**defaults)

    @classmethod
    def for_production(cls, **kwargs) -> "EnvironmentConfig":
        """Factory method para configuracion de produccion."""
        defaults = {
            'use_curriculum_costs': False,
            'use_enhanced': True,
        }
        defaults.update(kwargs)

        # En produccion, siempre usar risk manager
        if 'risk_manager' not in kwargs:
            defaults['risk_manager'] = RiskManagerConfig(enabled=True)

        return cls(**defaults)

    def copy(self, **overrides) -> "EnvironmentConfig":
        """Crear copia con overrides."""
        data = asdict(self)
        data.update(overrides)

        # Reconstruir dataclasses anidadas
        if 'cost_model' in data and isinstance(data['cost_model'], dict):
            data['cost_model'] = CostModelConfig(**data['cost_model'])
        if 'vol_scaling' in data and isinstance(data['vol_scaling'], dict):
            data['vol_scaling'] = VolatilityScalingConfig(**data['vol_scaling'])
        if 'regime_detection' in data and isinstance(data['regime_detection'], dict):
            data['regime_detection'] = RegimeDetectionConfig(**data['regime_detection'])
        if 'enhanced_features' in data and isinstance(data['enhanced_features'], dict):
            data['enhanced_features'] = EnhancedFeaturesConfig(**data['enhanced_features'])
        if 'risk_manager' in data and isinstance(data['risk_manager'], dict):
            data['risk_manager'] = RiskManagerConfig(**data['risk_manager'])

        return EnvironmentConfig(**data)


# =============================================================================
# PRESETS
# =============================================================================

class EnvironmentPreset(Enum):
    """Presets predefinidos para configuracion rapida."""
    DEFAULT = "default"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    CURRICULUM = "curriculum"
    FULL_COSTS = "full_costs"
    ENHANCED = "enhanced"
    PRODUCTION = "production"


PRESET_CONFIGS: Dict[str, Dict[str, Any]] = {
    # === DEFAULT ===
    # Configuracion balanceada para training general
    'default': {
        'initial_balance': 10_000,
        'max_position': 1.0,
        'episode_length': 1200,
        'max_drawdown_pct': 15.0,
        'use_curriculum_costs': True,
        'vol_scaling': VolatilityScalingConfig(enabled=True),
        'regime_detection': RegimeDetectionConfig(enabled=False),
        'use_enhanced': False,
    },

    # === CONSERVATIVE ===
    # Menor exposicion, mas proteccion
    'conservative': {
        'initial_balance': 10_000,
        'max_position': 0.5,
        'episode_length': 1200,
        'max_drawdown_pct': 10.0,
        'use_curriculum_costs': True,
        'vol_scaling': VolatilityScalingConfig(
            enabled=True,
            quantiles=[0.4, 0.6, 0.8],
            scale_factors=[1.0, 0.6, 0.3, 0.1],
        ),
        'regime_detection': RegimeDetectionConfig(
            enabled=True,
            crisis_multiplier=0.0,
            volatile_multiplier=0.3,
            protection_mode='min',
        ),
        'use_enhanced': False,
    },

    # === AGGRESSIVE ===
    # Mayor exposicion, menos restricciones
    'aggressive': {
        'initial_balance': 10_000,
        'max_position': 1.0,
        'episode_length': 1200,
        'max_drawdown_pct': 20.0,
        'use_curriculum_costs': True,
        'vol_scaling': VolatilityScalingConfig(
            enabled=True,
            quantiles=[0.7, 0.85, 0.95],
            scale_factors=[1.0, 0.85, 0.7, 0.5],
        ),
        'regime_detection': RegimeDetectionConfig(enabled=False),
        'use_enhanced': False,
    },

    # === CURRICULUM ===
    # Optimizado para curriculum learning
    'curriculum': {
        'initial_balance': 10_000,
        'max_position': 1.0,
        'episode_length': 1200,
        'max_drawdown_pct': 15.0,
        'use_curriculum_costs': True,  # Costos graduales
        'vol_scaling': VolatilityScalingConfig(enabled=True),
        'regime_detection': RegimeDetectionConfig(enabled=False),
        'use_enhanced': False,
    },

    # === FULL_COSTS ===
    # Costos reales completos (para validacion)
    'full_costs': {
        'initial_balance': 10_000,
        'max_position': 1.0,
        'episode_length': 1200,
        'max_drawdown_pct': 15.0,
        'use_curriculum_costs': False,  # Costos completos
        'vol_scaling': VolatilityScalingConfig(enabled=True),
        'regime_detection': RegimeDetectionConfig(enabled=False),
        'use_enhanced': False,
    },

    # === ENHANCED ===
    # Environment V19 Enhanced con todas las features
    'enhanced': {
        'initial_balance': 10_000,
        'max_position': 1.0,
        'episode_length': 1200,
        'max_drawdown_pct': 15.0,
        'use_curriculum_costs': True,
        'vol_scaling': VolatilityScalingConfig(enabled=True),
        'regime_detection': RegimeDetectionConfig(enabled=True, protection_mode='min'),
        'use_enhanced': True,
        'enhanced_features': EnhancedFeaturesConfig(
            use_regime_features=True,
            use_feedback_features=True,
        ),
    },

    # === PRODUCTION ===
    # Configuracion VALIDADA que produjo Sharpe 2.21-4.27
    # IMPORTANTE: Usa V19 base (NO Enhanced) con 32 features
    # CRITICO: crisis_multiplier=0.0 (NO operar en crisis) - igual que V1 defaults
    'production': {
        'initial_balance': 10_000,
        'max_position': 1.0,
        'episode_length': 400,              # V1: 400 (NO 1200)
        'max_drawdown_pct': 15.0,
        'use_curriculum_costs': True,       # V1: curriculum ENABLED
        'vol_scaling': VolatilityScalingConfig(enabled=True),
        'regime_detection': RegimeDetectionConfig(
            enabled=True,
            crisis_multiplier=0.0,          # V1 DEFAULT: NO operar en crisis
            volatile_multiplier=0.5,
            normal_multiplier=1.0,
            protection_mode='min',
        ),
        'use_enhanced': False,              # V1: usa V19 BASE (32 features)
        'risk_manager': RiskManagerConfig(enabled=False),  # V1: sin risk manager extra
    },

    # === PRODUCTION_ENHANCED ===
    # Version con Enhanced environment (41 features) - experimental
    'production_enhanced': {
        'initial_balance': 10_000,
        'max_position': 1.0,
        'episode_length': 1200,
        'max_drawdown_pct': 15.0,
        'use_curriculum_costs': False,
        'vol_scaling': VolatilityScalingConfig(enabled=True),
        'regime_detection': RegimeDetectionConfig(
            enabled=True,
            crisis_multiplier=0.0,
            volatile_multiplier=0.5,
            protection_mode='min',
        ),
        'use_enhanced': True,
        'enhanced_features': EnhancedFeaturesConfig(
            use_regime_features=True,
            use_feedback_features=True,
        ),
        'risk_manager': RiskManagerConfig(enabled=True),
    },
}


# =============================================================================
# ENVIRONMENT FACTORY
# =============================================================================

class EnvironmentFactory:
    """
    Factory para creacion simplificada de environments de trading.

    Esta clase proporciona una interfaz unificada para crear cualquier
    variante de environment (V19 base o Enhanced) con configuracion
    centralizada y presets predefinidos.

    PATRONES DE USO:

    1. Uso Simple (defaults):
       >>> factory = EnvironmentFactory(df)
       >>> train_env = factory.create_training_env()
       >>> val_env = factory.create_validation_env()

    2. Uso con Configuracion:
       >>> config = EnvironmentConfig(
       ...     initial_balance=50000,
       ...     use_enhanced=True,
       ... )
       >>> factory = EnvironmentFactory(df, config)
       >>> env = factory.create_training_env()

    3. Uso con Presets:
       >>> factory = EnvironmentFactory(df)
       >>> env = factory.create_from_preset('aggressive')

    4. Backward-compatible (static methods):
       >>> env = EnvironmentFactory.create_v19(df, initial_balance=10000)
       >>> env = EnvironmentFactory.create_v19_enhanced(df, use_regime_features=True)

    Args:
        df: DataFrame con datos de mercado (OHLCV, indicadores, etc.)
        config: Configuracion del environment (opcional, usa defaults)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[EnvironmentConfig] = None,
    ):
        self.df = df
        self.config = config or EnvironmentConfig()

        # Validar DataFrame
        self._validate_dataframe()

    def _validate_dataframe(self) -> None:
        """Validar que el DataFrame tiene las columnas necesarias."""
        required_cols = []

        if self.config.return_column not in self.df.columns:
            warnings.warn(
                f"Return column '{self.config.return_column}' not found in DataFrame. "
                "Environment will calculate returns from 'close' column."
            )

        if self.config.vol_scaling.enabled:
            if self.config.vol_scaling.feature_column not in self.df.columns:
                warnings.warn(
                    f"Volatility column '{self.config.vol_scaling.feature_column}' not found. "
                    "Volatility scaling may not work correctly."
                )

        if self.config.regime_detection.enabled:
            if self.config.regime_detection.vix_column not in self.df.columns:
                warnings.warn(
                    f"VIX column '{self.config.regime_detection.vix_column}' not found. "
                    "Regime detection may not work correctly."
                )

    def _build_env_kwargs(self, is_training: bool = True) -> Dict[str, Any]:
        """
        Construir kwargs para crear el environment.

        Este metodo traduce EnvironmentConfig a los parametros
        que esperan TradingEnvironmentV19 y TradingEnvironmentV19Enhanced.
        """
        cfg = self.config

        kwargs = {
            # Core settings
            'df': self.df,
            'initial_balance': cfg.initial_balance,
            'max_position': cfg.max_position,
            'episode_length': cfg.episode_length,
            'max_drawdown_pct': cfg.max_drawdown_pct,

            # Cost settings
            'cost_model': cfg.cost_model.to_cost_model() if cfg.cost_model else None,
            'use_curriculum_costs': cfg.use_curriculum_costs if is_training else False,

            # Feature columns
            'feature_columns': cfg.feature_columns,
            'volatility_column': cfg.volatility_column,
            'return_column': cfg.return_column,

            # Volatility scaling
            'use_vol_scaling': cfg.vol_scaling.enabled,
            'vol_scaling_config': cfg.vol_scaling.to_dict() if cfg.vol_scaling.enabled else None,
            'vol_feature_column': cfg.vol_scaling.feature_column,

            # Regime detection
            'use_regime_detection': cfg.regime_detection.enabled,
            'regime_detector': cfg.regime_detection.create_detector() if cfg.regime_detection.enabled else None,
            'vix_column': cfg.regime_detection.vix_column,
            'embi_column': cfg.regime_detection.embi_column,
            'protection_mode': cfg.regime_detection.protection_mode,

            # Reward function - crear SymmetricCurriculumReward si no se especifica
            'reward_function': self._get_reward_function(cfg, is_training),

            # Misc
            'verbose': cfg.verbose,
        }

        # Enhanced-specific kwargs
        if cfg.use_enhanced:
            kwargs.update({
                'use_regime_features': cfg.enhanced_features.use_regime_features,
                'use_feedback_features': cfg.enhanced_features.use_feedback_features,
                'use_risk_manager': cfg.risk_manager.enabled if not is_training else False,
                'risk_limits': cfg.risk_manager.to_risk_limits() if cfg.risk_manager.enabled else None,
            })

        return kwargs

    def _get_reward_function(self, cfg, is_training: bool):
        """Obtener reward function, creando SymmetricCurriculumReward si es necesario."""
        if cfg.reward_function is not None:
            return cfg.reward_function

        # Si use_curriculum_costs=True y es training, crear SymmetricCurriculumReward
        if cfg.use_curriculum_costs and is_training:
            try:
                from ..rewards import SymmetricCurriculumReward
                return SymmetricCurriculumReward(
                    total_timesteps=80_000,  # Default, se puede override
                    config=None
                )
            except ImportError:
                pass

        return None

    def create_training_env(self) -> Union["TradingEnvironmentV19", "TradingEnvironmentV19Enhanced"]:
        """
        Crear environment para training.

        - Usa curriculum costs si esta configurado
        - Risk manager deshabilitado

        Returns:
            TradingEnvironmentV19 o TradingEnvironmentV19Enhanced
        """
        kwargs = self._build_env_kwargs(is_training=True)

        if self.config.use_enhanced:
            from ..environment_v19_enhanced import TradingEnvironmentV19Enhanced
            return TradingEnvironmentV19Enhanced(**kwargs)
        else:
            from ..environment_v19 import TradingEnvironmentV19
            return TradingEnvironmentV19(**kwargs)

    def create_validation_env(self, use_full_costs: bool = False) -> Union["TradingEnvironmentV19", "TradingEnvironmentV19Enhanced"]:
        """
        Crear environment para validacion.

        Args:
            use_full_costs: Si True, usa costos completos. Si False, usa mismos
                           costos que training para comparacion justa.

        - Risk manager habilitado si esta configurado

        Returns:
            TradingEnvironmentV19 o TradingEnvironmentV19Enhanced
        """
        # Para comparacion justa con training, usar is_training=True por defecto
        # Solo usar costos completos si explicitamente se solicita
        kwargs = self._build_env_kwargs(is_training=not use_full_costs)

        if self.config.use_enhanced:
            from ..environment_v19_enhanced import TradingEnvironmentV19Enhanced
            # En validacion, habilitar risk manager si esta configurado
            kwargs['use_risk_manager'] = self.config.risk_manager.enabled
            return TradingEnvironmentV19Enhanced(**kwargs)
        else:
            from ..environment_v19 import TradingEnvironmentV19
            return TradingEnvironmentV19(**kwargs)

    def create_from_preset(
        self,
        preset: Union[str, EnvironmentPreset],
        is_training: bool = True,
        **overrides,
    ) -> Union["TradingEnvironmentV19", "TradingEnvironmentV19Enhanced"]:
        """
        Crear environment desde un preset predefinido.

        Args:
            preset: Nombre del preset ('default', 'conservative', 'aggressive', etc.)
            is_training: Si es para training (True) o validacion (False)
            **overrides: Overrides adicionales sobre el preset

        Returns:
            Environment configurado

        Example:
            >>> factory = EnvironmentFactory(df)
            >>> env = factory.create_from_preset('aggressive', initial_balance=50000)
        """
        preset_name = preset.value if isinstance(preset, EnvironmentPreset) else preset

        if preset_name not in PRESET_CONFIGS:
            raise ValueError(
                f"Unknown preset '{preset_name}'. "
                f"Available: {list(PRESET_CONFIGS.keys())}"
            )

        # Obtener config del preset
        preset_config = copy.deepcopy(PRESET_CONFIGS[preset_name])
        preset_config.update(overrides)

        # Crear EnvironmentConfig
        new_config = EnvironmentConfig(**preset_config)

        # Crear factory temporal con nueva config
        factory = EnvironmentFactory(self.df, new_config)

        if is_training:
            return factory.create_training_env()
        else:
            return factory.create_validation_env()

    def with_config(self, **overrides) -> "EnvironmentFactory":
        """
        Crear nueva factory con configuracion modificada.

        Args:
            **overrides: Campos a modificar en la configuracion

        Returns:
            Nueva instancia de EnvironmentFactory

        Example:
            >>> factory = EnvironmentFactory(df)
            >>> factory_high_balance = factory.with_config(initial_balance=100000)
        """
        new_config = self.config.copy(**overrides)
        return EnvironmentFactory(self.df, new_config)

    def get_observation_space_size(self) -> int:
        """
        Obtener tamano del observation space sin crear el environment.

        Util para configurar la red neuronal antes de crear el environment.
        """
        # Base state features
        n_state = 12

        # Enhanced features
        if self.config.use_enhanced:
            if self.config.enhanced_features.use_regime_features:
                n_state += 6
            if self.config.enhanced_features.use_feedback_features:
                n_state += 3

        # Market features
        if self.config.feature_columns:
            n_market = len(self.config.feature_columns)
        else:
            # Auto-detect
            exclude_cols = {'timestamp', 'date', 'time', 'symbol'}
            numeric_cols = [
                c for c in self.df.columns
                if c not in exclude_cols and self.df[c].dtype in ['float64', 'float32', 'int64']
            ]
            n_market = len(numeric_cols)

        return n_state + n_market

    def describe(self) -> str:
        """Describir la configuracion actual."""
        cfg = self.config

        lines = [
            "=" * 60,
            "EnvironmentFactory Configuration",
            "=" * 60,
            "",
            "CORE SETTINGS:",
            f"  Initial Balance: ${cfg.initial_balance:,.2f}",
            f"  Max Position: {cfg.max_position}",
            f"  Episode Length: {cfg.episode_length} bars",
            f"  Max Drawdown: {cfg.max_drawdown_pct}%",
            "",
            "COST SETTINGS:",
            f"  Curriculum Costs: {cfg.use_curriculum_costs}",
            f"  Base Spread: {cfg.cost_model.base_spread_bps} bps",
            f"  High Vol Spread: {cfg.cost_model.high_vol_spread_bps} bps",
            "",
            "VOLATILITY SCALING:",
            f"  Enabled: {cfg.vol_scaling.enabled}",
        ]

        if cfg.vol_scaling.enabled:
            lines.extend([
                f"  Lookback: {cfg.vol_scaling.lookback_window} bars",
                f"  Quantiles: {cfg.vol_scaling.quantiles}",
                f"  Scale Factors: {cfg.vol_scaling.scale_factors}",
            ])

        lines.extend([
            "",
            "REGIME DETECTION:",
            f"  Enabled: {cfg.regime_detection.enabled}",
        ])

        if cfg.regime_detection.enabled:
            lines.extend([
                f"  Protection Mode: {cfg.regime_detection.protection_mode}",
                f"  Crisis Multiplier: {cfg.regime_detection.crisis_multiplier}",
                f"  Volatile Multiplier: {cfg.regime_detection.volatile_multiplier}",
            ])

        lines.extend([
            "",
            "ENHANCED ENVIRONMENT:",
            f"  Use Enhanced: {cfg.use_enhanced}",
        ])

        if cfg.use_enhanced:
            lines.extend([
                f"  Regime Features: {cfg.enhanced_features.use_regime_features}",
                f"  Feedback Features: {cfg.enhanced_features.use_feedback_features}",
            ])

        lines.extend([
            "",
            "RISK MANAGER:",
            f"  Enabled: {cfg.risk_manager.enabled}",
        ])

        if cfg.risk_manager.enabled:
            lines.extend([
                f"  Max DD Warning: {cfg.risk_manager.max_drawdown_warning*100}%",
                f"  Max DD Reduce: {cfg.risk_manager.max_drawdown_reduce*100}%",
                f"  Max DD Stop: {cfg.risk_manager.max_drawdown_stop*100}%",
            ])

        lines.extend([
            "",
            "OBSERVATION SPACE:",
            f"  Estimated Size: {self.get_observation_space_size()} features",
            "",
            "=" * 60,
        ])

        return "\n".join(lines)

    # =========================================================================
    # STATIC METHODS FOR BACKWARD COMPATIBILITY
    # =========================================================================

    @staticmethod
    def create_v19(
        df: pd.DataFrame,
        initial_balance: float = 10_000,
        max_position: float = 1.0,
        episode_length: int = 1200,
        max_drawdown_pct: float = 15.0,
        use_curriculum_costs: bool = True,
        reward_function: Optional[Any] = None,
        feature_columns: Optional[List[str]] = None,
        use_vol_scaling: bool = True,
        vol_scaling_config: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        **kwargs,
    ) -> "TradingEnvironmentV19":
        """
        Crear TradingEnvironmentV19 directamente.

        Metodo estatico para backward compatibility con codigo existente.
        Equivalente a llamar directamente a TradingEnvironmentV19(...).

        Example:
            >>> env = EnvironmentFactory.create_v19(df, initial_balance=50000)
        """
        from ..environment_v19 import TradingEnvironmentV19

        return TradingEnvironmentV19(
            df=df,
            initial_balance=initial_balance,
            max_position=max_position,
            episode_length=episode_length,
            max_drawdown_pct=max_drawdown_pct,
            use_curriculum_costs=use_curriculum_costs,
            reward_function=reward_function,
            feature_columns=feature_columns,
            use_vol_scaling=use_vol_scaling,
            vol_scaling_config=vol_scaling_config,
            verbose=verbose,
            **kwargs,
        )

    @staticmethod
    def create_v19_enhanced(
        df: pd.DataFrame,
        initial_balance: float = 10_000,
        max_position: float = 1.0,
        episode_length: int = 1200,
        max_drawdown_pct: float = 15.0,
        use_curriculum_costs: bool = True,
        reward_function: Optional[Any] = None,
        feature_columns: Optional[List[str]] = None,
        use_vol_scaling: bool = True,
        use_regime_features: bool = True,
        use_feedback_features: bool = True,
        use_risk_manager: bool = False,
        verbose: int = 0,
        **kwargs,
    ) -> "TradingEnvironmentV19Enhanced":
        """
        Crear TradingEnvironmentV19Enhanced directamente.

        Metodo estatico para backward compatibility con codigo existente.

        Example:
            >>> env = EnvironmentFactory.create_v19_enhanced(df, use_regime_features=True)
        """
        from ..environment_v19_enhanced import TradingEnvironmentV19Enhanced

        return TradingEnvironmentV19Enhanced(
            df=df,
            initial_balance=initial_balance,
            max_position=max_position,
            episode_length=episode_length,
            max_drawdown_pct=max_drawdown_pct,
            use_curriculum_costs=use_curriculum_costs,
            reward_function=reward_function,
            feature_columns=feature_columns,
            use_vol_scaling=use_vol_scaling,
            use_regime_features=use_regime_features,
            use_feedback_features=use_feedback_features,
            use_risk_manager=use_risk_manager,
            verbose=verbose,
            **kwargs,
        )

    @staticmethod
    def list_presets() -> List[str]:
        """Listar presets disponibles."""
        return list(PRESET_CONFIGS.keys())

    @staticmethod
    def get_preset_config(preset: str) -> Dict[str, Any]:
        """Obtener configuracion de un preset."""
        if preset not in PRESET_CONFIGS:
            raise ValueError(f"Unknown preset '{preset}'")
        return copy.deepcopy(PRESET_CONFIGS[preset])


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_environment(
    df: pd.DataFrame,
    config: Optional[EnvironmentConfig] = None,
    preset: Optional[str] = None,
    is_training: bool = True,
    **kwargs,
) -> Union["TradingEnvironmentV19", "TradingEnvironmentV19Enhanced"]:
    """
    Funcion de conveniencia para crear environments.

    Args:
        df: DataFrame con datos de mercado
        config: Configuracion (opcional)
        preset: Nombre del preset (opcional)
        is_training: Si es para training
        **kwargs: Overrides adicionales

    Returns:
        Environment configurado

    Example:
        >>> env = create_environment(df, preset='aggressive')
        >>> env = create_environment(df, initial_balance=50000)
    """
    # Si hay kwargs, crear config con ellos
    if kwargs and config is None:
        config = EnvironmentConfig(**kwargs)

    factory = EnvironmentFactory(df, config)

    if preset:
        return factory.create_from_preset(preset, is_training=is_training)
    elif is_training:
        return factory.create_training_env()
    else:
        return factory.create_validation_env()


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("ENVIRONMENT FACTORY - USD/COP RL Trading System")
    print("=" * 70)

    # Create dummy data for testing
    np.random.seed(42)
    n_samples = 2000

    df = pd.DataFrame({
        'close': 4500 + np.cumsum(np.random.normal(0, 10, n_samples)),
        'close_return': np.random.normal(0, 0.001, n_samples),
        'log_ret_5m': np.random.normal(0, 0.001, n_samples),
        'volatility_pct': np.random.uniform(0.5, 2.0, n_samples),
        'atr_pct': np.random.uniform(0.001, 0.01, n_samples),
        'vix_z': np.random.normal(0, 1, n_samples),
        'embi_z': np.random.normal(0, 1, n_samples),
        'rsi_14': np.random.uniform(30, 70, n_samples),
    })

    print("\n1. Test: Default Factory")
    print("-" * 50)
    factory = EnvironmentFactory(df)
    print(factory.describe())

    print("\n2. Test: Create Training Environment")
    print("-" * 50)
    env = factory.create_training_env()
    print(f"  Type: {type(env).__name__}")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space}")

    print("\n3. Test: Create from Preset 'enhanced'")
    print("-" * 50)
    env_enhanced = factory.create_from_preset('enhanced')
    print(f"  Type: {type(env_enhanced).__name__}")
    print(f"  Observation space: {env_enhanced.observation_space.shape}")

    print("\n4. Test: Static method create_v19")
    print("-" * 50)
    env_v19 = EnvironmentFactory.create_v19(df, initial_balance=50000)
    print(f"  Type: {type(env_v19).__name__}")
    print(f"  Initial balance: ${env_v19.initial_balance:,}")

    print("\n5. Test: List Presets")
    print("-" * 50)
    presets = EnvironmentFactory.list_presets()
    print(f"  Available presets: {presets}")

    print("\n6. Test: Custom Config")
    print("-" * 50)
    config = EnvironmentConfig(
        initial_balance=100_000,
        max_position=0.5,
        episode_length=2400,
        use_enhanced=True,
        enhanced_features=EnhancedFeaturesConfig(
            use_regime_features=True,
            use_feedback_features=True,
        ),
    )
    factory_custom = EnvironmentFactory(df, config)
    env_custom = factory_custom.create_training_env()
    print(f"  Type: {type(env_custom).__name__}")
    print(f"  Initial balance: ${config.initial_balance:,}")
    print(f"  Observation space: {env_custom.observation_space.shape}")

    print("\n7. Test: Run a few steps")
    print("-" * 50)
    obs, info = env.reset()
    print(f"  Reset obs shape: {obs.shape}")

    for i in range(5):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"  Step {i+1}: action={action[0]:.3f}, reward={reward:.3f}, position={info['position']:.3f}")

    print("\n" + "=" * 70)
    print("EnvironmentFactory ready for use!")
    print("=" * 70)
