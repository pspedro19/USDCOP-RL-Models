"""
USD/COP RL Trading System - Environment V19
=============================================

Environment mejorado con estado enriquecido de 12 features + Position Sizing Dinámico.

PROBLEMAS QUE RESUELVE:
- Estado actual: solo 2 features (position, time_normalized)
- Modelo no puede aprender patrones complejos
- No hay awareness de drawdown, régimen, costos
- Modelo hace 87% HOLD en alta volatilidad (posición fija = riesgo excesivo)

SOLUCIÓN:
- 12 state features informativas
- Costos dinámicos integrados
- Early termination en drawdown extremo
- Sampling inteligente por régimen
- Position Sizing Dinámico con DOS capas de protección:
  * VolatilityScaler: basado en volatilidad realizada (percentil)
  * RegimeDetector: basado en VIX, EMBI, volatilidad
  * AMBOS usan MIN(vol_mult, regime_mult) - NO se multiplican

Author: Claude Code
Version: 1.2.0 (integrated RegimeDetector with MIN logic)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any, List, Union
from dataclasses import dataclass
from enum import IntEnum
from collections import deque
import warnings

# Lazy import for RegimeDetector to avoid circular imports
_regime_detector_module = None

def _get_regime_detector_class():
    """Lazy load RegimeDetector to avoid circular imports."""
    global _regime_detector_module
    if _regime_detector_module is None:
        try:
            # Try relative import first (package context)
            from . import regime_detector as _regime_detector_module
        except ImportError:
            try:
                # Fallback to absolute import (script context)
                import regime_detector as _regime_detector_module
            except ImportError:
                return None
    return _regime_detector_module.RegimeDetector if _regime_detector_module else None


class TradingAction(IntEnum):
    """Discretización de acciones para interpretación."""
    STRONG_SHORT = 0
    SHORT = 1
    HOLD = 2
    LONG = 3
    STRONG_LONG = 4


@dataclass
class MarketRegime:
    """Régimen de mercado actual."""
    volatility: str  # 'low', 'medium', 'high', 'crisis'
    trend: str  # 'bull', 'bear', 'sideways'
    session: str  # 'london', 'ny_am', 'ny_pm', 'asia', 'closed'


class VolatilityScaler:
    """
    Position sizing dinámico basado en volatilidad realizada.

    PROBLEMA:
    - Modelo hace HOLD 87% en períodos de alta volatilidad
    - Posiciones fijas exponen demasiado riesgo en crisis

    SOLUCIÓN:
    - Escalar posición inversamente a la volatilidad
    - Reducir exposición cuando vol_percentile > 0.5
    - Crisis (p90+) → 25% posición
    - Alta vol (p75+) → 50% posición
    - Media vol (p50+) → 75% posición
    - Normal vol → 100% posición

    Args:
        lookback_window: Ventana para cálculo de volatilidad histórica
        quantiles: Percentiles para buckets de volatilidad
        scale_factors: Factores de escala por bucket [0.25, 0.5, 0.75, 1.0]
    """

    def __init__(
        self,
        lookback_window: int = 60,  # 60 barras = 1 día en 15min
        quantiles: Optional[List[float]] = None,
        scale_factors: Optional[List[float]] = None,
    ):
        self.lookback_window = lookback_window
        self.quantiles = quantiles or [0.5, 0.75, 0.9]
        self.scale_factors = scale_factors or [1.0, 0.75, 0.5, 0.25]

        # Validación
        if len(self.scale_factors) != len(self.quantiles) + 1:
            raise ValueError(
                f"scale_factors ({len(self.scale_factors)}) debe tener "
                f"len(quantiles) + 1 ({len(self.quantiles) + 1}) elementos"
            )

        # Buffer circular para volatilidad histórica
        self.vol_history = deque(maxlen=lookback_window)

    def reset(self):
        """Reset del buffer de volatilidad."""
        self.vol_history.clear()

    def update(self, volatility: float):
        """
        Actualizar buffer de volatilidad histórica.

        Args:
            volatility: Volatilidad actual (ej: atr_pct)
        """
        self.vol_history.append(volatility)

    def scale_position(self, current_vol: float) -> float:
        """
        Escalar posición basado en volatilidad actual vs histórica.

        Args:
            current_vol: Volatilidad actual

        Returns:
            Factor de escala [0.25, 1.0]
        """
        # Si no hay suficiente historia, no escalar
        if len(self.vol_history) < 10:
            return 1.0

        # Calcular percentil de volatilidad actual
        vol_array = np.array(self.vol_history)
        vol_percentile = np.searchsorted(np.sort(vol_array), current_vol) / len(vol_array)

        # Determinar factor de escala por bucket
        for i, q in enumerate(self.quantiles):
            if vol_percentile <= q:
                return self.scale_factors[i]

        # Mayor que el último quantile
        return self.scale_factors[-1]

    def get_volatility_percentile(self, current_vol: float) -> float:
        """
        Obtener percentil de volatilidad para tracking.

        Args:
            current_vol: Volatilidad actual

        Returns:
            Percentil [0, 1]
        """
        if len(self.vol_history) < 10:
            return 0.5

        vol_array = np.array(self.vol_history)
        return float(np.searchsorted(np.sort(vol_array), current_vol) / len(vol_array))


class SETFXCostModel:
    """
    Modelo de costos basado en SET-FX Colombia.

    Costos reales:
    - Spread normal: 14-18 bps
    - Spread alta volatilidad: 25-36 bps
    - Slippage promedio: 2-5 bps
    """

    def __init__(
        self,
        base_spread_bps: float = 14.0,
        high_vol_spread_bps: float = 28.0,
        crisis_spread_bps: float = 45.0,
        slippage_bps: float = 3.0,
        volatility_threshold_high: float = 0.7,
        volatility_threshold_crisis: float = 0.9,
    ):
        self.base_spread = base_spread_bps / 10000
        self.high_vol_spread = high_vol_spread_bps / 10000
        self.crisis_spread = crisis_spread_bps / 10000
        self.slippage = slippage_bps / 10000
        self.vol_threshold_high = volatility_threshold_high
        self.vol_threshold_crisis = volatility_threshold_crisis

    def get_cost(self, volatility_percentile: float, position_change: float) -> float:
        """
        Calcular costo total de transacción.

        Args:
            volatility_percentile: Percentil de volatilidad [0, 1]
            position_change: Cambio absoluto en posición [0, 2]

        Returns:
            Costo como decimal (ej: 0.0020 = 20 bps)
        """
        # Determinar spread según volatilidad
        if volatility_percentile >= self.vol_threshold_crisis:
            spread = self.crisis_spread
        elif volatility_percentile >= self.vol_threshold_high:
            spread = self.high_vol_spread
        else:
            spread = self.base_spread

        # Costo total proporcional al cambio de posición
        total_cost = (spread + self.slippage) * abs(position_change)

        return total_cost

    def get_cost_bps(self, volatility_percentile: float) -> float:
        """Obtener spread actual en basis points."""
        if volatility_percentile >= self.vol_threshold_crisis:
            return self.crisis_spread * 10000
        elif volatility_percentile >= self.vol_threshold_high:
            return self.high_vol_spread * 10000
        return self.base_spread * 10000


class TradingEnvironmentV19(gym.Env):
    """
    Environment de trading V19 con estado enriquecido.

    STATE FEATURES (12):
    1. position: Posición actual normalizada [-1, 1]
    2. unrealized_pnl: PnL no realizado normalizado
    3. cumulative_return: Retorno acumulado del episodio
    4. current_drawdown: Drawdown actual
    5. max_drawdown_episode: Max drawdown del episodio
    6. regime_encoded: Régimen de mercado codificado
    7. session_phase: Fase de sesión [0-1]
    8. volatility_regime: Régimen de volatilidad [0-1]
    9. cost_regime: Costo actual normalizado
    10. position_duration: Duración de posición actual
    11. trade_count_normalized: Trades ejecutados normalizado
    12. time_remaining: Tiempo restante en episodio

    PLUS: Features de mercado del dataset (OHLCV, indicadores)

    Args:
        df: DataFrame con datos de mercado
        initial_balance: Balance inicial
        max_position: Posición máxima permitida
        episode_length: Longitud del episodio en barras
        max_drawdown_pct: Drawdown máximo antes de terminar
        cost_model: Modelo de costos (None = usar default)
        use_curriculum_costs: Si usar costos curriculum (0 al inicio)
        reward_function: Función de reward externa
        feature_columns: Columnas a usar como features de mercado
        volatility_column: Columna de volatilidad para costos
        return_column: Columna de retornos del mercado
        use_vol_scaling: Si usar position sizing dinámico por volatilidad
        vol_scaling_config: Config para VolatilityScaler (None = usar defaults)
        vol_feature_column: Columna con volatilidad realizada (default: 'atr_pct')
        verbose: Nivel de verbosidad
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        df,
        initial_balance: float = 10_000,
        max_position: float = 1.0,
        episode_length: int = 1200,
        max_drawdown_pct: float = 15.0,
        cost_model: Optional[SETFXCostModel] = None,
        use_curriculum_costs: bool = True,
        reward_function = None,
        feature_columns: Optional[List[str]] = None,
        volatility_column: str = 'volatility_pct',
        return_column: str = 'close_return',
        use_vol_scaling: bool = True,
        vol_scaling_config: Optional[Dict[str, Any]] = None,
        vol_feature_column: str = 'atr_pct',
        use_regime_detection: bool = False,
        regime_detector: Optional[Any] = None,
        vix_column: str = 'vix_z',
        embi_column: str = 'embi_z',
        protection_mode: str = 'min',  # 'min', 'multiply', 'vol_only', 'regime_only'
        verbose: int = 0,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.max_position = max_position
        self.episode_length = episode_length
        self.max_drawdown_pct = max_drawdown_pct / 100
        self.verbose = verbose

        # Modelo de costos
        self.cost_model = cost_model or SETFXCostModel()
        self.use_curriculum_costs = use_curriculum_costs
        self.curriculum_cost_multiplier = 0.0 if use_curriculum_costs else 1.0

        # Reward function externa
        self.reward_function = reward_function

        # Columnas
        self.volatility_column = volatility_column
        self.return_column = return_column
        self.vol_feature_column = vol_feature_column
        self.vix_column = vix_column
        self.embi_column = embi_column

        # Protection mode: how to combine vol_scaler and regime_detector
        # 'min': take minimum of both multipliers (RECOMMENDED)
        # 'multiply': multiply both (very conservative, may over-penalize)
        # 'vol_only': only use VolatilityScaler
        # 'regime_only': only use RegimeDetector
        self.protection_mode = protection_mode

        # Volatility Scaling
        self.use_vol_scaling = use_vol_scaling and protection_mode in ['min', 'multiply', 'vol_only']
        if self.use_vol_scaling:
            vol_config = vol_scaling_config or {}
            self.vol_scaler = VolatilityScaler(
                lookback_window=vol_config.get('lookback_window', 60),
                quantiles=vol_config.get('quantiles', [0.5, 0.75, 0.9]),
                scale_factors=vol_config.get('scale_factors', [1.0, 0.75, 0.5, 0.25]),
            )
        else:
            self.vol_scaler = None

        # Regime Detection
        self.use_regime_detection = use_regime_detection and protection_mode in ['min', 'multiply', 'regime_only']
        if self.use_regime_detection:
            if regime_detector is not None:
                self.regime_detector = regime_detector
            else:
                # Try to create default RegimeDetector
                RegimeDetectorClass = _get_regime_detector_class()
                if RegimeDetectorClass:
                    self.regime_detector = RegimeDetectorClass()
                else:
                    warnings.warn("RegimeDetector not available. Disabling regime detection.")
                    self.use_regime_detection = False
                    self.regime_detector = None
        else:
            self.regime_detector = None

        # Detectar features de mercado
        if feature_columns is None:
            # Auto-detectar columnas numéricas
            exclude_cols = ['timestamp', 'date', 'time', 'symbol']
            self.feature_columns = [
                c for c in df.columns
                if c not in exclude_cols and df[c].dtype in ['float64', 'float32', 'int64']
            ]
        else:
            self.feature_columns = feature_columns

        self.n_market_features = len(self.feature_columns)
        self.n_state_features = 12  # Features internas del estado

        # Espacios de observación y acción
        total_features = self.n_state_features + self.n_market_features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_features,),
            dtype=np.float32
        )

        # Acción continua [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # Pre-calcular estadísticas para normalización
        self._precompute_stats()

        # Inicializar estado
        self._reset_state()

    def _precompute_stats(self):
        """Pre-calcular estadísticas para normalización."""
        # Volatilidad
        if self.volatility_column in self.df.columns:
            vol = self.df[self.volatility_column].dropna()
            self.vol_percentiles = np.percentile(vol, [25, 50, 75, 90, 95])
        else:
            self.vol_percentiles = [0.5, 1.0, 1.5, 2.0, 2.5]

        # Returns
        if self.return_column in self.df.columns:
            ret = self.df[self.return_column].dropna()
            self.return_std = ret.std()
        else:
            self.return_std = 0.001

        # Feature stats para normalización
        self.feature_means = {}
        self.feature_stds = {}
        for col in self.feature_columns:
            self.feature_means[col] = float(self.df[col].mean())
            self.feature_stds[col] = float(self.df[col].std()) + 1e-8

    def _reset_state(self):
        """Resetear estado interno."""
        self.position = 0.0
        self.portfolio_value = self.initial_balance
        self.peak_value = self.initial_balance
        self.current_step = 0
        self.start_idx = 0
        self.entry_price = None
        self.unrealized_pnl = 0.0
        self.cumulative_return = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown_episode = 0.0
        self.position_duration = 0
        self.trade_count = 0
        self.last_trade_step = 0

        # Historial para métricas
        self.returns_history = []
        self.actions_history = []
        self.portfolio_history = [self.initial_balance]

        # Reset volatility scaler
        if self.vol_scaler is not None:
            self.vol_scaler.reset()

        # Reset regime detector history if it has one
        if self.regime_detector is not None and hasattr(self.regime_detector, 'regime_history'):
            self.regime_detector.regime_history = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Resetear environment para nuevo episodio.

        Options:
            start_idx: Índice de inicio específico
            regime: Régimen de mercado deseado ('high_vol', 'low_vol', 'trend', 'range')
        """
        super().reset(seed=seed)

        self._reset_state()

        # Determinar índice de inicio
        max_start = len(self.df) - self.episode_length - 1

        if options and 'start_idx' in options:
            self.start_idx = min(options['start_idx'], max_start)
        elif options and 'regime' in options:
            self.start_idx = self._sample_by_regime(options['regime'], max_start)
        else:
            self.start_idx = self.np_random.integers(0, max_start)

        self.current_step = 0

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def _sample_by_regime(self, regime: str, max_start: int) -> int:
        """Sampling inteligente por régimen de mercado."""
        if self.volatility_column not in self.df.columns:
            return self.np_random.integers(0, max_start)

        vol = self.df[self.volatility_column].values

        if regime == 'high_vol':
            # Buscar períodos de alta volatilidad
            threshold = np.percentile(vol, 80)
            candidates = np.where(vol > threshold)[0]
        elif regime == 'low_vol':
            threshold = np.percentile(vol, 20)
            candidates = np.where(vol < threshold)[0]
        elif regime == 'crisis':
            threshold = np.percentile(vol, 95)
            candidates = np.where(vol > threshold)[0]
        else:
            candidates = np.arange(max_start)

        # Filtrar candidatos válidos
        valid = candidates[candidates < max_start]

        if len(valid) == 0:
            return self.np_random.integers(0, max_start)

        return int(self.np_random.choice(valid))

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Ejecutar un step en el environment.

        Args:
            action: Acción continua [-1, 1] representando posición target

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Parsear acción
        action_value = float(np.clip(action[0], -1, 1))
        target_position = action_value * self.max_position

        # Obtener datos actuales
        idx = self.start_idx + self.current_step
        current_data = self.df.iloc[idx]
        next_data = self.df.iloc[idx + 1] if idx + 1 < len(self.df) else current_data

        # === POSITION PROTECTION (VolatilityScaler + RegimeDetector) ===
        vol_multiplier = 1.0
        regime_multiplier = 1.0
        vol_percentile_scaled = 0.5
        current_regime = "NORMAL"

        # Get VolatilityScaler multiplier
        if self.vol_scaler is not None and self.vol_feature_column in current_data.index:
            current_vol = current_data.get(self.vol_feature_column, 1.0)
            self.vol_scaler.update(current_vol)
            vol_multiplier = self.vol_scaler.scale_position(current_vol)
            vol_percentile_scaled = self.vol_scaler.get_volatility_percentile(current_vol)

        # Get RegimeDetector multiplier
        if self.regime_detector is not None:
            vix_z = current_data.get(self.vix_column, 0.0)
            embi_z = current_data.get(self.embi_column, 0.0)
            vol_pct = vol_percentile_scaled * 100  # Convert to 0-100 scale
            current_regime = self.regime_detector.detect_regime(vix_z, embi_z, vol_pct)
            regime_multiplier = self.regime_detector.get_position_multiplier(current_regime)

        # Combine multipliers based on protection_mode
        if self.protection_mode == 'min':
            # Take the more conservative (minimum) multiplier
            position_scale = min(vol_multiplier, regime_multiplier)
        elif self.protection_mode == 'multiply':
            # Multiply both (very conservative)
            position_scale = vol_multiplier * regime_multiplier
        elif self.protection_mode == 'vol_only':
            position_scale = vol_multiplier
        elif self.protection_mode == 'regime_only':
            position_scale = regime_multiplier
        else:
            position_scale = 1.0

        # Apply position scaling
        target_position *= position_scale

        # Obtener retorno del mercado
        market_return = self._get_market_return(current_data, next_data)

        # Calcular cambio de posición
        prev_position = self.position
        position_change = abs(target_position - prev_position)

        # Obtener volatilidad para costos
        volatility_pct = self._get_volatility_percentile(current_data)

        # Calcular costo de transacción
        transaction_cost = 0.0
        if position_change > 0.01:  # Umbral mínimo de cambio
            base_cost = self.cost_model.get_cost(volatility_pct, position_change)
            transaction_cost = base_cost * self.curriculum_cost_multiplier
            self.trade_count += 1
            self.last_trade_step = self.current_step
            self.position_duration = 0
        else:
            self.position_duration += 1

        # Actualizar posición
        self.position = target_position

        # Calcular PnL
        position_return = self.position * market_return
        net_return = position_return - transaction_cost

        # Actualizar portfolio
        prev_portfolio = self.portfolio_value
        self.portfolio_value *= (1 + net_return)

        # Actualizar unrealized PnL
        if abs(self.position) > 0.01:
            if self.entry_price is None:
                self.entry_price = current_data.get('close', 1.0)
            current_price = next_data.get('close', self.entry_price)
            self.unrealized_pnl = self.position * (current_price - self.entry_price) / self.entry_price
        else:
            self.entry_price = None
            self.unrealized_pnl = 0.0

        # Actualizar métricas
        step_return = (self.portfolio_value - prev_portfolio) / prev_portfolio
        self.returns_history.append(step_return)
        self.actions_history.append(action_value)
        self.portfolio_history.append(self.portfolio_value)

        self.cumulative_return = (self.portfolio_value - self.initial_balance) / self.initial_balance

        # Actualizar drawdown
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        self.current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        self.max_drawdown_episode = max(self.max_drawdown_episode, self.current_drawdown)

        # Calcular reward
        if self.reward_function is not None:
            reward, reward_info = self.reward_function.calculate(
                portfolio_return=step_return,
                market_return=market_return,
                portfolio_value=self.portfolio_value,
                position=self.position,
                prev_position=prev_position,
                volatility_percentile=volatility_pct,
            )
        else:
            # Reward simple
            reward = step_return * 100  # Escalar para gradientes
            reward_info = {}

        # Avanzar step
        self.current_step += 1

        # Verificar terminación
        terminated = False
        truncated = False

        # Early termination por drawdown extremo
        if self.current_drawdown > self.max_drawdown_pct:
            terminated = True
            if self.verbose > 0:
                print(f"Episode terminated: Drawdown {self.current_drawdown*100:.1f}% > {self.max_drawdown_pct*100:.1f}%")

        # Truncation por fin de episodio
        if self.current_step >= self.episode_length:
            truncated = True

        # Truncation por fin de datos
        if self.start_idx + self.current_step >= len(self.df) - 1:
            truncated = True

        # Obtener observación
        obs = self._get_observation()
        info = self._get_info()
        info['step_return'] = step_return
        info['market_return'] = market_return
        info['transaction_cost'] = transaction_cost
        info['reward_info'] = reward_info
        info['position_scale'] = position_scale
        info['vol_percentile_scaled'] = vol_percentile_scaled
        info['vol_multiplier'] = vol_multiplier
        info['regime_multiplier'] = regime_multiplier
        info['current_regime'] = current_regime
        info['protection_mode'] = self.protection_mode

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Construir vector de observación."""
        idx = self.start_idx + self.current_step
        current_data = self.df.iloc[idx]

        # === STATE FEATURES (12) ===
        state_features = [
            # 1. Posición actual
            self.position,

            # 2. Unrealized PnL normalizado
            np.clip(self.unrealized_pnl / 0.05, -1, 1),

            # 3. Retorno acumulado normalizado
            np.clip(self.cumulative_return / 0.10, -1, 1),

            # 4. Drawdown actual
            -self.current_drawdown / self.max_drawdown_pct,

            # 5. Max drawdown del episodio
            -self.max_drawdown_episode / self.max_drawdown_pct,

            # 6. Régimen de mercado codificado
            self._encode_regime(current_data),

            # 7. Fase de sesión
            self._get_session_phase(current_data),

            # 8. Régimen de volatilidad
            self._get_volatility_percentile(current_data),

            # 9. Costo actual normalizado
            self.curriculum_cost_multiplier,

            # 10. Duración de posición normalizada
            min(self.position_duration / 100, 1.0),

            # 11. Trade count normalizado
            min(self.trade_count / 50, 1.0),

            # 12. Tiempo restante
            1.0 - (self.current_step / self.episode_length),
        ]

        # === MARKET FEATURES ===
        market_features = []
        for col in self.feature_columns:
            value = current_data.get(col, 0)
            # Normalizar
            normalized = (value - self.feature_means[col]) / self.feature_stds[col]
            market_features.append(np.clip(normalized, -5, 5))

        # Combinar
        obs = np.array(state_features + market_features, dtype=np.float32)

        # Reemplazar NaN/Inf
        obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)

        return obs

    def _get_market_return(self, current_data, next_data) -> float:
        """Obtener retorno del mercado."""
        if self.return_column in self.df.columns:
            return float(next_data.get(self.return_column, 0))

        # Calcular desde precios
        current_price = current_data.get('close', 1.0)
        next_price = next_data.get('close', current_price)

        if current_price > 0:
            return (next_price - current_price) / current_price
        return 0.0

    def _get_volatility_percentile(self, data) -> float:
        """Obtener percentil de volatilidad."""
        if self.volatility_column not in self.df.columns:
            return 0.5

        vol = data.get(self.volatility_column, 1.0)

        # Calcular percentil
        for i, p in enumerate(self.vol_percentiles):
            if vol <= p:
                return i / len(self.vol_percentiles)

        return 1.0

    def _encode_regime(self, data) -> float:
        """Codificar régimen de mercado."""
        vol_pct = self._get_volatility_percentile(data)

        # Simple encoding basado en volatilidad
        if vol_pct >= 0.9:
            return 1.0  # Crisis
        elif vol_pct >= 0.7:
            return 0.5  # Alta volatilidad
        elif vol_pct <= 0.3:
            return -0.5  # Baja volatilidad
        return 0.0  # Normal

    def _get_session_phase(self, data) -> float:
        """Obtener fase de sesión de trading."""
        # Si tenemos columna de hora
        if 'hour' in data.index:
            hour = int(data['hour'])
        elif 'timestamp' in data.index:
            try:
                import pandas as pd
                ts = pd.Timestamp(data['timestamp'])
                hour = ts.hour
            except:
                return 0.5
        else:
            return 0.5

        # Sesiones de trading para Colombia (UTC-5)
        if 3 <= hour < 8:  # Londres
            return 0.2
        elif 8 <= hour < 12:  # NY AM
            return 0.8  # Mayor liquidez
        elif 12 <= hour < 16:  # NY PM
            return 0.6
        elif 16 <= hour < 20:  # Asia temprana
            return 0.3
        else:  # Cerrado/Asia tarde
            return 0.1

    def _get_info(self) -> Dict[str, Any]:
        """Obtener información adicional."""
        return {
            'portfolio': self.portfolio_value,
            'position': self.position,
            'cumulative_return': self.cumulative_return,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown_episode,
            'trade_count': self.trade_count,
            'step': self.current_step,
            'cost_multiplier': self.curriculum_cost_multiplier,
        }

    def set_curriculum_costs(self, normal_cost: float, high_vol_cost: float = None):
        """
        Método para que CostCurriculumCallback actualice los costos.

        Args:
            normal_cost: Costo normal como decimal (ej: 0.0025 = 25 bps)
            high_vol_cost: Costo alta volatilidad (opcional)
        """
        # Calcular multiplicador basado en costo objetivo vs costo base
        if self.cost_model.base_spread > 0:
            self.curriculum_cost_multiplier = normal_cost / self.cost_model.base_spread
        else:
            self.curriculum_cost_multiplier = 1.0 if normal_cost > 0 else 0.0

        self.curriculum_cost_multiplier = np.clip(self.curriculum_cost_multiplier, 0, 1)

    # Aliases para compatibilidad con callbacks
    @property
    def cost(self) -> float:
        """Alias para compatibilidad."""
        return self.cost_model.base_spread * self.curriculum_cost_multiplier

    @cost.setter
    def cost(self, value: float):
        """Setter para compatibilidad."""
        if self.cost_model.base_spread > 0:
            self.curriculum_cost_multiplier = value / self.cost_model.base_spread
        self.curriculum_cost_multiplier = np.clip(self.curriculum_cost_multiplier, 0, 1)

    @property
    def cost_normal(self) -> float:
        return self.cost

    @cost_normal.setter
    def cost_normal(self, value: float):
        self.cost = value

    @property
    def cost_high_vol(self) -> float:
        return self.cost_model.high_vol_spread * self.curriculum_cost_multiplier

    @cost_high_vol.setter
    def cost_high_vol(self, value: float):
        pass  # Ignorado, usamos el modelo

    def get_episode_metrics(self) -> Dict[str, float]:
        """Obtener métricas del episodio actual."""
        if len(self.returns_history) < 2:
            return {}

        returns = np.array(self.returns_history)
        actions = np.array(self.actions_history)

        # Sharpe (anualizado asumiendo 60 bars/día)
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252 * 60)
        else:
            sharpe = 0.0

        # Sortino
        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = returns.mean() / downside.std() * np.sqrt(252 * 60)
        else:
            sortino = 0.0

        # Distribución de acciones
        # Action classification - THRESHOLD 0.10 (was 0.15, changed to account for regime multiplier)
        ACTION_THRESHOLD = 0.10
        pct_hold = float((np.abs(actions) < ACTION_THRESHOLD).mean() * 100)
        pct_long = float((actions > ACTION_THRESHOLD).mean() * 100)
        pct_short = float((actions < -ACTION_THRESHOLD).mean() * 100)

        metrics = {
            'sharpe': sharpe,
            'sortino': sortino,
            'total_return': self.cumulative_return * 100,
            'max_drawdown': self.max_drawdown_episode * 100,
            'trade_count': self.trade_count,
            'pct_hold': pct_hold,
            'pct_long': pct_long,
            'pct_short': pct_short,
            'mean_action': float(actions.mean()),
            'std_action': float(actions.std()),
        }

        # Agregar métricas de volatility scaling si está habilitado
        if self.vol_scaler is not None and len(self.vol_scaler.vol_history) > 0:
            vol_array = np.array(self.vol_scaler.vol_history)
            metrics['vol_mean'] = float(vol_array.mean())
            metrics['vol_std'] = float(vol_array.std())
            metrics['vol_min'] = float(vol_array.min())
            metrics['vol_max'] = float(vol_array.max())

        return metrics

    def render(self, mode='human'):
        """Renderizar estado actual."""
        if mode == 'human':
            print(f"\nStep {self.current_step}/{self.episode_length}")
            print(f"  Portfolio: ${self.portfolio_value:,.2f}")
            print(f"  Position: {self.position:+.2f}")
            print(f"  Return: {self.cumulative_return*100:+.2f}%")
            print(f"  Drawdown: {self.current_drawdown*100:.2f}%")
            print(f"  Trades: {self.trade_count}")


def create_training_env(
    df,
    config,
    reward_function=None,
    feature_columns=None,
) -> TradingEnvironmentV19:
    """
    Factory function para crear environment de training.

    Args:
        df: DataFrame con datos
        config: TrainingConfigV19 o dict con configuración
        reward_function: Función de reward
        feature_columns: Columnas de features

    Returns:
        TradingEnvironmentV19 configurado
    """
    # Extraer configuración
    if hasattr(config, 'environment'):
        env_config = config.environment
        initial_balance = env_config.initial_balance
        max_position = env_config.max_position
        episode_length = env_config.episode_length
        max_drawdown = env_config.max_drawdown_pct
    else:
        initial_balance = config.get('initial_balance', 10000)
        max_position = config.get('max_position', 1.0)
        episode_length = config.get('episode_length', 1200)
        max_drawdown = config.get('max_drawdown_pct', 15.0)

    env = TradingEnvironmentV19(
        df=df,
        initial_balance=initial_balance,
        max_position=max_position,
        episode_length=episode_length,
        max_drawdown_pct=max_drawdown,
        use_curriculum_costs=True,
        reward_function=reward_function,
        feature_columns=feature_columns,
    )

    return env


def create_validation_env(
    df,
    config,
    reward_function=None,
    feature_columns=None,
) -> TradingEnvironmentV19:
    """
    Factory function para crear environment de validación.

    La diferencia principal es que usa costos COMPLETOS
    (no curriculum) para evaluar en condiciones reales.
    """
    if hasattr(config, 'environment'):
        env_config = config.environment
        initial_balance = env_config.initial_balance
        max_position = env_config.max_position
        episode_length = env_config.episode_length
        max_drawdown = env_config.max_drawdown_pct
    else:
        initial_balance = config.get('initial_balance', 10000)
        max_position = config.get('max_position', 1.0)
        episode_length = config.get('episode_length', 1200)
        max_drawdown = config.get('max_drawdown_pct', 15.0)

    env = TradingEnvironmentV19(
        df=df,
        initial_balance=initial_balance,
        max_position=max_position,
        episode_length=episode_length,
        max_drawdown_pct=max_drawdown,
        use_curriculum_costs=False,  # Costos completos
        reward_function=reward_function,
        feature_columns=feature_columns,
    )

    return env


# ============================================================================
# EJEMPLO DE USO: VOLATILITY SCALING
# ============================================================================
"""
EJEMPLO 1: Uso básico con configuración por default
----------------------------------------------------

import pandas as pd
from environment_v19 import TradingEnvironmentV19

# Cargar datos (debe tener columna 'atr_pct')
df = pd.read_parquet('dataset.parquet')

# Crear environment con volatility scaling habilitado
env = TradingEnvironmentV19(
    df=df,
    use_vol_scaling=True,  # Activar position sizing dinámico
    vol_feature_column='atr_pct',  # Columna de volatilidad realizada
)

# El environment ahora escalará automáticamente las posiciones:
# - Crisis (vol p90+) → 25% posición
# - Alta vol (p75+) → 50% posición
# - Media vol (p50+) → 75% posición
# - Normal vol → 100% posición


EJEMPLO 2: Configuración personalizada
---------------------------------------

# Configuración más agresiva (menos reducción en alta vol)
vol_config = {
    'lookback_window': 96,  # 1.5 días en 15min
    'quantiles': [0.6, 0.8, 0.95],  # Buckets más altos
    'scale_factors': [1.0, 0.8, 0.6, 0.4]  # Menos reducción
}

env = TradingEnvironmentV19(
    df=df,
    use_vol_scaling=True,
    vol_scaling_config=vol_config,
    vol_feature_column='atr_pct',
)


EJEMPLO 3: Configuración conservadora (más reducción)
------------------------------------------------------

# Para trading más defensivo en crisis
vol_config = {
    'lookback_window': 60,
    'quantiles': [0.4, 0.6, 0.8],  # Buckets más bajos
    'scale_factors': [1.0, 0.6, 0.3, 0.1]  # Más reducción
}

env = TradingEnvironmentV19(
    df=df,
    use_vol_scaling=True,
    vol_scaling_config=vol_config,
    vol_feature_column='atr_pct',
)


EJEMPLO 4: Desactivar volatility scaling
-----------------------------------------

# Para baseline o si quieres posiciones fijas
env = TradingEnvironmentV19(
    df=df,
    use_vol_scaling=False,  # Posiciones fijas
)


EJEMPLO 5: Acceder a métricas de scaling durante training
----------------------------------------------------------

obs, info = env.reset()

for step in range(1000):
    action = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)

    # Métricas de volatility scaling
    position_scale = info['position_scale']  # Factor aplicado [0.25-1.0]
    vol_percentile = info['vol_percentile_scaled']  # Percentil de vol [0-1]

    if done or truncated:
        metrics = env.get_episode_metrics()
        # Métricas incluyen: vol_mean, vol_std, vol_min, vol_max
        print(f"Vol mean: {metrics.get('vol_mean', 0):.4f}")
        print(f"Vol std: {metrics.get('vol_std', 0):.4f}")
        break


EJEMPLO 6: Training con SB3
----------------------------

from stable_baselines3 import PPO

# Environment wrapper para vectorización
def make_env():
    return TradingEnvironmentV19(
        df=df,
        use_vol_scaling=True,
        vol_feature_column='atr_pct',
    )

env = make_env()

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)


RESULTADO ESPERADO:
-------------------
- Menos HOLD en alta volatilidad (modelo puede tomar posiciones pequeñas)
- Mejor gestión de riesgo en períodos de crisis
- Sharpe ratio más estable across regímenes
- Menos early terminations por drawdown extremo
"""
