"""
USD/COP RL Trading System - Symmetric Curriculum Reward V2
==========================================================

Reward function que resuelve:
1. Colapso a política constante (LONG=100%)
2. Asimetría LONG/SHORT/HOLD
3. Desconexión costos training vs reales

Features:
- Curriculum learning en 3 fases
- Simetría forzada LONG/SHORT
- Costos graduales (0 -> 25 bps)
- Detección de comportamientos patológicos
- Sortino-based risk adjustment

Author: Claude Code
Version: 2.0.0
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import sys
from pathlib import Path

# Importar configuración
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from config.training_config_v19 import RewardConfig, TrainingPhase
except ImportError:
    # Fallback si no encuentra el módulo
    class TrainingPhase(Enum):
        EXPLORATION = 0
        TRANSITION = 1
        REALISTIC = 2

    @dataclass
    class RewardConfig:
        phase_boundaries: Tuple[float, float] = (0.30, 0.60)
        transition_target_cost_bps: float = 10.0
        realistic_min_cost_bps: float = 25.0
        realistic_max_cost_bps: float = 36.0
        symmetry_window: int = 60
        max_directional_bias: float = 0.30
        symmetry_penalty_scale: float = 2.0
        max_trades_per_bar: float = 0.05
        overtrading_lookback: int = 120
        overtrading_penalty: float = 0.5
        max_hold_duration: int = 36
        inactivity_penalty: float = 0.3
        reversal_threshold: int = 5
        churning_penalty: float = 0.4
        sortino_window: int = 60
        sortino_mar: float = 0.0
        reward_scale: float = 100.0
        clip_range: Tuple[float, float] = (-5.0, 5.0)


# =============================================================================
# CURRICULUM COST SCHEDULER
# =============================================================================

class CurriculumCostScheduler:
    """
    Implementa introducción gradual de costos.

    Fases:
    - Fase 1: 0 bps (exploración libre)
    - Fase 2: Linear 0 -> transition_target
    - Fase 3: Costos reales dinámicos (25-36 bps)
    """

    def __init__(self, config: RewardConfig):
        self.config = config
        self.phase1_end, self.phase2_end = config.phase_boundaries

    def get_phase(self, progress: float) -> TrainingPhase:
        """Obtener fase actual del training."""
        if progress < self.phase1_end:
            return TrainingPhase.EXPLORATION
        elif progress < self.phase2_end:
            return TrainingPhase.TRANSITION
        else:
            return TrainingPhase.REALISTIC

    def get_cost_bps(
        self,
        progress: float,
        volatility_percentile: float = 0.5,
    ) -> float:
        """
        Obtener costo actual en basis points.

        Args:
            progress: Progreso del training (0.0 a 1.0)
            volatility_percentile: Percentil de volatilidad actual

        Returns:
            Costo en basis points
        """
        phase = self.get_phase(progress)

        if phase == TrainingPhase.EXPLORATION:
            return 0.0

        elif phase == TrainingPhase.TRANSITION:
            phase_progress = (progress - self.phase1_end) / (self.phase2_end - self.phase1_end)
            return phase_progress * self.config.transition_target_cost_bps

        else:
            base = self.config.realistic_min_cost_bps
            range_bps = self.config.realistic_max_cost_bps - base
            return base + volatility_percentile * range_bps

    def get_cost_decimal(
        self,
        progress: float,
        volatility_percentile: float = 0.5,
    ) -> float:
        """Obtener costo como decimal (ej: 0.0025 para 25 bps)."""
        return self.get_cost_bps(progress, volatility_percentile) / 10000


# =============================================================================
# SYMMETRY TRACKER
# =============================================================================

class SymmetryTracker:
    """
    Rastrea y fuerza simetría direccional.

    Previene que el modelo desarrolle sesgo LONG o SHORT.
    """

    def __init__(self, config: RewardConfig):
        self.config = config
        self.positions: deque = deque(maxlen=config.symmetry_window)
        self.long_count = 0
        self.short_count = 0
        self.hold_count = 0

    def reset(self):
        """Reset del tracking."""
        self.positions.clear()
        self.long_count = 0
        self.short_count = 0
        self.hold_count = 0

    def add_position(self, position: float):
        """Agregar observación de posición."""
        self.positions.append(position)

        if position > 0.1:
            self.long_count += 1
        elif position < -0.1:
            self.short_count += 1
        else:
            self.hold_count += 1

    def get_directional_bias(self) -> float:
        """
        Calcular sesgo direccional.

        Returns:
            Media de posiciones. 0 = balanceado, +1 = todo LONG, -1 = todo SHORT
        """
        if len(self.positions) < 10:
            return 0.0
        return float(np.mean(list(self.positions)))

    def get_symmetry_penalty(self) -> float:
        """
        Calcular penalización por sesgo direccional.

        Returns:
            Penalización (0 si balanceado, negativo si sesgado)
        """
        bias = abs(self.get_directional_bias())

        if bias <= self.config.max_directional_bias:
            return 0.0

        excess = bias - self.config.max_directional_bias
        penalty = -excess ** 2 * self.config.symmetry_penalty_scale

        return penalty

    def get_exploration_bonus(self) -> float:
        """
        Bonus por usar ambas direcciones LONG y SHORT.

        Returns:
            Bonus (positivo si usa ambas direcciones)
        """
        if len(self.positions) < 20:
            return 0.0

        total = self.long_count + self.short_count + self.hold_count
        if total == 0:
            return 0.0

        pct_long = self.long_count / total
        pct_short = self.short_count / total

        # Máximo cuando es 50/50 LONG/SHORT
        if pct_long > 0.1 and pct_short > 0.1:
            balance = 1.0 - abs(pct_long - pct_short)
            return balance * 0.1

        return 0.0


# =============================================================================
# PATHOLOGICAL BEHAVIOR DETECTOR
# =============================================================================

class PathologicalBehaviorDetector:
    """
    Detecta y penaliza comportamientos patológicos.

    1. Overtrading (demasiados trades por tiempo)
    2. Inactividad (HOLD por demasiado tiempo)
    3. Churning (reversiones frecuentes LONG/SHORT)
    """

    def __init__(self, config: RewardConfig):
        self.config = config
        self.trade_history: deque = deque(maxlen=config.overtrading_lookback)
        self.position_history: deque = deque(maxlen=config.overtrading_lookback)
        self.hold_duration = 0
        self.reversals = 0
        self.last_direction = 0

    def reset(self):
        """Reset de todo el tracking."""
        self.trade_history.clear()
        self.position_history.clear()
        self.hold_duration = 0
        self.reversals = 0
        self.last_direction = 0

    def update(self, position: float, prev_position: float) -> Dict[str, float]:
        """
        Actualizar tracking y retornar penalizaciones.

        Returns:
            Dict con componentes de penalización
        """
        penalties = {
            'overtrading': 0.0,
            'inactivity': 0.0,
            'churning': 0.0,
        }

        # Detectar si hubo trade
        trade_occurred = abs(position - prev_position) > 0.5
        self.trade_history.append(1 if trade_occurred else 0)
        self.position_history.append(position)

        # === OVERTRADING ===
        if len(self.trade_history) >= 60:
            trade_rate = sum(self.trade_history) / len(self.trade_history)
            if trade_rate > self.config.max_trades_per_bar:
                excess = trade_rate - self.config.max_trades_per_bar
                penalties['overtrading'] = -excess * self.config.overtrading_penalty

        # === INACTIVIDAD ===
        if abs(position) < 0.1:  # HOLD
            self.hold_duration += 1
            if self.hold_duration > self.config.max_hold_duration:
                excess = self.hold_duration - self.config.max_hold_duration
                penalties['inactivity'] = -(excess / 12) * self.config.inactivity_penalty
        else:
            self.hold_duration = 0

        # === CHURNING ===
        current_direction = 1 if position > 0.1 else (-1 if position < -0.1 else 0)
        if current_direction != 0 and self.last_direction != 0:
            if current_direction != self.last_direction:
                self.reversals += 1
        if current_direction != 0:
            self.last_direction = current_direction

        if len(self.position_history) >= 60:
            if self.reversals > self.config.reversal_threshold:
                excess = self.reversals - self.config.reversal_threshold
                penalties['churning'] = -(excess / 5) * self.config.churning_penalty

            # Decay del contador de reversiones
            self.reversals = max(0, self.reversals - 0.1)

        return penalties


# =============================================================================
# ONLINE SORTINO CALCULATOR
# =============================================================================

class OnlineSortinoCalculator:
    """
    Calculador online de Sortino ratio.

    Mejor que Sharpe para EM currencies porque solo penaliza
    volatilidad de bajada, no de subida.
    """

    def __init__(self, config: RewardConfig):
        self.config = config
        self.returns: deque = deque(maxlen=config.sortino_window)

    def reset(self):
        """Reset del buffer de returns."""
        self.returns.clear()

    def add_return(self, ret: float):
        """Agregar observación de return."""
        self.returns.append(ret)

    def get_downside_deviation(self) -> float:
        """Calcular downside deviation."""
        if len(self.returns) < 10:
            return 1e-8

        returns = np.array(self.returns)
        downside = returns[returns < self.config.sortino_mar]

        if len(downside) < 3:
            return 1e-8

        return float(np.std(downside))

    def get_sortino_reward(self, current_return: float) -> float:
        """
        Calcular reward basado en Sortino.

        Args:
            current_return: Return del período actual

        Returns:
            Componente de reward basado en Sortino
        """
        self.add_return(current_return)

        if len(self.returns) < 20:
            return 0.0

        mean_return = np.mean(list(self.returns))
        downside_dev = self.get_downside_deviation()

        sortino = (mean_return - self.config.sortino_mar) / (downside_dev + 1e-8)

        return float(np.clip(sortino * 0.5, -2, 2))


# =============================================================================
# MAIN REWARD FUNCTION
# =============================================================================

class SymmetricCurriculumReward:
    """
    Reward function completa con curriculum learning simétrico.

    Resuelve:
    1. Colapso a política constante (LONG=100%)
    2. Asimetría LONG/SHORT/HOLD
    3. Desconexión costos training vs producción

    Features:
    - Curriculum learning (3 fases)
    - Reward shaping simétrico
    - Cost scheduling dinámico
    - Sortino-based risk adjustment
    - Detección de comportamientos patológicos
    """

    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        total_timesteps: int = 500_000,
    ):
        self.config = config or RewardConfig()
        self.total_timesteps = total_timesteps
        self.current_timestep = 0

        # Componentes
        self.cost_scheduler = CurriculumCostScheduler(self.config)
        self.symmetry_tracker = SymmetryTracker(self.config)
        self.pathological_detector = PathologicalBehaviorDetector(self.config)
        self.sortino_calculator = OnlineSortinoCalculator(self.config)

        # Tracking
        self.peak_portfolio = None
        self.component_history: List[Dict[str, float]] = []

        # Pesos por fase
        self.phase_weights = {
            TrainingPhase.EXPLORATION: {
                'direction': 0.50,
                'symmetry': 0.30,
                'exploration': 0.20,
            },
            TrainingPhase.TRANSITION: {
                'direction': 0.30,
                'pnl': 0.30,
                'symmetry': 0.20,
                'hold_penalty': 0.20,
            },
            TrainingPhase.REALISTIC: {
                'sortino': 0.40,
                'pnl': 0.25,
                'efficiency': 0.15,
                'drawdown': 0.20,
            },
        }

    def reset(self, initial_balance: float = 10000):
        """Reset para nuevo episodio."""
        self.symmetry_tracker.reset()
        self.pathological_detector.reset()
        self.sortino_calculator.reset()
        self.peak_portfolio = initial_balance
        self.component_history = []

    def set_timestep(self, timestep: int):
        """Actualizar timestep actual para curriculum."""
        self.current_timestep = timestep

    @property
    def progress(self) -> float:
        """Progreso del training (0 a 1)."""
        return min(self.current_timestep / self.total_timesteps, 1.0)

    @property
    def phase(self) -> TrainingPhase:
        """Fase actual del training."""
        return self.cost_scheduler.get_phase(self.progress)

    def _calculate_direction_reward(
        self,
        position: float,
        market_return: float,
    ) -> float:
        """
        Calcular reward por dirección correcta.

        SIMÉTRICO: mismo reward para LONG correcto que SHORT correcto.
        """
        if abs(position) < 0.1 or abs(market_return) < 1e-6:
            return 0.0

        correct = (position > 0 and market_return > 0) or \
                  (position < 0 and market_return < 0)

        magnitude = min(abs(market_return) * 1000, 1.0)

        if correct:
            return magnitude * 0.5
        else:
            return -magnitude * 0.3

    def _calculate_pnl_reward(
        self,
        portfolio_return: float,
        position: float,
    ) -> float:
        """
        Calcular reward basado en PnL.

        SIMÉTRICO: usa benchmark neutral (no buy-and-hold).
        """
        if abs(position) < 0.1:
            return 0.0

        scaled_return = portfolio_return * 100
        return float(np.clip(scaled_return, -2, 2))

    def _calculate_efficiency_reward(
        self,
        portfolio_return: float,
        trade_occurred: bool,
    ) -> float:
        """
        Reward por eficiencia de trading.

        Penaliza trades que no generan returns proporcionales.
        """
        if not trade_occurred:
            return 0.0

        current_cost = self.cost_scheduler.get_cost_bps(self.progress)
        return_bps = portfolio_return * 10000

        if return_bps > current_cost * 1.5:
            return 0.1
        elif return_bps > current_cost:
            return 0.0
        else:
            return -0.1

    def _calculate_drawdown_penalty(
        self,
        portfolio_value: float,
    ) -> float:
        """Calcular penalización por drawdown."""
        self.peak_portfolio = max(self.peak_portfolio, portfolio_value)
        drawdown = (self.peak_portfolio - portfolio_value) / self.peak_portfolio

        if drawdown < 0.05:
            return 0.0
        elif drawdown < 0.10:
            return -(drawdown - 0.05) * 2
        else:
            return -(drawdown - 0.05) * 5

    def calculate(
        self,
        portfolio_return: float,
        market_return: float,
        portfolio_value: float,
        position: float,
        prev_position: float,
        volatility_percentile: float = 0.5,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calcular reward con curriculum-aware y simetría.

        Args:
            portfolio_return: Return del portfolio este step
            market_return: Return raw del mercado
            portfolio_value: Valor actual del portfolio
            position: Posición actual (-1 a +1)
            prev_position: Posición anterior
            volatility_percentile: Percentil de volatilidad actual

        Returns:
            reward: Reward total
            components: Desglose de componentes del reward
        """
        components = {}
        phase = self.phase
        trade_occurred = abs(position - prev_position) > 0.5

        # Actualizar trackers
        self.symmetry_tracker.add_position(position)
        pathological_penalties = self.pathological_detector.update(position, prev_position)

        # === REWARD ESPECÍFICO POR FASE ===
        weights = self.phase_weights[phase]

        if phase == TrainingPhase.EXPLORATION:
            components['direction'] = self._calculate_direction_reward(
                position, market_return
            ) * weights['direction']

            components['symmetry'] = self.symmetry_tracker.get_symmetry_penalty() * \
                                      weights['symmetry']

            components['exploration'] = self.symmetry_tracker.get_exploration_bonus() * \
                                         weights['exploration']

        elif phase == TrainingPhase.TRANSITION:
            components['direction'] = self._calculate_direction_reward(
                position, market_return
            ) * weights['direction']

            components['pnl'] = self._calculate_pnl_reward(
                portfolio_return, position
            ) * weights['pnl']

            components['symmetry'] = self.symmetry_tracker.get_symmetry_penalty() * \
                                      weights['symmetry']

            if abs(position) < 0.1:
                components['hold_penalty'] = -0.1 * weights['hold_penalty']
            else:
                components['hold_penalty'] = 0.0

        else:  # REALISTIC
            components['sortino'] = self.sortino_calculator.get_sortino_reward(
                portfolio_return
            ) * weights['sortino']

            components['pnl'] = self._calculate_pnl_reward(
                portfolio_return, position
            ) * weights['pnl']

            components['efficiency'] = self._calculate_efficiency_reward(
                portfolio_return, trade_occurred
            ) * weights['efficiency']

            components['drawdown'] = self._calculate_drawdown_penalty(
                portfolio_value
            ) * weights['drawdown']

        # === PENALIZACIONES SIEMPRE ACTIVAS ===
        components['overtrading'] = pathological_penalties['overtrading']
        components['inactivity'] = pathological_penalties['inactivity']
        components['churning'] = pathological_penalties['churning']

        # === AGREGAR ===
        total_reward = sum(components.values())

        # Scale y clip
        total_reward *= self.config.reward_scale
        total_reward = float(np.clip(total_reward, *self.config.clip_range))

        components['total'] = total_reward
        components['phase'] = phase.value
        components['progress'] = self.progress
        components['current_cost_bps'] = self.cost_scheduler.get_cost_bps(
            self.progress, volatility_percentile
        )

        self.component_history.append(components)

        return total_reward, components

    def get_stats(self) -> Dict[str, float]:
        """Obtener estadísticas del reward."""
        if not self.component_history:
            return {}

        stats = {}

        for key in ['total', 'direction', 'pnl', 'symmetry', 'sortino']:
            values = [h.get(key, 0) for h in self.component_history]
            if values:
                stats[f'{key}_mean'] = float(np.mean(values))
                stats[f'{key}_std'] = float(np.std(values))

        stats['directional_bias'] = self.symmetry_tracker.get_directional_bias()
        stats['phase'] = self.phase.value
        stats['progress'] = self.progress

        return stats


# =============================================================================
# REWARD WRAPPER PARA GYMNASIUM
# =============================================================================

class CurriculumRewardWrapper:
    """
    Wrapper para integrar curriculum reward con Gymnasium environment.

    Reemplaza el reward del environment con el curriculum-aware symmetric reward.
    """

    def __init__(
        self,
        env,
        config: Optional[RewardConfig] = None,
        total_timesteps: int = 500_000,
    ):
        self.env = env
        self.reward_fn = SymmetricCurriculumReward(config, total_timesteps)
        self.global_timestep = 0

        # Proxy attributes
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs):
        """Reset environment y reward function."""
        self.reward_fn.reset(getattr(self.env, 'initial_balance', 10000))
        return self.env.reset(**kwargs)

    def step(self, action):
        """Step con curriculum reward."""
        obs, original_reward, terminated, truncated, info = self.env.step(action)

        self.global_timestep += 1
        self.reward_fn.set_timestep(self.global_timestep)

        # Obtener valores para calcular reward
        portfolio_return = info.get('step_return', 0)
        market_return = info.get('market_return', 0)
        portfolio_value = info.get('portfolio', 10000)
        position = info.get('position', 0)
        prev_position = getattr(self, '_prev_position', 0)

        # Obtener volatilidad si está disponible
        volatility_pct = 0.5
        if hasattr(self.env, 'df') and 'atr_pct' in self.env.df.columns:
            idx = getattr(self.env, 'start_idx', 0) + getattr(self.env, 'step_count', 0)
            if idx < len(self.env.df):
                volatility_pct = min(float(self.env.df['atr_pct'].iloc[idx]), 1.0)

        # Calcular curriculum reward
        reward, components = self.reward_fn.calculate(
            portfolio_return=portfolio_return,
            market_return=market_return,
            portfolio_value=portfolio_value,
            position=position,
            prev_position=prev_position,
            volatility_percentile=volatility_pct,
        )

        self._prev_position = position

        # Agregar componentes a info
        info['reward_components'] = components
        info['original_reward'] = original_reward
        info['curriculum_phase'] = self.reward_fn.phase.name
        info['current_cost_bps'] = components.get('current_cost_bps', 0)

        return obs, reward, terminated, truncated, info

    def set_total_timesteps(self, total: int):
        """Actualizar total timesteps para curriculum pacing."""
        self.reward_fn.total_timesteps = total


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("SYMMETRIC CURRICULUM REWARD - USD/COP RL Trading System")
    print("=" * 70)

    config = RewardConfig()
    reward_fn = SymmetricCurriculumReward(config, total_timesteps=500_000)

    print("\n1. CURRICULUM PHASES:")
    print("-" * 50)
    for progress in [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 1.0]:
        reward_fn.set_timestep(int(progress * 500_000))
        phase = reward_fn.phase.name
        cost = reward_fn.cost_scheduler.get_cost_bps(progress)
        print(f"  Progress {progress:.0%}: Phase={phase:12}, Cost={cost:.1f} bps")

    print("\n2. SYMMETRY TEST:")
    print("-" * 50)

    # Simular comportamiento sesgado
    reward_fn.reset(10000)
    for i in range(100):
        reward_fn.symmetry_tracker.add_position(0.8)

    bias = reward_fn.symmetry_tracker.get_directional_bias()
    penalty = reward_fn.symmetry_tracker.get_symmetry_penalty()
    print(f"  After 100 LONG positions: bias={bias:.3f}, penalty={penalty:.3f}")

    # Comportamiento balanceado
    reward_fn.reset(10000)
    for i in range(100):
        pos = 0.8 if i % 2 == 0 else -0.8
        reward_fn.symmetry_tracker.add_position(pos)

    bias = reward_fn.symmetry_tracker.get_directional_bias()
    penalty = reward_fn.symmetry_tracker.get_symmetry_penalty()
    bonus = reward_fn.symmetry_tracker.get_exploration_bonus()
    print(f"  After balanced trading: bias={bias:.3f}, penalty={penalty:.3f}, bonus={bonus:.3f}")

    print("\n3. REWARD COMPARISON BY PHASE:")
    print("-" * 50)

    test_cases = [
        ("LONG correct", 0.8, 0.001),
        ("SHORT correct", -0.8, -0.001),
        ("LONG wrong", 0.8, -0.001),
        ("SHORT wrong", -0.8, 0.001),
        ("HOLD", 0.0, 0.001),
    ]

    for phase_progress in [0.15, 0.45, 0.85]:
        reward_fn.set_timestep(int(phase_progress * 500_000))
        reward_fn.reset(10000)

        for _ in range(30):
            reward_fn.symmetry_tracker.add_position(0.0)

        print(f"\n  Phase: {reward_fn.phase.name} (progress={phase_progress:.0%})")

        for name, position, market_ret in test_cases:
            reward, components = reward_fn.calculate(
                portfolio_return=position * market_ret,
                market_return=market_ret,
                portfolio_value=10000,
                position=position,
                prev_position=0.0,
            )
            print(f"    {name:15}: reward={reward:+.3f}")

    print("\n" + "=" * 70)
    print("SymmetricCurriculumReward ready for integration")
    print("=" * 70)
