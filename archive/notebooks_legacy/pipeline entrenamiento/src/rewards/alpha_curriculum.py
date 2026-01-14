"""
USD/COP RL Trading System - Alpha Curriculum Reward
====================================================

Reward function basada en ALPHA (skill sobre mercado).

DIFERENCIAS VS SymmetricCurriculumReward:
1. NO penaliza HOLD - a veces no operar es correcto
2. Reward basada en ALPHA, no actividad
3. Fases más simples: direction → transition → alpha
4. Bonus por consistencia (mantener posición ganadora)

Fases:
- Fase 1 (0-30%):  Sin costos, aprende dirección
- Fase 2 (30-70%): Costos 0→60%, mezcla dirección+alpha
- Fase 3 (70-100%): Costos 100%, alpha puro risk-adjusted

Author: Claude Code
Version: 1.0.0
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class CurriculumPhase:
    """Configuración de cada fase del curriculum."""
    name: str
    cost_multiplier: float      # 0.0 a 1.0
    alpha_weight: float         # Peso del alpha en reward
    direction_weight: float     # Peso de acertar dirección
    consistency_bonus: float    # Bonus por mantener posición


class AlphaCurriculumReward:
    """
    Reward function con curriculum de 3 fases basada en Alpha.

    FILOSOFÍA:
    - El modelo debe generar ALPHA (retorno sobre el mercado)
    - NO penalizar HOLD - a veces la mejor decisión es no operar
    - Costos graduales para aprender timing primero
    - Simetría natural por diseño (alpha es simétrico)

    Args:
        final_cost: Costo final en decimal (0.0025 = 25 bps)
        total_steps: Total de steps de training
        phase_boundaries: Límites de las fases (0.3, 0.7)
        drawdown_threshold: Umbral de DD para penalización
        verbose: Nivel de verbosidad
    """

    PHASES = [
        CurriculumPhase(
            name="direction",
            cost_multiplier=0.0,
            alpha_weight=0.3,
            direction_weight=0.7,
            consistency_bonus=0.02
        ),
        CurriculumPhase(
            name="transition",
            cost_multiplier=0.6,
            alpha_weight=0.6,
            direction_weight=0.3,
            consistency_bonus=0.01
        ),
        CurriculumPhase(
            name="alpha",
            cost_multiplier=1.0,
            alpha_weight=0.9,
            direction_weight=0.1,
            consistency_bonus=0.005
        ),
    ]

    def __init__(
        self,
        final_cost: float = 0.0025,
        total_steps: int = 200_000,
        phase_boundaries: Tuple[float, float] = (0.3, 0.7),
        drawdown_threshold: float = 0.10,
        verbose: int = 0,
    ):
        self.final_cost = final_cost
        self.total_steps = total_steps
        self.phase_boundaries = phase_boundaries
        self.drawdown_threshold = drawdown_threshold
        self.verbose = verbose

        self.current_step = 0
        self.returns_buffer = deque(maxlen=60)
        self.peak_portfolio = None
        self.initial_balance = None

        # Tracking para diagnóstico
        self.long_correct = 0
        self.short_correct = 0
        self.total_long = 0
        self.total_short = 0
        self.total_hold = 0

        self._phase_logged = set()

    def reset(self, initial_balance: float = 10000):
        """Reset para nuevo episodio."""
        self.returns_buffer.clear()
        self.peak_portfolio = initial_balance
        self.initial_balance = initial_balance

    def get_current_phase(self) -> CurriculumPhase:
        """Determina la fase actual del curriculum."""
        progress = self.current_step / max(1, self.total_steps)

        if progress < self.phase_boundaries[0]:
            return self.PHASES[0]
        elif progress < self.phase_boundaries[1]:
            return self.PHASES[1]
        else:
            return self.PHASES[2]

    def get_current_cost(self) -> float:
        """Costo actual basado en fase del curriculum."""
        phase = self.get_current_phase()
        return self.final_cost * phase.cost_multiplier

    def calculate(
        self,
        portfolio_return: float,
        market_return: float,
        portfolio_value: float,
        position: float,
        prev_position: float,
        volatility_percentile: float = 0.5,
    ) -> Tuple[float, Dict]:
        """
        Calcular reward basada en alpha.

        Args:
            portfolio_return: Retorno del portfolio este step
            market_return: Retorno del mercado
            portfolio_value: Valor actual del portfolio
            position: Posición actual [-1, 0, +1]
            prev_position: Posición anterior
            volatility_percentile: Percentil de volatilidad [0, 1]

        Returns:
            Tuple de (reward, dict con componentes)
        """
        self.current_step += 1
        phase = self.get_current_phase()

        # Log cambio de fase
        if phase.name not in self._phase_logged and self.verbose > 0:
            print(f"\n[AlphaCurriculum] Entering {phase.name} phase at step {self.current_step:,}")
            self._phase_logged.add(phase.name)

        # === 1. ALPHA COMPONENT ===
        # Alpha = skill sobre el mercado (verdadero alpha)
        # Si mercado +1% y portfolio +1.5% → alpha = +0.5% (bueno)
        # Si mercado -1% y portfolio -0.5% → alpha = +0.5% (bueno, perdimos menos)
        alpha = portfolio_return - market_return
        # FIX: Scale alpha appropriately (was 100x, now 1000x to compete with costs)
        alpha_reward = alpha * 1000  # 1 bp alpha = 0.1 reward

        # === 2. DIRECTION COMPONENT ===
        direction_reward = 0.0

        if abs(position) > 0.1:  # Tengo posición
            direction_correct = (
                (position > 0 and market_return > 0) or
                (position < 0 and market_return < 0)
            )

            if direction_correct:
                # Bonus proporcional al movimiento del mercado
                direction_reward = 0.05 + abs(market_return) * 10
                if position > 0:
                    self.long_correct += 1
                else:
                    self.short_correct += 1
            else:
                # Penalización menor
                direction_reward = -0.025

            # Tracking
            if position > 0:
                self.total_long += 1
            else:
                self.total_short += 1
        else:
            # HOLD - NO PENALIZAR pero tampoco premiar
            self.total_hold += 1
            # FIX: Remove HOLD bonus - it was incentivizing inactivity
            # The agent should be neutral about HOLD, not rewarded for it
            direction_reward = 0.0

        # === 3. COST COMPONENT ===
        position_change = abs(position - prev_position)
        cost = self.get_current_cost()

        # Costo ajustado por volatilidad
        if volatility_percentile > 0.8:
            cost *= 1.5  # 50% más en alta volatilidad

        # FIX: Reduce cost scaling from 100x to 10x
        # Before: 0.0025 * 100 = 0.25 penalty (too harsh)
        # After:  0.0025 * 10 = 0.025 penalty (balanced with alpha)
        cost_penalty = -position_change * cost * 10

        # === 4. CONSISTENCY BONUS ===
        consistency = 0.0
        if position_change < 0.1 and abs(position) > 0.1:
            # Mantener posición ganadora
            if portfolio_return > 0:
                consistency = phase.consistency_bonus * 2
            else:
                consistency = phase.consistency_bonus

        # === 5. COMBINAR SEGÚN FASE ===
        reward = (
            phase.alpha_weight * alpha_reward +
            phase.direction_weight * direction_reward +
            cost_penalty +
            consistency
        )

        # === 6. DRAWDOWN PENALTY (solo en fase alpha) ===
        if phase.name == "alpha" and self.peak_portfolio is not None:
            self.peak_portfolio = max(self.peak_portfolio, portfolio_value)
            drawdown = (self.peak_portfolio - portfolio_value) / self.peak_portfolio

            if drawdown > self.drawdown_threshold:
                reward -= (drawdown - self.drawdown_threshold) * 2

        # Clip final
        reward = float(np.clip(reward, -2, 2))

        # Componentes para logging
        components = {
            'alpha': alpha_reward,
            'direction': direction_reward,
            'cost_penalty': cost_penalty,
            'consistency': consistency,
            'phase': phase.name,
            'total_reward': reward,
        }

        return reward, components

    def get_diagnostics(self) -> Dict:
        """Retorna métricas para diagnóstico."""
        phase = self.get_current_phase()

        return {
            'phase': phase.name,
            'current_cost_bps': self.get_current_cost() * 10000,
            'progress': self.current_step / max(1, self.total_steps),
            'long_accuracy': self.long_correct / max(1, self.total_long),
            'short_accuracy': self.short_correct / max(1, self.total_short),
            'long_short_ratio': self.total_long / max(1, self.total_short),
            'hold_pct': self.total_hold / max(1, self.total_long + self.total_short + self.total_hold),
        }

    def get_symmetry_stats(self) -> Dict:
        """Estadísticas de simetría LONG/SHORT."""
        total = self.total_long + self.total_short + self.total_hold
        if total == 0:
            return {'long_pct': 0, 'short_pct': 0, 'hold_pct': 0, 'is_balanced': True}

        long_pct = self.total_long / total * 100
        short_pct = self.total_short / total * 100
        hold_pct = self.total_hold / total * 100

        # Consideramos balanceado si LONG y SHORT están entre 20-60%
        is_balanced = (20 <= long_pct <= 60) and (20 <= short_pct <= 60)

        return {
            'long_pct': long_pct,
            'short_pct': short_pct,
            'hold_pct': hold_pct,
            'is_balanced': is_balanced,
        }


class AlphaCurriculumRewardV2(AlphaCurriculumReward):
    """
    Versión 2 con ajustes adicionales para 15 minutos.

    Cambios vs V1:
    - Menos sensible a ruido de mercado
    - Mayor peso a alpha en todas las fases
    - Bonus por ganar en tendencia
    """

    PHASES = [
        CurriculumPhase(
            name="direction",
            cost_multiplier=0.0,
            alpha_weight=0.4,
            direction_weight=0.6,
            consistency_bonus=0.03
        ),
        CurriculumPhase(
            name="transition",
            cost_multiplier=0.5,
            alpha_weight=0.7,
            direction_weight=0.2,
            consistency_bonus=0.02
        ),
        CurriculumPhase(
            name="alpha",
            cost_multiplier=1.0,
            alpha_weight=0.95,
            direction_weight=0.05,
            consistency_bonus=0.01
        ),
    ]

    def __init__(self, *args, trend_bonus: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.trend_bonus = trend_bonus
        self.returns_window = deque(maxlen=20)  # 5 horas en 15min

    def calculate(
        self,
        portfolio_return: float,
        market_return: float,
        portfolio_value: float,
        position: float,
        prev_position: float,
        volatility_percentile: float = 0.5,
    ) -> Tuple[float, Dict]:
        """Calcular con ajustes para 15min."""
        # Tracking de retornos para detectar tendencia
        self.returns_window.append(market_return)

        # Detectar tendencia
        if len(self.returns_window) >= 10:
            recent_returns = list(self.returns_window)[-10:]
            trend_direction = np.sign(sum(recent_returns))
            trend_strength = abs(sum(recent_returns)) / (np.std(recent_returns) + 1e-6)
        else:
            trend_direction = 0
            trend_strength = 0

        # Calcular reward base
        reward, components = super().calculate(
            portfolio_return=portfolio_return,
            market_return=market_return,
            portfolio_value=portfolio_value,
            position=position,
            prev_position=prev_position,
            volatility_percentile=volatility_percentile,
        )

        # Bonus por seguir tendencia fuerte
        if trend_strength > 2 and abs(position) > 0.1:
            if np.sign(position) == trend_direction:
                trend_bonus = self.trend_bonus * min(trend_strength / 5, 1)
                reward += trend_bonus
                components['trend_bonus'] = trend_bonus

        return reward, components
