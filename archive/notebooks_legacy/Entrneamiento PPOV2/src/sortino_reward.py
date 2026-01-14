"""
Sortino-Based Reward Function
=============================

Reward function que usa Sortino ratio en lugar de Sharpe.

PROBLEMA QUE RESUELVE:
- Sharpe penaliza TODA la varianza (incluyendo ganancias grandes)
- Esto desincentiva movimientos positivos grandes
- Didact AI logró Sortino 4.64 vs Sharpe 0.83

SOLUCIÓN:
- Sortino solo penaliza varianza NEGATIVA (downside)
- Premia: GANAR GRANDE
- Penaliza: PERDER (cualquier cantidad)
- Neutral sobre ganancias volátiles

BENEFICIO:
- Modelo aprende a capturar movimientos grandes favorables
- Mientras mantiene pérdidas controladas
- Más alineado con objetivo real de trading

Author: Claude Code
Version: 1.0.0
"""

import numpy as np
from typing import Tuple, Dict, Optional
from collections import deque
from dataclasses import dataclass


@dataclass
class SortinoConfig:
    """Configuración para Sortino reward."""
    window_size: int = 60  # Ventana rolling
    mar: float = 0.0  # Minimum Acceptable Return
    annualization: float = np.sqrt(252)  # Factor de anualización
    sortino_weight: float = 0.4
    pnl_weight: float = 0.4
    cost_weight: float = 0.2
    reward_scale: float = 100.0
    clip_range: Tuple[float, float] = (-5.0, 5.0)


class SortinoCalculator:
    """
    Calcula Sortino ratio para usar en reward function.

    Sortino = (Return - MAR) / Downside_Deviation

    Donde:
    - MAR = Minimum Acceptable Return (típicamente 0)
    - Downside_Deviation = std de retornos NEGATIVOS solamente

    Esto significa:
    - Retornos positivos grandes NO penalizan
    - Solo retornos negativos cuentan para la volatilidad
    """

    def __init__(
        self,
        window_size: int = 60,
        mar: float = 0.0,
        annualization: float = np.sqrt(252),
    ):
        self.window_size = window_size
        self.mar = mar
        self.annualization = annualization
        self.returns: deque = deque(maxlen=window_size)

    def reset(self):
        """Reset del buffer."""
        self.returns.clear()

    def update(self, portfolio_return: float) -> None:
        """Añadir nuevo retorno al buffer."""
        self.returns.append(portfolio_return)

    def get_downside_deviation(self) -> float:
        """
        Calcular downside deviation.

        Solo considera retornos por debajo del MAR.
        """
        if len(self.returns) < 10:
            return 1e-8  # Evitar división por cero

        returns = np.array(self.returns)
        downside = returns[returns < self.mar]

        if len(downside) < 3:
            return 1e-8

        return float(np.std(downside))

    def get_sortino(self) -> float:
        """
        Calcular Sortino ratio.

        Returns:
            Sortino ratio (clipped para estabilidad)
        """
        if len(self.returns) < 10:
            return 0.0

        returns = np.array(self.returns)
        mean_return = returns.mean()
        downside_dev = self.get_downside_deviation()

        if downside_dev < 1e-10:
            # No hay downside → Sortino muy alto si positive, 0 si negative
            return 3.0 if mean_return > 0 else 0.0

        sortino = (mean_return - self.mar) / downside_dev

        # Anualizar
        sortino *= self.annualization

        # Clip para estabilidad
        return float(np.clip(sortino, -3.0, 3.0))

    def get_reward_component(self) -> float:
        """
        Obtener componente de reward basado en Sortino.

        Escala Sortino a rango apropiado para reward.

        Returns:
            Reward component en [-1, 1]
        """
        sortino = self.get_sortino()
        # Convertir a reward (0 = Sortino 0, ±1 = Sortino ±3)
        return sortino / 3.0


class SortinoRewardFunction:
    """
    Reward function que usa Sortino en lugar de Sharpe.

    Componentes:
    1. Sortino component: Rolling Sortino ratio
    2. PnL component: Return directo del portfolio
    3. Cost component: Penalización por costos de transacción

    IMPORTANTE: Esta función puede usarse como drop-in replacement
    para otras reward functions. Implementa la misma interface.
    """

    def __init__(self, config: Optional[SortinoConfig] = None):
        self.config = config or SortinoConfig()

        # Calculador Sortino
        self.sortino_calc = SortinoCalculator(
            window_size=self.config.window_size,
            mar=self.config.mar,
            annualization=self.config.annualization,
        )

        # Tracking
        self.cumulative_cost = 0.0
        self.component_history = []
        self.peak_portfolio = None

    def reset(self, initial_balance: float = 10000):
        """Reset para nuevo episodio."""
        self.sortino_calc.reset()
        self.cumulative_cost = 0.0
        self.component_history = []
        self.peak_portfolio = initial_balance

    def calculate(
        self,
        portfolio_return: float,
        market_return: float,
        portfolio_value: float,
        position: float,
        prev_position: float,
        volatility_percentile: float = 0.5,
        transaction_cost: float = 0.0,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calcular reward.

        Args:
            portfolio_return: Return del portfolio este step
            market_return: Return raw del mercado (no usado directamente)
            portfolio_value: Valor actual del portfolio
            position: Posición actual
            prev_position: Posición anterior
            volatility_percentile: Percentil de volatilidad
            transaction_cost: Costo de transacción aplicado

        Returns:
            (reward, components_dict)
        """
        components = {}

        # 1. Actualizar Sortino
        self.sortino_calc.update(portfolio_return)
        self.cumulative_cost += transaction_cost

        # 2. Componente Sortino
        sortino_reward = self.sortino_calc.get_reward_component()
        components['sortino'] = sortino_reward * self.config.sortino_weight

        # 3. Componente PnL directo
        # Escalar retorno para que sea significativo
        pnl_reward = portfolio_return * 100
        pnl_reward = np.clip(pnl_reward, -1, 1)
        components['pnl'] = pnl_reward * self.config.pnl_weight

        # 4. Penalización por costos
        # Solo penalizar si hubo transacción
        if transaction_cost > 0:
            cost_penalty = -transaction_cost * 100  # Escalar
            cost_penalty = np.clip(cost_penalty, -1, 0)
            components['cost'] = cost_penalty * self.config.cost_weight
        else:
            components['cost'] = 0.0

        # 5. Drawdown penalty (adicional, bajo peso)
        if self.peak_portfolio is not None:
            self.peak_portfolio = max(self.peak_portfolio, portfolio_value)
            drawdown = (self.peak_portfolio - portfolio_value) / self.peak_portfolio

            if drawdown > 0.05:
                dd_penalty = -(drawdown - 0.05) * 2
                components['drawdown'] = dd_penalty * 0.1
            else:
                components['drawdown'] = 0.0
        else:
            self.peak_portfolio = portfolio_value
            components['drawdown'] = 0.0

        # Total
        total = sum(components.values())

        # Scale y clip
        total *= self.config.reward_scale
        total = float(np.clip(total, *self.config.clip_range))

        components['total'] = total
        components['sortino_raw'] = self.sortino_calc.get_sortino()

        self.component_history.append(components)

        return total, components

    def get_stats(self) -> Dict[str, float]:
        """Obtener estadísticas del reward function."""
        if not self.component_history:
            return {}

        stats = {}

        for key in ['total', 'sortino', 'pnl', 'cost', 'drawdown']:
            values = [h.get(key, 0) for h in self.component_history]
            if values:
                stats[f'{key}_mean'] = float(np.mean(values))
                stats[f'{key}_std'] = float(np.std(values))

        stats['sortino_current'] = self.sortino_calc.get_sortino()
        stats['cumulative_cost'] = self.cumulative_cost

        return stats


class HybridSharpesortinoReward:
    """
    Reward híbrido que combina Sharpe y Sortino.

    FASE 1 (exploration): Usa Sharpe (penaliza toda varianza)
    FASE 2 (transition): Blend de ambos
    FASE 3 (realistic): Usa Sortino (solo penaliza downside)

    Esto permite:
    - Early training: Reducir varianza general primero
    - Late training: Permitir upside volatility
    """

    def __init__(
        self,
        window_size: int = 60,
        phase_boundaries: Tuple[float, float] = (0.3, 0.6),
        total_timesteps: int = 500_000,
    ):
        self.window_size = window_size
        self.phase1_end, self.phase2_end = phase_boundaries
        self.total_timesteps = total_timesteps
        self.current_timestep = 0

        # Calculadores
        self.returns: deque = deque(maxlen=window_size)
        self.peak_portfolio = None

    def reset(self, initial_balance: float = 10000):
        """Reset para nuevo episodio."""
        self.returns.clear()
        self.peak_portfolio = initial_balance

    def set_timestep(self, timestep: int):
        """Actualizar timestep para curriculum."""
        self.current_timestep = timestep

    @property
    def progress(self) -> float:
        return min(self.current_timestep / self.total_timesteps, 1.0)

    def _get_sharpe(self) -> float:
        """Calcular Sharpe (penaliza toda varianza)."""
        if len(self.returns) < 10:
            return 0.0

        returns = np.array(self.returns)
        if returns.std() < 1e-10:
            return 0.0

        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        return float(np.clip(sharpe, -3, 3))

    def _get_sortino(self) -> float:
        """Calcular Sortino (solo penaliza downside)."""
        if len(self.returns) < 10:
            return 0.0

        returns = np.array(self.returns)
        mean_return = returns.mean()
        downside = returns[returns < 0]

        if len(downside) < 3:
            return 3.0 if mean_return > 0 else 0.0

        downside_std = downside.std()
        if downside_std < 1e-10:
            return 3.0 if mean_return > 0 else 0.0

        sortino = mean_return / downside_std * np.sqrt(252)
        return float(np.clip(sortino, -3, 3))

    def calculate(
        self,
        portfolio_return: float,
        market_return: float,
        portfolio_value: float,
        position: float,
        prev_position: float,
        volatility_percentile: float = 0.5,
    ) -> Tuple[float, Dict[str, float]]:
        """Calcular reward híbrido."""
        self.returns.append(portfolio_return)

        components = {}

        # Calcular métricas base
        sharpe = self._get_sharpe()
        sortino = self._get_sortino()

        # Determinar blend según fase
        progress = self.progress

        if progress < self.phase1_end:
            # FASE 1: Pure Sharpe
            ratio_reward = sharpe / 3.0
            components['sharpe'] = sharpe
            components['sortino'] = sortino
            components['blend'] = 'sharpe'
        elif progress < self.phase2_end:
            # FASE 2: Blend (linear transition)
            phase_progress = (progress - self.phase1_end) / (self.phase2_end - self.phase1_end)
            sharpe_weight = 1.0 - phase_progress
            sortino_weight = phase_progress
            ratio_reward = (sharpe * sharpe_weight + sortino * sortino_weight) / 3.0
            components['sharpe'] = sharpe
            components['sortino'] = sortino
            components['blend'] = f'hybrid_{phase_progress:.2f}'
        else:
            # FASE 3: Pure Sortino
            ratio_reward = sortino / 3.0
            components['sharpe'] = sharpe
            components['sortino'] = sortino
            components['blend'] = 'sortino'

        # PnL component
        pnl_reward = np.clip(portfolio_return * 100, -1, 1) * 0.4

        # Total
        total = ratio_reward * 0.6 + pnl_reward

        # Scale
        total *= 100
        total = float(np.clip(total, -5, 5))

        components['ratio_reward'] = ratio_reward
        components['pnl_reward'] = pnl_reward
        components['total'] = total
        components['progress'] = progress

        return total, components


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("SORTINO REWARD - USD/COP RL Trading System")
    print("=" * 70)

    # Test SortinoCalculator
    print("\n1. SortinoCalculator Test:")
    print("-" * 50)

    calc = SortinoCalculator(window_size=60)

    # Simular returns con más upside que downside
    np.random.seed(42)
    returns_with_upside = np.random.normal(0.001, 0.02, 50)  # Media positiva
    returns_with_upside[10:15] = 0.05  # Algunos gains grandes

    for ret in returns_with_upside:
        calc.update(ret)

    print(f"  Returns simulated: {len(returns_with_upside)}")
    print(f"  Mean return: {np.mean(returns_with_upside)*100:.4f}%")
    print(f"  Std return: {np.std(returns_with_upside)*100:.4f}%")
    print(f"  Downside std: {calc.get_downside_deviation()*100:.4f}%")
    print(f"  Sortino ratio: {calc.get_sortino():.3f}")

    # Comparar con Sharpe
    sharpe = np.mean(returns_with_upside) / np.std(returns_with_upside) * np.sqrt(252)
    print(f"  Sharpe ratio (comparison): {sharpe:.3f}")
    print(f"  Sortino > Sharpe: {calc.get_sortino() > sharpe}")

    # Test SortinoRewardFunction
    print("\n2. SortinoRewardFunction Test:")
    print("-" * 50)

    reward_fn = SortinoRewardFunction()
    reward_fn.reset(10000)

    test_scenarios = [
        ("Big win", 0.02, 0.0),
        ("Small win", 0.001, 0.0),
        ("Small loss", -0.001, 0.0),
        ("Big loss", -0.02, 0.0),
        ("Win with cost", 0.01, 0.002),
    ]

    for name, port_ret, cost in test_scenarios:
        reward, components = reward_fn.calculate(
            portfolio_return=port_ret,
            market_return=port_ret,
            portfolio_value=10000,
            position=0.5,
            prev_position=0.5,
            transaction_cost=cost,
        )
        print(f"\n  {name}:")
        print(f"    Portfolio return: {port_ret*100:.2f}%")
        print(f"    Reward: {reward:+.3f}")
        print(f"    Components: sortino={components.get('sortino', 0):.3f}, "
              f"pnl={components.get('pnl', 0):.3f}, cost={components.get('cost', 0):.3f}")

    print("\n" + "=" * 70)
    print("SortinoRewardFunction ready for integration")
    print("=" * 70)
