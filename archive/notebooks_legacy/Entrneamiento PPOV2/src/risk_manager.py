"""
Risk Manager con Kill Switches
==============================

Sistema de gestión de riesgo en tiempo real con kill switches automáticos.

PROBLEMA QUE RESUELVE:
- No hay protección automática contra drawdowns extremos en producción
- No hay alertas cuando el modelo se comporta de forma anómala
- No hay mecanismo para pausar trading automáticamente

SOLUCIÓN:
- Kill switches basados en métricas en tiempo real
- Alertas configurables
- Reducción automática de sizing en condiciones adversas
- Pausa automática en situaciones de crisis

KILL SWITCHES:
1. Max Drawdown Warning (3%) → Alerta
2. Max Drawdown Reduce (5%) → Reducir sizing 50%
3. Max Drawdown Stop (10%) → Pausar trading
4. Sharpe Rolling Negativo → Revisión
5. HOLD% Excesivo → Alerta
6. Pérdidas Consecutivas → Alerta

Author: Claude Code
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class RiskStatus(Enum):
    """Estados del risk manager."""
    ACTIVE = "ACTIVE"       # Trading normal
    REDUCED = "REDUCED"     # Sizing reducido
    PAUSED = "PAUSED"       # Trading pausado
    MANUAL_REVIEW = "MANUAL_REVIEW"  # Requiere revisión manual


@dataclass
class RiskLimits:
    """Límites de riesgo configurables."""
    # Drawdown limits
    max_drawdown_warning: float = 0.03   # 3% → Warning
    max_drawdown_reduce: float = 0.05    # 5% → Reducir sizing 50%
    max_drawdown_stop: float = 0.10      # 10% → Pausar trading

    # Performance limits
    min_sharpe_30d: float = 0.0          # Sharpe rolling < 0 → Review
    max_hold_pct: float = 0.95           # 95% HOLD por 5 días → Alert

    # Loss limits
    max_consecutive_losses: int = 10     # 10 pérdidas seguidas → Alert
    max_daily_loss: float = 0.02         # 2% pérdida diaria → Warning

    # Recovery
    auto_resume_after_recovery: bool = True  # Reanudar si métricas mejoran
    recovery_sharpe_threshold: float = 0.5   # Sharpe necesario para reanudar


@dataclass
class Alert:
    """Estructura de alerta."""
    timestamp: datetime
    level: str  # 'WARNING', 'CRITICAL', 'INFO'
    message: str
    metric: str
    value: float
    threshold: float


class RiskManager:
    """
    Gestor de riesgo con kill switches automáticos.

    Monitorea métricas en tiempo real y aplica restricciones automáticas.

    Args:
        limits: Configuración de límites
        rolling_window: Ventana en días para métricas rolling
        bars_per_day: Número de barras por día
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        rolling_window: int = 30,
        bars_per_day: int = 56,
    ):
        self.limits = limits or RiskLimits()
        self.rolling_window = rolling_window
        self.bars_per_day = bars_per_day

        # Buffers de historial
        buffer_size = rolling_window * bars_per_day
        self.returns_history: deque = deque(maxlen=buffer_size)
        self.actions_history: deque = deque(maxlen=buffer_size)
        self.portfolio_history: deque = deque(maxlen=buffer_size)

        # Estado
        self.peak_value = 10000.0
        self.current_value = 10000.0
        self.consecutive_losses = 0
        self.high_hold_days = 0
        self.daily_returns: List[float] = []

        self.status = RiskStatus.ACTIVE
        self.alerts: List[Alert] = []
        self.status_history: List[Tuple[datetime, RiskStatus]] = []

        # Tracking de días
        self.current_day_returns = 0.0
        self.bars_today = 0

    def reset(self, initial_value: float = 10000.0):
        """Reset del risk manager."""
        self.returns_history.clear()
        self.actions_history.clear()
        self.portfolio_history.clear()

        self.peak_value = initial_value
        self.current_value = initial_value
        self.consecutive_losses = 0
        self.high_hold_days = 0
        self.daily_returns = []

        self.status = RiskStatus.ACTIVE
        self.alerts = []
        self.status_history = []

        self.current_day_returns = 0.0
        self.bars_today = 0

    def update(
        self,
        portfolio_value: float,
        step_return: float,
        action: float,
    ) -> Tuple[RiskStatus, float, List[Alert]]:
        """
        Actualizar estado y verificar límites.

        Args:
            portfolio_value: Valor actual del portfolio
            step_return: Retorno de este step
            action: Acción tomada

        Returns:
            (status, position_multiplier, new_alerts)
        """
        now = datetime.now()
        new_alerts = []

        # Actualizar historial
        self.current_value = portfolio_value
        self.returns_history.append(step_return)
        self.actions_history.append(action)
        self.portfolio_history.append(portfolio_value)

        # Actualizar peak
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value

        # Calcular drawdown
        drawdown = (self.peak_value - portfolio_value) / self.peak_value

        # Tracking diario
        self.current_day_returns += step_return
        self.bars_today += 1

        if self.bars_today >= self.bars_per_day:
            self.daily_returns.append(self.current_day_returns)
            self.current_day_returns = 0.0
            self.bars_today = 0

        # Tracking de losses
        if step_return < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Determinar multiplier y status
        position_multiplier = 1.0

        # === DRAWDOWN CHECKS ===
        if drawdown > self.limits.max_drawdown_stop:
            self.status = RiskStatus.PAUSED
            position_multiplier = 0.0
            new_alerts.append(Alert(
                timestamp=now,
                level='CRITICAL',
                message=f"STOP: Drawdown {drawdown:.1%} > {self.limits.max_drawdown_stop:.1%}",
                metric='drawdown',
                value=drawdown,
                threshold=self.limits.max_drawdown_stop,
            ))

        elif drawdown > self.limits.max_drawdown_reduce:
            self.status = RiskStatus.REDUCED
            position_multiplier = 0.5
            new_alerts.append(Alert(
                timestamp=now,
                level='CRITICAL',
                message=f"REDUCE: Drawdown {drawdown:.1%} > {self.limits.max_drawdown_reduce:.1%}",
                metric='drawdown',
                value=drawdown,
                threshold=self.limits.max_drawdown_reduce,
            ))

        elif drawdown > self.limits.max_drawdown_warning:
            new_alerts.append(Alert(
                timestamp=now,
                level='WARNING',
                message=f"WARNING: Drawdown {drawdown:.1%}",
                metric='drawdown',
                value=drawdown,
                threshold=self.limits.max_drawdown_warning,
            ))
            # Keep current status unless PAUSED
            if self.status != RiskStatus.PAUSED:
                self.status = RiskStatus.ACTIVE

        else:
            # Recovery check
            if self.status == RiskStatus.REDUCED and self.limits.auto_resume_after_recovery:
                sharpe = self._calculate_rolling_sharpe()
                if sharpe >= self.limits.recovery_sharpe_threshold:
                    self.status = RiskStatus.ACTIVE
                    new_alerts.append(Alert(
                        timestamp=now,
                        level='INFO',
                        message=f"RECOVERED: Sharpe {sharpe:.2f} >= {self.limits.recovery_sharpe_threshold:.2f}",
                        metric='sharpe_recovery',
                        value=sharpe,
                        threshold=self.limits.recovery_sharpe_threshold,
                    ))

        # === SHARPE CHECK ===
        sharpe = self._calculate_rolling_sharpe()
        if sharpe < self.limits.min_sharpe_30d:
            new_alerts.append(Alert(
                timestamp=now,
                level='WARNING',
                message=f"LOW SHARPE: {sharpe:.2f} (30d rolling)",
                metric='sharpe_30d',
                value=sharpe,
                threshold=self.limits.min_sharpe_30d,
            ))

        # === HOLD CHECK ===
        hold_pct = self._calculate_hold_percentage()
        if hold_pct > self.limits.max_hold_pct:
            self.high_hold_days += 1 / self.bars_per_day  # Fracción de día
            if self.high_hold_days >= 5:
                new_alerts.append(Alert(
                    timestamp=now,
                    level='WARNING',
                    message=f"HIGH HOLD: {hold_pct:.1%} for {self.high_hold_days:.1f} days",
                    metric='hold_pct',
                    value=hold_pct,
                    threshold=self.limits.max_hold_pct,
                ))
        else:
            self.high_hold_days = 0

        # === CONSECUTIVE LOSSES ===
        if self.consecutive_losses >= self.limits.max_consecutive_losses:
            new_alerts.append(Alert(
                timestamp=now,
                level='WARNING',
                message=f"LOSING STREAK: {self.consecutive_losses} consecutive losses",
                metric='consecutive_losses',
                value=float(self.consecutive_losses),
                threshold=float(self.limits.max_consecutive_losses),
            ))

        # === DAILY LOSS CHECK ===
        if self.current_day_returns < -self.limits.max_daily_loss:
            new_alerts.append(Alert(
                timestamp=now,
                level='WARNING',
                message=f"DAILY LOSS: {self.current_day_returns:.2%} < -{self.limits.max_daily_loss:.2%}",
                metric='daily_loss',
                value=self.current_day_returns,
                threshold=-self.limits.max_daily_loss,
            ))

        # Guardar alertas y status
        self.alerts.extend(new_alerts)
        if new_alerts:
            self.status_history.append((now, self.status))

        return self.status, position_multiplier, new_alerts

    def _calculate_rolling_sharpe(self) -> float:
        """Calcular Sharpe rolling de 30 días."""
        if len(self.returns_history) < self.bars_per_day * 5:
            return 0.5  # Default neutral

        returns = np.array(self.returns_history)
        n_bars = len(returns)
        n_days = n_bars // self.bars_per_day

        if n_days < 2:
            return 0.5

        # Agregar a retornos diarios
        daily = returns[:n_days * self.bars_per_day].reshape(n_days, self.bars_per_day).sum(axis=1)

        if daily.std() < 1e-10:
            return 0.0

        return float(daily.mean() / daily.std() * np.sqrt(252))

    def _calculate_hold_percentage(self) -> float:
        """Calcular porcentaje de HOLD en acciones recientes."""
        if len(self.actions_history) < 100:
            return 0.5

        actions = np.array(self.actions_history)
        hold_pct = (np.abs(actions) < 0.1).mean()

        return float(hold_pct)

    def get_current_drawdown(self) -> float:
        """Obtener drawdown actual."""
        return (self.peak_value - self.current_value) / self.peak_value

    def get_report(self) -> Dict:
        """Obtener reporte completo de estado."""
        return {
            'status': self.status.value,
            'current_value': self.current_value,
            'peak_value': self.peak_value,
            'drawdown': self.get_current_drawdown(),
            'rolling_sharpe': self._calculate_rolling_sharpe(),
            'hold_pct': self._calculate_hold_percentage(),
            'consecutive_losses': self.consecutive_losses,
            'high_hold_days': self.high_hold_days,
            'total_alerts': len(self.alerts),
            'recent_alerts': [
                {'level': a.level, 'message': a.message}
                for a in self.alerts[-5:]
            ],
        }

    def should_trade(self) -> bool:
        """Verificar si se debería operar."""
        return self.status != RiskStatus.PAUSED

    def get_position_multiplier(self) -> float:
        """Obtener multiplicador de posición según status."""
        if self.status == RiskStatus.PAUSED:
            return 0.0
        elif self.status == RiskStatus.REDUCED:
            return 0.5
        return 1.0

    def force_resume(self) -> None:
        """Forzar reanudación del trading (requiere confirmación manual)."""
        if self.status == RiskStatus.PAUSED:
            self.status = RiskStatus.ACTIVE
            self.alerts.append(Alert(
                timestamp=datetime.now(),
                level='INFO',
                message="MANUAL RESUME: Trading resumed by operator",
                metric='manual_override',
                value=0.0,
                threshold=0.0,
            ))

    def get_alerts_summary(self) -> Dict[str, int]:
        """Resumen de alertas por nivel."""
        summary = {'WARNING': 0, 'CRITICAL': 0, 'INFO': 0}
        for alert in self.alerts:
            summary[alert.level] = summary.get(alert.level, 0) + 1
        return summary


# =============================================================================
# CONVENIENCE WRAPPER
# =============================================================================

class TradingLoopWithRiskManager:
    """
    Wrapper para integrar RiskManager en un loop de trading.

    Ejemplo de uso:
    ```python
    loop = TradingLoopWithRiskManager(env, model, risk_manager)
    results = loop.run_episode()
    ```
    """

    def __init__(
        self,
        env,
        model,
        risk_manager: Optional[RiskManager] = None,
    ):
        self.env = env
        self.model = model
        self.risk_manager = risk_manager or RiskManager()

    def run_episode(
        self,
        deterministic: bool = True,
        verbose: bool = False,
    ) -> Dict:
        """
        Ejecutar un episodio completo con risk management.

        Returns:
            Dict con métricas y alertas
        """
        obs, _ = self.env.reset()
        self.risk_manager.reset(self.env.initial_balance)

        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            # Verificar si debemos operar
            if not self.risk_manager.should_trade():
                if verbose:
                    print(f"Step {steps}: Trading PAUSED by risk manager")
                action = np.array([0.0])  # HOLD
            else:
                # Obtener acción del modelo
                action, _ = self.model.predict(obs, deterministic=deterministic)

                # Aplicar multiplier del risk manager
                multiplier = self.risk_manager.get_position_multiplier()
                action = action * multiplier

            # Ejecutar step
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Actualizar risk manager
            status, mult, alerts = self.risk_manager.update(
                portfolio_value=info.get('portfolio', 10000),
                step_return=info.get('step_return', 0),
                action=float(action[0]),
            )

            # Mostrar alertas
            if verbose and alerts:
                for alert in alerts:
                    print(f"  [{alert.level}] {alert.message}")

            total_reward += reward
            steps += 1
            done = terminated or truncated

        # Obtener reporte final
        report = self.risk_manager.get_report()
        report['total_reward'] = total_reward
        report['total_steps'] = steps

        if hasattr(self.env, 'get_episode_metrics'):
            report['episode_metrics'] = self.env.get_episode_metrics()

        return report


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("RISK MANAGER - USD/COP RL Trading System")
    print("=" * 70)

    # Test básico
    print("\n1. Basic Risk Manager Test:")
    print("-" * 50)

    rm = RiskManager(
        limits=RiskLimits(
            max_drawdown_warning=0.03,
            max_drawdown_reduce=0.05,
            max_drawdown_stop=0.10,
        )
    )

    # Simular trading con pérdidas graduales
    portfolio = 10000.0
    print(f"  Initial portfolio: ${portfolio:,.2f}")

    scenarios = [
        (9800, -0.02),   # 2% loss
        (9700, -0.01),   # 1% more
        (9500, -0.02),   # 2% more (now at 5% DD)
        (9200, -0.03),   # 3% more (now at 8% DD)
        (9000, -0.02),   # 2% more (now at 10% DD - STOP)
    ]

    for pv, ret in scenarios:
        status, mult, alerts = rm.update(pv, ret, 0.5)
        print(f"\n  Portfolio: ${pv:,.2f} | DD: {rm.get_current_drawdown():.1%}")
        print(f"  Status: {status.value} | Multiplier: {mult}")
        if alerts:
            for a in alerts:
                print(f"    [{a.level}] {a.message}")

    # Test report
    print("\n2. Risk Report:")
    print("-" * 50)
    report = rm.get_report()
    for key, value in report.items():
        if key != 'recent_alerts':
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("RiskManager ready for integration")
    print("=" * 70)
