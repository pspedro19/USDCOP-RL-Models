"""
USD/COP RL Trading System - Trading Metrics
=============================================

Cálculo correcto de métricas de trading.

PROBLEMA QUE RESUELVE:
- Sharpe calculado incorrectamente (sin anualizar, sin agregar a daily)
- Métricas solo en train set (no representativas)
- Falta de intervalos de confianza

Author: Claude Code
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats


@dataclass
class TradingMetrics:
    """Contenedor de métricas de trading."""

    # Retornos
    total_return: float
    annualized_return: float
    mean_daily_return: float
    std_daily_return: float

    # Ratios de riesgo-ajustado
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float

    # Drawdown
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int
    recovery_time: int

    # Trading
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    expectancy: float

    # Estabilidad
    stability: float
    tail_ratio: float
    var_95: float
    cvar_95: float

    # Distribución de acciones
    pct_long: float
    pct_short: float
    pct_hold: float
    action_std: float

    # Metadatos
    n_trades: int
    n_days: int
    bars_per_day: int

    def to_dict(self) -> Dict[str, float]:
        """Convertir a diccionario."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'omega_ratio': self.omega_ratio,
            'max_drawdown': self.max_drawdown,
            'avg_drawdown': self.avg_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'expectancy': self.expectancy,
            'stability': self.stability,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'pct_long': self.pct_long,
            'pct_short': self.pct_short,
            'pct_hold': self.pct_hold,
            'n_trades': self.n_trades,
        }

    def passes_acceptance(
        self,
        min_sharpe: float = 0.8,
        max_drawdown: float = 15.0,
        min_calmar: float = 0.5,
        min_profit_factor: float = 1.2,
        min_win_rate: float = 0.45,
        max_hold_pct: float = 80.0,
    ) -> Tuple[bool, List[str]]:
        """
        Verificar si las métricas pasan los criterios de aceptación.

        Returns:
            Tuple de (passed, list of failed criteria)
        """
        failures = []

        if self.sharpe_ratio < min_sharpe:
            failures.append(f"Sharpe {self.sharpe_ratio:.2f} < {min_sharpe}")

        if self.max_drawdown > max_drawdown:
            failures.append(f"MaxDD {self.max_drawdown:.1f}% > {max_drawdown}%")

        if self.calmar_ratio < min_calmar:
            failures.append(f"Calmar {self.calmar_ratio:.2f} < {min_calmar}")

        if self.profit_factor < min_profit_factor:
            failures.append(f"PF {self.profit_factor:.2f} < {min_profit_factor}")

        if self.win_rate < min_win_rate:
            failures.append(f"WinRate {self.win_rate:.1%} < {min_win_rate:.0%}")

        if self.pct_hold > max_hold_pct:
            failures.append(f"Hold {self.pct_hold:.1f}% > {max_hold_pct}%")

        return len(failures) == 0, failures


def calculate_all_metrics(
    returns: np.ndarray,
    actions: Optional[np.ndarray] = None,
    portfolio_values: Optional[np.ndarray] = None,
    bars_per_day: int = 60,
    trading_days_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> TradingMetrics:
    """
    Calcular todas las métricas de trading.

    IMPORTANTE: Los returns deben ser per-bar, no acumulados.

    Args:
        returns: Array de retornos por barra
        actions: Array de acciones [-1, 1]
        portfolio_values: Valores del portfolio (opcional)
        bars_per_day: Barras por día de trading
        trading_days_per_year: Días de trading por año
        risk_free_rate: Tasa libre de riesgo anualizada

    Returns:
        TradingMetrics con todas las métricas calculadas
    """
    returns = np.array(returns)
    n_bars = len(returns)
    n_days = n_bars // bars_per_day

    if n_days < 2:
        # Retornar métricas vacías si no hay suficientes datos
        return _empty_metrics(bars_per_day)

    # Agregar a retornos diarios
    daily_returns = returns[:n_days * bars_per_day].reshape(n_days, bars_per_day).sum(axis=1)

    # === RETORNOS ===
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (trading_days_per_year / n_days) - 1
    mean_daily = daily_returns.mean()
    std_daily = daily_returns.std()

    # === SHARPE RATIO ===
    # Correcto: usar retornos diarios, anualizar al final
    daily_rf = risk_free_rate / trading_days_per_year
    excess_daily = daily_returns - daily_rf

    if std_daily > 1e-10:
        sharpe = excess_daily.mean() / std_daily * np.sqrt(trading_days_per_year)
    else:
        sharpe = 0.0

    # === SORTINO RATIO ===
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 1 and downside.std() > 1e-10:
        sortino = mean_daily / downside.std() * np.sqrt(trading_days_per_year)
    else:
        sortino = 0.0

    # === DRAWDOWN ===
    if portfolio_values is not None:
        pv = np.array(portfolio_values)
    else:
        pv = (1 + returns).cumprod()

    peak = np.maximum.accumulate(pv)
    drawdown = (peak - pv) / (peak + 1e-10)
    max_dd = drawdown.max() * 100
    avg_dd = drawdown.mean() * 100

    # Drawdown duration
    dd_duration = _calculate_drawdown_duration(drawdown)
    max_dd_duration = dd_duration.max() if len(dd_duration) > 0 else 0
    recovery_time = _calculate_recovery_time(drawdown)

    # === CALMAR RATIO ===
    if max_dd > 0.1:  # Al menos 0.1% drawdown
        calmar = annualized_return / (max_dd / 100)
    else:
        calmar = annualized_return * 100  # Si no hay DD, escalar por 100

    # === OMEGA RATIO ===
    threshold = 0
    gains = daily_returns[daily_returns > threshold].sum()
    losses = abs(daily_returns[daily_returns <= threshold].sum())
    omega = gains / losses if losses > 1e-10 else 10.0

    # === TRADING METRICS ===
    if actions is not None:
        actions = np.array(actions)
        # Identificar trades (cambios de posición significativos)
        position_changes = np.abs(np.diff(actions, prepend=0))
        trade_mask = position_changes > 0.1

        # Retornos por trade
        trade_returns = returns[trade_mask]
        n_trades = len(trade_returns)

        if n_trades > 0:
            wins = trade_returns[trade_returns > 0]
            losses = trade_returns[trade_returns <= 0]

            win_rate = len(wins) / n_trades
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 0

            gross_profit = wins.sum() if len(wins) > 0 else 0
            gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-10
            profit_factor = gross_profit / gross_loss

            expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
        else:
            win_rate = 0.5
            avg_win = 0
            avg_loss = 0
            profit_factor = 1.0
            expectancy = 0
            n_trades = 0

        # Distribución de acciones
        # Action classification - THRESHOLD 0.10 (was 0.15, changed to account for regime multiplier)
        ACTION_THRESHOLD = 0.10
        pct_hold = (np.abs(actions) < ACTION_THRESHOLD).mean() * 100
        pct_long = (actions > ACTION_THRESHOLD).mean() * 100
        pct_short = (actions < -ACTION_THRESHOLD).mean() * 100
        action_std = actions.std()
    else:
        win_rate = 0.5
        avg_win = 0
        avg_loss = 0
        profit_factor = 1.0
        expectancy = 0
        n_trades = 0
        pct_hold = 33.3
        pct_long = 33.3
        pct_short = 33.3
        action_std = 0.5

    # === ESTABILIDAD ===
    # R² de regresión de equity curve
    if portfolio_values is not None and len(portfolio_values) > 2:
        x = np.arange(len(portfolio_values))
        slope, intercept, r_value, _, _ = stats.linregress(x, portfolio_values)
        stability = r_value ** 2
    else:
        stability = 0.5

    # === TAIL RATIOS ===
    if len(daily_returns) > 20:
        positive = daily_returns[daily_returns > 0]
        negative = daily_returns[daily_returns < 0]

        if len(positive) > 0 and len(negative) > 0:
            p95 = np.percentile(positive, 95)
            n95 = abs(np.percentile(negative, 5))
            tail_ratio = p95 / n95 if n95 > 1e-10 else 2.0
        else:
            tail_ratio = 1.0
    else:
        tail_ratio = 1.0

    # === VAR / CVAR ===
    if len(daily_returns) > 20:
        var_95 = -np.percentile(daily_returns, 5) * 100
        cvar_95 = -daily_returns[daily_returns <= np.percentile(daily_returns, 5)].mean() * 100
    else:
        var_95 = 0
        cvar_95 = 0

    return TradingMetrics(
        total_return=total_return * 100,
        annualized_return=annualized_return * 100,
        mean_daily_return=mean_daily * 100,
        std_daily_return=std_daily * 100,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        omega_ratio=omega,
        max_drawdown=max_dd,
        avg_drawdown=avg_dd,
        max_drawdown_duration=max_dd_duration,
        recovery_time=recovery_time,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win * 100,
        avg_loss=avg_loss * 100,
        expectancy=expectancy * 100,
        stability=stability,
        tail_ratio=tail_ratio,
        var_95=var_95,
        cvar_95=cvar_95,
        pct_long=pct_long,
        pct_short=pct_short,
        pct_hold=pct_hold,
        action_std=action_std,
        n_trades=n_trades,
        n_days=n_days,
        bars_per_day=bars_per_day,
    )


def _calculate_drawdown_duration(drawdown: np.ndarray) -> np.ndarray:
    """Calcular duraciones de drawdown."""
    in_dd = drawdown > 0
    durations = []
    current_duration = 0

    for is_in in in_dd:
        if is_in:
            current_duration += 1
        else:
            if current_duration > 0:
                durations.append(current_duration)
            current_duration = 0

    if current_duration > 0:
        durations.append(current_duration)

    return np.array(durations)


def _calculate_recovery_time(drawdown: np.ndarray) -> int:
    """Calcular tiempo de recuperación del último drawdown."""
    if len(drawdown) == 0:
        return 0

    # Buscar último pico
    peak_idx = np.argmax(np.maximum.accumulate(1 - drawdown))

    # Si no hay drawdown, retornar 0
    if drawdown[peak_idx:].max() < 0.001:
        return 0

    # Buscar siguiente recuperación (drawdown = 0)
    for i, dd in enumerate(drawdown[peak_idx:]):
        if dd < 0.001:
            return i

    # Si nunca se recuperó
    return len(drawdown) - peak_idx


def _empty_metrics(bars_per_day: int) -> TradingMetrics:
    """Retornar métricas vacías."""
    return TradingMetrics(
        total_return=0,
        annualized_return=0,
        mean_daily_return=0,
        std_daily_return=0,
        sharpe_ratio=0,
        sortino_ratio=0,
        calmar_ratio=0,
        omega_ratio=1,
        max_drawdown=0,
        avg_drawdown=0,
        max_drawdown_duration=0,
        recovery_time=0,
        win_rate=0.5,
        profit_factor=1,
        avg_win=0,
        avg_loss=0,
        expectancy=0,
        stability=0,
        tail_ratio=1,
        var_95=0,
        cvar_95=0,
        pct_long=33.3,
        pct_short=33.3,
        pct_hold=33.3,
        action_std=0.5,
        n_trades=0,
        n_days=0,
        bars_per_day=bars_per_day,
    )


class MetricsAggregator:
    """
    Agregar métricas de múltiples evaluaciones.

    Útil para multi-seed training o bootstrap.
    """

    def __init__(self):
        self.metrics_list: List[TradingMetrics] = []

    def add(self, metrics: TradingMetrics):
        """Agregar métricas."""
        self.metrics_list.append(metrics)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Obtener resumen estadístico.

        Returns:
            Dict con mean, std, min, max para cada métrica
        """
        if not self.metrics_list:
            return {}

        summary = {}
        metrics_dict = [m.to_dict() for m in self.metrics_list]

        for key in metrics_dict[0].keys():
            values = [m[key] for m in metrics_dict]
            summary[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
            }

        return summary

    def get_confidence_interval(
        self,
        metric: str,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Obtener intervalo de confianza para una métrica.

        Args:
            metric: Nombre de la métrica
            confidence: Nivel de confianza

        Returns:
            Tuple de (lower, upper)
        """
        values = [m.to_dict()[metric] for m in self.metrics_list]

        if len(values) < 2:
            return (values[0], values[0]) if values else (0, 0)

        alpha = 1 - confidence
        lower = np.percentile(values, alpha / 2 * 100)
        upper = np.percentile(values, (1 - alpha / 2) * 100)

        return (lower, upper)
