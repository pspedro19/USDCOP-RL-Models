"""
USD/COP RL Trading System - Action Distribution Callback
=========================================================

Monitorea la distribución de acciones durante training
y detecta colapso a política degenerada.

PROBLEMA QUE RESUELVE:
- Modelo colapsa a HOLD (100% inactividad)
- Modelo tiene bias extremo (solo LONG o solo SHORT)
- Entropy muy baja (acciones concentradas en un valor)

ALERTA SI:
- pct_hold > 70% (demasiado conservador)
- pct_hold < 10% (demasiado agresivo)
- std_action < 0.05 (colapso a valor fijo)
- bias > 0.4 (sesgo LONG/SHORT)

Author: Claude Code
Version: 1.0.0
"""

import numpy as np
from typing import List
from stable_baselines3.common.callbacks import BaseCallback


class ActionDistributionCallback(BaseCallback):
    """
    Monitorea la distribución de acciones durante training.

    Args:
        log_freq: Frecuencia de logging
        window_size: Tamaño de ventana para estadísticas
        collapse_threshold: Umbral de std para detectar colapso
        hold_warning_threshold: % de HOLD para warning
        bias_warning_threshold: Sesgo direccional para warning
        verbose: Nivel de verbosidad
    """

    def __init__(
        self,
        log_freq: int = 1_000,
        window_size: int = 5_000,
        collapse_threshold: float = 0.05,
        hold_warning_threshold: float = 70.0,
        bias_warning_threshold: float = 0.4,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.window_size = window_size
        self.collapse_threshold = collapse_threshold
        self.hold_warning_threshold = hold_warning_threshold
        self.bias_warning_threshold = bias_warning_threshold

        self.actions_buffer: List[float] = []
        self.collapse_warnings = 0
        self.bias_warnings = 0

    def _on_step(self) -> bool:
        """Llamado en cada step del training."""
        # Capturar acciones
        if 'actions' in self.locals:
            actions = self.locals['actions']
            if hasattr(actions, 'flatten'):
                actions = actions.flatten()
            self.actions_buffer.extend([float(a) for a in actions])

            # Mantener window size
            if len(self.actions_buffer) > self.window_size:
                self.actions_buffer = self.actions_buffer[-self.window_size:]

        if self.n_calls % self.log_freq != 0:
            return True

        if len(self.actions_buffer) < 100:
            return True

        actions = np.array(self.actions_buffer)

        # === CALCULAR MÉTRICAS ===
        mean_action = float(actions.mean())
        std_action = float(actions.std())
        # Action classification - THRESHOLD 0.10 (was 0.15, changed to account for regime multiplier)
        ACTION_THRESHOLD = 0.10
        pct_hold = float((np.abs(actions) < ACTION_THRESHOLD).mean() * 100)
        pct_long = float((actions > ACTION_THRESHOLD).mean() * 100)
        pct_short = float((actions < -ACTION_THRESHOLD).mean() * 100)
        bias = (pct_long - pct_short) / 100  # [-1, 1]

        # === LOG A TENSORBOARD ===
        if self.logger is not None:
            self.logger.record("train/mean_action", mean_action)
            self.logger.record("train/std_action", std_action)
            self.logger.record("train/pct_hold", pct_hold)
            self.logger.record("train/pct_long", pct_long)
            self.logger.record("train/pct_short", pct_short)
            self.logger.record("train/action_bias", bias)
            self.logger.record("train/collapse_warnings", self.collapse_warnings)

        # === DETECTAR PROBLEMAS ===
        problems = []

        # 1. Colapso a valor fijo
        if std_action < self.collapse_threshold:
            problems.append(f"COLLAPSE: std_action={std_action:.4f} < {self.collapse_threshold}")
            self.collapse_warnings += 1

        # 2. Demasiado conservador (HOLD)
        if pct_hold > self.hold_warning_threshold:
            problems.append(f"TOO_CONSERVATIVE: {pct_hold:.1f}% HOLD")

        # 3. Sesgo extremo
        if abs(bias) > self.bias_warning_threshold:
            direction = "LONG" if bias > 0 else "SHORT"
            problems.append(f"EXTREME_BIAS: {abs(bias)*100:.1f}% toward {direction}")
            self.bias_warnings += 1

        # 4. Todo en una dirección
        if pct_long > 90:
            problems.append(f"ALL_LONG: {pct_long:.1f}% LONG positions")
        elif pct_short > 90:
            problems.append(f"ALL_SHORT: {pct_short:.1f}% SHORT positions")

        # === REPORTAR ===
        if problems and self.verbose > 0:
            print(f"\n[{self.n_calls:,}] ⚠️ ACTION DISTRIBUTION WARNING:")
            for p in problems:
                print(f"  - {p}")
            print(f"  Distribution: HOLD={pct_hold:.1f}%, LONG={pct_long:.1f}%, SHORT={pct_short:.1f}%")
            print(f"  Stats: mean={mean_action:.3f}, std={std_action:.3f}")

        # === ADVERTENCIA SI COLAPSO PERSISTENTE ===
        if self.collapse_warnings >= 5 and self.verbose > 0:
            print(f"\n{'='*60}")
            print("CRITICAL: Model collapsed {self.collapse_warnings} times!")
            print("Consider:")
            print("  1. Increase entropy coefficient")
            print("  2. Check reward function symmetry")
            print("  3. Reduce learning rate")
            print("  4. Add exploration noise")
            print(f"{'='*60}")

        return True

    def get_stats(self) -> dict:
        """Obtener estadísticas actuales."""
        if len(self.actions_buffer) < 10:
            return {}

        actions = np.array(self.actions_buffer)
        ACTION_THRESHOLD = 0.10  # Was 0.15, changed to account for regime multiplier
        return {
            'mean_action': float(actions.mean()),
            'std_action': float(actions.std()),
            'pct_hold': float((np.abs(actions) < ACTION_THRESHOLD).mean() * 100),
            'pct_long': float((actions > ACTION_THRESHOLD).mean() * 100),
            'pct_short': float((actions < -ACTION_THRESHOLD).mean() * 100),
            'collapse_warnings': self.collapse_warnings,
            'bias_warnings': self.bias_warnings,
        }

    def is_healthy(self) -> bool:
        """Verificar si la distribución de acciones es saludable."""
        stats = self.get_stats()
        if not stats:
            return True

        # Criterios de salud
        healthy = True

        if stats.get('std_action', 1.0) < self.collapse_threshold:
            healthy = False

        if stats.get('pct_hold', 0) > self.hold_warning_threshold:
            healthy = False

        bias = (stats.get('pct_long', 50) - stats.get('pct_short', 50)) / 100
        if abs(bias) > self.bias_warning_threshold:
            healthy = False

        return healthy
