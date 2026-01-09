"""
USD/COP RL Trading System - Cost Curriculum Callback
=====================================================

Introduce costos de transacción progresivamente.

FILOSOFÍA:
- Inicio: Sin costos (facilita aprendizaje de timing)
- Mitad: Costos parciales (aprende a filtrar señales débiles)
- Final: Costos completos (production-ready)

PROBLEMA QUE RESUELVE:
- Con costos desde el inicio, el modelo aprende a NO operar
- Sin costos, el modelo sobreopera (whipsaw)
- Curriculum permite aprender CUÁNDO operar primero

Author: Claude Code
Version: 1.0.0
"""

import numpy as np
from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback


class CostCurriculumCallback(BaseCallback):
    """
    Introduce costos de transacción progresivamente.

    RECOMENDACIÓN:
    - warmup_steps: 30_000 (sin costos, exploración pura)
    - rampup_steps: 70_000 (costos incrementales)
    - final_cost: 0.0025 (25 bps normal)
    - crisis_multiplier: 2.4 (60 bps en crisis)

    Args:
        env: Environment de training
        warmup_steps: Steps sin costos
        rampup_steps: Steps para incrementar costos
        final_cost: Costo final (decimal, ej: 0.0025 = 25 bps)
        crisis_multiplier: Multiplicador para alta volatilidad
        log_freq: Frecuencia de logging
        verbose: Nivel de verbosidad
    """

    def __init__(
        self,
        env,
        warmup_steps: int = 30_000,
        rampup_steps: int = 70_000,
        final_cost: float = 0.0025,
        crisis_multiplier: float = 2.4,
        log_freq: int = 5_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.env = env
        self.warmup_steps = warmup_steps
        self.rampup_steps = rampup_steps
        self.final_cost = final_cost
        self.crisis_multiplier = crisis_multiplier
        self.log_freq = log_freq

        self._last_phase = None
        self._phase_logged = set()

    def _on_training_start(self) -> None:
        """Configurar costos iniciales en cero."""
        self._set_env_costs(0.0, 0.0)
        if self.verbose > 0:
            print(f"\n[CostCurriculum] Starting with ZERO costs")
            print(f"  Warmup: {self.warmup_steps:,} steps")
            print(f"  Rampup: {self.rampup_steps:,} steps")
            print(f"  Final: {self.final_cost * 10000:.1f} bps")

    def _on_step(self) -> bool:
        """Actualizar costos en cada step."""
        steps = self.num_timesteps
        phase, current_cost = self._calculate_cost(steps)

        # Actualizar environment
        high_vol_cost = current_cost * self.crisis_multiplier
        self._set_env_costs(current_cost, high_vol_cost)

        # Log cambio de fase
        if phase != self._last_phase:
            if self.verbose > 0 and phase not in self._phase_logged:
                print(f"\n[{steps:,}] Cost Curriculum: Entering {phase} phase")
                print(f"  Normal cost: {current_cost * 10000:.1f} bps")
                print(f"  High-vol cost: {high_vol_cost * 10000:.1f} bps")
                self._phase_logged.add(phase)
            self._last_phase = phase

        # Log periódico
        if self.n_calls % self.log_freq == 0:
            if self.logger is not None:
                self.logger.record("train/cost_bps", current_cost * 10000)
                self.logger.record("train/cost_phase", phase)

        return True

    def _calculate_cost(self, steps: int) -> tuple:
        """
        Calcular costo actual según curriculum.

        Returns:
            Tuple de (fase, costo_decimal)
        """
        if steps < self.warmup_steps:
            # Fase 1: Sin costos
            return "warmup", 0.0

        elif steps < self.warmup_steps + self.rampup_steps:
            # Fase 2: Ramp-up lineal
            progress = (steps - self.warmup_steps) / self.rampup_steps
            cost = self.final_cost * progress
            return "rampup", cost

        else:
            # Fase 3: Costos completos
            return "full", self.final_cost

    def _set_env_costs(self, normal_cost: float, high_vol_cost: float):
        """Actualizar costos en el environment."""
        # Intentar diferentes estructuras de environment
        envs_to_update = []

        # VecEnv (DummyVecEnv, SubprocVecEnv)
        if hasattr(self.env, 'envs'):
            envs_to_update = self.env.envs
        # Monitor wrapper
        elif hasattr(self.env, 'env'):
            envs_to_update = [self.env.env]
        # Environment directo
        else:
            envs_to_update = [self.env]

        for e in envs_to_update:
            # Unwrap si es necesario
            actual_env = e
            while hasattr(actual_env, 'env'):
                if hasattr(actual_env, 'cost') or hasattr(actual_env, 'cost_normal'):
                    break
                actual_env = actual_env.env

            # Actualizar costos
            if hasattr(actual_env, 'cost'):
                actual_env.cost = normal_cost
            if hasattr(actual_env, 'cost_normal'):
                actual_env.cost_normal = normal_cost
            if hasattr(actual_env, 'cost_high_vol'):
                actual_env.cost_high_vol = high_vol_cost

    def get_current_cost_bps(self) -> float:
        """Obtener costo actual en basis points."""
        _, cost = self._calculate_cost(self.num_timesteps)
        return cost * 10000

    def _on_training_end(self) -> None:
        """Log final."""
        if self.verbose > 0:
            final_cost = self.get_current_cost_bps()
            print(f"\n[CostCurriculum] Training ended with cost: {final_cost:.1f} bps")
