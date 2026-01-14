"""
USD/COP RL Trading System - Entropy Scheduler Callback
=======================================================

Schedule de entropy coefficient para evitar colapso a política constante.

FILOSOFÍA:
- Inicio: entropy alta (0.10) para exploración
- Final: entropy baja (0.01) para explotación
- Transición: coseno suave o lineal

SOLUCIÓN AL PROBLEMA:
- Config actual: ent_coef=0.05 FIJO
- Esto es demasiado bajo al inicio -> colapso inmediato
- Demasiado alto al final -> acciones erráticas

Author: Claude Code
Version: 1.0.0
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EntropySchedulerCallback(BaseCallback):
    """
    Scheduler de entropy coefficient con curriculum.

    RECOMENDACIÓN:
    - init_ent=0.10: Forzar exploración inicial
    - final_ent=0.01: Permitir convergencia
    - warmup_fraction=0.2: Mantener alta en primer 20%

    Args:
        init_ent: Entropy inicial (alto para exploración)
        final_ent: Entropy final (bajo para explotación)
        schedule_type: Tipo de schedule ('linear', 'cosine', 'warmup_cosine')
        warmup_fraction: Fracción de training para warmup
        log_freq: Frecuencia de logging
        verbose: Nivel de verbosidad
    """

    def __init__(
        self,
        init_ent: float = 0.10,
        final_ent: float = 0.01,
        schedule_type: str = 'warmup_cosine',
        warmup_fraction: float = 0.2,
        log_freq: int = 1000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.init_ent = init_ent
        self.final_ent = final_ent
        self.schedule_type = schedule_type
        self.warmup_fraction = warmup_fraction
        self.log_freq = log_freq

        self._last_logged_ent = None

    def _on_training_start(self) -> None:
        """Configurar entropy inicial."""
        if hasattr(self.model, 'ent_coef'):
            # Para PPO
            self.model.ent_coef = self.init_ent
            if self.verbose > 0:
                print(f"\n[EntropyScheduler] Initial entropy set to {self.init_ent}")
        elif hasattr(self.model, 'ent_coef_tensor'):
            # SAC tiene auto-tuning de entropy - no modificamos
            if self.verbose > 0:
                print(f"\n[EntropyScheduler] SAC detected - using automatic entropy tuning")
                print(f"  (Manual scheduling disabled for SAC)")

    def _on_step(self) -> bool:
        """Actualizar entropy en cada step."""
        # Calcular progreso [0, 1]
        if hasattr(self.model, '_total_timesteps') and self.model._total_timesteps > 0:
            progress = self.num_timesteps / self.model._total_timesteps
        else:
            progress = 0.0

        # Calcular nuevo entropy según schedule
        new_ent = self._calculate_entropy(progress)

        # Aplicar al modelo
        if hasattr(self.model, 'ent_coef'):
            self.model.ent_coef = max(new_ent, self.final_ent)

        # Log periódico
        if self.n_calls % self.log_freq == 0:
            if self.logger is not None:
                self.logger.record("train/ent_coef", new_ent)
                self.logger.record("train/ent_progress", progress)

            if self.verbose > 1 and new_ent != self._last_logged_ent:
                print(f"[{self.n_calls:,}] Entropy: {new_ent:.4f} (progress: {progress:.1%})")
                self._last_logged_ent = new_ent

        return True

    def _calculate_entropy(self, progress: float) -> float:
        """
        Calcular entropy según el schedule seleccionado.

        Args:
            progress: Progreso del training [0, 1]

        Returns:
            Nuevo valor de entropy
        """
        if self.schedule_type == 'linear':
            # Decay lineal
            new_ent = self.init_ent - progress * (self.init_ent - self.final_ent)

        elif self.schedule_type == 'cosine':
            # Cosine annealing (más suave)
            new_ent = self.final_ent + (self.init_ent - self.final_ent) * \
                      (1 + np.cos(np.pi * progress)) / 2

        elif self.schedule_type == 'warmup_cosine':
            # Warmup: mantener alto, luego cosine
            if progress < self.warmup_fraction:
                new_ent = self.init_ent
            else:
                adjusted_progress = (progress - self.warmup_fraction) / (1 - self.warmup_fraction)
                new_ent = self.final_ent + (self.init_ent - self.final_ent) * \
                          (1 + np.cos(np.pi * adjusted_progress)) / 2

        elif self.schedule_type == 'exponential':
            # Decay exponencial
            decay_rate = np.log(self.final_ent / self.init_ent)
            new_ent = self.init_ent * np.exp(decay_rate * progress)

        elif self.schedule_type == 'step':
            # Step decay cada 25%
            if progress < 0.25:
                new_ent = self.init_ent
            elif progress < 0.50:
                new_ent = self.init_ent * 0.5
            elif progress < 0.75:
                new_ent = self.init_ent * 0.25
            else:
                new_ent = self.final_ent

        else:
            # Default: mantener inicial
            new_ent = self.init_ent

        return float(max(new_ent, self.final_ent))

    def _on_training_end(self) -> None:
        """Log final de entropy."""
        if self.verbose > 0:
            final_ent = getattr(self.model, 'ent_coef', self.final_ent)
            print(f"\n[EntropyScheduler] Final entropy: {final_ent:.4f}")
