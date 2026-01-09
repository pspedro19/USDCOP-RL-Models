"""
USD/COP RL Trading System - Sharpe Evaluation Callback
=======================================================

Early stopping basado en Sharpe ratio del validation set.

A diferencia del EvalCallback de SB3 que usa mean_reward,
este callback calcula Sharpe anualizado correctamente.

Author: Claude Code
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback


class SharpeEvalCallback(BaseCallback):
    """
    Early stopping basado en Sharpe ratio del validation set.

    CRÍTICO: A diferencia del EvalCallback de SB3 que usa mean_reward,
    este callback calcula Sharpe anualizado correctamente.

    Args:
        eval_env: Environment de validación (SEPARADO del training)
        eval_freq: Frecuencia de evaluación (cada N steps)
        n_eval_episodes: Episodios para calcular Sharpe
        patience: Epochs sin mejora antes de early stop
        min_delta: Mejora mínima para considerar "progreso"
        min_sharpe: Sharpe mínimo para guardar modelo
        best_model_save_path: Donde guardar el mejor modelo
        bars_per_day: Barras por día para anualizar Sharpe
        verbose: Nivel de verbosidad
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 5_000,
        n_eval_episodes: int = 5,
        patience: int = 10,
        min_delta: float = 0.05,
        min_sharpe: float = 0.3,
        best_model_save_path: str = "./models/best",
        bars_per_day: int = 60,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.patience = patience
        self.min_delta = min_delta
        self.min_sharpe = min_sharpe
        self.best_model_save_path = Path(best_model_save_path)
        self.bars_per_day = bars_per_day

        # Estado interno
        self.best_sharpe = -np.inf
        self.no_improvement_count = 0
        self.eval_history: List[Dict] = []

        # Crear directorio si no existe
        self.best_model_save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        """Llamado en cada step del training."""

        if self.n_calls % self.eval_freq != 0:
            return True

        # === EVALUAR EN VALIDATION SET ===
        metrics = self._evaluate_validation()

        sharpe = metrics['sharpe']
        sortino = metrics['sortino']
        max_dd = metrics['max_dd']
        pct_hold = metrics['pct_hold']
        pct_long = metrics['pct_long']
        pct_short = metrics['pct_short']

        # Log a TensorBoard si está disponible
        if self.logger is not None:
            self.logger.record("eval/sharpe", sharpe)
            self.logger.record("eval/sortino", sortino)
            self.logger.record("eval/max_dd", max_dd)
            self.logger.record("eval/pct_hold", pct_hold)
            self.logger.record("eval/pct_long", pct_long)
            self.logger.record("eval/pct_short", pct_short)

        # === EARLY STOPPING LOGIC ===
        improvement = sharpe - self.best_sharpe

        if improvement > self.min_delta:
            # Mejora significativa
            self.best_sharpe = sharpe
            self.no_improvement_count = 0

            # Guardar checkpoint si supera mínimo
            if sharpe >= self.min_sharpe:
                save_path = self.best_model_save_path / f"best_sharpe_{sharpe:.3f}"
                self.model.save(str(save_path))
                if self.verbose > 0:
                    print(f"\n[{self.n_calls:,}] ✓ NEW BEST: Sharpe={sharpe:.3f}, saved to {save_path}")
        else:
            self.no_improvement_count += 1
            if self.verbose > 0:
                print(f"\n[{self.n_calls:,}] No improvement ({self.no_improvement_count}/{self.patience}), "
                      f"Sharpe={sharpe:.3f}, Best={self.best_sharpe:.3f}")

        # === CHECK EARLY STOP ===
        if self.no_improvement_count >= self.patience:
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"EARLY STOPPING: {self.patience} evals without improvement")
                print(f"Best Sharpe achieved: {self.best_sharpe:.3f}")
                print(f"{'='*60}")
            return False  # DETENER TRAINING

        # === CHECK COLLAPSE ===
        if pct_hold > 80:
            if self.verbose > 0:
                print(f"\n[WARNING] Model collapsing to HOLD ({pct_hold:.1f}%)")
                print("  Consider increasing entropy coefficient")

        if pct_long > 90 or pct_short > 90:
            direction = "LONG" if pct_long > 90 else "SHORT"
            if self.verbose > 0:
                print(f"\n[WARNING] Model biased to {direction} ({max(pct_long, pct_short):.1f}%)")

        self.eval_history.append(metrics)
        return True

    def _evaluate_validation(self) -> Dict[str, float]:
        """
        Ejecutar n episodios en validation env y calcular métricas.
        """
        all_returns = []
        all_actions = []
        all_portfolios = []

        for episode in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            episode_returns = []
            episode_actions = []
            portfolio_values = [getattr(self.eval_env, 'initial_balance', 10000)]
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)

                episode_returns.append(info.get('step_return', 0))
                episode_actions.append(float(action[0] if hasattr(action, '__len__') else action))
                portfolio_values.append(info.get('portfolio', portfolio_values[-1]))

                done = terminated or truncated

            all_returns.extend(episode_returns)
            all_actions.extend(episode_actions)
            all_portfolios.append(portfolio_values)

        returns = np.array(all_returns)
        actions = np.array(all_actions)

        # === CALCULAR SHARPE CORRECTAMENTE ===
        sharpe = self._calculate_sharpe(returns)
        sortino = self._calculate_sortino(returns)
        max_dd = self._calculate_max_dd(all_portfolios)

        # Distribución de acciones
        # Action classification - THRESHOLD 0.10 (was 0.15, changed to account for regime multiplier)
        ACTION_THRESHOLD = 0.10
        pct_hold = float((np.abs(actions) < ACTION_THRESHOLD).mean() * 100)
        pct_long = float((actions > ACTION_THRESHOLD).mean() * 100)
        pct_short = float((actions < -ACTION_THRESHOLD).mean() * 100)

        return {
            'sharpe': sharpe,
            'sortino': sortino,
            'max_dd': max_dd,
            'pct_hold': pct_hold,
            'pct_long': pct_long,
            'pct_short': pct_short,
            'mean_return': float(returns.mean() * 100),
            'std_return': float(returns.std() * 100),
            'n_samples': len(returns),
        }

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Sharpe anualizado correctamente: agregar a daily PRIMERO."""
        if len(returns) < self.bars_per_day * 2:
            return 0.0

        n_days = len(returns) // self.bars_per_day
        if n_days == 0:
            return 0.0

        # Agregar a returns diarios
        daily_returns = returns[:n_days * self.bars_per_day].reshape(
            n_days, self.bars_per_day
        ).sum(axis=1)

        if np.std(daily_returns) < 0.0005:
            return 0.0

        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        return float(sharpe)

    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """Sortino ratio (penaliza solo downside)."""
        if len(returns) < self.bars_per_day * 2:
            return 0.0

        n_days = len(returns) // self.bars_per_day
        if n_days == 0:
            return 0.0

        daily_returns = returns[:n_days * self.bars_per_day].reshape(
            n_days, self.bars_per_day
        ).sum(axis=1)

        downside = daily_returns[daily_returns < 0]
        if len(downside) < 2:
            return 0.0

        downside_std = np.std(downside)
        if downside_std < 1e-10:
            return 0.0

        sortino = np.mean(daily_returns) / downside_std * np.sqrt(252)
        return float(sortino)

    def _calculate_max_dd(self, all_portfolios: List[List[float]]) -> float:
        """Maximum Drawdown promedio de todos los episodios."""
        max_dds = []

        for portfolio_values in all_portfolios:
            pv = np.array(portfolio_values)
            peak = np.maximum.accumulate(pv)
            drawdown = (peak - pv) / (peak + 1e-10)
            max_dds.append(np.max(drawdown))

        return float(np.mean(max_dds)) * 100

    def get_best_model_path(self) -> Optional[Path]:
        """Obtener path del mejor modelo guardado."""
        if self.best_sharpe >= self.min_sharpe:
            return self.best_model_save_path / f"best_sharpe_{self.best_sharpe:.3f}.zip"
        return None
