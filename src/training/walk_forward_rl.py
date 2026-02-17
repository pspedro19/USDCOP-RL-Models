"""
Walk-Forward RL Training
========================

FASE 5: Integrates walk-forward validation with RL training.

Walk-forward for RL:
- Train on expanding/rolling windows
- Evaluate on held-out period
- Select best model based on OOS performance
- Track performance degradation across windows

This module adapts the backtesting walk-forward validator for RL training.

Author: Trading Team
Version: 1.0.0
Date: 2026-02-02
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple
import json

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardRLWindow:
    """A single walk-forward window for RL training."""
    window_id: int
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime

    # Data sizes
    train_rows: int = 0
    val_rows: int = 0

    # Training results
    model_path: Optional[str] = None
    training_timesteps: int = 0
    training_duration_seconds: float = 0.0

    # Metrics
    train_reward: float = 0.0
    val_reward: float = 0.0
    val_sharpe: float = 0.0
    val_return: float = 0.0
    val_win_rate: float = 0.0
    val_max_drawdown: float = 0.0

    # Action distribution
    action_distribution: Dict[str, float] = field(default_factory=dict)

    # Status
    success: bool = False
    error: Optional[str] = None

    @property
    def degradation_ratio(self) -> float:
        """Ratio of validation to training reward (measures overfitting)."""
        if self.train_reward != 0:
            return self.val_reward / self.train_reward
        return 0.0

    @property
    def train_days(self) -> int:
        """Number of days in training period."""
        return (self.train_end - self.train_start).days

    @property
    def val_days(self) -> int:
        """Number of days in validation period."""
        return (self.val_end - self.val_start).days

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "window_id": self.window_id,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "val_start": self.val_start.isoformat(),
            "val_end": self.val_end.isoformat(),
            "train_rows": self.train_rows,
            "val_rows": self.val_rows,
            "train_days": self.train_days,
            "val_days": self.val_days,
            "model_path": self.model_path,
            "training_timesteps": self.training_timesteps,
            "training_duration_seconds": self.training_duration_seconds,
            "train_reward": self.train_reward,
            "val_reward": self.val_reward,
            "val_sharpe": self.val_sharpe,
            "val_return": self.val_return,
            "val_win_rate": self.val_win_rate,
            "val_max_drawdown": self.val_max_drawdown,
            "degradation_ratio": self.degradation_ratio,
            "action_distribution": self.action_distribution,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class WalkForwardRLReport:
    """Complete walk-forward RL training report."""
    windows: List[WalkForwardRLWindow]
    method: str  # "rolling" or "anchored"

    # Aggregate metrics
    mean_val_reward: float = 0.0
    std_val_reward: float = 0.0
    mean_val_sharpe: float = 0.0
    std_val_sharpe: float = 0.0
    mean_degradation: float = 0.0
    consistency_score: float = 0.0  # % windows with positive val reward

    # Best model selection
    best_window_id: int = -1
    best_val_sharpe: float = float('-inf')
    best_model_path: Optional[str] = None

    # Summary
    total_windows: int = 0
    successful_windows: int = 0
    total_train_days: int = 0
    total_val_days: int = 0

    def select_best_model(self) -> Optional[WalkForwardRLWindow]:
        """Select the best model based on validation Sharpe."""
        best = None
        for w in self.windows:
            if w.success and w.val_sharpe > self.best_val_sharpe:
                self.best_val_sharpe = w.val_sharpe
                self.best_window_id = w.window_id
                self.best_model_path = w.model_path
                best = w
        return best

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "total_windows": self.total_windows,
            "successful_windows": self.successful_windows,
            "mean_val_reward": self.mean_val_reward,
            "std_val_reward": self.std_val_reward,
            "mean_val_sharpe": self.mean_val_sharpe,
            "std_val_sharpe": self.std_val_sharpe,
            "mean_degradation": self.mean_degradation,
            "consistency_score": self.consistency_score,
            "best_window_id": self.best_window_id,
            "best_val_sharpe": self.best_val_sharpe,
            "best_model_path": self.best_model_path,
            "total_train_days": self.total_train_days,
            "total_val_days": self.total_val_days,
            "windows": [w.to_dict() for w in self.windows],
        }


class WalkForwardRLTrainer:
    """
    Walk-forward trainer for RL models.

    FASE 5: Integrates walk-forward validation with RL training.

    Usage:
        trainer = WalkForwardRLTrainer(
            train_period_months=12,
            val_period_months=3,
            method="rolling",
        )

        report = trainer.run(
            data=df,
            train_func=my_train_function,
            eval_func=my_eval_function,
            output_dir=Path("models/walk_forward"),
        )

        # Get best model
        best = report.select_best_model()
        print(f"Best model: {best.model_path} (Sharpe: {best.val_sharpe:.2f})")
    """

    def __init__(
        self,
        train_period_months: int = 12,
        val_period_months: int = 3,
        step_months: Optional[int] = None,
        method: str = "rolling",  # "rolling" or "anchored"
        min_train_rows: int = 50000,
        date_column: str = "timestamp",
    ):
        """
        Initialize walk-forward RL trainer.

        Args:
            train_period_months: Training window size in months
            val_period_months: Validation window size in months
            step_months: Step size between windows (default: val_period_months)
            method: "rolling" (fixed window) or "anchored" (expanding window)
            min_train_rows: Minimum rows required for training
            date_column: Column containing timestamps
        """
        self.train_period_months = train_period_months
        self.val_period_months = val_period_months
        self.step_months = step_months or val_period_months
        self.method = method
        self.min_train_rows = min_train_rows
        self.date_column = date_column

        self.windows: List[WalkForwardRLWindow] = []

    def generate_windows(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[WalkForwardRLWindow]:
        """
        Generate walk-forward windows.

        Args:
            start_date: Start of data period
            end_date: End of data period

        Returns:
            List of WalkForwardRLWindow objects
        """
        self.windows = []
        window_id = 0

        # Convert months to days (approximate)
        train_days = self.train_period_months * 30
        val_days = self.val_period_months * 30
        step_days = self.step_months * 30

        if self.method == "anchored":
            # Anchored: training always starts from start_date, expands over time
            current_train_start = start_date
            current_train_end = start_date + timedelta(days=train_days)

            while current_train_end + timedelta(days=val_days) <= end_date:
                window = WalkForwardRLWindow(
                    window_id=window_id,
                    train_start=current_train_start,
                    train_end=current_train_end,
                    val_start=current_train_end,
                    val_end=current_train_end + timedelta(days=val_days),
                )
                self.windows.append(window)

                # Expand training window
                current_train_end += timedelta(days=step_days)
                window_id += 1

        else:
            # Rolling: training window moves forward
            current_train_start = start_date

            while current_train_start + timedelta(days=train_days + val_days) <= end_date:
                train_end = current_train_start + timedelta(days=train_days)
                val_end = train_end + timedelta(days=val_days)

                window = WalkForwardRLWindow(
                    window_id=window_id,
                    train_start=current_train_start,
                    train_end=train_end,
                    val_start=train_end,
                    val_end=val_end,
                )
                self.windows.append(window)

                current_train_start += timedelta(days=step_days)
                window_id += 1

        logger.info(
            f"[FASE-5] Generated {len(self.windows)} walk-forward windows "
            f"({self.method} method, {self.train_period_months}m train / "
            f"{self.val_period_months}m val)"
        )

        return self.windows

    def run(
        self,
        data: pd.DataFrame,
        train_func: Callable[[pd.DataFrame, Path, int], Dict[str, Any]],
        eval_func: Callable[[Path, pd.DataFrame], Dict[str, float]],
        output_dir: Path,
        timesteps_per_window: int = 100_000,
    ) -> WalkForwardRLReport:
        """
        Run walk-forward RL training.

        Args:
            data: Complete dataset with timestamp column
            train_func: Function(train_data, output_dir, timesteps) -> {"model_path", "reward", ...}
            eval_func: Function(model_path, val_data) -> {"sharpe", "return", "win_rate", ...}
            output_dir: Base output directory for models
            timesteps_per_window: Training timesteps per window

        Returns:
            WalkForwardRLReport with all results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate windows if not already done
        if not self.windows:
            start_date = pd.to_datetime(data[self.date_column].min())
            end_date = pd.to_datetime(data[self.date_column].max())
            self.generate_windows(start_date, end_date)

        # Run training for each window
        for window in self.windows:
            logger.info(f"[FASE-5] Window {window.window_id}: "
                       f"Train {window.train_start.date()} to {window.train_end.date()}, "
                       f"Val {window.val_start.date()} to {window.val_end.date()}")

            try:
                # Filter data
                train_mask = (
                    (pd.to_datetime(data[self.date_column]) >= window.train_start) &
                    (pd.to_datetime(data[self.date_column]) < window.train_end)
                )
                val_mask = (
                    (pd.to_datetime(data[self.date_column]) >= window.val_start) &
                    (pd.to_datetime(data[self.date_column]) < window.val_end)
                )

                train_data = data[train_mask].copy()
                val_data = data[val_mask].copy()

                window.train_rows = len(train_data)
                window.val_rows = len(val_data)

                # Check minimum data
                if window.train_rows < self.min_train_rows:
                    window.error = f"Insufficient training data: {window.train_rows} < {self.min_train_rows}"
                    logger.warning(f"[FASE-5] Window {window.window_id}: {window.error}")
                    continue

                if window.val_rows == 0:
                    window.error = "No validation data"
                    logger.warning(f"[FASE-5] Window {window.window_id}: {window.error}")
                    continue

                # Create window output directory
                window_dir = output_dir / f"window_{window.window_id}"
                window_dir.mkdir(parents=True, exist_ok=True)

                # Train
                logger.info(f"[FASE-5] Training window {window.window_id} "
                           f"({window.train_rows} rows, {timesteps_per_window} timesteps)...")

                train_result = train_func(train_data, window_dir, timesteps_per_window)

                window.model_path = train_result.get("model_path")
                window.train_reward = train_result.get("best_reward", 0.0)
                window.training_timesteps = train_result.get("timesteps", timesteps_per_window)
                window.training_duration_seconds = train_result.get("duration", 0.0)
                window.action_distribution = train_result.get("action_distribution", {})

                # Evaluate on validation
                if window.model_path:
                    logger.info(f"[FASE-5] Evaluating window {window.window_id} on validation set...")

                    eval_result = eval_func(Path(window.model_path), val_data)

                    window.val_reward = eval_result.get("mean_reward", 0.0)
                    window.val_sharpe = eval_result.get("sharpe", 0.0)
                    window.val_return = eval_result.get("total_return", 0.0)
                    window.val_win_rate = eval_result.get("win_rate", 0.0)
                    window.val_max_drawdown = eval_result.get("max_drawdown", 0.0)

                window.success = True

                logger.info(
                    f"[FASE-5] Window {window.window_id} complete: "
                    f"Train reward={window.train_reward:.2f}, "
                    f"Val Sharpe={window.val_sharpe:.2f}, "
                    f"Degradation={window.degradation_ratio:.2f}"
                )

            except Exception as e:
                window.error = str(e)
                logger.error(f"[FASE-5] Window {window.window_id} failed: {e}")

        # Generate report
        return self._generate_report(output_dir)

    def _generate_report(self, output_dir: Path) -> WalkForwardRLReport:
        """Generate comprehensive walk-forward report."""
        successful = [w for w in self.windows if w.success]

        if not successful:
            logger.warning("[FASE-5] No successful windows!")
            return WalkForwardRLReport(
                windows=self.windows,
                method=self.method,
                total_windows=len(self.windows),
            )

        # Calculate aggregate metrics
        val_rewards = [w.val_reward for w in successful]
        val_sharpes = [w.val_sharpe for w in successful]
        degradations = [w.degradation_ratio for w in successful if w.degradation_ratio != 0]

        report = WalkForwardRLReport(
            windows=self.windows,
            method=self.method,
            total_windows=len(self.windows),
            successful_windows=len(successful),
            mean_val_reward=float(np.mean(val_rewards)),
            std_val_reward=float(np.std(val_rewards)),
            mean_val_sharpe=float(np.mean(val_sharpes)),
            std_val_sharpe=float(np.std(val_sharpes)),
            mean_degradation=float(np.mean(degradations)) if degradations else 0.0,
            consistency_score=sum(1 for s in val_sharpes if s > 0) / len(val_sharpes),
            total_train_days=sum(w.train_days for w in successful),
            total_val_days=sum(w.val_days for w in successful),
        )

        # Select best model
        report.select_best_model()

        # Save report
        report_path = output_dir / "walk_forward_report.json"
        with open(report_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(
            f"[FASE-5] Walk-Forward Report: "
            f"{report.successful_windows}/{report.total_windows} successful, "
            f"Mean Val Sharpe={report.mean_val_sharpe:.2f} (+/- {report.std_val_sharpe:.2f}), "
            f"Best Window={report.best_window_id} (Sharpe={report.best_val_sharpe:.2f}), "
            f"Consistency={report.consistency_score:.1%}"
        )

        return report


# =============================================================================
# Convenience function
# =============================================================================

def create_walk_forward_trainer_from_ssot() -> WalkForwardRLTrainer:
    """
    Create a WalkForwardRLTrainer using SSOT configuration.

    Returns:
        WalkForwardRLTrainer configured from experiment_ssot.yaml
    """
    try:
        from src.config.experiment_loader import load_experiment_config
        config = load_experiment_config()
        rolling = config.pipeline.rolling

        return WalkForwardRLTrainer(
            train_period_months=rolling.window_months - rolling.validation_months,
            val_period_months=rolling.validation_months,
            method="rolling" if rolling.enabled else "anchored",
            min_train_rows=rolling.min_train_rows,
        )
    except (ImportError, FileNotFoundError) as e:
        logger.warning(f"[FASE-5] Could not load SSOT, using defaults: {e}")
        return WalkForwardRLTrainer()


__all__ = [
    "WalkForwardRLWindow",
    "WalkForwardRLReport",
    "WalkForwardRLTrainer",
    "create_walk_forward_trainer_from_ssot",
]
