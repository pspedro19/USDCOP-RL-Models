# pipeline_limpio_regresion/forecasting/backtest.py
"""
Backtesting module for model evaluation.

Implements walk-forward validation and performance analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
from datetime import datetime

from ..core.config import HORIZONS
from ..evaluation.metrics import calculate_all_metrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from backtesting a model."""
    model_name: str
    horizon: int
    n_predictions: int
    metrics: Dict[str, float]
    predictions: np.ndarray
    actuals: np.ndarray
    dates: List[datetime]
    cumulative_return: float
    max_drawdown: float


class Backtester:
    """
    Performs walk-forward backtesting for regression models.

    Features:
    - Walk-forward validation
    - Performance metrics calculation
    - Drawdown analysis
    - Strategy returns
    """

    def __init__(self, initial_train_pct: float = 0.5, step_size: int = 1):
        self.initial_train_pct = initial_train_pct
        self.step_size = step_size

        self.results: List[BacktestResult] = []

    def backtest_model(
        self,
        model_class: Any,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str = "Model",
        horizon: int = 1,
        retrain_frequency: int = 22  # Retrain monthly
    ) -> BacktestResult:
        """
        Run walk-forward backtest for a model.

        Args:
            model_class: Model class to instantiate
            X: Features DataFrame
            y: Target series
            model_name: Model name
            horizon: Prediction horizon
            retrain_frequency: How often to retrain

        Returns:
            BacktestResult with metrics and predictions
        """
        n = len(X)
        initial_train = int(n * self.initial_train_pct)

        predictions = []
        actuals = []
        dates = []

        model = None
        last_train_idx = 0

        for i in range(initial_train, n - horizon, self.step_size):
            # Retrain if needed
            if model is None or (i - last_train_idx) >= retrain_frequency:
                X_train = X.iloc[:i].values
                y_train = y.iloc[:i].values

                # Clean NaN
                mask = np.isfinite(y_train)
                X_train_clean = X_train[mask]
                y_train_clean = y_train[mask]

                if len(X_train_clean) > 10:
                    model = model_class()
                    model.fit(X_train_clean, y_train_clean)
                    last_train_idx = i

            if model is None:
                continue

            # Predict
            X_test = X.iloc[i:i+1].values
            y_pred = model.predict(X_test)[0]

            # Get actual
            y_actual = y.iloc[i + horizon] if (i + horizon) < n else y.iloc[i]

            if np.isfinite(y_pred) and np.isfinite(y_actual):
                predictions.append(y_pred)
                actuals.append(y_actual)
                dates.append(X.index[i])

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Calculate metrics
        metrics = calculate_all_metrics(actuals, predictions)

        # Calculate strategy returns
        strategy_returns = actuals * np.sign(predictions)
        cumulative_return = np.sum(strategy_returns)

        # Calculate max drawdown
        cumsum = np.cumsum(strategy_returns)
        running_max = np.maximum.accumulate(cumsum)
        drawdown = running_max - cumsum
        max_drawdown = np.max(drawdown)

        result = BacktestResult(
            model_name=model_name,
            horizon=horizon,
            n_predictions=len(predictions),
            metrics=metrics,
            predictions=predictions,
            actuals=actuals,
            dates=dates,
            cumulative_return=cumulative_return,
            max_drawdown=max_drawdown
        )

        self.results.append(result)
        return result

    def backtest_all_models(
        self,
        model_configs: Dict[str, Any],
        X: pd.DataFrame,
        y_by_horizon: Dict[int, pd.Series],
        X_scaled: pd.DataFrame = None
    ) -> List[BacktestResult]:
        """
        Backtest all models across all horizons.

        Args:
            model_configs: Dictionary of model_name -> model_class
            X: Features DataFrame
            y_by_horizon: Target by horizon
            X_scaled: Scaled features for models that require it

        Returns:
            List of BacktestResults
        """
        results = []

        for model_name, model_class in model_configs.items():
            for horizon, y in y_by_horizon.items():
                logger.info(f"Backtesting {model_name} h={horizon}")

                # Use scaled features if needed
                features = X_scaled if model_name == 'ridge' and X_scaled is not None else X

                try:
                    result = self.backtest_model(
                        model_class, features, y,
                        model_name=model_name,
                        horizon=horizon
                    )
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Backtest failed for {model_name} h={horizon}: {e}")

        return results

    def get_results_df(self) -> pd.DataFrame:
        """Get all results as DataFrame."""
        rows = []
        for r in self.results:
            row = {
                'model': r.model_name,
                'horizon': r.horizon,
                'n_predictions': r.n_predictions,
                'cumulative_return': r.cumulative_return,
                'max_drawdown': r.max_drawdown,
                **r.metrics
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def print_summary(self):
        """Print backtest summary."""
        print("\n" + "=" * 80)
        print("BACKTEST SUMMARY")
        print("=" * 80)

        df = self.get_results_df()
        if df.empty:
            print("No results")
            return

        print(f"\n{'Model':<15}{'H':>4}{'N':>8}{'DA':>10}{'Return':>12}{'MaxDD':>10}")
        print("-" * 60)

        for _, row in df.iterrows():
            print(
                f"{row['model']:<15}{row['horizon']:>4}{row['n_predictions']:>8}"
                f"{row.get('direction_accuracy', 0)*100:>9.1f}%"
                f"{row['cumulative_return']*100:>11.2f}%"
                f"{row['max_drawdown']*100:>9.2f}%"
            )

        # Summary stats
        print("\n--- SUMMARY BY MODEL ---")
        summary = df.groupby('model').agg({
            'direction_accuracy': 'mean',
            'cumulative_return': 'mean',
            'max_drawdown': 'mean'
        })

        for model, row in summary.iterrows():
            print(
                f"{model}: DA={row['direction_accuracy']*100:.1f}% | "
                f"Avg Return={row['cumulative_return']*100:.2f}% | "
                f"Avg MaxDD={row['max_drawdown']*100:.2f}%"
            )

        print("=" * 80)


def run_backtest(
    X: pd.DataFrame,
    y_by_horizon: Dict[int, pd.Series],
    model_configs: Dict[str, Any],
    X_scaled: pd.DataFrame = None,
    initial_train_pct: float = 0.5
) -> Tuple[List[BacktestResult], pd.DataFrame]:
    """
    Convenience function to run backtest.

    Args:
        X: Features
        y_by_horizon: Targets by horizon
        model_configs: Model configurations
        X_scaled: Scaled features
        initial_train_pct: Initial training percentage

    Returns:
        Tuple of (results_list, results_dataframe)
    """
    backtester = Backtester(initial_train_pct=initial_train_pct)

    results = backtester.backtest_all_models(
        model_configs, X, y_by_horizon, X_scaled
    )

    backtester.print_summary()

    return results, backtester.get_results_df()
