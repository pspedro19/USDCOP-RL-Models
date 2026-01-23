"""
Backtest Engine
===============

Historical backtesting for forecasting models.

@version 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import logging

from src.forecasting.models.base import BaseModel
from src.forecasting.evaluation.metrics import Metrics, MetricsResult

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results."""
    model_name: str
    horizon: int
    metrics: MetricsResult
    predictions: np.ndarray
    actuals: np.ndarray
    dates: Optional[List] = None
    feature_importance: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BacktestEngine:
    """
    Engine for running historical backtests on forecasting models.

    Supports:
    - Train/test split
    - Walk-forward validation
    - Multiple horizons
    - Feature importance tracking
    """

    def __init__(
        self,
        train_ratio: float = 0.8,
        use_scaling: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize backtest engine.

        Args:
            train_ratio: Ratio of data to use for training
            use_scaling: Whether to scale features for models that need it
            verbose: Whether to print progress
        """
        self.train_ratio = train_ratio
        self.use_scaling = use_scaling
        self.verbose = verbose
        self._scaler = None

    def run(
        self,
        model: BaseModel,
        X: np.ndarray,
        y: np.ndarray,
        horizon: int = 1,
        dates: Optional[List] = None,
    ) -> BacktestResult:
        """
        Run a single backtest.

        Args:
            model: Model instance to evaluate
            X: Feature matrix
            y: Target values
            horizon: Prediction horizon
            dates: Optional date index

        Returns:
            BacktestResult with metrics and predictions
        """
        # Split data
        split_idx = int(len(X) * self.train_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        test_dates = dates[split_idx:] if dates is not None else None

        # Scale if needed
        if self.use_scaling and model.requires_scaling:
            X_train, X_test = self._scale_features(X_train, X_test)

        # Train model
        if self.verbose:
            logger.info(f"Training {model.name} on {len(X_train)} samples...")

        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Compute metrics
        metrics = Metrics.compute_all(y_test, predictions)

        if self.verbose:
            logger.info(f"  DA: {metrics.direction_accuracy:.2f}%, RMSE: {metrics.rmse:.4f}")

        return BacktestResult(
            model_name=model.name,
            horizon=horizon,
            metrics=metrics,
            predictions=predictions,
            actuals=y_test,
            dates=test_dates,
            feature_importance=model.get_feature_importance(),
            metadata={
                'train_size': len(X_train),
                'test_size': len(X_test),
                'train_ratio': self.train_ratio,
            },
        )

    def run_multiple_horizons(
        self,
        model: BaseModel,
        X: np.ndarray,
        targets: Dict[int, np.ndarray],
        dates: Optional[List] = None,
    ) -> Dict[int, BacktestResult]:
        """
        Run backtests for multiple horizons.

        Args:
            model: Model instance (will be re-trained for each horizon)
            X: Feature matrix
            targets: Dict mapping horizon to target values
            dates: Optional date index

        Returns:
            Dict mapping horizon to BacktestResult
        """
        results = {}

        for horizon, y in targets.items():
            if self.verbose:
                logger.info(f"\n=== Horizon {horizon} ===")

            # Create fresh model instance for each horizon
            from src.forecasting.models import ModelFactory
            fresh_model = ModelFactory.create(model.name, params=model.get_params())

            result = self.run(fresh_model, X, y, horizon=horizon, dates=dates)
            results[horizon] = result

        return results

    def _scale_features(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler.

        Fits on training data, transforms both train and test.

        Args:
            X_train: Training features
            X_test: Test features

        Returns:
            Tuple of (scaled_train, scaled_test)
        """
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled = self._scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

    @staticmethod
    def results_to_dataframe(results: Dict[int, BacktestResult]) -> pd.DataFrame:
        """
        Convert multiple backtest results to a DataFrame.

        Args:
            results: Dict mapping horizon to BacktestResult

        Returns:
            DataFrame with metrics for each horizon
        """
        rows = []
        for horizon, result in results.items():
            row = {
                'model_name': result.model_name,
                'horizon': horizon,
                'direction_accuracy': result.metrics.direction_accuracy,
                'rmse': result.metrics.rmse,
                'mae': result.metrics.mae,
                'r2': result.metrics.r2,
                'sample_count': result.metrics.sample_count,
            }
            rows.append(row)

        return pd.DataFrame(rows)
