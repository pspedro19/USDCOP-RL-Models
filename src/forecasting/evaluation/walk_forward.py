"""
Walk-Forward Validation
=======================

Expanding window validation for time series models.

@version 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Generator, Tuple
from dataclasses import dataclass, field
import logging

from src.forecasting.models.base import BaseModel
from src.forecasting.evaluation.metrics import Metrics, MetricsResult

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Result for a single validation fold."""
    fold_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    metrics: MetricsResult
    predictions: np.ndarray
    actuals: np.ndarray


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward validation results."""
    model_name: str
    horizon: int
    n_folds: int
    fold_results: List[FoldResult]
    avg_metrics: MetricsResult
    std_metrics: Dict[str, float]
    all_predictions: np.ndarray
    all_actuals: np.ndarray


class WalkForwardValidator:
    """
    Walk-forward (expanding window) cross-validation for time series.

    Unlike standard k-fold CV, walk-forward:
    - Respects temporal ordering
    - Uses expanding training windows
    - Avoids look-ahead bias

    Example splits with 5 folds:
    - Fold 1: Train [0-60%], Test [60-68%]
    - Fold 2: Train [0-68%], Test [68-76%]
    - Fold 3: Train [0-76%], Test [76-84%]
    - Fold 4: Train [0-84%], Test [84-92%]
    - Fold 5: Train [0-92%], Test [92-100%]
    """

    def __init__(
        self,
        n_folds: int = 5,
        initial_train_ratio: float = 0.6,
        gap: int = 0,
        use_scaling: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize walk-forward validator.

        Args:
            n_folds: Number of validation folds
            initial_train_ratio: Ratio of data for initial training
            gap: Number of samples to skip between train and test (purging)
            use_scaling: Whether to scale features
            verbose: Whether to print progress
        """
        self.n_folds = n_folds
        self.initial_train_ratio = initial_train_ratio
        self.gap = gap
        self.use_scaling = use_scaling
        self.verbose = verbose
        self._scaler = None

    def get_splits(
        self,
        n_samples: int,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test split indices.

        Args:
            n_samples: Total number of samples

        Yields:
            Tuples of (train_indices, test_indices)
        """
        initial_train_size = int(n_samples * self.initial_train_ratio)
        test_size = (n_samples - initial_train_size) // self.n_folds

        for fold in range(self.n_folds):
            train_end = initial_train_size + (fold * test_size)
            test_start = train_end + self.gap
            test_end = min(test_start + test_size, n_samples)

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx

    def validate(
        self,
        model: BaseModel,
        X: np.ndarray,
        y: np.ndarray,
        horizon: int = 1,
    ) -> WalkForwardResult:
        """
        Run walk-forward validation.

        Args:
            model: Model instance (will be re-trained for each fold)
            X: Feature matrix
            y: Target values
            horizon: Prediction horizon

        Returns:
            WalkForwardResult with aggregated metrics
        """
        from src.forecasting.models import ModelFactory

        fold_results = []
        all_predictions = []
        all_actuals = []

        for fold_idx, (train_idx, test_idx) in enumerate(self.get_splits(len(X))):
            if self.verbose:
                logger.info(f"Fold {fold_idx + 1}/{self.n_folds}: "
                          f"Train [{train_idx[0]}-{train_idx[-1]}], "
                          f"Test [{test_idx[0]}-{test_idx[-1]}]")

            # Split data
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Scale if needed
            if self.use_scaling and model.requires_scaling:
                X_train, X_test = self._scale_features(X_train, X_test)

            # Create fresh model instance
            fresh_model = ModelFactory.create(model.name, params=model.get_params())

            # Train and predict
            fresh_model.fit(X_train, y_train)
            predictions = fresh_model.predict(X_test)

            # Compute fold metrics
            fold_metrics = Metrics.compute_all(y_test, predictions)

            fold_result = FoldResult(
                fold_idx=fold_idx,
                train_start=int(train_idx[0]),
                train_end=int(train_idx[-1]),
                test_start=int(test_idx[0]),
                test_end=int(test_idx[-1]),
                metrics=fold_metrics,
                predictions=predictions,
                actuals=y_test,
            )
            fold_results.append(fold_result)
            all_predictions.extend(predictions)
            all_actuals.extend(y_test)

            if self.verbose:
                logger.info(f"  DA: {fold_metrics.direction_accuracy:.2f}%, "
                          f"RMSE: {fold_metrics.rmse:.4f}")

        # Aggregate metrics
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)

        avg_metrics = Metrics.compute_all(all_actuals, all_predictions)

        # Compute std of metrics across folds
        das = [f.metrics.direction_accuracy for f in fold_results]
        rmses = [f.metrics.rmse for f in fold_results]
        std_metrics = {
            'direction_accuracy_std': float(np.std(das)),
            'rmse_std': float(np.std(rmses)),
        }

        return WalkForwardResult(
            model_name=model.name,
            horizon=horizon,
            n_folds=self.n_folds,
            fold_results=fold_results,
            avg_metrics=avg_metrics,
            std_metrics=std_metrics,
            all_predictions=all_predictions,
            all_actuals=all_actuals,
        )

    def _scale_features(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features using StandardScaler."""
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

    @staticmethod
    def results_to_dataframe(result: WalkForwardResult) -> pd.DataFrame:
        """Convert fold results to DataFrame."""
        rows = []
        for fold in result.fold_results:
            row = {
                'model_name': result.model_name,
                'horizon': result.horizon,
                'fold': fold.fold_idx + 1,
                'train_samples': fold.train_end - fold.train_start + 1,
                'test_samples': fold.test_end - fold.test_start + 1,
                'direction_accuracy': fold.metrics.direction_accuracy,
                'rmse': fold.metrics.rmse,
                'mae': fold.metrics.mae,
            }
            rows.append(row)

        return pd.DataFrame(rows)
