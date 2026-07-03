"""
Bayesian Ridge Regression Model
===============================

Bayesian linear regression with uncertainty estimation.

@version 1.0.0
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.linear_model import BayesianRidge
import logging

from src.forecasting.models.base import BaseModel

logger = logging.getLogger(__name__)


class BayesianRidgeModel(BaseModel):
    """
    Bayesian Ridge Regression model wrapper.

    Attributes:
        - Requires feature scaling
        - Provides uncertainty estimates
        - Automatic regularization via evidence maximization
    """

    def __init__(self, name: str = 'bayesian_ridge', params: Optional[Dict[str, Any]] = None):
        default_params = {
            'alpha_1': 1e-6,
            'alpha_2': 1e-6,
            'lambda_1': 1e-6,
            'lambda_2': 1e-6,
            'max_iter': 300,
            'fit_intercept': True,
        }
        merged_params = {**default_params, **(params or {})}
        super().__init__(name, merged_params)

    @property
    def requires_scaling(self) -> bool:
        return True

    @property
    def supports_early_stopping(self) -> bool:
        return False

    def _create_model(self) -> BayesianRidge:
        """Create sklearn BayesianRidge instance."""
        return BayesianRidge(**self.params)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> 'BayesianRidgeModel':
        """Fit Bayesian Ridge model."""
        if hasattr(X, 'columns'):
            self._feature_names = list(X.columns)
        elif hasattr(X, 'shape') and len(X.shape) > 1:
            self._feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        self._model = self._create_model()
        self._model.fit(X, y)
        self._is_fitted = True

        self._training_metrics = {
            'alpha': self._model.alpha_,
            'lambda': self._model.lambda_,
            'n_iter': self._model.n_iter_,
        }

        logger.debug(f"Fitted BayesianRidge with {self._model.n_iter_} iterations")
        return self

    def predict(self, X: np.ndarray, return_std: bool = False) -> np.ndarray:
        """
        Generate predictions with optional uncertainty.

        Args:
            X: Features array
            return_std: If True, return (predictions, std)

        Returns:
            Predictions array, or (predictions, std) if return_std=True
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if return_std:
            return self._model.predict(X, return_std=True)
        return self._model.predict(X)

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions with uncertainty bounds.

        Returns:
            Tuple of (predictions, standard_deviation)
        """
        return self.predict(X, return_std=True)

    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def set_params(self, **params) -> 'BayesianRidgeModel':
        self.params.update(params)
        return self

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance based on coefficient magnitude."""
        if not self._is_fitted:
            return None

        coefs = self._model.coef_
        if self._feature_names and len(self._feature_names) == len(coefs):
            importance = {k: abs(v) for k, v in zip(self._feature_names, coefs)}
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        return None

    @staticmethod
    def suggest_params(trial, horizon: int = 1) -> Dict[str, Any]:
        """Suggest parameters for Optuna trial."""
        return {
            'alpha_1': trial.suggest_float('alpha_1', 1e-8, 1e-4, log=True),
            'alpha_2': trial.suggest_float('alpha_2', 1e-8, 1e-4, log=True),
            'lambda_1': trial.suggest_float('lambda_1', 1e-8, 1e-4, log=True),
            'lambda_2': trial.suggest_float('lambda_2', 1e-8, 1e-4, log=True),
            'max_iter': 300,
            'fit_intercept': True,
        }
