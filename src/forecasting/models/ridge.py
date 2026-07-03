"""
Ridge Regression Model
======================

L2-regularized linear regression for USD/COP forecasting.
Best performing model with DA ~60.6%.

@version 1.0.0
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import Ridge
import logging

from src.forecasting.models.base import BaseModel

logger = logging.getLogger(__name__)


class RidgeModel(BaseModel):
    """
    Ridge Regression model wrapper.

    Attributes:
        - Requires feature scaling (StandardScaler recommended)
        - L2 regularization prevents overfitting
        - Best for multi-horizon forecasting
        - Interpretable via coefficients
    """

    def __init__(self, name: str = 'ridge', params: Optional[Dict[str, Any]] = None):
        default_params = {'alpha': 1.0, 'fit_intercept': True}
        merged_params = {**default_params, **(params or {})}
        super().__init__(name, merged_params)

    @property
    def requires_scaling(self) -> bool:
        return True

    @property
    def supports_early_stopping(self) -> bool:
        return False

    def _create_model(self) -> Ridge:
        """Create sklearn Ridge instance."""
        return Ridge(**self.params)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> 'RidgeModel':
        """
        Fit Ridge model.

        Args:
            X: Features array (should be scaled)
            y: Target array
            X_val: Ignored (no early stopping)
            y_val: Ignored (no early stopping)

        Returns:
            self for method chaining
        """
        # Store feature names if X is a DataFrame
        if hasattr(X, 'columns'):
            self._feature_names = list(X.columns)
        elif hasattr(X, 'shape') and len(X.shape) > 1:
            self._feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        self._model = self._create_model()
        self._model.fit(X, y)
        self._is_fitted = True

        # Store training metrics
        self._training_metrics = {
            'alpha': self.params.get('alpha'),
            'n_features': X.shape[1] if hasattr(X, 'shape') else None,
            'n_samples': len(y),
        }

        logger.debug(f"Fitted Ridge with alpha={self.params.get('alpha', 1.0)}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Features array (should be scaled)

        Returns:
            Predictions array
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._model.predict(X)

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.params.copy()

    def set_params(self, **params) -> 'RidgeModel':
        """Set model parameters."""
        self.params.update(params)
        return self

    def get_coefficients(self) -> Dict[str, float]:
        """
        Get Ridge coefficients for interpretation.

        Returns:
            Dictionary mapping feature names to coefficient values.
        """
        if not self._is_fitted:
            return {}

        coefs = self._model.coef_
        if hasattr(coefs, 'ndim') and coefs.ndim > 1:
            coefs = coefs.flatten()

        if self._feature_names and len(self._feature_names) == len(coefs):
            return dict(zip(self._feature_names, coefs.tolist()))
        return {'coefficients': coefs.tolist()}

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance based on absolute coefficient values.

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        coefs = self.get_coefficients()
        if not coefs or 'coefficients' in coefs:
            return None

        importance = {k: abs(v) for k, v in coefs.items()}
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def get_intercept(self) -> float:
        """Get model intercept."""
        if not self._is_fitted:
            return 0.0
        return float(self._model.intercept_)

    @staticmethod
    def get_optuna_space() -> Dict[str, tuple]:
        """Get Optuna hyperparameter search space."""
        return {
            'alpha': ('float_log', 1e-4, 100.0)
        }

    @staticmethod
    def suggest_params(trial, horizon: int = 1) -> Dict[str, Any]:
        """
        Suggest parameters for Optuna trial.

        Args:
            trial: Optuna trial object
            horizon: Prediction horizon (longer horizons need more regularization)

        Returns:
            Dictionary of suggested parameters
        """
        if horizon <= 5:
            alpha_min, alpha_max = 0.1, 50.0
        elif horizon <= 15:
            alpha_min, alpha_max = 1.0, 100.0
        else:
            alpha_min, alpha_max = 5.0, 200.0

        return {
            'alpha': trial.suggest_float('alpha', alpha_min, alpha_max, log=True),
            'fit_intercept': True
        }
