"""
ARD Regression Model
====================

Automatic Relevance Determination regression for forecasting.
ARD automatically prunes features with low relevance.

@version 1.0.0
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

from .base import BaseModel

logger = logging.getLogger(__name__)


class ARDModel(BaseModel):
    """
    ARD (Automatic Relevance Determination) Regression.

    ARD is a sparse Bayesian model that automatically identifies
    and down-weights irrelevant features.

    Advantages:
    - Automatic feature selection
    - Uncertainty estimation
    - Robust to multicollinearity
    """

    def __init__(self, name: str = "ard", params: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, params=params or {})

        # Default parameters
        self._default_params = {
            'max_iter': 300,
            'tol': 1e-3,
            'alpha_1': 1e-6,
            'alpha_2': 1e-6,
            'lambda_1': 1e-6,
            'lambda_2': 1e-6,
            'fit_intercept': True,
            'verbose': False,
        }

        # Merge with provided params
        self.params = {**self._default_params, **self.params}
        self._scaler = None

    @property
    def requires_scaling(self) -> bool:
        """ARD requires scaled features."""
        return True

    @property
    def supports_early_stopping(self) -> bool:
        """ARD uses iterative fitting but not early stopping."""
        return False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> 'ARDModel':
        """
        Train the ARD model.

        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
            X_val: Validation features (unused)
            y_val: Validation targets (unused)

        Returns:
            self for chaining
        """
        from sklearn.linear_model import ARDRegression
        from sklearn.preprocessing import StandardScaler

        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Create and fit model
        self._model = ARDRegression(
            max_iter=self.params['max_iter'],
            tol=self.params['tol'],
            alpha_1=self.params['alpha_1'],
            alpha_2=self.params['alpha_2'],
            lambda_1=self.params['lambda_1'],
            lambda_2=self.params['lambda_2'],
            fit_intercept=self.params['fit_intercept'],
            verbose=self.params['verbose'],
        )

        self._model.fit(X_scaled, y)
        self._is_fitted = True

        # Store training metrics
        self._training_metrics = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_iterations': self._model.n_iter_,
        }

        logger.info(f"ARD fitted: {self._model.n_iter_} iterations")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predictions (n_samples,)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.params.copy()

    def set_params(self, **params) -> 'ARDModel':
        """Set model parameters."""
        self.params.update(params)
        return self

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance based on precision of weights.

        In ARD, lambda_ represents the precision of each weight.
        Higher precision means the feature is more relevant.
        """
        if not self._is_fitted or self._feature_names is None:
            return None

        # Get lambdas (precision of weights)
        lambdas = self._model.lambda_

        # Convert to importance (inverse of lambda)
        importance = 1.0 / (lambdas + 1e-10)
        importance = importance / importance.sum()  # Normalize

        return dict(zip(self._feature_names, importance))
