"""
Hybrid Models
=============

Hybrid models that combine boosting with linear regression.
The ensemble uses weighted average of predictions.

@version 1.0.0
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

from .base import BaseModel

logger = logging.getLogger(__name__)


class HybridBaseModel(BaseModel):
    """
    Base class for hybrid models.

    Combines a boosting model with Ridge regression.
    Final prediction = (1-alpha) * boosting + alpha * linear
    """

    def __init__(
        self,
        name: str,
        boosting_class: type,
        params: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name=name, params=params or {})

        self._boosting_class = boosting_class
        self._boosting_model = None
        self._linear_model = None
        self._scaler = None

        # Hybrid-specific params
        self._default_params = {
            'alpha': 0.3,  # Weight for linear model
            'linear_alpha': 1.0,  # Ridge regularization
        }
        self.params = {**self._default_params, **self.params}

    @property
    def requires_scaling(self) -> bool:
        """Hybrid models use scaling for linear component."""
        return True

    @property
    def supports_early_stopping(self) -> bool:
        """Boosting component supports early stopping."""
        return True

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> 'HybridBaseModel':
        """
        Train both models.

        Args:
            X: Training features
            y: Training targets
            X_val: Validation features for early stopping
            y_val: Validation targets for early stopping

        Returns:
            self for chaining
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge

        # Scale features for linear model
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Train boosting model (filter out hybrid-specific params)
        hybrid_keys = {'alpha', 'linear_alpha'}
        boosting_params = {k: v for k, v in self.params.items() if k not in hybrid_keys}
        self._boosting_model = self._boosting_class(
            name=f"{self.name}_boosting",
            params=boosting_params
        )

        if X_val is not None and y_val is not None:
            self._boosting_model.fit(X, y, X_val, y_val, **kwargs)
        else:
            self._boosting_model.fit(X, y, **kwargs)

        # Train linear model on scaled features
        self._linear_model = Ridge(alpha=self.params.get('linear_alpha', 1.0))
        self._linear_model.fit(X_scaled, y)

        self._is_fitted = True

        logger.info(f"Hybrid model fitted: alpha={self.params['alpha']}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions.

        Args:
            X: Features

        Returns:
            Weighted average of predictions
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Boosting prediction (unscaled)
        boost_pred = self._boosting_model.predict(X)

        # Linear prediction (scaled)
        X_scaled = self._scaler.transform(X)
        linear_pred = self._linear_model.predict(X_scaled)

        # Weighted ensemble
        alpha = self.params.get('alpha', 0.3)
        return (1 - alpha) * boost_pred + alpha * linear_pred

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.params.copy()

    def set_params(self, **params) -> 'HybridBaseModel':
        """Set model parameters."""
        self.params.update(params)
        return self

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from boosting component."""
        if self._boosting_model is not None:
            return self._boosting_model.get_feature_importance()
        return None


class XGBoostHybridModel(HybridBaseModel):
    """XGBoost + Ridge hybrid model."""

    def __init__(self, name: str = "hybrid_xgboost", params: Optional[Dict[str, Any]] = None):
        from .xgboost import XGBoostModel
        super().__init__(name=name, boosting_class=XGBoostModel, params=params)


class LightGBMHybridModel(HybridBaseModel):
    """LightGBM + Ridge hybrid model."""

    def __init__(self, name: str = "hybrid_lightgbm", params: Optional[Dict[str, Any]] = None):
        from .lightgbm import LightGBMModel
        super().__init__(name=name, boosting_class=LightGBMModel, params=params)


class CatBoostHybridModel(HybridBaseModel):
    """CatBoost + Ridge hybrid model."""

    def __init__(self, name: str = "hybrid_catboost", params: Optional[Dict[str, Any]] = None):
        from .catboost import CatBoostModel
        super().__init__(name=name, boosting_class=CatBoostModel, params=params)


__all__ = [
    'HybridBaseModel',
    'XGBoostHybridModel',
    'LightGBMHybridModel',
    'CatBoostHybridModel',
]
