"""
CatBoost Regression Model
=========================

Categorical Boosting for USD/COP forecasting.
Handles categorical features natively.

@version 1.0.0
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

from src.forecasting.models.base import BaseModel

logger = logging.getLogger(__name__)


class CatBoostModel(BaseModel):
    """
    CatBoost Regression model wrapper.

    Attributes:
        - Does not require feature scaling
        - Native categorical feature support
        - Ordered boosting (reduces overfitting)
        - Supports early stopping
    """

    def __init__(self, name: str = 'catboost', params: Optional[Dict[str, Any]] = None):
        default_params = {
            'iterations': 100,
            'depth': 4,
            'learning_rate': 0.05,
            'l2_leaf_reg': 3.0,
            'verbose': False,
            'allow_writing_files': False,
        }
        merged_params = {**default_params, **(params or {})}
        super().__init__(name, merged_params)

    @property
    def requires_scaling(self) -> bool:
        return False

    @property
    def supports_early_stopping(self) -> bool:
        return True

    def _create_model(self):
        """Create CatBoost instance."""
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**self.params)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 15,
        **kwargs
    ) -> 'CatBoostModel':
        """
        Fit CatBoost model with optional early stopping.

        Args:
            X: Features array
            y: Target array
            X_val: Validation features for early stopping
            y_val: Validation target for early stopping
            early_stopping_rounds: Rounds without improvement to stop

        Returns:
            self for method chaining
        """
        if hasattr(X, 'columns'):
            self._feature_names = list(X.columns)
        elif hasattr(X, 'shape') and len(X.shape) > 1:
            self._feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        self._model = self._create_model()

        fit_params = {'verbose': False}
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = (X_val, y_val)
            fit_params['early_stopping_rounds'] = early_stopping_rounds

        self._model.fit(X, y, **fit_params)
        self._is_fitted = True

        if hasattr(self._model, 'best_iteration_') and self._model.best_iteration_ is not None:
            self._training_metrics['best_iteration'] = self._model.best_iteration_
            logger.debug(f"CatBoost early stopped at iteration {self._model.best_iteration_}")

        self._training_metrics['iterations'] = self.params.get('iterations')
        logger.debug(f"Fitted CatBoost with {self.params.get('iterations', 100)} iterations")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions with variance scaling."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        preds = self._model.predict(X)

        # Variance scaling
        pred_std = np.std(preds)
        pred_mean = np.mean(preds)
        min_pred_std = 0.005
        max_scale_factor = 10.0

        if pred_std < min_pred_std and pred_std > 1e-8:
            raw_scale_factor = min_pred_std / pred_std
            scale_factor = min(raw_scale_factor, max_scale_factor)
            preds = pred_mean + (preds - pred_mean) * scale_factor
            logger.debug(f"CatBoost: Scaled predictions by {scale_factor:.1f}x")
        elif pred_std <= 1e-8:
            logger.warning(f"CatBoost: Predictions are constant")

        return preds

    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def set_params(self, **params) -> 'CatBoostModel':
        self.params.update(params)
        return self

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores."""
        if not self._is_fitted:
            return None

        importance = self._model.get_feature_importance()
        if self._feature_names and len(self._feature_names) == len(importance):
            result = dict(zip(self._feature_names, importance.tolist()))
            return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
        return None

    @staticmethod
    def get_optuna_space() -> Dict[str, tuple]:
        """Get Optuna hyperparameter search space."""
        return {
            'iterations': ('int', 50, 200),
            'depth': ('int', 2, 5),
            'learning_rate': ('float_log', 0.01, 0.1),
            'l2_leaf_reg': ('float_log', 1.0, 10.0),
            'subsample': ('float', 0.6, 0.9),
            'colsample_bylevel': ('float', 0.6, 0.9),
        }

    @staticmethod
    def suggest_params(trial, horizon: int = 1) -> Dict[str, Any]:
        """Suggest parameters for Optuna trial."""
        if horizon >= 15:
            return {
                'iterations': trial.suggest_int('iterations', 200, 500),
                'depth': trial.suggest_int('depth', 3, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 20.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 0.8),
                'verbose': False,
                'allow_writing_files': False,
            }
        else:
            return {
                'iterations': trial.suggest_int('iterations', 100, 300),
                'depth': trial.suggest_int('depth', 3, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 0.85),
                'verbose': False,
                'allow_writing_files': False,
            }
