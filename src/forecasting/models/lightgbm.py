"""
LightGBM Regression Model
=========================

Light Gradient Boosting Machine for USD/COP forecasting.
Faster training than XGBoost with comparable accuracy.

@version 1.0.0
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

from src.forecasting.models.base import BaseModel

logger = logging.getLogger(__name__)


class LightGBMModel(BaseModel):
    """
    LightGBM Regression model wrapper.

    Attributes:
        - Does not require feature scaling
        - Histogram-based gradient boosting
        - Faster training than XGBoost
        - Supports early stopping
    """

    def __init__(self, name: str = 'lightgbm', params: Optional[Dict[str, Any]] = None):
        default_params = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 1.0,
            'reg_lambda': 1.0,
            'min_child_samples': 20,
            'verbosity': -1,
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
        """Create LightGBM instance."""
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**self.params)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 15,
        **kwargs
    ) -> 'LightGBMModel':
        """
        Fit LightGBM model with optional early stopping.

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

        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['callbacks'] = [
                self._get_early_stopping_callback(early_stopping_rounds)
            ]

        self._model.fit(X, y, **fit_params)
        self._is_fitted = True

        if hasattr(self._model, 'best_iteration_') and self._model.best_iteration_ is not None:
            self._training_metrics['best_iteration'] = self._model.best_iteration_
            logger.debug(f"LightGBM early stopped at iteration {self._model.best_iteration_}")

        self._training_metrics['n_estimators'] = self.params.get('n_estimators')
        logger.debug(f"Fitted LightGBM with {self.params.get('n_estimators', 100)} estimators")
        return self

    def _get_early_stopping_callback(self, rounds: int):
        """Get LightGBM early stopping callback."""
        from lightgbm import early_stopping
        return early_stopping(stopping_rounds=rounds, verbose=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions with variance scaling.

        LightGBM can predict near-zero when regularization is strong.
        """
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
            logger.debug(f"LightGBM: Scaled predictions by {scale_factor:.1f}x")
        elif pred_std <= 1e-8:
            logger.warning(f"LightGBM: Predictions are constant")

        return preds

    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def set_params(self, **params) -> 'LightGBMModel':
        self.params.update(params)
        return self

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores."""
        if not self._is_fitted:
            return None

        importance = self._model.feature_importances_
        if self._feature_names and len(self._feature_names) == len(importance):
            result = dict(zip(self._feature_names, importance.tolist()))
            return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
        return None

    @staticmethod
    def get_optuna_space() -> Dict[str, tuple]:
        """Get Optuna hyperparameter search space."""
        return {
            'n_estimators': ('int', 50, 200),
            'max_depth': ('int', 2, 5),
            'learning_rate': ('float_log', 0.01, 0.1),
            'subsample': ('float', 0.6, 0.9),
            'colsample_bytree': ('float', 0.6, 0.9),
            'reg_alpha': ('float_log', 0.1, 10.0),
            'reg_lambda': ('float_log', 0.1, 10.0),
            'min_child_samples': ('int', 10, 50),
        }

    @staticmethod
    def suggest_params(trial, horizon: int = 1) -> Dict[str, Any]:
        """Suggest parameters for Optuna trial."""
        if horizon >= 15:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 200, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 0.8),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 20.0, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 20, 50),
                'verbosity': -1,
            }
        else:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 0.85),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 20.0, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 30),
                'verbosity': -1,
            }
