"""
XGBoost Regression Model
========================

Gradient boosting model for USD/COP forecasting.
Does not require feature scaling.

@version 1.0.0
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

from src.forecasting.models.base import BaseModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """
    XGBoost Regression model wrapper.

    Attributes:
        - Does not require feature scaling
        - Gradient boosting with regularization
        - Supports early stopping
        - Good for non-linear relationships
    """

    def __init__(self, name: str = 'xgboost', params: Optional[Dict[str, Any]] = None):
        default_params = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 1.0,
            'reg_lambda': 1.0,
            'verbosity': 0,
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
        """Create XGBoost instance."""
        from xgboost import XGBRegressor
        return XGBRegressor(**self.params)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 15,
        **kwargs
    ) -> 'XGBoostModel':
        """
        Fit XGBoost model with optional early stopping.

        Args:
            X: Features array
            y: Target array
            X_val: Validation features for early stopping
            y_val: Validation target for early stopping
            early_stopping_rounds: Rounds without improvement to stop

        Returns:
            self for method chaining
        """
        # Store feature names
        if hasattr(X, 'columns'):
            self._feature_names = list(X.columns)
        elif hasattr(X, 'shape') and len(X.shape) > 1:
            self._feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # Adaptive early stopping based on learning rate
        lr = self.params.get('learning_rate', 0.1)
        if lr < 0.03:
            effective_early_stopping = max(50, int(early_stopping_rounds * 3))
        else:
            effective_early_stopping = early_stopping_rounds

        self._model = self._create_model()

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            self._model.set_params(early_stopping_rounds=effective_early_stopping)
            self._model.fit(X, y, eval_set=eval_set, verbose=False)

            if hasattr(self._model, 'best_iteration') and self._model.best_iteration is not None:
                logger.debug(f"XGBoost early stopped at iteration {self._model.best_iteration}")
                self._training_metrics['best_iteration'] = self._model.best_iteration
        else:
            self._model.fit(X, y)

        self._is_fitted = True
        self._training_metrics['n_estimators'] = self.params.get('n_estimators')
        self._training_metrics['n_features'] = X.shape[1] if hasattr(X, 'shape') else None

        logger.debug(f"Fitted XGBoost with {self.params.get('n_estimators', 100)} estimators")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions with variance scaling.

        XGBoost can predict near-zero when regularization is strong.
        This method scales predictions to ensure minimum variance.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        preds = self._model.predict(X)

        # Variance scaling to prevent collapse
        pred_std = np.std(preds)
        pred_mean = np.mean(preds)
        min_pred_std = 0.005
        max_scale_factor = 10.0

        if pred_std < min_pred_std and pred_std > 1e-8:
            raw_scale_factor = min_pred_std / pred_std
            scale_factor = min(raw_scale_factor, max_scale_factor)
            preds = pred_mean + (preds - pred_mean) * scale_factor

            if raw_scale_factor > max_scale_factor:
                logger.warning(f"XGBoost: Scale factor capped at {max_scale_factor}x")
            else:
                logger.debug(f"XGBoost: Scaled predictions by {scale_factor:.1f}x")
        elif pred_std <= 1e-8:
            logger.warning(f"XGBoost: Predictions are constant (std={pred_std:.2e})")

        return preds

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.params.copy()

    def set_params(self, **params) -> 'XGBoostModel':
        """Set model parameters."""
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
            'min_child_weight': ('int', 3, 10),
            'gamma': ('float_log', 0.01, 1.0),
        }

    @staticmethod
    def suggest_params(trial, horizon: int = 1) -> Dict[str, Any]:
        """
        Suggest parameters for Optuna trial.

        Args:
            trial: Optuna trial object
            horizon: Prediction horizon

        Returns:
            Dictionary of suggested parameters
        """
        if horizon >= 15:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 200, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 0.8),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 20.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
                'gamma': trial.suggest_float('gamma', 0.01, 0.2, log=True),
                'verbosity': 0,
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
                'min_child_weight': trial.suggest_int('min_child_weight', 3, 15),
                'gamma': trial.suggest_float('gamma', 0.01, 0.5, log=True),
                'verbosity': 0,
            }
