# pipeline_limpio_regresion/validation/metrics.py
"""
Custom metrics for early stopping based on Direction Accuracy.

FIXED: Compatible with sklearn API for XGBoost, LightGBM, CatBoost.
"""

import numpy as np
from typing import Tuple, Any, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Direction Accuracy."""
    if len(y_true) == 0:
        return 0.0
    correct = np.sign(y_true) == np.sign(y_pred)
    return float(np.mean(correct))


# =============================================================================
# XGBoost custom metric (sklearn API compatible)
# =============================================================================
def xgb_direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float]:
    """
    XGBoost custom metric for Direction Accuracy (sklearn API).

    When using XGBRegressor with eval_metric, the function receives
    (y_true, y_pred) as numpy arrays directly.

    Note: XGBoost minimizes by default, so we return 1 - DA.
    Higher DA = lower metric value = better.
    """
    da = direction_accuracy(y_true, y_pred)
    # Return (name, value) - XGBoost minimizes, so return 1-DA
    return 'direction_accuracy', 1.0 - da


# =============================================================================
# LightGBM custom metric (sklearn API compatible)
# =============================================================================
def lgb_direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
    """
    LightGBM custom metric for Direction Accuracy (sklearn API).

    When using LGBMRegressor with eval_metric, the function receives
    (y_true, y_pred) as numpy arrays directly.

    Returns: (name, value, is_higher_better)
    """
    da = direction_accuracy(y_true, y_pred)
    # Return (name, value, is_higher_better)
    return 'direction_accuracy', da, True


# =============================================================================
# CatBoost custom metric
# =============================================================================
class CatBoostDirectionAccuracy:
    """
    CatBoost custom metric for Direction Accuracy.
    """

    @staticmethod
    def get_final_error(error, weight):
        return error

    @staticmethod
    def is_max_optimal():
        # Higher DA is better
        return True

    @staticmethod
    def evaluate(approxes, target, weight=None):
        """
        Evaluate Direction Accuracy.

        Args:
            approxes: list of lists (predictions for each dimension)
            target: list of true values
            weight: optional weights

        Returns:
            (score, weight) tuple
        """
        y_pred = np.array(approxes[0])
        y_true = np.array(target)

        da = direction_accuracy(y_true, y_pred)

        return da, 1.0


# =============================================================================
# Aggregate metrics calculation
# =============================================================================
def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all regression metrics for model evaluation.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Filter out NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'r2': np.nan,
            'direction_accuracy': np.nan,
            'mse': np.nan
        }

    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
        'direction_accuracy': float(direction_accuracy(y_true, y_pred)),
        'mse': float(mean_squared_error(y_true, y_pred))
    }
