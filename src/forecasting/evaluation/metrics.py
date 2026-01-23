"""
Evaluation Metrics
==================

Metrics for evaluating forecasting model performance.

@version 1.0.0
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class MetricsResult:
    """Container for evaluation metrics."""
    direction_accuracy: float
    rmse: float
    mae: float
    mape: Optional[float]
    r2: float
    sharpe_ratio: Optional[float]
    max_drawdown: Optional[float]
    sample_count: int


class Metrics:
    """
    Collection of evaluation metrics for forecasting models.

    All metrics are computed comparing predictions against actuals.
    Direction accuracy is the primary metric for trading models.
    """

    @staticmethod
    def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate direction accuracy (DA).

        Measures the percentage of correctly predicted directions.
        This is the most important metric for trading applications.

        Args:
            y_true: Actual returns
            y_pred: Predicted returns

        Returns:
            Direction accuracy as percentage (0-100)
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.0

        # Compare signs (direction)
        true_signs = np.sign(y_true)
        pred_signs = np.sign(y_pred)

        # Count correct directions
        correct = np.sum(true_signs == pred_signs)
        total = len(y_true)

        return (correct / total) * 100 if total > 0 else 0.0

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error.

        Args:
            y_true: Actual values
            y_pred: Predicted values

        Returns:
            RMSE value
        """
        if len(y_true) == 0:
            return 0.0
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error.

        Args:
            y_true: Actual values
            y_pred: Predicted values

        Returns:
            MAE value
        """
        if len(y_true) == 0:
            return 0.0
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
        """
        Calculate Mean Absolute Percentage Error.

        Args:
            y_true: Actual values
            y_pred: Predicted values

        Returns:
            MAPE value or None if division by zero
        """
        if len(y_true) == 0:
            return None

        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return None

        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate R-squared (coefficient of determination).

        Args:
            y_true: Actual values
            y_pred: Predicted values

        Returns:
            R2 value
        """
        if len(y_true) == 0:
            return 0.0

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot == 0:
            return 0.0

        return float(1 - (ss_res / ss_tot))

    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate annualized Sharpe ratio.

        Args:
            returns: Array of period returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year (252 for daily)

        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - (risk_free_rate / periods_per_year)
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns)

        if std_return == 0:
            return 0.0

        return float((mean_return / std_return) * np.sqrt(periods_per_year))

    @staticmethod
    def max_drawdown(equity_curve: np.ndarray) -> float:
        """
        Calculate maximum drawdown.

        Args:
            equity_curve: Cumulative returns or equity values

        Returns:
            Maximum drawdown as percentage
        """
        if len(equity_curve) == 0:
            return 0.0

        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak

        return float(np.min(drawdown) * 100)

    @classmethod
    def compute_all(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        returns: Optional[np.ndarray] = None,
        equity_curve: Optional[np.ndarray] = None,
    ) -> MetricsResult:
        """
        Compute all metrics at once.

        Args:
            y_true: Actual values
            y_pred: Predicted values
            returns: Optional returns for Sharpe calculation
            equity_curve: Optional equity curve for drawdown

        Returns:
            MetricsResult with all computed metrics
        """
        return MetricsResult(
            direction_accuracy=cls.direction_accuracy(y_true, y_pred),
            rmse=cls.rmse(y_true, y_pred),
            mae=cls.mae(y_true, y_pred),
            mape=cls.mape(y_true, y_pred),
            r2=cls.r2(y_true, y_pred),
            sharpe_ratio=cls.sharpe_ratio(returns) if returns is not None else None,
            max_drawdown=cls.max_drawdown(equity_curve) if equity_curve is not None else None,
            sample_count=len(y_true),
        )

    @classmethod
    def to_dict(cls, result: MetricsResult) -> Dict[str, Any]:
        """Convert MetricsResult to dictionary."""
        return {
            'direction_accuracy': result.direction_accuracy,
            'rmse': result.rmse,
            'mae': result.mae,
            'mape': result.mape,
            'r2': result.r2,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'sample_count': result.sample_count,
        }
