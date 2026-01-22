"""
Walk-Forward Backtesting Module
===============================

Implements rolling window validation and walk-forward optimization
for more realistic backtesting of trading strategies.

P1: Walk-Forward Backtesting

Features:
- Rolling window validation
- Out-of-sample testing periods
- Statistical significance reporting
- Anchored vs. rolling walk-forward
- Performance stability analysis

Author: Trading Team
Version: 1.0.0
Date: 2026-01-17
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable, Tuple, Generator
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WalkForwardMethod(Enum):
    """Walk-forward validation methods."""
    ROLLING = "rolling"      # Fixed-size rolling window
    ANCHORED = "anchored"    # Expanding window (anchored to start)
    SLIDING = "sliding"      # Sliding window with overlap


@dataclass
class WalkForwardWindow:
    """A single walk-forward window (training + testing period)."""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    # Training data size
    train_samples: int = 0

    # Testing data size
    test_samples: int = 0

    # Results
    train_sharpe: Optional[float] = None
    test_sharpe: Optional[float] = None
    train_returns: Optional[float] = None
    test_returns: Optional[float] = None
    train_win_rate: Optional[float] = None
    test_win_rate: Optional[float] = None
    max_drawdown: Optional[float] = None

    # Additional metrics
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def train_period_days(self) -> int:
        """Number of days in training period."""
        return (self.train_end - self.train_start).days

    @property
    def test_period_days(self) -> int:
        """Number of days in testing period."""
        return (self.test_end - self.test_start).days

    @property
    def degradation_ratio(self) -> Optional[float]:
        """Ratio of test to train Sharpe (measures overfitting)."""
        if self.train_sharpe and self.test_sharpe:
            return self.test_sharpe / self.train_sharpe if self.train_sharpe != 0 else 0
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "window_id": self.window_id,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "train_period_days": self.train_period_days,
            "test_period_days": self.test_period_days,
            "train_sharpe": self.train_sharpe,
            "test_sharpe": self.test_sharpe,
            "train_returns": self.train_returns,
            "test_returns": self.test_returns,
            "train_win_rate": self.train_win_rate,
            "test_win_rate": self.test_win_rate,
            "max_drawdown": self.max_drawdown,
            "degradation_ratio": self.degradation_ratio,
            "metrics": self.metrics,
        }


@dataclass
class WalkForwardReport:
    """Complete walk-forward validation report."""
    method: str
    total_windows: int
    windows: List[WalkForwardWindow]

    # Aggregate metrics
    mean_test_sharpe: float
    std_test_sharpe: float
    mean_degradation_ratio: float
    consistency_score: float  # % of windows where test Sharpe > 0

    # Statistical tests
    is_statistically_significant: bool
    t_statistic: Optional[float]
    p_value: Optional[float]

    # Summary
    total_train_days: int
    total_test_days: int
    start_date: datetime
    end_date: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "total_windows": self.total_windows,
            "windows": [w.to_dict() for w in self.windows],
            "mean_test_sharpe": self.mean_test_sharpe,
            "std_test_sharpe": self.std_test_sharpe,
            "mean_degradation_ratio": self.mean_degradation_ratio,
            "consistency_score": self.consistency_score,
            "is_statistically_significant": self.is_statistically_significant,
            "t_statistic": self.t_statistic,
            "p_value": self.p_value,
            "total_train_days": self.total_train_days,
            "total_test_days": self.total_test_days,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
        }


class WalkForwardValidator:
    """
    Walk-forward validation for trading strategies.

    Walk-forward analysis divides the data into multiple train/test
    periods to simulate realistic out-of-sample performance.

    Usage:
        validator = WalkForwardValidator(
            train_period_days=90,
            test_period_days=30,
            method=WalkForwardMethod.ROLLING
        )

        # Generate windows
        windows = validator.generate_windows(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2024, 1, 1)
        )

        # Run validation
        for window in windows:
            train_data = data[(data['date'] >= window.train_start) &
                             (data['date'] < window.train_end)]
            test_data = data[(data['date'] >= window.test_start) &
                            (data['date'] < window.test_end)]

            # Train model on train_data
            # Evaluate on test_data
            window.test_sharpe = calculate_sharpe(results)

        # Generate report
        report = validator.generate_report()
    """

    def __init__(
        self,
        train_period_days: int = 90,
        test_period_days: int = 30,
        step_days: Optional[int] = None,
        method: WalkForwardMethod = WalkForwardMethod.ROLLING,
        min_train_samples: int = 100,
    ):
        """
        Initialize walk-forward validator.

        Args:
            train_period_days: Length of training period in days
            test_period_days: Length of testing period in days
            step_days: Days to step forward (default: test_period_days)
            method: Walk-forward method (rolling, anchored, sliding)
            min_train_samples: Minimum samples required for training
        """
        self.train_period_days = train_period_days
        self.test_period_days = test_period_days
        self.step_days = step_days or test_period_days
        self.method = method
        self.min_train_samples = min_train_samples

        self.windows: List[WalkForwardWindow] = []

    def generate_windows(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[WalkForwardWindow]:
        """
        Generate walk-forward windows.

        Args:
            start_date: Start of data period
            end_date: End of data period

        Returns:
            List of WalkForwardWindow objects
        """
        self.windows = []
        window_id = 0

        if self.method == WalkForwardMethod.ANCHORED:
            # Anchored: training always starts from start_date
            current_train_start = start_date
            current_train_end = start_date + timedelta(days=self.train_period_days)

            while current_train_end + timedelta(days=self.test_period_days) <= end_date:
                window = WalkForwardWindow(
                    window_id=window_id,
                    train_start=current_train_start,
                    train_end=current_train_end,
                    test_start=current_train_end,
                    test_end=current_train_end + timedelta(days=self.test_period_days),
                )
                self.windows.append(window)

                # Expand training window, move test forward
                current_train_end += timedelta(days=self.step_days)
                window_id += 1

        else:
            # Rolling or Sliding: training window moves forward
            current_train_start = start_date

            while current_train_start + timedelta(
                days=self.train_period_days + self.test_period_days
            ) <= end_date:
                train_end = current_train_start + timedelta(days=self.train_period_days)
                test_end = train_end + timedelta(days=self.test_period_days)

                window = WalkForwardWindow(
                    window_id=window_id,
                    train_start=current_train_start,
                    train_end=train_end,
                    test_start=train_end,
                    test_end=test_end,
                )
                self.windows.append(window)

                current_train_start += timedelta(days=self.step_days)
                window_id += 1

        logger.info(
            f"Generated {len(self.windows)} walk-forward windows "
            f"({self.method.value} method)"
        )

        return self.windows

    def iterate_windows(
        self,
        data: pd.DataFrame,
        date_column: str = "timestamp",
    ) -> Generator[Tuple[WalkForwardWindow, pd.DataFrame, pd.DataFrame], None, None]:
        """
        Iterate through windows yielding train and test data.

        Args:
            data: DataFrame with trading data
            date_column: Column containing timestamps

        Yields:
            Tuple of (window, train_data, test_data)
        """
        for window in self.windows:
            # Filter data for this window
            train_mask = (
                (data[date_column] >= window.train_start) &
                (data[date_column] < window.train_end)
            )
            test_mask = (
                (data[date_column] >= window.test_start) &
                (data[date_column] < window.test_end)
            )

            train_data = data[train_mask].copy()
            test_data = data[test_mask].copy()

            window.train_samples = len(train_data)
            window.test_samples = len(test_data)

            yield window, train_data, test_data

    def run_validation(
        self,
        data: pd.DataFrame,
        train_func: Callable[[pd.DataFrame], Any],
        evaluate_func: Callable[[Any, pd.DataFrame], Dict[str, float]],
        date_column: str = "timestamp",
    ) -> WalkForwardReport:
        """
        Run complete walk-forward validation.

        Args:
            data: DataFrame with all data
            train_func: Function that trains model on data, returns model
            evaluate_func: Function that evaluates model on data, returns metrics dict
            date_column: Column containing timestamps

        Returns:
            WalkForwardReport with complete results
        """
        for window, train_data, test_data in self.iterate_windows(data, date_column):
            if len(train_data) < self.min_train_samples:
                logger.warning(
                    f"Window {window.window_id}: Insufficient training samples "
                    f"({len(train_data)} < {self.min_train_samples})"
                )
                continue

            try:
                # Train model
                model = train_func(train_data)

                # Evaluate on train
                train_metrics = evaluate_func(model, train_data)
                window.train_sharpe = train_metrics.get("sharpe_ratio")
                window.train_returns = train_metrics.get("total_return")
                window.train_win_rate = train_metrics.get("win_rate")

                # Evaluate on test
                test_metrics = evaluate_func(model, test_data)
                window.test_sharpe = test_metrics.get("sharpe_ratio")
                window.test_returns = test_metrics.get("total_return")
                window.test_win_rate = test_metrics.get("win_rate")
                window.max_drawdown = test_metrics.get("max_drawdown")

                # Store all metrics
                window.metrics = {
                    "train": train_metrics,
                    "test": test_metrics,
                }

                logger.info(
                    f"Window {window.window_id}: "
                    f"Train Sharpe={window.train_sharpe:.2f}, "
                    f"Test Sharpe={window.test_sharpe:.2f}"
                )

            except Exception as e:
                logger.error(f"Window {window.window_id} failed: {e}")

        return self.generate_report()

    def generate_report(self) -> WalkForwardReport:
        """
        Generate comprehensive walk-forward report.

        Returns:
            WalkForwardReport with aggregate statistics
        """
        # Filter windows with valid results
        valid_windows = [
            w for w in self.windows
            if w.test_sharpe is not None
        ]

        if not valid_windows:
            logger.warning("No valid windows for report generation")
            return WalkForwardReport(
                method=self.method.value,
                total_windows=len(self.windows),
                windows=self.windows,
                mean_test_sharpe=0.0,
                std_test_sharpe=0.0,
                mean_degradation_ratio=0.0,
                consistency_score=0.0,
                is_statistically_significant=False,
                t_statistic=None,
                p_value=None,
                total_train_days=0,
                total_test_days=0,
                start_date=self.windows[0].train_start if self.windows else datetime.now(),
                end_date=self.windows[-1].test_end if self.windows else datetime.now(),
            )

        # Calculate aggregate metrics
        test_sharpes = [w.test_sharpe for w in valid_windows]
        degradation_ratios = [
            w.degradation_ratio for w in valid_windows
            if w.degradation_ratio is not None
        ]

        mean_test_sharpe = float(np.mean(test_sharpes))
        std_test_sharpe = float(np.std(test_sharpes))

        # Consistency: % of windows with positive test Sharpe
        positive_windows = sum(1 for s in test_sharpes if s > 0)
        consistency_score = positive_windows / len(test_sharpes) if test_sharpes else 0

        # Statistical significance: t-test against zero
        from scipy import stats
        t_stat, p_value = stats.ttest_1samp(test_sharpes, 0)

        # Total days
        total_train_days = sum(w.train_period_days for w in valid_windows)
        total_test_days = sum(w.test_period_days for w in valid_windows)

        report = WalkForwardReport(
            method=self.method.value,
            total_windows=len(self.windows),
            windows=self.windows,
            mean_test_sharpe=mean_test_sharpe,
            std_test_sharpe=std_test_sharpe,
            mean_degradation_ratio=float(np.mean(degradation_ratios)) if degradation_ratios else 0.0,
            consistency_score=consistency_score,
            is_statistically_significant=p_value < 0.05 if p_value else False,
            t_statistic=float(t_stat),
            p_value=float(p_value),
            total_train_days=total_train_days,
            total_test_days=total_test_days,
            start_date=self.windows[0].train_start,
            end_date=self.windows[-1].test_end,
        )

        logger.info(
            f"Walk-Forward Report: "
            f"Mean Test Sharpe={mean_test_sharpe:.2f} (+/- {std_test_sharpe:.2f}), "
            f"Consistency={consistency_score:.1%}, "
            f"Significant={report.is_statistically_significant}"
        )

        return report


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_walk_forward(
    returns: pd.Series,
    train_days: int = 90,
    test_days: int = 30,
) -> Dict[str, Any]:
    """
    Quick walk-forward analysis on a returns series.

    Args:
        returns: Series of daily returns
        train_days: Training period length
        test_days: Testing period length

    Returns:
        Dictionary with walk-forward statistics
    """
    validator = WalkForwardValidator(
        train_period_days=train_days,
        test_period_days=test_days,
    )

    # Create DataFrame from series
    if isinstance(returns.index, pd.DatetimeIndex):
        data = pd.DataFrame({
            "timestamp": returns.index,
            "returns": returns.values
        })
    else:
        data = pd.DataFrame({
            "timestamp": pd.date_range(
                start="2020-01-01",
                periods=len(returns),
                freq="D"
            ),
            "returns": returns.values
        })

    start_date = data["timestamp"].min()
    end_date = data["timestamp"].max()
    validator.generate_windows(start_date, end_date)

    results = []
    for window, train_data, test_data in validator.iterate_windows(data):
        if len(test_data) > 0:
            test_returns = test_data["returns"]
            sharpe = (
                test_returns.mean() / test_returns.std() * np.sqrt(252)
                if test_returns.std() > 0 else 0
            )
            window.test_sharpe = sharpe
            results.append(sharpe)

    return {
        "windows": len(results),
        "mean_sharpe": float(np.mean(results)) if results else 0,
        "std_sharpe": float(np.std(results)) if results else 0,
        "min_sharpe": float(np.min(results)) if results else 0,
        "max_sharpe": float(np.max(results)) if results else 0,
        "positive_windows": sum(1 for r in results if r > 0),
        "consistency": sum(1 for r in results if r > 0) / len(results) if results else 0,
    }


__all__ = [
    "WalkForwardMethod",
    "WalkForwardWindow",
    "WalkForwardReport",
    "WalkForwardValidator",
    "quick_walk_forward",
]
