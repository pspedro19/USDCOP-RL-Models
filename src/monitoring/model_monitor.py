"""
Model Monitor for RL Trading Models
====================================

Detects model degradation and drift in production RL models by:
1. Tracking action distribution and comparing vs baseline (KL divergence)
2. Detecting stuck/repetitive behavior
3. Monitoring rolling Sharpe ratio performance

Usage:
    monitor = ModelMonitor(window_size=100)

    # Set baseline from backtest
    monitor.set_baseline(backtest_actions)

    # In production loop
    monitor.record_action(action)
    monitor.record_pnl(pnl)

    # Check health
    health = monitor.get_health_status()
    if health['status'] == 'critical':
        # Alert or disable model
"""

import numpy as np
from collections import deque
from typing import List, Optional, Dict, Any
from scipy.stats import entropy
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    Detect degradation and drift in RL trading models.

    Monitors:
    - Action distribution drift via KL divergence
    - Stuck/repetitive behavior patterns
    - Rolling Sharpe ratio degradation

    Health Status Thresholds:
    - healthy: KL < 0.3, no stuck behavior
    - warning: 0.3 <= KL <= 0.5
    - critical: KL > 0.5 or stuck behavior detected
    """

    # Thresholds for health status
    KL_THRESHOLD_WARNING = 0.3
    KL_THRESHOLD_CRITICAL = 0.5
    STUCK_THRESHOLD_RATIO = 0.9  # 90% same action = stuck
    MIN_SAMPLES_FOR_KL = 20  # Minimum samples before KL is reliable

    def __init__(self, window_size: int = 100):
        """
        Initialize the ModelMonitor.

        Args:
            window_size: Number of recent actions/pnl values to track
        """
        self.window_size = window_size
        self.action_history: deque = deque(maxlen=window_size)
        self.pnl_history: deque = deque(maxlen=window_size)
        self.baseline_action_dist: Optional[np.ndarray] = None
        self._baseline_bins: Optional[np.ndarray] = None
        self._num_bins = 20  # Number of bins for discretizing continuous actions

        logger.info(f"ModelMonitor initialized with window_size={window_size}")

    def record_action(self, action: float) -> None:
        """
        Record a model action for drift detection.

        Args:
            action: The action value from the model (typically in [-1, 1])
        """
        self.action_history.append(float(action))

    def record_pnl(self, pnl: float) -> None:
        """
        Record PnL from a completed trade for performance monitoring.

        Args:
            pnl: Profit/Loss value (in USD or percentage)
        """
        self.pnl_history.append(float(pnl))

    def set_baseline(self, actions: List[float]) -> None:
        """
        Establish baseline action distribution from backtest/historical data.

        The baseline is used to detect drift via KL divergence.

        Args:
            actions: List of historical actions from backtest
        """
        if not actions or len(actions) < self.MIN_SAMPLES_FOR_KL:
            logger.warning(
                f"Insufficient baseline actions ({len(actions) if actions else 0}), "
                f"need at least {self.MIN_SAMPLES_FOR_KL}"
            )
            return

        actions_array = np.array(actions, dtype=np.float64)

        # Determine bin edges based on action range
        # Typically actions are in [-1, 1] but we adapt to actual range
        action_min = min(actions_array.min(), -1.0)
        action_max = max(actions_array.max(), 1.0)

        # Create bins with small buffer
        self._baseline_bins = np.linspace(
            action_min - 0.01,
            action_max + 0.01,
            self._num_bins + 1
        )

        # Compute histogram (probability distribution)
        hist, _ = np.histogram(actions_array, bins=self._baseline_bins, density=True)

        # Add small epsilon to avoid zero probabilities (for KL stability)
        epsilon = 1e-10
        self.baseline_action_dist = hist + epsilon
        self.baseline_action_dist /= self.baseline_action_dist.sum()  # Normalize

        logger.info(
            f"Baseline set from {len(actions)} actions, "
            f"range=[{action_min:.3f}, {action_max:.3f}]"
        )

    def check_action_drift(self) -> float:
        """
        Calculate KL divergence between current action distribution and baseline.

        Returns:
            KL divergence value:
            - 0.0: Distributions are identical
            - > 0.3: Warning level drift
            - > 0.5: Critical level drift
            - -1.0: Cannot calculate (no baseline or insufficient data)
        """
        if self.baseline_action_dist is None:
            logger.debug("No baseline set, cannot calculate KL divergence")
            return -1.0

        if len(self.action_history) < self.MIN_SAMPLES_FOR_KL:
            logger.debug(
                f"Insufficient action history ({len(self.action_history)}/{self.MIN_SAMPLES_FOR_KL})"
            )
            return -1.0

        try:
            # Get current action distribution
            current_actions = np.array(list(self.action_history), dtype=np.float64)

            # Clip actions to baseline bin range
            current_actions = np.clip(
                current_actions,
                self._baseline_bins[0],
                self._baseline_bins[-1]
            )

            # Compute histogram with same bins
            hist, _ = np.histogram(
                current_actions,
                bins=self._baseline_bins,
                density=True
            )

            # Add epsilon and normalize
            epsilon = 1e-10
            current_dist = hist + epsilon
            current_dist /= current_dist.sum()

            # Calculate KL divergence: KL(current || baseline)
            kl_div = entropy(current_dist, self.baseline_action_dist)

            return float(kl_div)

        except Exception as e:
            logger.error(f"Error calculating KL divergence: {e}")
            return -1.0

    def check_stuck_behavior(self) -> bool:
        """
        Detect if model is generating the same action repeatedly.

        A model is considered "stuck" if > 90% of recent actions are identical
        (within tolerance for floating point).

        Returns:
            True if stuck behavior detected, False otherwise
        """
        if len(self.action_history) < 10:
            return False

        actions = np.array(list(self.action_history), dtype=np.float64)

        # Round to 2 decimal places for comparison (handles float precision)
        rounded_actions = np.round(actions, 2)

        # Find the most common action
        unique, counts = np.unique(rounded_actions, return_counts=True)
        max_count = counts.max()

        # Check if most common action dominates
        stuck_ratio = max_count / len(actions)

        if stuck_ratio >= self.STUCK_THRESHOLD_RATIO:
            most_common = unique[counts.argmax()]
            logger.warning(
                f"Stuck behavior detected: {stuck_ratio:.1%} of actions are {most_common}"
            )
            return True

        return False

    def get_rolling_sharpe(self) -> float:
        """
        Calculate rolling Sharpe ratio from recent PnL history.

        Uses annualized Sharpe assuming 5-minute bars (252 * 78 periods per year).

        Returns:
            Sharpe ratio, or 0.0 if insufficient data
        """
        if len(self.pnl_history) < 5:
            return 0.0

        pnl_array = np.array(list(self.pnl_history), dtype=np.float64)

        mean_return = np.mean(pnl_array)
        std_return = np.std(pnl_array, ddof=1)  # Sample std

        if std_return < 1e-10:
            # Avoid division by zero
            return 0.0 if mean_return == 0 else (10.0 if mean_return > 0 else -10.0)

        # Annualization factor: assuming 5-min bars
        # ~78 bars per trading day * 252 trading days
        annualization = np.sqrt(252 * 78)

        sharpe = (mean_return / std_return) * annualization

        return float(np.clip(sharpe, -10.0, 10.0))  # Cap extreme values

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status for the monitored model.

        Returns:
            Dictionary containing:
            - action_drift_kl: KL divergence value
            - stuck_behavior: Boolean indicating stuck state
            - rolling_sharpe: Recent Sharpe ratio
            - actions_recorded: Number of actions in buffer
            - trades_recorded: Number of PnL values in buffer
            - status: "healthy", "warning", or "critical"
            - timestamp: ISO format timestamp
            - details: Human-readable status explanation
        """
        kl_divergence = self.check_action_drift()
        stuck = self.check_stuck_behavior()
        sharpe = self.get_rolling_sharpe()

        # Determine overall status
        if stuck:
            status = "critical"
            details = "Model is stuck - generating repetitive actions"
        elif kl_divergence > self.KL_THRESHOLD_CRITICAL:
            status = "critical"
            details = f"High action drift detected (KL={kl_divergence:.3f})"
        elif kl_divergence > self.KL_THRESHOLD_WARNING:
            status = "warning"
            details = f"Moderate action drift detected (KL={kl_divergence:.3f})"
        elif kl_divergence < 0:
            status = "healthy"
            details = "Insufficient data for drift detection"
        else:
            status = "healthy"
            details = "Model operating within normal parameters"

        # Add Sharpe warning if applicable
        if sharpe < -1.0 and len(self.pnl_history) >= 20:
            if status == "healthy":
                status = "warning"
                details = f"Poor rolling performance (Sharpe={sharpe:.2f})"
            else:
                details += f"; Poor rolling performance (Sharpe={sharpe:.2f})"

        return {
            "action_drift_kl": round(kl_divergence, 4) if kl_divergence >= 0 else None,
            "stuck_behavior": stuck,
            "rolling_sharpe": round(sharpe, 3),
            "actions_recorded": len(self.action_history),
            "trades_recorded": len(self.pnl_history),
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details,
            "thresholds": {
                "kl_warning": self.KL_THRESHOLD_WARNING,
                "kl_critical": self.KL_THRESHOLD_CRITICAL,
                "stuck_ratio": self.STUCK_THRESHOLD_RATIO
            }
        }

    def reset(self) -> None:
        """Clear all history buffers (useful for model retraining)."""
        self.action_history.clear()
        self.pnl_history.clear()
        logger.info("ModelMonitor history cleared")

    def reset_baseline(self) -> None:
        """Clear baseline distribution (requires new set_baseline call)."""
        self.baseline_action_dist = None
        self._baseline_bins = None
        logger.info("ModelMonitor baseline cleared")


# Convenience function for creating pre-configured monitors
def create_model_monitor(
    window_size: int = 100,
    baseline_actions: Optional[List[float]] = None
) -> ModelMonitor:
    """
    Factory function to create a configured ModelMonitor.

    Args:
        window_size: Size of rolling window for tracking
        baseline_actions: Optional list of baseline actions to set immediately

    Returns:
        Configured ModelMonitor instance
    """
    monitor = ModelMonitor(window_size=window_size)

    if baseline_actions:
        monitor.set_baseline(baseline_actions)

    return monitor
