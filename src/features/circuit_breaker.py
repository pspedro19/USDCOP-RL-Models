"""
Feature Circuit Breaker (Phase 13)
==================================

Circuit breaker that monitors feature quality and prevents trading
when data quality falls below acceptable thresholds.

Triggers when:
- >20% of features are NaN/Inf
- Same feature fails 5 times consecutively

Actions when triggered:
- Force HOLD (no trading)
- Log alert
- Emit prometheus metric
- Enter cooldown period (15 minutes)

Author: Trading Team
Version: 1.0.0
Created: 2025-01-14
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class FeatureCircuitBreakerError(Exception):
    """Raised when circuit breaker is triggered."""

    def __init__(self, nan_ratio: float, affected_features: List[str]):
        self.nan_ratio = nan_ratio
        self.affected_features = affected_features
        super().__init__(
            f"Feature circuit breaker triggered: {nan_ratio:.1%} NaN "
            f"(threshold: 20%). Affected: {affected_features}"
        )


@dataclass
class CircuitBreakerState:
    """Current state of the circuit breaker."""
    is_open: bool = False
    opened_at: Optional[datetime] = None
    nan_ratio: float = 0.0
    affected_features: List[str] = field(default_factory=list)
    consecutive_failures: int = 0


@dataclass
class CircuitBreakerConfig:
    """Configuration for the feature circuit breaker."""
    max_nan_ratio: float = 0.20  # 20% threshold
    max_consecutive_failures: int = 5
    cooldown_minutes: int = 15
    warning_threshold: float = 0.15  # 15% for warnings


class FeatureCircuitBreaker:
    """
    Circuit breaker that monitors feature quality.

    Triggers when:
    - >20% of features are NaN/Inf
    - Same feature fails 5 times consecutively

    Actions when triggered:
    - Force HOLD (no trading)
    - Log alert
    - Emit prometheus metric
    - Enter cooldown period

    Usage:
        cb = FeatureCircuitBreaker()

        try:
            cb.check(features)
            # Safe to trade
        except FeatureCircuitBreakerError as e:
            # Force HOLD, log alert
            logger.error(f"Circuit breaker: {e}")
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize the circuit breaker.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitBreakerState()
        self._feature_failure_counts: Dict[str, int] = {}

        logger.info(
            f"FeatureCircuitBreaker initialized: "
            f"max_nan_ratio={self.config.max_nan_ratio:.0%}, "
            f"cooldown={self.config.cooldown_minutes}min"
        )

    def check(self, features: Dict[str, float]) -> None:
        """
        Check feature quality and raise if circuit breaker triggered.

        Args:
            features: Dict of feature_name -> value

        Raises:
            FeatureCircuitBreakerError: If quality threshold exceeded
        """
        # Check if in cooldown
        if self._is_in_cooldown():
            raise FeatureCircuitBreakerError(
                self._state.nan_ratio,
                self._state.affected_features
            )

        # Count NaN/Inf features
        nan_features = []
        for name, value in features.items():
            if self._is_invalid(value):
                nan_features.append(name)
                self._increment_failure(name)
            else:
                self._reset_failure(name)

        nan_ratio = len(nan_features) / len(features) if features else 0

        # Check for warnings
        if nan_ratio > self.config.warning_threshold:
            logger.warning(
                f"Feature quality warning: {nan_ratio:.1%} NaN "
                f"(warning threshold: {self.config.warning_threshold:.0%}). "
                f"Affected: {nan_features}"
            )

        # Check thresholds
        should_trigger = False
        trigger_reason = ""

        if nan_ratio > self.config.max_nan_ratio:
            should_trigger = True
            trigger_reason = f"NaN ratio {nan_ratio:.1%} > threshold {self.config.max_nan_ratio:.0%}"
            logger.error(trigger_reason)

        # Check consecutive failures
        for name, count in self._feature_failure_counts.items():
            if count >= self.config.max_consecutive_failures:
                should_trigger = True
                trigger_reason = f"Feature '{name}' failed {count} consecutive times"
                logger.error(trigger_reason)
                if name not in nan_features:
                    nan_features.append(name)

        if should_trigger:
            self._open_circuit(nan_ratio, nan_features)
            raise FeatureCircuitBreakerError(nan_ratio, nan_features)

        # Update metrics
        self._emit_metrics(nan_ratio, nan_features)

    def _is_invalid(self, value: Any) -> bool:
        """Check if a value is invalid (NaN or Inf)."""
        if value is None:
            return True
        try:
            return np.isnan(value) or np.isinf(value)
        except (TypeError, ValueError):
            return True

    def _is_in_cooldown(self) -> bool:
        """Check if still in cooldown period."""
        if not self._state.is_open:
            return False

        elapsed = datetime.now() - self._state.opened_at
        if elapsed > timedelta(minutes=self.config.cooldown_minutes):
            self._close_circuit()
            return False

        remaining = timedelta(minutes=self.config.cooldown_minutes) - elapsed
        logger.debug(
            f"Circuit breaker in cooldown: {remaining.seconds}s remaining"
        )
        return True

    def _open_circuit(self, nan_ratio: float, affected: List[str]) -> None:
        """Open the circuit breaker."""
        self._state = CircuitBreakerState(
            is_open=True,
            opened_at=datetime.now(),
            nan_ratio=nan_ratio,
            affected_features=affected
        )
        logger.warning(
            f"🔴 Circuit breaker OPENED: {nan_ratio:.1%} NaN, "
            f"affected={affected}, "
            f"cooldown={self.config.cooldown_minutes}min"
        )

    def _close_circuit(self) -> None:
        """Close the circuit breaker."""
        logger.info("🟢 Circuit breaker CLOSED: resuming normal operation")
        self._state = CircuitBreakerState()
        self._feature_failure_counts.clear()

    def _increment_failure(self, feature_name: str) -> None:
        """Increment failure count for a feature."""
        self._feature_failure_counts[feature_name] = \
            self._feature_failure_counts.get(feature_name, 0) + 1

    def _reset_failure(self, feature_name: str) -> None:
        """Reset failure count for a feature."""
        self._feature_failure_counts[feature_name] = 0

    def _emit_metrics(self, nan_ratio: float, nan_features: List[str]) -> None:
        """Emit prometheus metrics."""
        try:
            from services.common.prometheus_metrics import (
                FEATURE_NAN_RATIO,
                CIRCUIT_BREAKER_STATE,
            )
            FEATURE_NAN_RATIO.set(nan_ratio)
            CIRCUIT_BREAKER_STATE.set(1 if self._state.is_open else 0)
        except ImportError:
            pass  # Metrics not available

    @property
    def is_open(self) -> bool:
        """Check if circuit breaker is currently open."""
        return self._is_in_cooldown()

    def get_state(self) -> Dict[str, Any]:
        """Get current state for debugging/monitoring."""
        return {
            "is_open": self._state.is_open,
            "opened_at": self._state.opened_at.isoformat() if self._state.opened_at else None,
            "nan_ratio": self._state.nan_ratio,
            "affected_features": self._state.affected_features,
            "feature_failure_counts": dict(self._feature_failure_counts),
            "config": {
                "max_nan_ratio": self.config.max_nan_ratio,
                "max_consecutive_failures": self.config.max_consecutive_failures,
                "cooldown_minutes": self.config.cooldown_minutes,
            }
        }

    def reset(self) -> None:
        """Manually reset the circuit breaker (use with caution)."""
        logger.warning("Circuit breaker manually reset")
        self._state = CircuitBreakerState()
        self._feature_failure_counts.clear()


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_circuit_breaker: Optional[FeatureCircuitBreaker] = None


def get_circuit_breaker(
    config: Optional[CircuitBreakerConfig] = None
) -> FeatureCircuitBreaker:
    """Get or create the global circuit breaker instance."""
    global _circuit_breaker
    if _circuit_breaker is None:
        _circuit_breaker = FeatureCircuitBreaker(config)
    return _circuit_breaker
