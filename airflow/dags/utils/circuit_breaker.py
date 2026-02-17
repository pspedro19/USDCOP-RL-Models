# -*- coding: utf-8 -*-
"""
Circuit Breaker Pattern Implementation
======================================
Prevents cascading failures by temporarily stopping requests to failing services.

Contract: CTR-L0-CIRCUIT-BREAKER-001

States:
    CLOSED: Normal operation, requests pass through
    OPEN: Failing, requests are rejected immediately
    HALF_OPEN: Testing if service recovered, limited requests allowed

Usage:
    from utils.circuit_breaker import CircuitBreakerRegistry, CircuitBreakerConfig

    # Get or create a circuit breaker
    cb = CircuitBreakerRegistry().get_or_create(
        "twelvedata_ohlcv",
        CircuitBreakerConfig(failure_threshold=3, timeout_seconds=300)
    )

    # Use with decorator
    @cb.protect
    def fetch_from_api():
        ...

    # Or use directly
    if cb.can_execute():
        try:
            result = risky_operation()
            cb.record_success()
        except Exception as e:
            cb.record_failure(e)

Version: 1.0.0
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""

    failure_threshold: int = 5
    """Number of failures before opening the circuit."""

    success_threshold: int = 3
    """Number of successes in half-open before closing."""

    timeout_seconds: int = 60
    """Time to wait before transitioning from open to half-open."""

    half_open_max_calls: int = 3
    """Maximum concurrent calls allowed in half-open state."""

    excluded_exceptions: tuple = ()
    """Exception types that should NOT trip the breaker."""

    reset_timeout_seconds: int = 300
    """Time after which failure count resets if no failures."""


@dataclass
class CircuitBreakerStats:
    """Statistics for a circuit breaker."""

    total_calls: int = 0
    total_successes: int = 0
    total_failures: int = 0
    total_rejections: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_state_change: Optional[datetime] = None
    state_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'total_calls': self.total_calls,
            'total_successes': self.total_successes,
            'total_failures': self.total_failures,
            'total_rejections': self.total_rejections,
            'consecutive_failures': self.consecutive_failures,
            'consecutive_successes': self.consecutive_successes,
            'success_rate': round(
                self.total_successes / max(self.total_calls, 1), 3
            ),
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'last_success_time': self.last_success_time.isoformat() if self.last_success_time else None,
        }


class CircuitBreaker:
    """
    Circuit breaker implementation.

    Protects against cascading failures by temporarily stopping
    requests to a failing service.
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.

        Args:
            name: Unique identifier for this circuit
            config: Configuration (uses defaults if not provided)
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._lock = threading.RLock()
        self._half_open_calls = 0

        logger.info(
            "[CIRCUIT] Initialized circuit breaker '%s' with threshold=%d, timeout=%ds",
            name, self.config.failure_threshold, self.config.timeout_seconds
        )

    @property
    def state(self) -> CircuitState:
        """Get current state, checking for timeout transitions."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_transition_to_half_open():
                    self._transition_to(CircuitState.HALF_OPEN)
            return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        """Get current statistics."""
        return self._stats

    def _should_transition_to_half_open(self) -> bool:
        """Check if we should transition from OPEN to HALF_OPEN."""
        if self._stats.last_failure_time is None:
            return True

        elapsed = (datetime.utcnow() - self._stats.last_failure_time).total_seconds()
        return elapsed >= self.config.timeout_seconds

    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._stats.last_state_change = datetime.utcnow()
        self._half_open_calls = 0

        self._stats.state_history.append({
            'from': old_state.value,
            'to': new_state.value,
            'timestamp': self._stats.last_state_change.isoformat(),
        })

        # Keep only last 20 state changes
        if len(self._stats.state_history) > 20:
            self._stats.state_history = self._stats.state_history[-20:]

        logger.info(
            "[CIRCUIT] %s: %s -> %s",
            self.name, old_state.value, new_state.value
        )

        # Export metrics if available
        try:
            from services.metrics_exporter import get_metrics
            get_metrics().record_circuit_breaker_state(self.name, new_state.value)
        except ImportError:
            pass

    def can_execute(self) -> bool:
        """
        Check if a request can be executed.

        Returns:
            True if request is allowed, False otherwise
        """
        current_state = self.state  # This may trigger timeout transition

        with self._lock:
            if current_state == CircuitState.CLOSED:
                return True

            if current_state == CircuitState.OPEN:
                self._stats.total_rejections += 1
                return False

            if current_state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                self._stats.total_rejections += 1
                return False

        return False

    def record_success(self):
        """Record a successful call."""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.total_successes += 1
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0
            self._stats.last_success_time = datetime.utcnow()

            if self._state == CircuitState.HALF_OPEN:
                if self._stats.consecutive_successes >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

            logger.debug(
                "[CIRCUIT] %s: Success (consecutive: %d)",
                self.name, self._stats.consecutive_successes
            )

    def record_failure(self, exception: Optional[Exception] = None):
        """Record a failed call."""
        # Check if exception is excluded
        if exception and isinstance(exception, self.config.excluded_exceptions):
            logger.debug(
                "[CIRCUIT] %s: Excluded exception %s, not recording failure",
                self.name, type(exception).__name__
            )
            return

        with self._lock:
            self._stats.total_calls += 1
            self._stats.total_failures += 1
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0
            self._stats.last_failure_time = datetime.utcnow()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately opens
                self._transition_to(CircuitState.OPEN)

            elif self._state == CircuitState.CLOSED:
                if self._stats.consecutive_failures >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

            logger.warning(
                "[CIRCUIT] %s: Failure (consecutive: %d, state: %s)",
                self.name, self._stats.consecutive_failures, self._state.value
            )

            # Export metrics
            try:
                from services.metrics_exporter import get_metrics
                get_metrics().record_circuit_breaker_failure(self.name)
            except ImportError:
                pass

    def reset(self):
        """Reset the circuit breaker to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._stats.consecutive_failures = 0
            self._stats.consecutive_successes = 0
            logger.info("[CIRCUIT] %s: Reset to CLOSED", self.name)

    def protect(self, func: Callable) -> Callable:
        """
        Decorator to protect a function with this circuit breaker.

        Usage:
            @circuit_breaker.protect
            def risky_function():
                ...
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.can_execute():
                raise CircuitOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. Request rejected."
                )

            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise

        return wrapper


class CircuitOpenError(Exception):
    """Raised when a request is rejected because the circuit is open."""
    pass


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Singleton pattern ensures consistent state across DAGs.
    """

    _instance: Optional['CircuitBreakerRegistry'] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._breakers: Dict[str, CircuitBreaker] = {}
                cls._instance._config_path: Optional[Path] = None
        return cls._instance

    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """
        Get an existing circuit breaker or create a new one.

        Args:
            name: Unique name for the circuit breaker
            config: Configuration (only used if creating new)

        Returns:
            CircuitBreaker instance
        """
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, config)
        return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self._breakers.get(name)

    def get_all(self) -> Dict[str, CircuitBreaker]:
        """Get all registered circuit breakers."""
        return self._breakers.copy()

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {
            name: {
                'state': cb.state.value,
                'stats': cb.stats.to_dict(),
            }
            for name, cb in self._breakers.items()
        }

    def get_open_breakers(self) -> List[str]:
        """Get names of all open circuit breakers."""
        return [
            name for name, cb in self._breakers.items()
            if cb.state == CircuitState.OPEN
        ]

    def reset_all(self):
        """Reset all circuit breakers."""
        for cb in self._breakers.values():
            cb.reset()

    def save_state(self, path: Optional[Path] = None):
        """Save circuit breaker states to file."""
        save_path = path or self._config_path
        if not save_path:
            return

        state = {
            name: {
                'state': cb.state.value,
                'stats': cb.stats.to_dict(),
                'config': asdict(cb.config),
            }
            for name, cb in self._breakers.items()
        }

        try:
            with open(save_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            logger.info("[CIRCUIT] Saved state to %s", save_path)
        except Exception as e:
            logger.error("[CIRCUIT] Failed to save state: %s", e)


# =============================================================================
# DEFAULT CIRCUIT BREAKER CONFIGURATIONS
# =============================================================================

DEFAULT_CONFIGS = {
    'twelvedata_ohlcv': CircuitBreakerConfig(
        failure_threshold=3,
        timeout_seconds=300,  # 5 minutes
        success_threshold=2,
    ),
    'fred_api': CircuitBreakerConfig(
        failure_threshold=5,
        timeout_seconds=600,  # 10 minutes
        success_threshold=3,
    ),
    'investing_scraper': CircuitBreakerConfig(
        failure_threshold=3,
        timeout_seconds=600,  # 10 minutes
        success_threshold=2,
    ),
    'suameca_api': CircuitBreakerConfig(
        failure_threshold=4,
        timeout_seconds=300,
        success_threshold=2,
    ),
    'bcrp_api': CircuitBreakerConfig(
        failure_threshold=4,
        timeout_seconds=300,
        success_threshold=2,
    ),
}


def get_circuit_breaker(name: str) -> CircuitBreaker:
    """
    Get a circuit breaker with default config for known services.

    Args:
        name: Service name (e.g., 'twelvedata_ohlcv')

    Returns:
        CircuitBreaker instance
    """
    registry = CircuitBreakerRegistry()
    config = DEFAULT_CONFIGS.get(name)
    return registry.get_or_create(name, config)
