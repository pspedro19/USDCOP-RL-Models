"""
Circuit Breaker Pattern Implementation
======================================

Implements the circuit breaker pattern to prevent cascading failures
in distributed systems. Protects services by failing fast when
downstream dependencies are unhealthy.

P2: Circuit Breaker

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Failure threshold exceeded, requests fail immediately
- HALF_OPEN: Testing if service has recovered

Author: Trading Team
Version: 1.0.0
Date: 2026-01-17
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, Generic
from collections import deque
import threading

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    def __init__(self, circuit_name: str, state: CircuitState, message: str = ""):
        self.circuit_name = circuit_name
        self.state = state
        super().__init__(
            message or f"Circuit breaker '{circuit_name}' is {state.value}"
        )


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0  # Requests rejected when open
    state_transitions: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_state_change: Optional[datetime] = None
    current_state: CircuitState = CircuitState.CLOSED

    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        total = self.successful_requests + self.failed_requests
        if total == 0:
            return 0.0
        return self.failed_requests / total

    @property
    def success_rate(self) -> float:
        """Calculate current success rate."""
        return 1.0 - self.failure_rate

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "rejected_requests": self.rejected_requests,
            "failure_rate": self.failure_rate,
            "success_rate": self.success_rate,
            "state_transitions": self.state_transitions,
            "current_state": self.current_state.value,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
            "last_state_change": self.last_state_change.isoformat() if self.last_state_change else None,
        }


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 3  # Successes in half-open before closing
    timeout_seconds: float = 30.0  # Time in open state before half-open
    failure_rate_threshold: float = 0.5  # Failure rate to trigger open
    window_size: int = 10  # Sliding window for failure rate calculation
    excluded_exceptions: tuple = ()  # Exceptions that don't count as failures


T = TypeVar('T')


class CircuitBreaker:
    """
    Circuit breaker implementation for protecting service calls.

    Usage:
        # Create circuit breaker
        cb = CircuitBreaker(
            name="database",
            config=CircuitBreakerConfig(
                failure_threshold=5,
                timeout_seconds=30.0
            )
        )

        # Use as decorator
        @cb
        def call_database():
            return db.query(...)

        # Or use with context manager style
        try:
            with cb.protect():
                result = call_database()
        except CircuitBreakerError:
            return fallback_value

        # Or call directly
        result = cb.call(call_database)
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback: Optional[Callable[[], Any]] = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Unique name for this circuit breaker
            config: Configuration options
            fallback: Optional fallback function when circuit is open
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.fallback = fallback

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._last_state_change: Optional[float] = None

        # Sliding window for failure rate
        self._recent_results: deque = deque(maxlen=self.config.window_size)

        # Statistics
        self.stats = CircuitBreakerStats(current_state=self._state)

        # Thread safety
        self._lock = threading.RLock()

        logger.info(f"CircuitBreaker '{name}' initialized in {self._state.value} state")

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self.state == CircuitState.HALF_OPEN

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._last_state_change = time.time()

        self.stats.state_transitions += 1
        self.stats.current_state = new_state
        self.stats.last_state_change = datetime.now(timezone.utc)

        logger.warning(
            f"CircuitBreaker '{self.name}' state transition: "
            f"{old_state.value} -> {new_state.value}"
        )

    def _should_allow_request(self) -> bool:
        """Determine if a request should be allowed."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if timeout has passed
                if self._last_failure_time is not None:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.config.timeout_seconds:
                        self._transition_to(CircuitState.HALF_OPEN)
                        self._success_count = 0
                        return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                # Allow limited requests to test recovery
                return True

            return False

    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self.stats.successful_requests += 1
            self.stats.last_success_time = datetime.now(timezone.utc)
            self._recent_results.append(True)

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                    self._failure_count = 0

    def _record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        # Check if this exception should be excluded
        if isinstance(exception, self.config.excluded_exceptions):
            return

        with self._lock:
            self.stats.failed_requests += 1
            self.stats.last_failure_time = datetime.now(timezone.utc)
            self._last_failure_time = time.time()
            self._recent_results.append(False)
            self._failure_count += 1

            if self._state == CircuitState.CLOSED:
                # Check failure threshold
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
                # Also check failure rate
                elif len(self._recent_results) >= self.config.window_size:
                    failure_rate = 1 - (sum(self._recent_results) / len(self._recent_results))
                    if failure_rate >= self.config.failure_rate_threshold:
                        self._transition_to(CircuitState.OPEN)

            elif self._state == CircuitState.HALF_OPEN:
                # Single failure in half-open goes back to open
                self._transition_to(CircuitState.OPEN)

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function or fallback

        Raises:
            CircuitBreakerError: If circuit is open and no fallback provided
        """
        self.stats.total_requests += 1

        if not self._should_allow_request():
            self.stats.rejected_requests += 1
            if self.fallback:
                return self.fallback()
            raise CircuitBreakerError(self.name, self._state)

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise

    async def call_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute an async function through the circuit breaker.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of the function or fallback
        """
        self.stats.total_requests += 1

        if not self._should_allow_request():
            self.stats.rejected_requests += 1
            if self.fallback:
                return self.fallback()
            raise CircuitBreakerError(self.name, self._state)

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise

    def __call__(self, func: Callable) -> Callable:
        """
        Use circuit breaker as a decorator.

        Usage:
            @circuit_breaker
            def my_function():
                ...
        """
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self.call_async(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self.call(func, *args, **kwargs)
            return sync_wrapper

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
            self._recent_results.clear()
            logger.info(f"CircuitBreaker '{self.name}' manually reset")

    def force_open(self) -> None:
        """Manually force the circuit breaker to open state."""
        with self._lock:
            self._transition_to(CircuitState.OPEN)
            self._last_failure_time = time.time()
            logger.warning(f"CircuitBreaker '{self.name}' manually opened")

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "stats": self.stats.to_dict(),
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "success_threshold": self.config.success_threshold,
                    "timeout_seconds": self.config.timeout_seconds,
                    "failure_rate_threshold": self.config.failure_rate_threshold,
                    "window_size": self.config.window_size,
                },
            }


# =============================================================================
# Circuit Breaker Registry
# =============================================================================

class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Usage:
        registry = CircuitBreakerRegistry()
        db_breaker = registry.get_or_create("database")
        api_breaker = registry.get_or_create("external_api")

        # Get all statuses
        statuses = registry.get_all_statuses()
    """

    _instance: Optional['CircuitBreakerRegistry'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'CircuitBreakerRegistry':
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._breakers: Dict[str, CircuitBreaker] = {}
        return cls._instance

    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback: Optional[Callable] = None,
    ) -> CircuitBreaker:
        """
        Get existing circuit breaker or create new one.

        Args:
            name: Circuit breaker name
            config: Configuration (only used if creating new)
            fallback: Fallback function (only used if creating new)

        Returns:
            CircuitBreaker instance
        """
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, config, fallback)
        return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._breakers.get(name)

    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {
            name: breaker.get_status()
            for name, breaker in self._breakers.items()
        }

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()


# =============================================================================
# Convenience Functions
# =============================================================================

def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout_seconds: float = 30.0,
    fallback: Optional[Callable] = None,
) -> Callable:
    """
    Decorator to apply circuit breaker to a function.

    Usage:
        @circuit_breaker("database", failure_threshold=3)
        def query_database():
            ...
    """
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        timeout_seconds=timeout_seconds,
    )
    registry = CircuitBreakerRegistry()
    breaker = registry.get_or_create(name, config, fallback)
    return breaker


def get_circuit_breaker(name: str) -> Optional[CircuitBreaker]:
    """Get a circuit breaker from the global registry."""
    return CircuitBreakerRegistry().get(name)


def get_all_circuit_breaker_statuses() -> Dict[str, Dict[str, Any]]:
    """Get status of all circuit breakers in the registry."""
    return CircuitBreakerRegistry().get_all_statuses()


__all__ = [
    "CircuitState",
    "CircuitBreakerError",
    "CircuitBreakerStats",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "circuit_breaker",
    "get_circuit_breaker",
    "get_all_circuit_breaker_statuses",
]
