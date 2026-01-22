"""
Decorators Module - Resilience Patterns
=======================================

Provides decorators for building resilient systems with automatic
retry, timing measurement, and circuit breaker patterns.

Components:
- with_retry: Retry failed operations with exponential backoff
- with_timing: Measure and log execution time
- with_circuit_breaker: Circuit breaker for cascading failure prevention
- with_timeout: Add timeout to function execution
- with_fallback: Use fallback on failure

Usage:
    from src.core.decorators import (
        with_retry, with_timing, with_circuit_breaker,
        CircuitBreakerOpenError
    )

    @with_retry(max_attempts=3)
    @with_timing
    def fetch_data():
        return api.get()

    @with_circuit_breaker(failure_threshold=5, reset_timeout=60)
    def call_external_service():
        return requests.get(url)

Author: USD/COP Trading System
Version: 1.0.0
"""

from .resilience import (
    # Decorators
    with_retry,
    with_timing,
    with_circuit_breaker,
    with_timeout,
    with_fallback,

    # Circuit Breaker
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerStats,
    CircuitBreakerOpenError,

    # Utilities
    get_circuit_breaker,
    reset_all_circuit_breakers,
)

__all__ = [
    # Decorators
    "with_retry",
    "with_timing",
    "with_circuit_breaker",
    "with_timeout",
    "with_fallback",

    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerState",
    "CircuitBreakerStats",
    "CircuitBreakerOpenError",

    # Utilities
    "get_circuit_breaker",
    "reset_all_circuit_breakers",
]
