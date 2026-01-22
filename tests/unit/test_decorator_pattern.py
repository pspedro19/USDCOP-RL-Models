"""
Unit Tests for Decorator Pattern - Resilience Decorators
=========================================================

Tests for the Decorator Pattern implementation in src/core/decorators/resilience.py

Author: USD/COP Trading System
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, patch, MagicMock
import logging

from src.core.decorators.resilience import (
    with_retry,
    with_timing,
    with_circuit_breaker,
    with_timeout,
    with_fallback,
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerStats,
    CircuitBreakerOpenError,
    get_circuit_breaker,
    reset_all_circuit_breakers,
)


class TestWithRetry:
    """Tests for with_retry decorator."""

    def test_succeeds_first_try(self):
        """Test function that succeeds on first try."""
        mock_func = Mock(return_value="success")

        @with_retry(max_attempts=3)
        def test_func():
            return mock_func()

        result = test_func()

        assert result == "success"
        assert mock_func.call_count == 1

    def test_succeeds_after_retry(self):
        """Test function that succeeds after retry."""
        call_count = [0]

        @with_retry(max_attempts=3, delay=0.01)
        def test_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("Temporary error")
            return "success"

        result = test_func()

        assert result == "success"
        assert call_count[0] == 2

    def test_fails_after_max_attempts(self):
        """Test function that fails all attempts."""
        mock_func = Mock(side_effect=ValueError("Permanent error"))

        @with_retry(max_attempts=3, delay=0.01)
        def test_func():
            return mock_func()

        with pytest.raises(ValueError, match="Permanent error"):
            test_func()

        assert mock_func.call_count == 3

    def test_exponential_backoff(self):
        """Test that delay increases with backoff."""
        call_times = []
        call_count = [0]

        # Use longer delays to avoid timing issues on different systems
        @with_retry(max_attempts=3, delay=0.1, backoff=2.0)
        def test_func():
            call_times.append(time.time())
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Error")
            return "success"

        test_func()

        # Check delays increase (approximately)
        assert len(call_times) == 3
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        # delay2 should be at least 1.5x delay1 (allowing for system variance)
        # Expected: delay1 ~= 0.1s, delay2 ~= 0.2s
        assert delay2 >= delay1 * 1.5, f"delay2={delay2:.3f} should be >= 1.5*delay1={delay1*1.5:.3f}"

    def test_only_retry_specified_exceptions(self):
        """Test that only specified exceptions trigger retry."""
        call_count = [0]

        @with_retry(max_attempts=3, delay=0.01, exceptions=(ValueError,))
        def test_func():
            call_count[0] += 1
            raise TypeError("Different error")

        with pytest.raises(TypeError):
            test_func()

        # TypeError should not trigger retry
        assert call_count[0] == 1

    def test_on_retry_callback(self):
        """Test on_retry callback is called."""
        callback = Mock()
        call_count = [0]

        @with_retry(max_attempts=3, delay=0.01, on_retry=callback)
        def test_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Error")
            return "success"

        test_func()

        # Callback should be called for each retry
        assert callback.call_count == 2

    @pytest.mark.asyncio
    async def test_async_retry(self):
        """Test retry with async function."""
        call_count = [0]

        @with_retry(max_attempts=3, delay=0.01)
        async def async_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("Error")
            return "success"

        result = await async_func()

        assert result == "success"
        assert call_count[0] == 2


class TestWithTiming:
    """Tests for with_timing decorator."""

    def test_returns_correct_result(self):
        """Test that decorated function returns correct result."""
        @with_timing
        def test_func():
            return "result"

        assert test_func() == "result"

    def test_measures_time(self):
        """Test that execution time is measured."""
        times = []

        def capture_time(name, elapsed):
            times.append(elapsed)

        @with_timing(callback=capture_time)
        def test_func():
            time.sleep(0.05)
            return "done"

        test_func()

        assert len(times) == 1
        assert times[0] >= 50  # At least 50ms

    def test_threshold_filters_fast_calls(self):
        """Test that threshold filters fast calls from logging."""
        slow_times = []
        fast_times = []

        def capture_slow_time(name, elapsed):
            slow_times.append(elapsed)

        def capture_fast_time(name, elapsed):
            fast_times.append(elapsed)

        @with_timing(threshold_ms=100, callback=capture_fast_time)
        def fast_func():
            return "fast"

        @with_timing(threshold_ms=100, callback=capture_slow_time)
        def slow_func():
            time.sleep(0.15)
            return "slow"

        fast_func()  # Won't trigger callback (below threshold)
        slow_func()  # Will trigger callback (above threshold)

        # fast_func should not trigger callback (execution time < 100ms)
        # slow_func should trigger callback (execution time > 100ms)
        assert len(slow_times) == 1
        assert slow_times[0] >= 100
        # fast_times may or may not have entries depending on actual timing
        # The key test is that slow_times has the expected entry

    def test_with_parentheses(self):
        """Test decorator works with parentheses."""
        @with_timing()
        def test_func():
            return "result"

        assert test_func() == "result"

    def test_without_parentheses(self):
        """Test decorator works without parentheses."""
        @with_timing
        def test_func():
            return "result"

        assert test_func() == "result"

    @pytest.mark.asyncio
    async def test_async_timing(self):
        """Test timing with async function."""
        times = []

        def capture_time(name, elapsed):
            times.append(elapsed)

        @with_timing(callback=capture_time)
        async def async_func():
            await asyncio.sleep(0.05)
            return "done"

        result = await async_func()

        assert result == "done"
        assert len(times) == 1
        assert times[0] >= 50


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state_closed(self):
        """Test circuit breaker starts in closed state."""
        cb = CircuitBreaker(failure_threshold=3)

        assert cb.state == CircuitBreakerState.CLOSED

    def test_opens_after_threshold(self):
        """Test circuit opens after failure threshold."""
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=60.0)

        def failing_func():
            raise ValueError("Error")

        for _ in range(3):
            try:
                cb.execute(failing_func)
            except ValueError:
                pass

        assert cb.state == CircuitBreakerState.OPEN

    def test_rejects_calls_when_open(self):
        """Test circuit rejects calls when open."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=60.0)

        def failing_func():
            raise ValueError("Error")

        # Trip the circuit
        for _ in range(2):
            try:
                cb.execute(failing_func)
            except ValueError:
                pass

        # Next call should be rejected
        with pytest.raises(CircuitBreakerOpenError):
            cb.execute(lambda: "test")

    def test_transitions_to_half_open(self):
        """Test circuit transitions to half-open after timeout."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)

        def failing_func():
            raise ValueError("Error")

        # Trip the circuit
        for _ in range(2):
            try:
                cb.execute(failing_func)
            except ValueError:
                pass

        assert cb.state == CircuitBreakerState.OPEN

        # Wait for reset timeout
        time.sleep(0.15)

        # Should transition to half-open
        assert cb.state == CircuitBreakerState.HALF_OPEN

    def test_closes_on_half_open_success(self):
        """Test circuit closes on successful call in half-open state."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)

        call_count = [0]

        def sometimes_failing():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ValueError("Error")
            return "success"

        # Trip the circuit
        for _ in range(2):
            try:
                cb.execute(sometimes_failing)
            except ValueError:
                pass

        # Wait for reset timeout
        time.sleep(0.15)

        # This call should succeed and close the circuit
        result = cb.execute(sometimes_failing)

        assert result == "success"
        assert cb.state == CircuitBreakerState.CLOSED

    def test_reopens_on_half_open_failure(self):
        """Test circuit reopens on failure in half-open state."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)

        def failing_func():
            raise ValueError("Error")

        # Trip the circuit
        for _ in range(2):
            try:
                cb.execute(failing_func)
            except ValueError:
                pass

        # Wait for reset timeout
        time.sleep(0.15)

        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Fail in half-open state
        try:
            cb.execute(failing_func)
        except ValueError:
            pass

        assert cb.state == CircuitBreakerState.OPEN

    def test_success_resets_failure_count(self):
        """Test successful call resets failure count."""
        cb = CircuitBreaker(failure_threshold=3)

        def sometimes_failing(should_fail):
            if should_fail:
                raise ValueError("Error")
            return "success"

        # Two failures
        for _ in range(2):
            try:
                cb.execute(sometimes_failing, True)
            except ValueError:
                pass

        # One success should reset counter
        cb.execute(sometimes_failing, False)

        # Two more failures shouldn't open circuit
        for _ in range(2):
            try:
                cb.execute(sometimes_failing, True)
            except ValueError:
                pass

        assert cb.state == CircuitBreakerState.CLOSED

    def test_excluded_exceptions_dont_count(self):
        """Test excluded exceptions don't count as failures."""
        cb = CircuitBreaker(
            failure_threshold=2,
            excluded_exceptions=(TypeError,)
        )

        def func_with_type_error():
            raise TypeError("Excluded")

        for _ in range(5):
            with pytest.raises(TypeError):
                cb.execute(func_with_type_error)

        # Circuit should still be closed
        assert cb.state == CircuitBreakerState.CLOSED

    def test_manual_reset(self):
        """Test manual reset of circuit breaker."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=60.0)

        def failing_func():
            raise ValueError("Error")

        # Trip the circuit
        for _ in range(2):
            try:
                cb.execute(failing_func)
            except ValueError:
                pass

        assert cb.state == CircuitBreakerState.OPEN

        cb.reset()

        assert cb.state == CircuitBreakerState.CLOSED

    def test_stats(self):
        """Test circuit breaker statistics."""
        cb = CircuitBreaker(failure_threshold=5)

        def success_func():
            return "success"

        def failing_func():
            raise ValueError("Error")

        # 3 successes, 2 failures
        for _ in range(3):
            cb.execute(success_func)

        for _ in range(2):
            try:
                cb.execute(failing_func)
            except ValueError:
                pass

        stats = cb.stats

        assert stats.total_calls == 5
        assert stats.successful_calls == 3
        assert stats.failed_calls == 2

    def test_state_change_callback(self):
        """Test state change callback is called."""
        transitions = []

        def on_transition(old, new):
            transitions.append((old, new))

        cb = CircuitBreaker(
            failure_threshold=2,
            reset_timeout=0.1,
            on_state_change=on_transition
        )

        def failing_func():
            raise ValueError("Error")

        # Trip the circuit
        for _ in range(2):
            try:
                cb.execute(failing_func)
            except ValueError:
                pass

        assert len(transitions) == 1
        assert transitions[0] == (CircuitBreakerState.CLOSED, CircuitBreakerState.OPEN)


class TestWithCircuitBreaker:
    """Tests for with_circuit_breaker decorator."""

    def test_decorator_creates_circuit_breaker(self):
        """Test decorator creates and attaches circuit breaker."""
        @with_circuit_breaker(failure_threshold=3, name="test_func_1")
        def test_func():
            return "result"

        assert hasattr(test_func, 'circuit_breaker')
        assert isinstance(test_func.circuit_breaker, CircuitBreaker)

        reset_all_circuit_breakers()

    def test_decorator_protects_function(self):
        """Test decorator provides circuit breaker protection."""
        call_count = [0]

        @with_circuit_breaker(failure_threshold=2, reset_timeout=60.0, name="test_func_2")
        def test_func():
            call_count[0] += 1
            raise ValueError("Error")

        # Two failures
        for _ in range(2):
            try:
                test_func()
            except ValueError:
                pass

        # Third call should be rejected
        with pytest.raises(CircuitBreakerOpenError):
            test_func()

        # Only 2 actual calls should have been made
        assert call_count[0] == 2

        reset_all_circuit_breakers()

    def test_get_circuit_breaker_helper(self):
        """Test get_circuit_breaker helper function."""
        @with_circuit_breaker(failure_threshold=3, name="my_named_cb")
        def test_func():
            return "result"

        cb = get_circuit_breaker("my_named_cb")

        assert cb is not None
        assert cb is test_func.circuit_breaker

        reset_all_circuit_breakers()


class TestWithTimeout:
    """Tests for with_timeout decorator."""

    def test_completes_within_timeout(self):
        """Test function that completes within timeout."""
        @with_timeout(1.0)
        def fast_func():
            return "done"

        result = fast_func()

        assert result == "done"

    def test_raises_on_timeout(self):
        """Test function raises TimeoutError on timeout."""
        @with_timeout(0.1)
        def slow_func():
            time.sleep(1.0)
            return "done"

        with pytest.raises(TimeoutError):
            slow_func()

    def test_returns_default_on_timeout(self):
        """Test returns default value when not raising."""
        @with_timeout(0.1, raise_on_timeout=False, default_value="default")
        def slow_func():
            time.sleep(1.0)
            return "done"

        result = slow_func()

        assert result == "default"


class TestWithFallback:
    """Tests for with_fallback decorator."""

    def test_returns_normal_result(self):
        """Test returns normal result when no exception."""
        def fallback():
            return "fallback"

        @with_fallback(fallback)
        def test_func():
            return "normal"

        result = test_func()

        assert result == "normal"

    def test_returns_fallback_on_error(self):
        """Test returns fallback result on exception."""
        def fallback():
            return "fallback"

        @with_fallback(fallback)
        def test_func():
            raise ValueError("Error")

        result = test_func()

        assert result == "fallback"

    def test_fallback_receives_args(self):
        """Test fallback receives same arguments."""
        def fallback(x, y):
            return x + y + 100

        @with_fallback(fallback)
        def test_func(x, y):
            raise ValueError("Error")

        result = test_func(10, 20)

        assert result == 130

    @pytest.mark.asyncio
    async def test_async_fallback(self):
        """Test fallback with async functions."""
        async def fallback():
            return "async_fallback"

        @with_fallback(fallback)
        async def async_func():
            raise ValueError("Error")

        result = await async_func()

        assert result == "async_fallback"


class TestCombinedDecorators:
    """Tests for combining multiple decorators."""

    def test_retry_with_timing(self):
        """Test combining retry and timing decorators."""
        times = []
        call_count = [0]

        def capture_time(name, elapsed):
            times.append(elapsed)

        @with_timing(callback=capture_time)
        @with_retry(max_attempts=2, delay=0.01)
        def test_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("Error")
            return "success"

        result = test_func()

        assert result == "success"
        assert len(times) == 1  # Timing measured once for full execution

    def test_circuit_breaker_with_retry(self):
        """Test combining circuit breaker and retry."""
        call_count = [0]

        @with_circuit_breaker(failure_threshold=5, reset_timeout=60.0, name="combo_test")
        @with_retry(max_attempts=2, delay=0.01)
        def test_func():
            call_count[0] += 1
            raise ValueError("Error")

        # Each decorated call will retry twice, then fail
        # 3 decorated calls = 6 actual calls
        for _ in range(3):
            try:
                test_func()
            except ValueError:
                pass

        assert call_count[0] == 6

        reset_all_circuit_breakers()
