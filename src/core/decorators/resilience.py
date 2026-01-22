"""
Resilience Decorators - Retry, Timing, and Circuit Breaker Patterns
===================================================================

This module provides decorators for building resilient systems:
- with_retry: Automatically retry failed operations with exponential backoff
- with_timing: Measure and log execution time
- with_circuit_breaker: Prevent cascading failures with circuit breaker pattern

Usage:
    from src.core.decorators import (
        with_retry, with_timing, with_circuit_breaker,
        CircuitBreakerOpenError
    )

    @with_retry(max_attempts=3, delay=1.0)
    @with_timing
    def fetch_price_data():
        # May fail due to network issues
        return api.get_prices()

    @with_circuit_breaker(failure_threshold=5, reset_timeout=60.0)
    def external_api_call():
        # Protects against cascading failures
        return api.call()

Author: USD/COP Trading System
Version: 1.0.0
"""

import functools
import time
import logging
import threading
import asyncio
from typing import (
    Any,
    Callable,
    Optional,
    Type,
    TypeVar,
    Union,
    Tuple,
    List
)
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class CircuitBreakerState(Enum):
    """States for circuit breaker."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""

    def __init__(
        self,
        message: str = "Circuit breaker is open",
        reset_time: Optional[datetime] = None
    ):
        super().__init__(message)
        self.reset_time = reset_time
        self.message = message


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    state_changed_at: datetime = field(default_factory=datetime.now)


def with_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    logger_instance: Optional[logging.Logger] = None
) -> Callable[[F], F]:
    """
    Decorator for retrying failed operations with exponential backoff.

    Automatically retries a function if it raises an exception,
    with configurable delay and exponential backoff.

    Args:
        max_attempts: Maximum number of attempts (default: 3)
        delay: Initial delay between attempts in seconds (default: 1.0)
        backoff: Backoff multiplier for delay (default: 2.0)
        exceptions: Tuple of exception types to retry on
        on_retry: Optional callback function(exception, attempt_number)
        logger_instance: Optional logger for retry messages

    Returns:
        Decorated function

    Example:
        @with_retry(max_attempts=3, delay=1.0, backoff=2.0)
        def unreliable_api_call():
            return api.fetch()

        # Will retry up to 3 times with delays of 1s, 2s, 4s
    """
    log = logger_instance or logger

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None
            current_delay = delay

            for attempt in range(1, max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 1:
                        log.info(
                            f"'{func.__name__}' succeeded on attempt {attempt}"
                        )
                    return result

                except exceptions as e:
                    last_exception = e
                    log.warning(
                        f"'{func.__name__}' attempt {attempt}/{max_attempts} "
                        f"failed: {type(e).__name__}: {e}"
                    )

                    # Call retry callback if provided
                    if on_retry:
                        try:
                            on_retry(e, attempt)
                        except Exception as callback_error:
                            log.error(
                                f"Retry callback failed: {callback_error}"
                            )

                    # Don't sleep after last attempt
                    if attempt < max_attempts:
                        log.debug(
                            f"Retrying '{func.__name__}' in {current_delay:.1f}s"
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff

            # All attempts failed
            log.error(
                f"'{func.__name__}' failed after {max_attempts} attempts"
            )
            raise last_exception

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None
            current_delay = delay

            for attempt in range(1, max_attempts + 1):
                try:
                    result = await func(*args, **kwargs)
                    if attempt > 1:
                        log.info(
                            f"'{func.__name__}' succeeded on attempt {attempt}"
                        )
                    return result

                except exceptions as e:
                    last_exception = e
                    log.warning(
                        f"'{func.__name__}' attempt {attempt}/{max_attempts} "
                        f"failed: {type(e).__name__}: {e}"
                    )

                    if on_retry:
                        try:
                            on_retry(e, attempt)
                        except Exception as callback_error:
                            log.error(
                                f"Retry callback failed: {callback_error}"
                            )

                    if attempt < max_attempts:
                        log.debug(
                            f"Retrying '{func.__name__}' in {current_delay:.1f}s"
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff

            log.error(
                f"'{func.__name__}' failed after {max_attempts} attempts"
            )
            raise last_exception

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def with_timing(
    func: Optional[F] = None,
    *,
    logger_instance: Optional[logging.Logger] = None,
    log_level: int = logging.DEBUG,
    threshold_ms: Optional[float] = None,
    callback: Optional[Callable[[str, float], None]] = None
) -> Union[F, Callable[[F], F]]:
    """
    Decorator for measuring and logging execution time.

    Records the execution time of a function and optionally logs
    it or calls a callback. Can be used with or without parentheses.

    Args:
        func: Function to decorate (for use without parentheses)
        logger_instance: Optional logger for timing messages
        log_level: Log level for timing messages (default: DEBUG)
        threshold_ms: Only log if execution exceeds this threshold
        callback: Optional callback(func_name, elapsed_ms)

    Returns:
        Decorated function

    Example:
        @with_timing
        def my_function():
            time.sleep(0.1)

        @with_timing(threshold_ms=100)
        def fast_function():
            pass  # Won't log if under 100ms
    """
    log = logger_instance or logger

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = (time.perf_counter() - start_time) * 1000  # ms

                should_log = threshold_ms is None or elapsed >= threshold_ms
                if should_log:
                    log.log(
                        log_level,
                        f"'{fn.__name__}' executed in {elapsed:.2f}ms"
                    )

                if callback:
                    try:
                        callback(fn.__name__, elapsed)
                    except Exception as e:
                        log.error(f"Timing callback failed: {e}")

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            try:
                return await fn(*args, **kwargs)
            finally:
                elapsed = (time.perf_counter() - start_time) * 1000  # ms

                should_log = threshold_ms is None or elapsed >= threshold_ms
                if should_log:
                    log.log(
                        log_level,
                        f"'{fn.__name__}' executed in {elapsed:.2f}ms"
                    )

                if callback:
                    try:
                        callback(fn.__name__, elapsed)
                    except Exception as e:
                        log.error(f"Timing callback failed: {e}")

        if asyncio.iscoroutinefunction(fn):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    # Allow use with or without parentheses
    if func is not None:
        return decorator(func)
    return decorator


class CircuitBreaker:
    """
    Circuit Breaker implementation for protecting against cascading failures.

    The circuit breaker has three states:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Too many failures, calls are rejected immediately
    - HALF_OPEN: Testing if service recovered, allows one call through

    Thread-safe implementation suitable for concurrent access.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_max_calls: int = 1,
        excluded_exceptions: Tuple[Type[Exception], ...] = (),
        on_state_change: Optional[Callable[[CircuitBreakerState, CircuitBreakerState], None]] = None
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Seconds before attempting to close circuit
            half_open_max_calls: Max calls allowed in half-open state
            excluded_exceptions: Exceptions that don't count as failures
            on_state_change: Callback(old_state, new_state) on transitions
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls
        self.excluded_exceptions = excluded_exceptions
        self.on_state_change = on_state_change

        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time: Optional[datetime] = None
        self._lock = threading.RLock()

        # Statistics
        self._stats = CircuitBreakerStats()

    @property
    def state(self) -> CircuitBreakerState:
        """Get current state, checking for timeout transitions."""
        with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(CircuitBreakerState.HALF_OPEN)
            return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        """Get current statistics."""
        with self._lock:
            self._stats.state = self._state
            return CircuitBreakerStats(
                total_calls=self._stats.total_calls,
                successful_calls=self._stats.successful_calls,
                failed_calls=self._stats.failed_calls,
                rejected_calls=self._stats.rejected_calls,
                last_failure_time=self._stats.last_failure_time,
                last_success_time=self._stats.last_success_time,
                state=self._state,
                state_changed_at=self._stats.state_changed_at
            )

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return True
        elapsed = (datetime.now() - self._last_failure_time).total_seconds()
        return elapsed >= self.reset_timeout

    def _transition_to(self, new_state: CircuitBreakerState) -> None:
        """
        Transition to a new state.

        Args:
            new_state: State to transition to
        """
        old_state = self._state
        if old_state == new_state:
            return

        self._state = new_state
        self._stats.state_changed_at = datetime.now()

        if new_state == CircuitBreakerState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
        elif new_state == CircuitBreakerState.HALF_OPEN:
            self._half_open_calls = 0

        logger.info(
            f"Circuit breaker transitioned: {old_state.value} -> {new_state.value}"
        )

        if self.on_state_change:
            try:
                self.on_state_change(old_state, new_state)
            except Exception as e:
                logger.error(f"State change callback failed: {e}")

    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._stats.successful_calls += 1
            self._stats.last_success_time = datetime.now()
            self._success_count += 1

            if self._state == CircuitBreakerState.HALF_OPEN:
                # Success in half-open state, close the circuit
                self._transition_to(CircuitBreakerState.CLOSED)
            elif self._state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def _record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._stats.failed_calls += 1
            self._stats.last_failure_time = datetime.now()
            self._last_failure_time = datetime.now()
            self._failure_count += 1

            if self._state == CircuitBreakerState.HALF_OPEN:
                # Failure in half-open state, open the circuit again
                self._transition_to(CircuitBreakerState.OPEN)
            elif self._state == CircuitBreakerState.CLOSED:
                # Check if we've reached the failure threshold
                if self._failure_count >= self.failure_threshold:
                    self._transition_to(CircuitBreakerState.OPEN)

    def can_execute(self) -> bool:
        """
        Check if a call can be executed.

        Returns:
            True if call is allowed, False otherwise
        """
        with self._lock:
            state = self.state  # Triggers timeout check

            if state == CircuitBreakerState.CLOSED:
                return True
            elif state == CircuitBreakerState.OPEN:
                return False
            elif state == CircuitBreakerState.HALF_OPEN:
                return self._half_open_calls < self.half_open_max_calls

        return False

    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        with self._lock:
            self._stats.total_calls += 1
            state = self.state

            if state == CircuitBreakerState.OPEN:
                self._stats.rejected_calls += 1
                reset_time = None
                if self._last_failure_time:
                    reset_time = self._last_failure_time + timedelta(
                        seconds=self.reset_timeout
                    )
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. Resets at {reset_time}",
                    reset_time=reset_time
                )

            if state == CircuitBreakerState.HALF_OPEN:
                self._half_open_calls += 1

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except self.excluded_exceptions:
            # Don't count excluded exceptions as failures
            raise
        except Exception:
            self._record_failure()
            raise

    async def execute_async(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """
        Execute an async function through the circuit breaker.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        with self._lock:
            self._stats.total_calls += 1
            state = self.state

            if state == CircuitBreakerState.OPEN:
                self._stats.rejected_calls += 1
                reset_time = None
                if self._last_failure_time:
                    reset_time = self._last_failure_time + timedelta(
                        seconds=self.reset_timeout
                    )
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. Resets at {reset_time}",
                    reset_time=reset_time
                )

            if state == CircuitBreakerState.HALF_OPEN:
                self._half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except self.excluded_exceptions:
            raise
        except Exception:
            self._record_failure()
            raise

    def reset(self) -> None:
        """Force reset the circuit breaker to closed state."""
        with self._lock:
            self._transition_to(CircuitBreakerState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._last_failure_time = None
            logger.info("Circuit breaker manually reset")

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(state={self.state.value}, "
            f"failures={self._failure_count}/{self.failure_threshold})"
        )


# Registry for circuit breakers (for decorator use)
_circuit_breakers: dict[str, CircuitBreaker] = {}
_cb_lock = threading.Lock()


def with_circuit_breaker(
    failure_threshold: int = 5,
    reset_timeout: float = 60.0,
    half_open_max_calls: int = 1,
    excluded_exceptions: Tuple[Type[Exception], ...] = (),
    name: Optional[str] = None,
    on_state_change: Optional[Callable[[CircuitBreakerState, CircuitBreakerState], None]] = None
) -> Callable[[F], F]:
    """
    Decorator for applying circuit breaker pattern to a function.

    Wraps a function with circuit breaker protection to prevent
    cascading failures when external dependencies are failing.

    Args:
        failure_threshold: Number of failures before opening circuit
        reset_timeout: Seconds before attempting to close circuit
        half_open_max_calls: Max calls in half-open state
        excluded_exceptions: Exceptions that don't count as failures
        name: Optional name for the circuit breaker (defaults to func name)
        on_state_change: Callback on state transitions

    Returns:
        Decorated function

    Example:
        @with_circuit_breaker(failure_threshold=5, reset_timeout=60.0)
        def call_external_api():
            return requests.get("https://api.example.com/data")

        # After 5 failures, calls will raise CircuitBreakerOpenError
        # for 60 seconds before attempting recovery
    """
    def decorator(func: F) -> F:
        cb_name = name or func.__name__

        # Get or create circuit breaker for this function
        with _cb_lock:
            if cb_name not in _circuit_breakers:
                _circuit_breakers[cb_name] = CircuitBreaker(
                    failure_threshold=failure_threshold,
                    reset_timeout=reset_timeout,
                    half_open_max_calls=half_open_max_calls,
                    excluded_exceptions=excluded_exceptions,
                    on_state_change=on_state_change
                )
            cb = _circuit_breakers[cb_name]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return cb.execute(func, *args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return await cb.execute_async(func, *args, **kwargs)

        # Attach circuit breaker to wrapper for inspection
        wrapper = async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        wrapper.circuit_breaker = cb  # type: ignore

        return wrapper  # type: ignore

    return decorator


def get_circuit_breaker(name: str) -> Optional[CircuitBreaker]:
    """
    Get a circuit breaker by name.

    Args:
        name: Name of the circuit breaker

    Returns:
        CircuitBreaker instance or None
    """
    return _circuit_breakers.get(name)


def reset_all_circuit_breakers() -> None:
    """Reset all circuit breakers to closed state."""
    with _cb_lock:
        for cb in _circuit_breakers.values():
            cb.reset()
    logger.info(f"Reset {len(_circuit_breakers)} circuit breakers")


# Additional utility decorators

def with_timeout(
    timeout_seconds: float,
    raise_on_timeout: bool = True,
    default_value: Any = None
) -> Callable[[F], F]:
    """
    Decorator for adding timeout to function execution.

    Args:
        timeout_seconds: Maximum execution time in seconds
        raise_on_timeout: If True, raise TimeoutError; else return default
        default_value: Value to return on timeout if not raising

    Returns:
        Decorated function

    Example:
        @with_timeout(5.0)
        def slow_operation():
            time.sleep(10)  # Will timeout after 5 seconds
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout_seconds)
                except concurrent.futures.TimeoutError:
                    if raise_on_timeout:
                        raise TimeoutError(
                            f"'{func.__name__}' timed out after {timeout_seconds}s"
                        )
                    logger.warning(
                        f"'{func.__name__}' timed out, returning default"
                    )
                    return default_value

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                if raise_on_timeout:
                    raise TimeoutError(
                        f"'{func.__name__}' timed out after {timeout_seconds}s"
                    )
                logger.warning(
                    f"'{func.__name__}' timed out, returning default"
                )
                return default_value

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def with_fallback(fallback_func: Callable[..., T]) -> Callable[[F], F]:
    """
    Decorator that calls a fallback function on failure.

    Args:
        fallback_func: Function to call if primary fails

    Returns:
        Decorated function

    Example:
        def cached_price():
            return cache.get("price", 0.0)

        @with_fallback(cached_price)
        def get_live_price():
            return api.fetch_price()
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"'{func.__name__}' failed, using fallback: {e}"
                )
                return fallback_func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"'{func.__name__}' failed, using fallback: {e}"
                )
                if asyncio.iscoroutinefunction(fallback_func):
                    return await fallback_func(*args, **kwargs)
                return fallback_func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator
