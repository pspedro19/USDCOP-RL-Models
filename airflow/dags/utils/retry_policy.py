# -*- coding: utf-8 -*-
"""
Retry Policy with Exponential Backoff
=====================================
Standardized retry mechanisms for all data extraction operations.

Contract: CTR-L0-RETRY-001

This module provides:
1. Exponential backoff with jitter
2. Circuit breaker integration
3. DLQ integration for permanent failures
4. Configurable retry strategies per source

Usage:
    from utils.retry_policy import (
        retry_with_backoff,
        RetryConfig,
        get_retry_decorator,
    )

    @retry_with_backoff(max_attempts=5, base_delay=2)
    def fetch_data():
        ...

    # Or with custom config
    config = RetryConfig.for_source('fred')

    @get_retry_decorator(config)
    def extract_fred_data():
        ...

Version: 1.0.0
"""

import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

logger = logging.getLogger(__name__)


# =============================================================================
# RETRY CONFIGURATION
# =============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 5
    base_delay_seconds: float = 2.0
    max_delay_seconds: float = 300.0  # 5 minutes max
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_range: Tuple[float, float] = (0.5, 1.5)

    # Exceptions to retry on
    retry_exceptions: Tuple[Type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        IOError,
    )

    # Exceptions to NOT retry (fail immediately)
    fatal_exceptions: Tuple[Type[Exception], ...] = (
        ValueError,
        TypeError,
        KeyError,
    )

    # Callback for logging/monitoring
    on_retry: Optional[Callable[[int, Exception, float], None]] = None
    on_failure: Optional[Callable[[Exception, int], None]] = None
    on_success: Optional[Callable[[int], None]] = None

    # DLQ integration
    dlq_on_permanent_failure: bool = True
    source_name: str = "unknown"
    variable_name: str = "unknown"

    @classmethod
    def for_source(cls, source: str) -> 'RetryConfig':
        """Get recommended retry config for a data source."""
        configs = {
            'fred': cls(
                max_attempts=5,
                base_delay_seconds=2.0,
                max_delay_seconds=120.0,
                source_name='fred',
            ),
            'investing': cls(
                max_attempts=5,
                base_delay_seconds=5.0,  # Slower - rate limited
                max_delay_seconds=300.0,
                source_name='investing',
            ),
            'suameca': cls(
                max_attempts=4,
                base_delay_seconds=3.0,
                max_delay_seconds=180.0,
                source_name='suameca',
            ),
            'bcrp': cls(
                max_attempts=4,
                base_delay_seconds=2.0,
                max_delay_seconds=120.0,
                source_name='bcrp',
            ),
            'twelvedata': cls(
                max_attempts=3,
                base_delay_seconds=10.0,  # API rate limit
                max_delay_seconds=60.0,
                source_name='twelvedata',
            ),
            'fedesarrollo': cls(
                max_attempts=3,
                base_delay_seconds=5.0,
                max_delay_seconds=120.0,
                source_name='fedesarrollo',
            ),
            'dane': cls(
                max_attempts=3,
                base_delay_seconds=5.0,
                max_delay_seconds=120.0,
                source_name='dane',
            ),
        }
        return configs.get(source, cls(source_name=source))


# =============================================================================
# RETRY STATISTICS
# =============================================================================

@dataclass
class RetryStats:
    """Statistics for retry operations."""
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_delay_seconds: float = 0.0
    last_attempt_time: Optional[datetime] = None
    last_error: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)

    def record_attempt(self, success: bool, delay: float, error: Optional[str] = None):
        """Record a retry attempt."""
        self.total_attempts += 1
        self.total_delay_seconds += delay
        self.last_attempt_time = datetime.utcnow()

        if success:
            self.successful_attempts += 1
        else:
            self.failed_attempts += 1
            self.last_error = error

        self.history.append({
            'timestamp': self.last_attempt_time.isoformat(),
            'success': success,
            'delay_seconds': delay,
            'error': error,
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_attempts': self.total_attempts,
            'successful_attempts': self.successful_attempts,
            'failed_attempts': self.failed_attempts,
            'total_delay_seconds': round(self.total_delay_seconds, 2),
            'success_rate': round(
                self.successful_attempts / max(self.total_attempts, 1), 3
            ),
            'last_error': self.last_error,
        }


# =============================================================================
# RETRY DECORATORS
# =============================================================================

def calculate_delay(
    attempt: int,
    config: RetryConfig,
) -> float:
    """
    Calculate delay for next retry with exponential backoff and jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration

    Returns:
        Delay in seconds
    """
    # Exponential backoff: base_delay * (exponential_base ^ attempt)
    delay = config.base_delay_seconds * (config.exponential_base ** attempt)

    # Cap at max delay
    delay = min(delay, config.max_delay_seconds)

    # Add jitter to prevent thundering herd
    if config.jitter:
        jitter_min, jitter_max = config.jitter_range
        delay *= random.uniform(jitter_min, jitter_max)

    return delay


def retry_with_backoff(
    max_attempts: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 300.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    fatal_exceptions: Tuple[Type[Exception], ...] = (),
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_attempts: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Whether to add jitter to delays
        retry_exceptions: Exception types to retry on
        fatal_exceptions: Exception types to fail immediately on
        on_retry: Callback(attempt, exception, delay) called before each retry

    Returns:
        Decorated function
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay_seconds=base_delay,
        max_delay_seconds=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retry_exceptions=retry_exceptions,
        fatal_exceptions=fatal_exceptions,
        on_retry=on_retry,
    )

    return get_retry_decorator(config)


def get_retry_decorator(config: RetryConfig):
    """
    Create a retry decorator from a RetryConfig.

    Args:
        config: Retry configuration

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            stats = RetryStats()
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    result = func(*args, **kwargs)

                    # Success - record and return
                    stats.record_attempt(True, 0)
                    if config.on_success:
                        config.on_success(attempt + 1)

                    logger.debug(
                        "[RETRY] %s succeeded on attempt %d",
                        func.__name__, attempt + 1
                    )
                    return result

                except config.fatal_exceptions as e:
                    # Fatal exception - don't retry
                    logger.error(
                        "[RETRY] %s fatal error (no retry): %s",
                        func.__name__, str(e)
                    )
                    raise

                except config.retry_exceptions as e:
                    last_exception = e
                    delay = calculate_delay(attempt, config)
                    stats.record_attempt(False, delay, str(e))

                    if attempt < config.max_attempts - 1:
                        # More attempts remaining
                        logger.warning(
                            "[RETRY] %s attempt %d/%d failed: %s. "
                            "Retrying in %.1fs...",
                            func.__name__, attempt + 1, config.max_attempts,
                            str(e)[:100], delay
                        )

                        if config.on_retry:
                            config.on_retry(attempt + 1, e, delay)

                        time.sleep(delay)
                    else:
                        # Final attempt failed
                        logger.error(
                            "[RETRY] %s FAILED after %d attempts. Last error: %s",
                            func.__name__, config.max_attempts, str(e)
                        )

                except Exception as e:
                    # Unexpected exception - check if retryable
                    if isinstance(e, config.retry_exceptions):
                        last_exception = e
                        delay = calculate_delay(attempt, config)
                        stats.record_attempt(False, delay, str(e))

                        if attempt < config.max_attempts - 1:
                            time.sleep(delay)
                        continue

                    # Not retryable
                    logger.error(
                        "[RETRY] %s unexpected error: %s",
                        func.__name__, str(e)
                    )
                    raise

            # All retries exhausted
            if config.on_failure:
                config.on_failure(last_exception, config.max_attempts)

            # Save to DLQ if configured
            if config.dlq_on_permanent_failure:
                _save_to_dlq(
                    config.source_name,
                    config.variable_name,
                    last_exception,
                    {'function': func.__name__, 'stats': stats.to_dict()}
                )

            raise last_exception

        # Attach stats to wrapper for inspection
        wrapper._retry_stats = RetryStats()
        return wrapper

    return decorator


def _save_to_dlq(
    source: str,
    variable: str,
    exception: Exception,
    payload: Dict[str, Any],
):
    """Save failed extraction to DLQ."""
    try:
        from services.dlq_service import get_dlq_service
        dlq = get_dlq_service()
        dlq.save_failed_extraction(
            source=source,
            variable=variable,
            error=str(exception),
            payload=payload,
            error_type=type(exception).__name__,
        )
    except ImportError:
        logger.warning("[RETRY] DLQ service not available for saving failure")
    except Exception as e:
        logger.error("[RETRY] Failed to save to DLQ: %s", e)


# =============================================================================
# ASYNC RETRY (for asyncio)
# =============================================================================

def async_retry_with_backoff(
    max_attempts: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 300.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Async decorator for retrying coroutines with exponential backoff.

    Usage:
        @async_retry_with_backoff(max_attempts=3)
        async def fetch_data():
            ...
    """
    import asyncio

    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay_seconds=base_delay,
        max_delay_seconds=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retry_exceptions=retry_exceptions,
    )

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)

                except config.retry_exceptions as e:
                    last_exception = e
                    delay = calculate_delay(attempt, config)

                    if attempt < config.max_attempts - 1:
                        logger.warning(
                            "[ASYNC-RETRY] %s attempt %d/%d failed. "
                            "Retrying in %.1fs...",
                            func.__name__, attempt + 1, config.max_attempts, delay
                        )
                        await asyncio.sleep(delay)

            raise last_exception

        return wrapper

    return decorator


# =============================================================================
# CONTEXT MANAGER FOR MANUAL RETRY
# =============================================================================

class RetryContext:
    """
    Context manager for manual retry control.

    Usage:
        with RetryContext(max_attempts=5, source='fred') as ctx:
            for attempt in ctx:
                try:
                    result = risky_operation()
                    ctx.success()
                    break
                except ConnectionError as e:
                    ctx.fail(e)
    """

    def __init__(
        self,
        max_attempts: int = 5,
        base_delay: float = 2.0,
        source: str = 'unknown',
        variable: str = 'unknown',
    ):
        self.config = RetryConfig(
            max_attempts=max_attempts,
            base_delay_seconds=base_delay,
            source_name=source,
            variable_name=variable,
        )
        self.stats = RetryStats()
        self.current_attempt = 0
        self._succeeded = False
        self._last_exception = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._succeeded and self._last_exception:
            _save_to_dlq(
                self.config.source_name,
                self.config.variable_name,
                self._last_exception,
                {'stats': self.stats.to_dict()}
            )
        return False

    def __iter__(self):
        for attempt in range(self.config.max_attempts):
            self.current_attempt = attempt + 1
            yield attempt + 1

            if self._succeeded:
                break

    def success(self):
        """Mark current attempt as successful."""
        self._succeeded = True
        self.stats.record_attempt(True, 0)

    def fail(self, exception: Exception):
        """Mark current attempt as failed and sleep before next."""
        self._last_exception = exception
        delay = calculate_delay(self.current_attempt - 1, self.config)
        self.stats.record_attempt(False, delay, str(exception))

        if self.current_attempt < self.config.max_attempts:
            logger.warning(
                "[RETRY-CTX] Attempt %d/%d failed: %s. Sleeping %.1fs...",
                self.current_attempt, self.config.max_attempts,
                str(exception)[:100], delay
            )
            time.sleep(delay)

    @property
    def succeeded(self) -> bool:
        return self._succeeded


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def retry_extraction(
    source: str,
    variable: str,
    func: Callable,
    *args,
    **kwargs,
) -> Any:
    """
    Retry an extraction function with source-specific config.

    Args:
        source: Data source name (fred, investing, etc.)
        variable: Variable being extracted
        func: Function to call
        *args: Arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Result of func

    Raises:
        Exception: If all retries fail
    """
    config = RetryConfig.for_source(source)
    config.variable_name = variable

    @get_retry_decorator(config)
    def wrapper():
        return func(*args, **kwargs)

    return wrapper()
