"""
Resilience Package
=================
Circuit breakers, fallback strategies, and resilience patterns.
"""

from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
from .breaker_registry import CircuitBreakerRegistry
from .fallback_strategies import FallbackStrategy, FallbackManager

# Additional resilience utilities
import time
import logging
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0
):
    """Decorator for retrying functions with exponential backoff"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = base_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(min(delay, max_delay))
                        delay *= exponential_base
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}")
            
            raise last_exception
        
        return wrapper
    return decorator


__all__ = [
    'CircuitBreaker',
    'CircuitBreakerConfig',
    'CircuitState', 
    'CircuitBreakerRegistry',
    'FallbackStrategy',
    'FallbackManager',
    'retry_with_backoff'
]
