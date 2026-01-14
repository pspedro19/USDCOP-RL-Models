"""
Middleware package for USDCOP Inference API.

This package provides security and observability middleware following
Clean Code and SOLID principles:

- RateLimitMiddleware: Prevents API abuse (Strategy Pattern)
- CorrelationIdMiddleware: Request tracing (Single Responsibility)
- RequestLoggingMiddleware: Structured logging (Observer Pattern)

Usage:
    from middleware import (
        RateLimitMiddleware,
        CorrelationIdMiddleware,
        RequestLoggingMiddleware,
        setup_middleware
    )

    setup_middleware(app)
"""

from .rate_limit import RateLimitMiddleware, RateLimitStrategy, create_rate_limiter
from .correlation import CorrelationIdMiddleware
from .logging import RequestLoggingMiddleware
from .setup import setup_middleware
from .errors import (
    ErrorCode,
    ErrorDetail,
    APIException,
    ValidationException,
    NotFoundError,
    ModelNotFoundError,
    DatabaseError,
    InferenceError,
    RateLimitError,
    setup_exception_handlers,
)

__all__ = [
    # Middleware
    "RateLimitMiddleware",
    "RateLimitStrategy",
    "create_rate_limiter",
    "CorrelationIdMiddleware",
    "RequestLoggingMiddleware",
    "setup_middleware",
    # Error handling
    "ErrorCode",
    "ErrorDetail",
    "APIException",
    "ValidationException",
    "NotFoundError",
    "ModelNotFoundError",
    "DatabaseError",
    "InferenceError",
    "RateLimitError",
    "setup_exception_handlers",
]
