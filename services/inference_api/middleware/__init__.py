"""
Middleware package for USDCOP Inference API.

This package provides security and observability middleware following
Clean Code and SOLID principles:

- AuthMiddleware: API Key and JWT authentication (P0-3)
- RateLimitMiddleware: Prevents API abuse (Strategy Pattern)
- AuthenticatedRateLimitMiddleware: Per-user rate limiting
- CorrelationIdMiddleware: Request tracing (Single Responsibility)
- RequestLoggingMiddleware: Structured logging (Observer Pattern)
- SecurityHeadersMiddleware: OWASP security headers (P0 Security)
- RequestSizeLimitMiddleware: DoS protection via size limits

Usage:
    from middleware import (
        AuthMiddleware,
        RateLimitMiddleware,
        CorrelationIdMiddleware,
        RequestLoggingMiddleware,
        SecurityHeadersMiddleware,
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

# Security middleware (P0 Security)
from .security_headers import (
    SecurityHeadersMiddleware,
    RequestSizeLimitMiddleware,
    create_security_middleware_stack,
)

# Authentication middleware (P0-3)
from .auth import (
    AuthMiddleware,
    generate_api_key,
    create_jwt_token,
    get_key_prefix,
    API_KEY_HEADER,
)

# Enhanced rate limiting with per-user limits
from .rate_limiter import (
    AuthenticatedRateLimiter,
    AuthenticatedRateLimitMiddleware,
    AsyncTokenBucket,
    create_authenticated_rate_limiter,
)

__all__ = [
    # Authentication (P0-3)
    "AuthMiddleware",
    "generate_api_key",
    "create_jwt_token",
    "get_key_prefix",
    "API_KEY_HEADER",
    # Rate Limiting
    "RateLimitMiddleware",
    "RateLimitStrategy",
    "create_rate_limiter",
    "AuthenticatedRateLimiter",
    "AuthenticatedRateLimitMiddleware",
    "AsyncTokenBucket",
    "create_authenticated_rate_limiter",
    # Security Headers (P0 Security)
    "SecurityHeadersMiddleware",
    "RequestSizeLimitMiddleware",
    "create_security_middleware_stack",
    # Observability
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
