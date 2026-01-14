"""
Middleware Setup Module.

This module provides a single function to configure all middleware
in the correct order, following the Facade Pattern.

Order matters:
1. CorrelationIdMiddleware - First to generate ID for all other middleware
2. RequestLoggingMiddleware - Second to log with correlation ID
3. RateLimitMiddleware - Third to reject before processing

Usage:
    from middleware import setup_middleware

    app = FastAPI()
    setup_middleware(app)
"""

import logging
import os
from fastapi import FastAPI

from .correlation import CorrelationIdMiddleware, CorrelationIdFilter
from .logging import RequestLoggingMiddleware
from .rate_limit import RateLimitMiddleware, create_rate_limiter

logger = logging.getLogger(__name__)


def setup_middleware(
    app: FastAPI,
    enable_rate_limiting: bool = True,
    enable_request_logging: bool = True,
    enable_correlation_id: bool = True,
    rate_limit_requests_per_minute: int = 100,
    rate_limit_burst_size: int = 20,
    rate_limit_strategy: str = "token_bucket",
    rate_limit_exclude_paths: list = None,
) -> None:
    """Configure all middleware for the FastAPI application.

    Facade Pattern: Single entry point for middleware configuration.

    Args:
        app: FastAPI application instance
        enable_rate_limiting: Enable rate limiting middleware
        enable_request_logging: Enable request logging middleware
        enable_correlation_id: Enable correlation ID middleware
        rate_limit_requests_per_minute: Max requests per minute
        rate_limit_burst_size: Max burst size for token bucket
        rate_limit_strategy: Rate limiting algorithm
        rate_limit_exclude_paths: Paths to exclude from rate limiting

    Example:
        app = FastAPI()
        setup_middleware(
            app,
            enable_rate_limiting=True,
            rate_limit_requests_per_minute=100
        )
    """
    # Load from environment variables (override parameters)
    enable_rate_limiting = enable_rate_limiting and os.environ.get(
        "ENABLE_RATE_LIMITING", "true"
    ).lower() == "true"

    enable_request_logging = enable_request_logging and os.environ.get(
        "ENABLE_REQUEST_LOGGING", "true"
    ).lower() == "true"

    rate_limit_requests_per_minute = int(os.environ.get(
        "RATE_LIMIT_REQUESTS_PER_MINUTE",
        str(rate_limit_requests_per_minute)
    ))

    rate_limit_burst_size = int(os.environ.get(
        "RATE_LIMIT_BURST",
        str(rate_limit_burst_size)
    ))

    # Default exclude paths
    if rate_limit_exclude_paths is None:
        rate_limit_exclude_paths = [
            "/health",
            "/v1/health",
            "/api/v1/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/",
        ]

    # Setup logging filter for correlation ID
    if enable_correlation_id:
        _setup_logging_filter()

    # Add middleware in REVERSE order (last added = first executed)

    # 3. Rate limiting (last to add = first to check)
    if enable_rate_limiting:
        strategy = create_rate_limiter(
            strategy=rate_limit_strategy,
            requests_per_minute=rate_limit_requests_per_minute,
            burst_size=rate_limit_burst_size,
        )

        app.add_middleware(
            RateLimitMiddleware,
            strategy=strategy,
            exclude_paths=rate_limit_exclude_paths,
            enabled=True,
        )

        logger.info(
            f"Rate limiting enabled: {rate_limit_requests_per_minute} req/min, "
            f"strategy={rate_limit_strategy}"
        )

    # 2. Request logging
    if enable_request_logging:
        app.add_middleware(
            RequestLoggingMiddleware,
            exclude_paths=set(rate_limit_exclude_paths),
            slow_request_threshold_ms=1000.0,
        )

        logger.info("Request logging enabled")

    # 1. Correlation ID (first to add = last to check, but generates ID)
    if enable_correlation_id:
        app.add_middleware(CorrelationIdMiddleware)
        logger.info("Correlation ID tracking enabled")

    logger.info(
        f"Middleware setup complete: "
        f"rate_limiting={enable_rate_limiting}, "
        f"request_logging={enable_request_logging}, "
        f"correlation_id={enable_correlation_id}"
    )


def _setup_logging_filter() -> None:
    """Add correlation ID filter to root logger."""
    root_logger = logging.getLogger()

    # Check if filter already added
    for handler in root_logger.handlers:
        for filter_ in handler.filters:
            if isinstance(filter_, CorrelationIdFilter):
                return

    # Add filter to all handlers
    correlation_filter = CorrelationIdFilter()
    for handler in root_logger.handlers:
        handler.addFilter(correlation_filter)
