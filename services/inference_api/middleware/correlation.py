"""
Correlation ID Middleware for Request Tracing.

This module implements request correlation for distributed tracing,
following the Single Responsibility Principle.

Features:
- Generates or propagates X-Request-ID headers
- Stores correlation ID in request state for logging
- Adds correlation ID to response headers

Usage:
    app.add_middleware(CorrelationIdMiddleware)

    # In route handlers:
    correlation_id = request.state.correlation_id
"""

import uuid
import logging
from contextvars import ContextVar
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Context variable for correlation ID (thread-safe)
correlation_id_var: ContextVar[Optional[str]] = ContextVar(
    "correlation_id", default=None
)


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID from context.

    Use this in logging handlers or service calls.

    Example:
        logger.info(f"Processing request", extra={"correlation_id": get_correlation_id()})
    """
    return correlation_id_var.get()


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware for propagating correlation IDs across requests.

    Single Responsibility: Only handles correlation ID management.

    The correlation ID is:
    1. Read from incoming X-Request-ID header (if present)
    2. Generated as UUID4 if not present
    3. Stored in request.state.correlation_id
    4. Added to response headers
    5. Stored in context variable for logging

    Example:
        app.add_middleware(CorrelationIdMiddleware)

        @app.get("/api/v1/data")
        async def get_data(request: Request):
            # Access correlation ID
            correlation_id = request.state.correlation_id
            logger.info(f"Processing request {correlation_id}")
            return {"correlation_id": correlation_id}
    """

    HEADER_NAME = "X-Request-ID"

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with correlation ID tracking."""
        # Get or generate correlation ID
        correlation_id = request.headers.get(self.HEADER_NAME)

        if not correlation_id:
            correlation_id = str(uuid.uuid4())

        # Store in request state for route handlers
        request.state.correlation_id = correlation_id

        # Store in context variable for logging
        token = correlation_id_var.set(correlation_id)

        try:
            # Process request
            response = await call_next(request)

            # Add correlation ID to response
            response.headers[self.HEADER_NAME] = correlation_id

            return response

        finally:
            # Reset context variable
            correlation_id_var.reset(token)


class CorrelationIdFilter(logging.Filter):
    """Logging filter that adds correlation ID to log records.

    Use with Python's logging to automatically include correlation ID.

    Example:
        handler = logging.StreamHandler()
        handler.addFilter(CorrelationIdFilter())
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(correlation_id)s - %(message)s'
        ))
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation_id to log record."""
        record.correlation_id = get_correlation_id() or "no-correlation-id"
        return True
