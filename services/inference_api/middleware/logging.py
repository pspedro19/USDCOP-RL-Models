"""
Request Logging Middleware for Observability.

This module implements structured request/response logging,
following the Observer Pattern for non-intrusive monitoring.

Features:
- Logs all requests with timing information
- Includes correlation ID for tracing
- Configurable log level and format
- Excludes health check endpoints from verbose logging

SOLID Principles:
- SRP: Only handles request logging
- OCP: Log format can be extended without modification
"""

import time
import os
import logging
from typing import Optional, Set

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .correlation import get_correlation_id

logger = logging.getLogger("usdcop.requests")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request/response logging.

    Single Responsibility: Only handles request logging.
    Observer Pattern: Monitors requests without affecting behavior.

    Example:
        app.add_middleware(
            RequestLoggingMiddleware,
            exclude_paths={"/health", "/v1/health"},
            log_request_body=False,
            log_response_body=False
        )
    """

    def __init__(
        self,
        app,
        exclude_paths: Optional[Set[str]] = None,
        log_request_body: bool = False,
        log_response_body: bool = False,
        slow_request_threshold_ms: float = 1000.0,
    ):
        super().__init__(app)

        self.exclude_paths = exclude_paths or {
            "/health",
            "/v1/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico",
        }

        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.slow_request_threshold_ms = slow_request_threshold_ms

        self.enabled = os.environ.get(
            "ENABLE_REQUEST_LOGGING", "true"
        ).lower() == "true"

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with logging."""
        if not self.enabled:
            return await call_next(request)

        path = request.url.path

        # Skip excluded paths for verbose logging
        is_excluded = path in self.exclude_paths

        # Capture start time
        start_time = time.perf_counter()

        # Get correlation ID
        correlation_id = get_correlation_id() or getattr(
            request.state, "correlation_id", "unknown"
        )

        # Log request (unless excluded)
        if not is_excluded:
            self._log_request(request, correlation_id)

        # Process request
        try:
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log response
            if not is_excluded or response.status_code >= 400:
                self._log_response(
                    request, response, duration_ms, correlation_id
                )

            # Warn on slow requests
            if duration_ms > self.slow_request_threshold_ms:
                logger.warning(
                    f"Slow request: {request.method} {path} "
                    f"took {duration_ms:.2f}ms "
                    f"(threshold: {self.slow_request_threshold_ms}ms) "
                    f"[{correlation_id}]"
                )

            return response

        except Exception as e:
            # Log exception
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"Request failed: {request.method} {path} "
                f"after {duration_ms:.2f}ms - {type(e).__name__}: {str(e)} "
                f"[{correlation_id}]",
                exc_info=True
            )
            raise

    def _log_request(self, request: Request, correlation_id: str) -> None:
        """Log incoming request details."""
        client_ip = self._get_client_ip(request)

        log_data = {
            "event": "request_start",
            "method": request.method,
            "path": request.url.path,
            "query": str(request.query_params) if request.query_params else None,
            "client_ip": client_ip,
            "user_agent": request.headers.get("User-Agent", "unknown"),
            "correlation_id": correlation_id,
        }

        logger.info(
            f"→ {request.method} {request.url.path} "
            f"from {client_ip} [{correlation_id}]",
            extra=log_data
        )

    def _log_response(
        self,
        request: Request,
        response: Response,
        duration_ms: float,
        correlation_id: str
    ) -> None:
        """Log response details."""
        log_data = {
            "event": "request_end",
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
            "correlation_id": correlation_id,
        }

        # Choose log level based on status code
        if response.status_code >= 500:
            log_func = logger.error
            status_emoji = "✗"
        elif response.status_code >= 400:
            log_func = logger.warning
            status_emoji = "⚠"
        else:
            log_func = logger.info
            status_emoji = "✓"

        log_func(
            f"← {status_emoji} {response.status_code} {request.method} "
            f"{request.url.path} ({duration_ms:.2f}ms) [{correlation_id}]",
            extra=log_data
        )

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, handling proxies."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        if request.client:
            return request.client.host

        return "unknown"
