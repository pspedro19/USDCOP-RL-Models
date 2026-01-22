"""
Security Headers Middleware
===========================

Adds essential security headers to all HTTP responses to protect
against common web vulnerabilities.

P0 Security: OWASP Security Headers

Headers Added:
    - X-Content-Type-Options: Prevents MIME type sniffing
    - X-Frame-Options: Prevents clickjacking attacks
    - Strict-Transport-Security: Enforces HTTPS connections
    - Content-Security-Policy: Controls resource loading
    - X-XSS-Protection: Legacy XSS filter (for older browsers)
    - Referrer-Policy: Controls referrer information
    - Permissions-Policy: Controls browser features
    - Cache-Control: Prevents caching of sensitive responses

Author: Trading Team
Version: 1.0.0
Date: 2026-01-17
"""

import logging
import os
from typing import Callable, Optional, List

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds security headers to all responses.

    This middleware implements OWASP security header recommendations
    to protect against common web vulnerabilities like XSS, clickjacking,
    and MIME-type sniffing.

    Usage:
        from middleware.security_headers import SecurityHeadersMiddleware

        app.add_middleware(SecurityHeadersMiddleware)

        # Or with custom configuration:
        app.add_middleware(
            SecurityHeadersMiddleware,
            csp_directives="default-src 'self'; script-src 'self'",
            hsts_max_age=63072000,
        )
    """

    def __init__(
        self,
        app: ASGIApp,
        csp_directives: Optional[str] = None,
        hsts_max_age: int = 31536000,
        include_subdomains: bool = True,
        exclude_paths: Optional[List[str]] = None,
    ):
        """
        Initialize the security headers middleware.

        Args:
            app: The ASGI application
            csp_directives: Custom Content-Security-Policy directives
            hsts_max_age: Max age for HSTS header (seconds, default: 1 year)
            include_subdomains: Include subdomains in HSTS
            exclude_paths: Paths to exclude from security headers
        """
        super().__init__(app)

        self.hsts_max_age = hsts_max_age
        self.include_subdomains = include_subdomains
        self.exclude_paths = exclude_paths or []

        # Default CSP for API services (relaxed for API usage)
        self.csp_directives = csp_directives or self._default_api_csp()

        # Check if we're in production
        self.is_production = os.getenv("ENVIRONMENT", "development").lower() == "production"

        logger.info(
            f"SecurityHeadersMiddleware initialized: "
            f"production={self.is_production}, hsts_max_age={hsts_max_age}"
        )

    def _default_api_csp(self) -> str:
        """
        Return default Content-Security-Policy for API services.

        This is more permissive than a web application CSP because
        API services primarily return JSON, not HTML.
        """
        return "; ".join([
            "default-src 'self'",
            "frame-ancestors 'none'",
            "form-action 'self'",
            "base-uri 'self'",
        ])

    def _should_add_headers(self, path: str) -> bool:
        """Check if security headers should be added for this path."""
        for exclude_path in self.exclude_paths:
            if path.startswith(exclude_path):
                return False
        return True

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """
        Process request and add security headers to response.

        Args:
            request: The incoming request
            call_next: The next middleware/handler

        Returns:
            Response with security headers added
        """
        response = await call_next(request)

        # Skip excluded paths
        if not self._should_add_headers(request.url.path):
            return response

        # ========================================
        # OWASP Recommended Security Headers
        # ========================================

        # Prevent MIME type sniffing
        # Stops browsers from trying to detect content type
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Prevent clickjacking by disabling iframes
        # DENY: Page cannot be displayed in any iframe
        response.headers["X-Frame-Options"] = "DENY"

        # XSS Protection for legacy browsers
        # Modern browsers have built-in protection, but this helps older ones
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Control how much referrer information is sent
        # strict-origin-when-cross-origin: Full URL for same-origin, origin only for cross-origin
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Content Security Policy
        # Controls which resources the browser can load
        response.headers["Content-Security-Policy"] = self.csp_directives

        # Permissions Policy (formerly Feature-Policy)
        # Disables potentially dangerous browser features
        response.headers["Permissions-Policy"] = (
            "accelerometer=(), "
            "camera=(), "
            "geolocation=(), "
            "gyroscope=(), "
            "magnetometer=(), "
            "microphone=(), "
            "payment=(), "
            "usb=()"
        )

        # HTTP Strict Transport Security (HSTS)
        # Only add in production to avoid issues with local development
        if self.is_production:
            hsts_value = f"max-age={self.hsts_max_age}"
            if self.include_subdomains:
                hsts_value += "; includeSubDomains"
            response.headers["Strict-Transport-Security"] = hsts_value

        # Cache-Control for API responses
        # Prevent caching of sensitive API responses
        if request.url.path.startswith("/api/") or request.url.path.startswith("/v1/"):
            if "Cache-Control" not in response.headers:
                response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
                response.headers["Pragma"] = "no-cache"

        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to limit request body size.

    Prevents denial-of-service attacks by rejecting requests
    with bodies that exceed a specified size limit.

    Usage:
        app.add_middleware(
            RequestSizeLimitMiddleware,
            max_content_length=1_000_000  # 1MB
        )
    """

    def __init__(
        self,
        app: ASGIApp,
        max_content_length: int = 1_000_000,  # 1MB default
    ):
        """
        Initialize the request size limit middleware.

        Args:
            app: The ASGI application
            max_content_length: Maximum allowed request body size in bytes
        """
        super().__init__(app)
        self.max_content_length = max_content_length
        logger.info(f"RequestSizeLimitMiddleware initialized: max_size={max_content_length} bytes")

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """
        Check request size and reject if too large.

        Args:
            request: The incoming request
            call_next: The next middleware/handler

        Returns:
            Response or 413 error if request too large
        """
        content_length = request.headers.get("content-length")

        if content_length:
            try:
                if int(content_length) > self.max_content_length:
                    logger.warning(
                        f"Request too large: {content_length} bytes "
                        f"(max: {self.max_content_length})"
                    )
                    from starlette.responses import JSONResponse
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": "Request Entity Too Large",
                            "detail": f"Request body exceeds maximum allowed size of {self.max_content_length} bytes",
                            "max_size_bytes": self.max_content_length,
                        }
                    )
            except ValueError:
                pass  # Invalid content-length header, let the framework handle it

        return await call_next(request)


def create_security_middleware_stack(
    app: ASGIApp,
    production_mode: bool = False,
    max_request_size: int = 1_000_000,
    custom_csp: Optional[str] = None,
) -> None:
    """
    Convenience function to add all security middleware to an app.

    Args:
        app: The FastAPI/Starlette application
        production_mode: Whether running in production (enables HSTS)
        max_request_size: Maximum request body size in bytes
        custom_csp: Custom Content-Security-Policy directives

    Example:
        from middleware.security_headers import create_security_middleware_stack

        create_security_middleware_stack(
            app,
            production_mode=True,
            max_request_size=5_000_000  # 5MB
        )
    """
    # Request size limit (should be first to reject large requests early)
    app.add_middleware(
        RequestSizeLimitMiddleware,
        max_content_length=max_request_size,
    )

    # Security headers
    app.add_middleware(
        SecurityHeadersMiddleware,
        csp_directives=custom_csp,
    )

    logger.info("Security middleware stack configured")


__all__ = [
    "SecurityHeadersMiddleware",
    "RequestSizeLimitMiddleware",
    "create_security_middleware_stack",
]
