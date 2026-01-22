"""
Tracing Middleware for FastAPI Inference API.

This module provides OpenTelemetry-based request tracing middleware that
integrates with the existing correlation ID system.

Features:
    - Automatic span creation for HTTP requests
    - trace_id in response headers (X-Trace-ID)
    - Error recording with stack traces
    - Latency recording
    - Integration with existing X-Request-ID correlation

Usage:
    from services.inference_api.core.tracing_middleware import (
        TracingMiddleware,
        setup_tracing_middleware
    )

    # Setup all tracing
    setup_tracing_middleware(app, service_name="inference-api")

    # Or add middleware manually
    app.add_middleware(TracingMiddleware)
"""

import time
import logging
from typing import Callable, Optional, Set
from urllib.parse import urlparse

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class TracingMiddleware(BaseHTTPMiddleware):
    """
    OpenTelemetry tracing middleware for FastAPI.

    Creates spans for incoming HTTP requests with:
        - Request attributes (method, path, query, headers)
        - Response attributes (status code, content type)
        - Timing information (latency in ms)
        - Error details (on 4xx/5xx responses)
        - Trace ID in response headers

    Integrates with existing correlation ID middleware by using
    the X-Request-ID as a span attribute.

    Attributes:
        exclude_paths: Set of paths to exclude from tracing
        service_name: Name of the service for span naming
        record_query_params: Whether to record query parameters
        record_headers: List of headers to record

    Example:
        app.add_middleware(
            TracingMiddleware,
            exclude_paths={"/health", "/metrics"},
            service_name="inference-api"
        )
    """

    def __init__(
        self,
        app: ASGIApp,
        exclude_paths: Optional[Set[str]] = None,
        service_name: str = "inference-api",
        record_query_params: bool = True,
        record_headers: Optional[list] = None,
    ):
        """
        Initialize tracing middleware.

        Args:
            app: ASGI application
            exclude_paths: Paths to exclude from tracing
            service_name: Service name for span naming
            record_query_params: Whether to record query parameters
            record_headers: List of request headers to record
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or {
            "/health",
            "/api/v1/health",
            "/v1/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico",
        }
        self.service_name = service_name
        self.record_query_params = record_query_params
        self.record_headers = record_headers or [
            "user-agent",
            "content-type",
            "accept",
        ]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with tracing.

        Creates a span for the request if tracing is enabled and the path
        is not excluded.
        """
        # Skip excluded paths
        if self._should_skip(request.url.path):
            return await call_next(request)

        try:
            from opentelemetry import trace
            from opentelemetry.trace import StatusCode, SpanKind
        except ImportError:
            # OpenTelemetry not available, pass through
            return await call_next(request)

        tracer = trace.get_tracer(self.service_name)

        # Extract trace context from incoming request headers
        context = self._extract_context(dict(request.headers))

        # Create span name from HTTP method and path
        span_name = f"{request.method} {self._get_route_pattern(request)}"

        with tracer.start_as_current_span(
            span_name,
            context=context,
            kind=SpanKind.SERVER,
        ) as span:
            # Record request attributes
            self._record_request_attributes(span, request)

            start_time = time.perf_counter()
            response = None
            error = None

            try:
                response = await call_next(request)

                # Record response attributes
                self._record_response_attributes(span, response)

                return response

            except Exception as e:
                error = e
                self._record_error(span, e)
                raise

            finally:
                # Record latency
                latency_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("http.latency_ms", latency_ms)

                # Add trace ID to response headers
                if response is not None:
                    self._add_trace_headers(span, response)

                # Log slow requests
                if latency_ms > 1000:  # > 1 second
                    logger.warning(
                        f"Slow request: {request.method} {request.url.path} "
                        f"took {latency_ms:.2f}ms"
                    )

    def _should_skip(self, path: str) -> bool:
        """Check if path should be skipped from tracing."""
        # Exact match
        if path in self.exclude_paths:
            return True

        # Prefix match for docs/static
        for exclude_path in self.exclude_paths:
            if path.startswith(exclude_path):
                return True

        return False

    def _get_route_pattern(self, request: Request) -> str:
        """
        Get the route pattern for the request.

        Attempts to get the actual route pattern (e.g., /users/{user_id})
        rather than the resolved path (e.g., /users/123).
        """
        # Try to get route from request scope
        route = request.scope.get("route")
        if route and hasattr(route, "path"):
            return route.path

        # Fall back to actual path
        return request.url.path

    def _extract_context(self, headers: dict) -> Optional[object]:
        """Extract trace context from request headers."""
        try:
            from opentelemetry import propagate
            return propagate.extract(headers)
        except Exception:
            return None

    def _record_request_attributes(self, span, request: Request) -> None:
        """Record request attributes on the span."""
        # Standard HTTP attributes (OpenTelemetry semantic conventions)
        span.set_attribute("http.method", request.method)
        span.set_attribute("http.scheme", request.url.scheme)
        span.set_attribute("http.host", request.url.hostname or "")
        span.set_attribute("http.target", str(request.url.path))
        span.set_attribute("http.url", str(request.url))

        # Client information
        if request.client:
            span.set_attribute("http.client_ip", request.client.host)

        # Query parameters (if enabled)
        if self.record_query_params and request.url.query:
            span.set_attribute("http.query_string", request.url.query)

        # Selected headers
        for header_name in self.record_headers:
            value = request.headers.get(header_name)
            if value:
                # Sanitize header name for attribute key
                attr_name = f"http.request.header.{header_name.replace('-', '_')}"
                span.set_attribute(attr_name, value)

        # Correlation ID (from existing middleware)
        correlation_id = getattr(request.state, "correlation_id", None)
        if correlation_id:
            span.set_attribute("correlation_id", correlation_id)

        # Content length
        content_length = request.headers.get("content-length")
        if content_length:
            span.set_attribute("http.request_content_length", int(content_length))

    def _record_response_attributes(self, span, response: Response) -> None:
        """Record response attributes on the span."""
        from opentelemetry.trace import StatusCode

        span.set_attribute("http.status_code", response.status_code)

        # Content type
        content_type = response.headers.get("content-type")
        if content_type:
            span.set_attribute("http.response.content_type", content_type)

        # Content length
        content_length = response.headers.get("content-length")
        if content_length:
            span.set_attribute("http.response_content_length", int(content_length))

        # Set span status based on HTTP status code
        if response.status_code >= 500:
            span.set_status(StatusCode.ERROR, f"HTTP {response.status_code}")
        elif response.status_code >= 400:
            span.set_status(StatusCode.ERROR, f"HTTP {response.status_code}")
        else:
            span.set_status(StatusCode.OK)

    def _record_error(self, span, exception: Exception) -> None:
        """Record error details on the span."""
        from opentelemetry.trace import StatusCode

        span.record_exception(exception)
        span.set_status(StatusCode.ERROR, str(exception))
        span.set_attribute("error.type", type(exception).__name__)
        span.set_attribute("error.message", str(exception))

    def _add_trace_headers(self, span, response: Response) -> None:
        """Add trace context headers to response."""
        try:
            # Get trace ID from current span
            span_context = span.get_span_context()
            if span_context.is_valid:
                trace_id = format(span_context.trace_id, "032x")
                span_id = format(span_context.span_id, "016x")

                response.headers["X-Trace-ID"] = trace_id
                response.headers["X-Span-ID"] = span_id

        except Exception as e:
            logger.debug(f"Could not add trace headers: {e}")


def setup_tracing_middleware(
    app: FastAPI,
    service_name: str = "inference-api",
    jaeger_endpoint: Optional[str] = None,
    sampling_rate: float = 0.1,
    enabled: bool = True,
    exclude_paths: Optional[Set[str]] = None,
) -> bool:
    """
    Setup complete tracing infrastructure for FastAPI application.

    This function:
    1. Initializes OpenTelemetry with Jaeger exporter
    2. Adds tracing middleware to the application
    3. Configures auto-instrumentation

    Args:
        app: FastAPI application instance
        service_name: Name of the service
        jaeger_endpoint: Jaeger collector endpoint
        sampling_rate: Trace sampling rate (0.0-1.0)
        enabled: Whether tracing is enabled
        exclude_paths: Paths to exclude from tracing

    Returns:
        True if setup was successful

    Example:
        app = FastAPI()
        setup_tracing_middleware(
            app,
            service_name="inference-api",
            jaeger_endpoint="http://jaeger:14268/api/traces",
            sampling_rate=0.1
        )
    """
    import os

    # Check if enabled
    enabled = enabled and os.environ.get("OTEL_ENABLED", "true").lower() == "true"

    if not enabled:
        logger.info("Tracing is disabled")
        return False

    try:
        # Initialize OpenTelemetry
        import sys
        sys.path.insert(0, str(app.state.get("project_root", "")))

        try:
            from src.shared.tracing import init_tracing
        except ImportError:
            # Try relative import for when running as module
            try:
                from shared.tracing import init_tracing
            except ImportError:
                logger.warning("Could not import tracing module")
                return False

        success = init_tracing(
            service_name=service_name,
            jaeger_endpoint=jaeger_endpoint,
            sampling_rate=sampling_rate,
        )

        if not success:
            logger.warning("OpenTelemetry initialization failed")
            return False

        # Add tracing middleware
        # Note: Must be added in reverse order - last added is first executed
        # Add it AFTER correlation ID middleware but BEFORE rate limiting
        app.add_middleware(
            TracingMiddleware,
            exclude_paths=exclude_paths,
            service_name=service_name,
        )

        logger.info(
            f"Tracing middleware setup complete: "
            f"service={service_name}, sampling_rate={sampling_rate}"
        )

        return True

    except ImportError as e:
        logger.warning(f"OpenTelemetry packages not installed: {e}")
        return False

    except Exception as e:
        logger.error(f"Failed to setup tracing middleware: {e}")
        return False


def instrument_fastapi(app: FastAPI) -> None:
    """
    Apply OpenTelemetry auto-instrumentation to FastAPI app.

    This provides automatic tracing for all routes without needing
    the custom middleware.

    Args:
        app: FastAPI application to instrument

    Example:
        app = FastAPI()
        instrument_fastapi(app)
    """
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls="health,docs,redoc,openapi.json",
        )
        logger.info("FastAPI auto-instrumentation applied")

    except ImportError:
        logger.debug("FastAPI instrumentation not available")

    except Exception as e:
        logger.warning(f"FastAPI instrumentation failed: {e}")


class SpanContextMiddleware(BaseHTTPMiddleware):
    """
    Lightweight middleware for span context propagation.

    Use this instead of TracingMiddleware when you only need
    context propagation without automatic span creation.

    The auto-instrumentation handles span creation, so this middleware
    just ensures context is properly extracted and correlation IDs
    are linked.

    Example:
        app.add_middleware(SpanContextMiddleware)
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Extract and propagate span context."""
        try:
            from opentelemetry import trace, propagate

            # Extract context from headers
            context = propagate.extract(dict(request.headers))

            # Get current span
            span = trace.get_current_span()

            # Link correlation ID to span
            correlation_id = getattr(request.state, "correlation_id", None)
            if span and correlation_id:
                span.set_attribute("correlation_id", correlation_id)

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Span context propagation error: {e}")

        return await call_next(request)
