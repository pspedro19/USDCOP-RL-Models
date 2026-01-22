"""
OpenTelemetry Setup Module for USDCOP Trading Project.

This module provides centralized OpenTelemetry configuration for distributed
tracing across all services in the trading platform.

Features:
    - TracerProvider with Jaeger exporter
    - BatchSpanProcessor for efficient span export
    - Resource attributes for service identification
    - Auto-instrumentation for FastAPI, requests, psycopg2, redis
    - Context propagation for distributed tracing
    - Configurable sampling strategies

Usage:
    from shared.tracing import init_tracing, get_tracer

    # Initialize at application startup
    init_tracing(
        service_name="inference-api",
        jaeger_endpoint="http://jaeger:14268/api/traces"
    )

    # Get a tracer for creating spans
    tracer = get_tracer(__name__)

Environment Variables:
    OTEL_EXPORTER_JAEGER_ENDPOINT: Jaeger collector endpoint
    OTEL_SERVICE_NAME: Service name override
    OTEL_SERVICE_VERSION: Service version
    OTEL_SAMPLING_RATE: Sampling rate (0.0-1.0)
    OTEL_ENABLED: Enable/disable tracing (default: true)
"""

import os
import logging
from typing import Optional, Dict, Any
from contextvars import ContextVar

logger = logging.getLogger(__name__)

# Context variable for current span (thread-safe)
_current_span_context: ContextVar[Optional[Any]] = ContextVar(
    "current_span_context", default=None
)

# Global tracer provider (initialized once)
_tracer_provider: Optional[Any] = None
_initialized: bool = False


def init_tracing(
    service_name: str,
    jaeger_endpoint: Optional[str] = None,
    service_version: str = "1.0.0",
    sampling_rate: float = 0.1,
    additional_attributes: Optional[Dict[str, str]] = None,
    enabled: bool = True,
) -> bool:
    """
    Initialize OpenTelemetry tracing with Jaeger exporter.

    This function sets up the complete tracing infrastructure including:
    - TracerProvider with resource attributes
    - JaegerExporter for sending traces to Jaeger
    - BatchSpanProcessor for efficient batching
    - Auto-instrumentation for common libraries

    Args:
        service_name: Name of the service (e.g., "inference-api")
        jaeger_endpoint: Jaeger collector HTTP endpoint
            (default: from OTEL_EXPORTER_JAEGER_ENDPOINT or http://localhost:14268/api/traces)
        service_version: Version of the service
        sampling_rate: Probability of sampling (0.0-1.0)
        additional_attributes: Extra resource attributes to add
        enabled: Whether tracing is enabled

    Returns:
        True if initialization was successful, False otherwise

    Example:
        success = init_tracing(
            service_name="inference-api",
            jaeger_endpoint="http://jaeger:14268/api/traces",
            service_version="2.0.0",
            sampling_rate=0.1,
            additional_attributes={"deployment.environment": "production"}
        )
    """
    global _tracer_provider, _initialized

    # Check if tracing should be enabled
    enabled = enabled and os.environ.get("OTEL_ENABLED", "true").lower() == "true"

    if not enabled:
        logger.info("OpenTelemetry tracing is disabled")
        _initialized = True
        return True

    if _initialized:
        logger.warning("OpenTelemetry already initialized, skipping")
        return True

    try:
        # Import OpenTelemetry components
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased, ParentBased
        from opentelemetry.propagate import set_global_textmap
        from opentelemetry.propagators.b3 import B3MultiFormat

        # Get configuration from environment or parameters
        jaeger_endpoint = jaeger_endpoint or os.environ.get(
            "OTEL_EXPORTER_JAEGER_ENDPOINT",
            "http://localhost:14268/api/traces"
        )
        service_name = os.environ.get("OTEL_SERVICE_NAME", service_name)
        service_version = os.environ.get("OTEL_SERVICE_VERSION", service_version)
        sampling_rate = float(os.environ.get("OTEL_SAMPLING_RATE", str(sampling_rate)))

        # Build resource attributes
        resource_attributes = {
            SERVICE_NAME: service_name,
            SERVICE_VERSION: service_version,
            "service.namespace": "usdcop-trading",
            "deployment.environment": os.environ.get("DEPLOYMENT_ENV", "development"),
        }

        # Add custom attributes
        if additional_attributes:
            resource_attributes.update(additional_attributes)

        resource = Resource.create(resource_attributes)

        # Configure sampler (ParentBased respects parent sampling decision)
        root_sampler = TraceIdRatioBased(sampling_rate)
        sampler = ParentBased(root=root_sampler)

        # Create TracerProvider
        _tracer_provider = TracerProvider(
            resource=resource,
            sampler=sampler,
        )

        # Parse Jaeger endpoint to extract host/port
        # Jaeger thrift exporter uses host:port, not full URL
        jaeger_host = "localhost"
        jaeger_port = 6831  # Default UDP agent port

        if jaeger_endpoint:
            # Handle HTTP collector endpoint format
            if "/api/traces" in jaeger_endpoint:
                # HTTP collector format: http://host:14268/api/traces
                from urllib.parse import urlparse
                parsed = urlparse(jaeger_endpoint)
                jaeger_host = parsed.hostname or "localhost"
                # Use thrift port for direct exporter
                jaeger_port = 6831
            else:
                # Direct host:port format
                parts = jaeger_endpoint.replace("http://", "").replace("https://", "").split(":")
                jaeger_host = parts[0]
                jaeger_port = int(parts[1]) if len(parts) > 1 else 6831

        # Configure Jaeger exporter (using UDP agent)
        jaeger_exporter = JaegerExporter(
            agent_host_name=jaeger_host,
            agent_port=jaeger_port,
        )

        # Add BatchSpanProcessor for efficient export
        # Optimized for low latency (~1-2ms overhead)
        span_processor = BatchSpanProcessor(
            jaeger_exporter,
            max_queue_size=2048,
            max_export_batch_size=512,
            schedule_delay_millis=5000,  # 5 second batches
            export_timeout_millis=30000,
        )

        _tracer_provider.add_span_processor(span_processor)

        # Set as global tracer provider
        trace.set_tracer_provider(_tracer_provider)

        # Configure context propagation (B3 for compatibility)
        set_global_textmap(B3MultiFormat())

        # Auto-instrument libraries
        _setup_auto_instrumentation()

        _initialized = True
        logger.info(
            f"OpenTelemetry initialized: service={service_name}, "
            f"version={service_version}, jaeger={jaeger_host}:{jaeger_port}, "
            f"sampling={sampling_rate}"
        )

        return True

    except ImportError as e:
        logger.warning(
            f"OpenTelemetry packages not installed, tracing disabled: {e}"
        )
        _initialized = True
        return False

    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry: {e}")
        _initialized = True
        return False


def _setup_auto_instrumentation() -> None:
    """
    Configure auto-instrumentation for common libraries.

    Instruments:
        - FastAPI: HTTP requests and responses
        - requests: Outbound HTTP calls
        - psycopg2: PostgreSQL database queries
        - redis: Redis operations
        - asyncpg: Async PostgreSQL operations
    """
    try:
        # FastAPI instrumentation
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor().instrument()
        logger.debug("FastAPI auto-instrumentation enabled")
    except ImportError:
        logger.debug("FastAPI instrumentation not available")
    except Exception as e:
        logger.warning(f"FastAPI instrumentation failed: {e}")

    try:
        # Requests instrumentation
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
        RequestsInstrumentor().instrument()
        logger.debug("Requests auto-instrumentation enabled")
    except ImportError:
        logger.debug("Requests instrumentation not available")
    except Exception as e:
        logger.warning(f"Requests instrumentation failed: {e}")

    try:
        # psycopg2 instrumentation
        from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
        Psycopg2Instrumentor().instrument()
        logger.debug("psycopg2 auto-instrumentation enabled")
    except ImportError:
        logger.debug("psycopg2 instrumentation not available")
    except Exception as e:
        logger.warning(f"psycopg2 instrumentation failed: {e}")

    try:
        # Redis instrumentation
        from opentelemetry.instrumentation.redis import RedisInstrumentor
        RedisInstrumentor().instrument()
        logger.debug("Redis auto-instrumentation enabled")
    except ImportError:
        logger.debug("Redis instrumentation not available")
    except Exception as e:
        logger.warning(f"Redis instrumentation failed: {e}")

    try:
        # asyncpg instrumentation
        from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
        AsyncPGInstrumentor().instrument()
        logger.debug("asyncpg auto-instrumentation enabled")
    except ImportError:
        logger.debug("asyncpg instrumentation not available")
    except Exception as e:
        logger.warning(f"asyncpg instrumentation failed: {e}")


def get_tracer(name: str = __name__) -> Any:
    """
    Get a tracer instance for creating spans.

    Args:
        name: Name of the tracer (usually __name__ of the calling module)

    Returns:
        Tracer instance (or NoOpTracer if not initialized)

    Example:
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("my_operation") as span:
            span.set_attribute("key", "value")
            # ... do work ...
    """
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except ImportError:
        return _NoOpTracer()


def get_current_span() -> Any:
    """
    Get the currently active span.

    Returns:
        Current span or None if no active span

    Example:
        span = get_current_span()
        if span:
            span.set_attribute("custom.attribute", "value")
    """
    try:
        from opentelemetry import trace
        return trace.get_current_span()
    except ImportError:
        return None


def add_span_attribute(key: str, value: Any) -> None:
    """
    Add an attribute to the current span.

    This is a convenience function for adding attributes without
    needing to explicitly get the current span.

    Args:
        key: Attribute key
        value: Attribute value (str, int, float, bool, or list thereof)

    Example:
        add_span_attribute("user.id", "12345")
        add_span_attribute("request.size", 1024)
    """
    span = get_current_span()
    if span and hasattr(span, 'set_attribute'):
        span.set_attribute(key, value)


def add_span_event(name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
    """
    Add an event to the current span.

    Events are useful for recording significant occurrences during
    the span's lifetime.

    Args:
        name: Event name
        attributes: Optional event attributes

    Example:
        add_span_event("cache_miss", {"cache_key": "user:123"})
    """
    span = get_current_span()
    if span and hasattr(span, 'add_event'):
        span.add_event(name, attributes=attributes or {})


def record_exception(exception: Exception, attributes: Optional[Dict[str, Any]] = None) -> None:
    """
    Record an exception in the current span.

    Args:
        exception: The exception to record
        attributes: Additional attributes to add

    Example:
        try:
            risky_operation()
        except Exception as e:
            record_exception(e, {"operation": "risky_operation"})
            raise
    """
    span = get_current_span()
    if span and hasattr(span, 'record_exception'):
        span.record_exception(exception, attributes=attributes or {})
        try:
            from opentelemetry.trace import StatusCode
            span.set_status(StatusCode.ERROR, str(exception))
        except ImportError:
            pass


def get_trace_context() -> Dict[str, str]:
    """
    Get the current trace context for propagation.

    Returns a dictionary suitable for injection into HTTP headers
    or message metadata.

    Returns:
        Dictionary with trace context headers

    Example:
        headers = get_trace_context()
        response = requests.get(url, headers=headers)
    """
    try:
        from opentelemetry import propagate
        carrier: Dict[str, str] = {}
        propagate.inject(carrier)
        return carrier
    except ImportError:
        return {}


def inject_trace_context(carrier: Dict[str, str]) -> None:
    """
    Inject trace context into a carrier (e.g., HTTP headers).

    Args:
        carrier: Dictionary to inject trace context into

    Example:
        headers = {"Content-Type": "application/json"}
        inject_trace_context(headers)
        # headers now contains trace context
    """
    try:
        from opentelemetry import propagate
        propagate.inject(carrier)
    except ImportError:
        pass


def extract_trace_context(carrier: Dict[str, str]) -> Any:
    """
    Extract trace context from a carrier.

    Args:
        carrier: Dictionary containing trace context

    Returns:
        Context object for continuing the trace

    Example:
        context = extract_trace_context(request.headers)
        with tracer.start_as_current_span("child", context=context):
            pass
    """
    try:
        from opentelemetry import propagate
        return propagate.extract(carrier)
    except ImportError:
        return None


def shutdown_tracing() -> None:
    """
    Gracefully shutdown the tracer provider.

    Call this during application shutdown to ensure all pending
    spans are exported.

    Example:
        @app.on_event("shutdown")
        def shutdown():
            shutdown_tracing()
    """
    global _tracer_provider, _initialized

    if _tracer_provider:
        try:
            _tracer_provider.shutdown()
            logger.info("OpenTelemetry tracing shut down")
        except Exception as e:
            logger.error(f"Error shutting down tracing: {e}")

    _tracer_provider = None
    _initialized = False


class _NoOpTracer:
    """No-operation tracer when OpenTelemetry is not available."""

    def start_as_current_span(self, name: str, **kwargs):
        """Return a no-op context manager."""
        return _NoOpSpanContext()

    def start_span(self, name: str, **kwargs):
        """Return a no-op span."""
        return _NoOpSpan()


class _NoOpSpanContext:
    """No-operation span context manager."""

    def __enter__(self):
        return _NoOpSpan()

    def __exit__(self, *args):
        pass


class _NoOpSpan:
    """No-operation span when tracing is disabled."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def add_event(self, name: str, attributes: Optional[Dict] = None) -> None:
        pass

    def record_exception(self, exception: Exception, attributes: Optional[Dict] = None) -> None:
        pass

    def set_status(self, status: Any, description: str = None) -> None:
        pass

    def end(self) -> None:
        pass
