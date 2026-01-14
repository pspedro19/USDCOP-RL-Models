"""
Distributed Tracing Setup using OpenTelemetry.

This module configures OpenTelemetry for distributed tracing across
all USDCOP services. Traces are exported to Jaeger.

Design Patterns:
- Singleton Pattern: Single tracer instance per service
- Facade Pattern: Simple interface for tracing setup

Usage:
    from common.tracing import setup_tracing, get_tracer

    # Setup in main.py
    tracer = setup_tracing("inference-api")

    # Use in routes
    with tracer.start_as_current_span("inference") as span:
        span.set_attribute("model_id", model_id)
        result = model.predict(obs)

Contract: CTR-OBS-001
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Track initialized tracers
_tracers: Dict[str, Any] = {}
_initialized = False


def setup_tracing(
    service_name: str,
    jaeger_host: str = None,
    jaeger_port: int = None,
    sample_rate: float = 1.0,
) -> Any:
    """
    Configure OpenTelemetry tracing for a service.

    Args:
        service_name: Name of the service (appears in Jaeger UI)
        jaeger_host: Jaeger agent host (default from env)
        jaeger_port: Jaeger agent port (default from env)
        sample_rate: Sampling rate 0.0-1.0 (default: 1.0 = all traces)

    Returns:
        Configured tracer instance

    Example:
        tracer = setup_tracing("inference-api")
        with tracer.start_as_current_span("inference"):
            ...
    """
    global _initialized

    # Get configuration from environment
    jaeger_host = jaeger_host or os.environ.get("JAEGER_HOST", "jaeger")
    jaeger_port = jaeger_port or int(os.environ.get("JAEGER_PORT", "6831"))

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter

        # Only initialize once
        if not _initialized:
            # Create resource with service name
            resource = Resource.create({
                "service.name": service_name,
                "service.version": os.environ.get("SERVICE_VERSION", "1.0.0"),
            })

            # Create provider
            provider = TracerProvider(resource=resource)

            # Configure Jaeger exporter
            jaeger_exporter = JaegerExporter(
                agent_host_name=jaeger_host,
                agent_port=jaeger_port,
            )

            # Add batch processor for efficient export
            provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))

            # Set as global provider
            trace.set_tracer_provider(provider)

            _initialized = True
            logger.info(
                f"OpenTelemetry tracing initialized for {service_name} "
                f"(Jaeger: {jaeger_host}:{jaeger_port})"
            )

        # Get tracer for this service
        tracer = trace.get_tracer(service_name)
        _tracers[service_name] = tracer

        return tracer

    except ImportError:
        logger.warning(
            "OpenTelemetry not installed. Install with: "
            "pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-jaeger"
        )
        return NoOpTracer()


def get_tracer(service_name: str = "default") -> Any:
    """
    Get an existing tracer by service name.

    Args:
        service_name: Service name used in setup_tracing

    Returns:
        Tracer instance or NoOpTracer if not initialized
    """
    if service_name in _tracers:
        return _tracers[service_name]

    # Try to get from OpenTelemetry directly
    try:
        from opentelemetry import trace
        return trace.get_tracer(service_name)
    except ImportError:
        return NoOpTracer()


class NoOpTracer:
    """
    No-operation tracer for when OpenTelemetry is not available.

    Provides the same interface but does nothing.
    """

    @contextmanager
    def start_as_current_span(self, name: str, **kwargs):
        """Context manager that does nothing."""
        yield NoOpSpan()

    def start_span(self, name: str, **kwargs):
        """Start a span that does nothing."""
        return NoOpSpan()


class NoOpSpan:
    """No-operation span."""

    def set_attribute(self, key: str, value: Any) -> None:
        """Do nothing."""
        pass

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        """Do nothing."""
        pass

    def add_event(self, name: str, attributes: Dict[str, Any] = None) -> None:
        """Do nothing."""
        pass

    def record_exception(self, exception: Exception) -> None:
        """Do nothing."""
        pass

    def set_status(self, status: Any) -> None:
        """Do nothing."""
        pass

    def end(self) -> None:
        """Do nothing."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# =============================================================================
# CONVENIENCE DECORATORS
# =============================================================================

def trace_function(span_name: str = None, service_name: str = "default"):
    """
    Decorator to trace a function.

    Args:
        span_name: Name for the span (default: function name)
        service_name: Service name for tracer lookup

    Example:
        @trace_function("my_operation")
        def do_something():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = get_tracer(service_name)
            name = span_name or func.__name__

            with tracer.start_as_current_span(name) as span:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    raise
        return wrapper
    return decorator


def trace_async_function(span_name: str = None, service_name: str = "default"):
    """
    Decorator to trace an async function.

    Example:
        @trace_async_function("async_operation")
        async def do_something_async():
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            tracer = get_tracer(service_name)
            name = span_name or func.__name__

            with tracer.start_as_current_span(name) as span:
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    raise
        return wrapper
    return decorator
