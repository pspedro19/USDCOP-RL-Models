"""
Distributed Tracing Module for USDCOP Trading Project.

This module provides OpenTelemetry-based distributed tracing capabilities
for the USDCOP trading platform, with Jaeger as the trace backend.

Features:
    - Automatic trace context propagation
    - Low-overhead tracing (~1-2ms per request)
    - ML-specific span builders
    - Decorators for easy function instrumentation
    - Auto-instrumentation for FastAPI, databases, and HTTP clients

Components:
    - otel_setup: Core OpenTelemetry configuration
    - decorators: Function/method decorators and ML span builders

Quick Start:
    # Initialize tracing at application startup
    from shared.tracing import init_tracing

    init_tracing(
        service_name="inference-api",
        jaeger_endpoint="http://jaeger:14268/api/traces",
        sampling_rate=0.1
    )

    # Use decorators to trace functions
    from shared.tracing import traced, traced_async

    @traced(name="process_data")
    def process_data(data):
        return transform(data)

    # Use ML span builder for ML operations
    from shared.tracing import MLSpanBuilder

    ml_span = MLSpanBuilder()
    with ml_span.inference("ppo_primary", feature_count=15) as span:
        result = model.predict(features)
        span.set_attribute("ml.confidence", result.confidence)

    # Add attributes to current span
    from shared.tracing import add_span_attribute, add_ml_attributes

    add_span_attribute("request.user_id", "12345")
    add_ml_attributes(model_id="ppo_primary", confidence=0.85)

Environment Variables:
    OTEL_ENABLED: Enable/disable tracing (default: true)
    OTEL_SERVICE_NAME: Override service name
    OTEL_SERVICE_VERSION: Service version
    OTEL_SAMPLING_RATE: Sampling rate (0.0-1.0)
    OTEL_EXPORTER_JAEGER_ENDPOINT: Jaeger collector endpoint
    DEPLOYMENT_ENV: Deployment environment (development, staging, production)
"""

# Core setup functions
from .otel_setup import (
    init_tracing,
    get_tracer,
    get_current_span,
    add_span_attribute,
    add_span_event,
    record_exception,
    get_trace_context,
    inject_trace_context,
    extract_trace_context,
    shutdown_tracing,
)

# Decorators
from .decorators import (
    traced,
    traced_async,
    MLSpanBuilder,
    add_ml_attributes,
    add_trading_attributes,
)

__all__ = [
    # Setup
    "init_tracing",
    "shutdown_tracing",
    # Tracer access
    "get_tracer",
    "get_current_span",
    # Span manipulation
    "add_span_attribute",
    "add_span_event",
    "record_exception",
    # Context propagation
    "get_trace_context",
    "inject_trace_context",
    "extract_trace_context",
    # Decorators
    "traced",
    "traced_async",
    # ML utilities
    "MLSpanBuilder",
    "add_ml_attributes",
    "add_trading_attributes",
]

__version__ = "1.0.0"
