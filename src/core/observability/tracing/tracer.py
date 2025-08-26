"""
OpenTelemetry Tracer Setup
==========================
Configures tracing with Jaeger exporter and provides decorators.
"""
import os
import functools
from typing import Optional, Dict, Any, Callable
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Global tracer instance
_tracer: Optional[trace.Tracer] = None

def init_tracer(service_name: str, service_version: Optional[str] = None) -> trace.Tracer:
    """Initialize OpenTelemetry tracer with Jaeger exporter."""
    global _tracer
    
    if _tracer is not None:
        return _tracer
    
    # Create resource with service information
    resource = Resource.create({
        "service.name": service_name,
        "service.version": service_version or "dev",
        "deployment.environment": os.getenv("APP_ENV", "dev")
    })
    
    # Create tracer provider
    provider = TracerProvider(resource=resource)
    
    # Add OTLP exporter (Jaeger)
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4317")
    otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
    processor = BatchSpanProcessor(otlp_exporter)
    provider.add_span_processor(processor)
    
    # Set global tracer provider
    trace.set_tracer_provider(provider)
    
    # Instrument logging
    LoggingInstrumentor().instrument(set_logging_format=True)
    
    # Create and store tracer
    _tracer = trace.get_tracer(service_name)
    
    return _tracer

def get_tracer() -> trace.Tracer:
    """Get the global tracer instance."""
    if _tracer is None:
        raise RuntimeError("Tracer not initialized. Call init_tracer() first.")
    return _tracer

def trace_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Decorator to trace function execution with span."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(name, attributes=attributes or {}) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_status(trace.StatusCode.OK)
                    return result
                except Exception as e:
                    span.set_status(trace.StatusCode.ERROR, str(e))
                    span.record_exception(e)
                    raise
        return wrapper
    return decorator

def instrument_fastapi(app):
    """Instrument FastAPI app with OpenTelemetry."""
    FastAPIInstrumentor.instrument_app(app)

def get_current_span() -> Optional[trace.Span]:
    """Get current active span."""
    return trace.get_current_span()

def get_trace_context() -> Dict[str, str]:
    """Get current trace context as dict."""
    span = get_current_span()
    if span and span.get_span_context().is_valid:
        ctx = span.get_span_context()
        return {
            "trace_id": f"{ctx.trace_id:032x}",
            "span_id": f"{ctx.span_id:016x}"
        }
    return {"trace_id": None, "span_id": None}


# Wrapper class for backward compatibility
class Tracer:
    """Tracer wrapper for test compatibility"""
    
    def __init__(self, service_name: str = "test"):
        self.service_name = service_name
        self.tracer = init_tracer(service_name)
    
    def start_span(self, name: str):
        """Start a new span"""
        if self.tracer:
            return self.tracer.start_span(name)
        return None
