"""
Trace Context Propagation
========================
Utilities for propagating trace context across service boundaries.
"""
import uuid
from typing import Dict, Optional, Any
from opentelemetry import trace
from opentelemetry.propagate import inject, extract
from opentelemetry.trace import SpanContext, TraceFlags
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

def get_trace_context() -> Dict[str, str]:
    """Get current trace context as dict for propagation."""
    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        ctx = span.get_span_context()
        return {
            "trace_id": f"{ctx.trace_id:032x}",
            "span_id": f"{ctx.span_id:016x}",
            "trace_flags": str(ctx.trace_flags)
        }
    return {"trace_id": None, "span_id": None, "trace_flags": None}

def inject_trace_context(carrier: Dict[str, Any]) -> None:
    """Inject trace context into carrier (e.g., HTTP headers, message metadata)."""
    inject(carrier)

def extract_trace_context(carrier: Dict[str, Any]) -> Optional[SpanContext]:
    """Extract trace context from carrier."""
    return extract(carrier)

def create_correlation_id() -> str:
    """Create a unique correlation ID for request tracking."""
    return str(uuid.uuid4())

def get_request_context() -> Dict[str, str]:
    """Get complete request context including trace and correlation IDs."""
    trace_ctx = get_trace_context()
    correlation_id = create_correlation_id()
    
    return {
        **trace_ctx,
        "correlation_id": correlation_id,
        "request_id": correlation_id
    }

def set_span_attributes(attributes: Dict[str, Any]) -> None:
    """Set attributes on current span."""
    span = trace.get_current_span()
    if span:
        for key, value in attributes.items():
            span.set_attribute(key, value)

def add_span_event(name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
    """Add event to current span."""
    span = trace.get_current_span()
    if span:
        span.add_event(name, attributes or {})

def record_exception(exception: Exception, attributes: Optional[Dict[str, Any]] = None) -> None:
    """Record exception on current span."""
    span = trace.get_current_span()
    if span:
        span.record_exception(exception, attributes or {})
