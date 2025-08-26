"""
Prometheus Metrics Registry
==========================
Central registry for all Prometheus metrics with HTTP endpoint.
"""
import os
import time
from typing import Dict, Any, Optional
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, REGISTRY, 
    generate_latest, start_http_server, CONTENT_TYPE_LATEST
)
from prometheus_client.core import CollectorRegistry
from prometheus_client.exposition import generate_latest as generate_latest_custom

class MetricsRegistry:
    """Central registry for Prometheus metrics."""
    
    def __init__(self, service_name: str = "unknown"):
        self.service_name = service_name
        self.registry = CollectorRegistry()
        self.metrics: Dict[str, Any] = {}
        
        # Initialize common metrics
        self._init_common_metrics()
    
    def _init_common_metrics(self):
        """Initialize common metrics used across services."""
        # Request metrics
        self.metrics['request_total'] = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['service', 'method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.metrics['request_duration'] = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['service', 'method', 'endpoint'],
            registry=self.registry
        )
        
        # Error metrics
        self.metrics['errors_total'] = Counter(
            'errors_total',
            'Total errors',
            ['service', 'component', 'operation', 'type'],
            registry=self.registry
        )
        
        # Circuit breaker metrics
        self.metrics['circuit_state'] = Gauge(
            'circuit_state',
            'Circuit breaker state (0=closed, 1=half_open, 2=open)',
            ['service', 'source'],
            registry=self.registry
        )
        
        # Queue metrics
        self.metrics['queue_depth'] = Gauge(
            'queue_depth',
            'Queue depth',
            ['service', 'queue'],
            registry=self.registry
        )
        
        # System metrics (moved to TechnicalMetrics class)
        # Note: CPU and memory metrics are now handled by TechnicalMetrics
        # to avoid duplicate metric definitions
    
    def counter(self, name: str, description: str = "", labelnames: list = None, **kwargs) -> Counter:
        """Create or get a counter metric."""
        if name not in self.metrics:
            self.metrics[name] = Counter(
                name, description, labelnames or [], 
                registry=self.registry, **kwargs
            )
        return self.metrics[name]
    
    def register_counter(self, name: str, description: str = "", labelnames: list = None, **kwargs) -> Counter:
        """Register a counter metric (alias for counter method)."""
        return self.counter(name, description, labelnames, **kwargs)
    
    def gauge(self, name: str, description: str = "", labelnames: list = None, **kwargs) -> Gauge:
        """Create or get a gauge metric."""
        if name not in self.metrics:
            self.metrics[name] = Gauge(
                name, description, labelnames or [], 
                registry=self.registry, **kwargs
            )
        return self.metrics[name]
    
    def histogram(self, name: str, description: str = "", labelnames: list = None, **kwargs) -> Histogram:
        """Create or get a histogram metric."""
        if name not in self.metrics:
            self.metrics[name] = Histogram(
                name, description, labelnames or [], 
                registry=self.registry, **kwargs
            )
        return self.metrics[name]
    
    def summary(self, name: str, description: str = "", labelnames: list = None, **kwargs) -> Summary:
        """Create or get a summary metric."""
        if name not in self.metrics:
            self.metrics[name] = Summary(
                name, description, labelnames or [], 
                registry=self.registry, **kwargs
            )
        return self.metrics[name]
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics."""
        self.metrics['request_total'].labels(
            self.service_name, method, endpoint, status
        ).inc()
        
        self.metrics['request_duration'].labels(
            self.service_name, method, endpoint
        ).observe(duration)
    
    def record_error(self, component: str, operation: str, error_type: str):
        """Record error metrics."""
        self.metrics['errors_total'].labels(
            self.service_name, component, operation, error_type
        ).inc()
    
    def set_circuit_state(self, source: str, state: int):
        """Set circuit breaker state."""
        self.metrics['circuit_state'].labels(self.service_name, source).set(state)
    
    def set_queue_depth(self, queue: str, depth: int):
        """Set queue depth."""
        self.metrics['queue_depth'].labels(self.service_name, queue).set(depth)
    
    def set_system_metrics(self, cpu_percent: float, memory_bytes: int):
        """Set system resource metrics."""
        # Note: System metrics are now handled by TechnicalMetrics class
        # to avoid duplicate metric definitions
        pass
    
    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format."""
        return generate_latest_custom(self.registry).decode('utf-8')
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as a dictionary for debugging."""
        metrics_data = {}
        for name, metric in self.metrics.items():
            if hasattr(metric, '_metrics'):
                metrics_data[name] = {
                    'type': type(metric).__name__,
                    'samples': list(metric._metrics.values())
                }
        return metrics_data

# Global registry instance
_global_registry: Optional[MetricsRegistry] = None

def get_metrics(service_name: str = None) -> MetricsRegistry:
    """Get the global metrics registry."""
    global _global_registry
    if _global_registry is None:
        service_name = service_name or os.getenv("OTEL_SERVICE_NAME", "unknown")
        _global_registry = MetricsRegistry(service_name)
    return _global_registry

def start_metrics_server(port: int = 8001, addr: str = "0.0.0.0"):
    """Start HTTP server for metrics endpoint."""
    start_http_server(port, addr)
    print(f"Metrics server started on {addr}:{port}")

def create_metrics_app():
    """Create WSGI/ASGI app for metrics endpoint."""
    def metrics_app(environ, start_response):
        if environ['PATH_INFO'] == '/metrics':
            data = get_metrics().get_metrics_text()
            start_response('200 OK', [
                ('Content-Type', CONTENT_TYPE_LATEST),
                ('Content-Length', str(len(data)))
            ])
            return [data.encode('utf-8')]
        else:
            start_response('404 Not Found', [('Content-Type', 'text/plain')])
            return [b'Not Found']
    
    return metrics_app


# Alias for backward compatibility
PrometheusRegistry = MetricsRegistry
