"""
Technical Metrics
================
System and operational metrics for monitoring.
"""
import time
import psutil
from typing import Dict, Any, Optional
from .prometheus_registry import MetricsRegistry

class TechnicalMetrics:
    """Technical metrics for system monitoring."""
    
    def __init__(self, metrics_registry: MetricsRegistry):
        self.registry = metrics_registry
        self._init_technical_metrics()
    
    def _init_technical_metrics(self):
        """Initialize technical metrics."""
        # Data pipeline metrics
        self.pipeline_records = self.registry.counter(
            'pipeline_records_total',
            'Total records processed by pipeline',
            ['stage', 'status', 'symbol']
        )
        
        self.pipeline_duration = self.registry.histogram(
            'pipeline_stage_duration_seconds',
            'Pipeline stage duration',
            ['stage', 'symbol']
        )
        
        self.pipeline_errors = self.registry.counter(
            'pipeline_errors_total',
            'Total pipeline errors',
            ['stage', 'error_type', 'symbol']
        )
        
        # Data source metrics
        self.data_fetch_duration = self.registry.histogram(
            'data_fetch_duration_seconds',
            'Data fetch duration',
            ['source', 'operation', 'symbol']
        )
        
        self.data_fetch_errors = self.registry.counter(
            'data_fetch_errors_total',
            'Total data fetch errors',
            ['source', 'operation', 'symbol']
        )
        
        self.stream_staleness = self.registry.gauge(
            'stream_staleness_seconds',
            'Data stream staleness in seconds',
            ['source', 'symbol']
        )
        
        # Circuit breaker metrics
        self.circuit_transitions = self.registry.counter(
            'circuit_transitions_total',
            'Circuit breaker state transitions',
            ['source', 'from_state', 'to_state', 'reason']
        )
        
        self.circuit_failure_rate = self.registry.gauge(
            'circuit_failure_rate',
            'Circuit breaker failure rate',
            ['source']
        )
        
        # Queue and backpressure metrics
        self.queue_operations = self.registry.counter(
            'queue_operations_total',
            'Queue operations (put/get/drop)',
            ['queue', 'operation', 'result']
        )
        
        self.queue_wait_time = self.registry.histogram(
            'queue_wait_time_seconds',
            'Time spent waiting in queues',
            ['queue']
        )
        
        # SAGA metrics
        self.saga_transactions = self.registry.counter(
            'saga_transactions_total',
            'Total SAGA transactions',
            ['type', 'status']
        )
        
        self.saga_duration = self.registry.histogram(
            'saga_duration_seconds',
            'SAGA transaction duration',
            ['type']
        )
        
        self.saga_steps = self.registry.counter(
            'saga_steps_total',
            'Total SAGA steps executed',
            ['type', 'step', 'status']
        )
        
        # Resource utilization metrics
        self.cpu_usage = self.registry.gauge(
            'cpu_usage_percent',
            'CPU usage percentage',
            ['service', 'core']
        )
        
        self.memory_usage = self.registry.gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            ['service', 'type']
        )
        
        self.disk_usage = self.registry.gauge(
            'disk_usage_bytes',
            'Disk usage in bytes',
            ['service', 'mount_point']
        )
        
        self.network_io = self.registry.counter(
            'network_io_bytes_total',
            'Network I/O in bytes',
            ['service', 'direction', 'interface']
        )
        
        # Event metrics
        self.events_published = self.registry.counter(
            'events_published_total',
            'Total events published',
            ['type', 'source', 'status']
        )
        
        self.events_processed = self.registry.counter(
            'events_processed_total',
            'Total events processed',
            ['type', 'consumer', 'status']
        )
        
        self.event_latency = self.registry.histogram(
            'event_latency_seconds',
            'Event processing latency',
            ['type', 'consumer']
        )
    
    def record_pipeline_operation(self, stage: str, status: str, symbol: str, duration: float = None):
        """Record pipeline operation metrics."""
        self.pipeline_records.labels(stage=stage, status=status, symbol=symbol).inc()
        
        if duration is not None:
            self.pipeline_duration.labels(stage=stage, symbol=symbol).observe(duration)
    
    def record_pipeline_error(self, stage: str, error_type: str, symbol: str):
        """Record pipeline error metrics."""
        self.pipeline_errors.labels(stage=stage, error_type=error_type, symbol=symbol).inc()
    
    def record_data_fetch(self, source: str, operation: str, symbol: str, duration: float, success: bool):
        """Record data fetch metrics."""
        if success:
            self.data_fetch_duration.labels(source=source, operation=operation, symbol=symbol).observe(duration)
        else:
            self.data_fetch_errors.labels(source=source, operation=operation, symbol=symbol).inc()
    
    def update_stream_staleness(self, source: str, symbol: str, staleness_seconds: float):
        """Update stream staleness metric."""
        self.stream_staleness.labels(source=source, symbol=symbol).set(staleness_seconds)
    
    def record_circuit_transition(self, source: str, from_state: str, to_state: str, reason: str):
        """Record circuit breaker state transition."""
        self.circuit_transitions.labels(
            source=source, from_state=from_state, to_state=to_state, reason=reason
        ).inc()
    
    def update_circuit_failure_rate(self, source: str, failure_rate: float):
        """Update circuit breaker failure rate."""
        self.circuit_failure_rate.labels(source=source).set(failure_rate)
    
    def record_queue_operation(self, queue: str, operation: str, result: str, wait_time: float = None):
        """Record queue operation metrics."""
        self.queue_operations.labels(queue=queue, operation=operation, result=result).inc()
        
        if wait_time is not None:
            self.queue_wait_time.labels(queue=queue).observe(wait_time)
    
    def record_saga_transaction(self, saga_type: str, status: str, duration: float = None):
        """Record SAGA transaction metrics."""
        self.saga_transactions.labels(type=saga_type, status=status).inc()
        
        if duration is not None:
            self.saga_duration.labels(type=saga_type).observe(duration)
    
    def record_saga_step(self, saga_type: str, step: str, status: str):
        """Record SAGA step metrics."""
        self.saga_steps.labels(type=saga_type, step=step, status=status).inc()
    
    def update_resource_metrics(self, service: str):
        """Update system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.labels(service=service, core="total").set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.labels(service=service, type="total").set(memory.total)
            self.memory_usage.labels(service=service, type="available").set(memory.available)
            self.memory_usage.labels(service=service, type="used").set(memory.used)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.disk_usage.labels(service=service, mount_point="/").set(disk.used)
            
        except Exception as e:
            # Log error but don't fail
            pass
    
    def record_event_operation(self, event_type: str, source: str, status: str, 
                              consumer: str = None, latency: float = None):
        """Record event operation metrics."""
        self.events_published.labels(type=event_type, source=source, status=status).inc()
        
        if consumer and latency is not None:
            self.events_processed.labels(type=event_type, consumer=consumer, status=status).inc()
            self.event_latency.labels(type=event_type, consumer=consumer).observe(latency)
    
    def get_system_health_summary(self, service: str) -> Dict[str, Any]:
        """Get system health summary."""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'service': service,
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': (disk.used / disk.total) * 100,
                'disk_used_gb': disk.used / (1024**3),
                'disk_total_gb': disk.total / (1024**3)
            }
        except Exception:
            return {'service': service, 'error': 'Unable to collect system metrics'}
