# -*- coding: utf-8 -*-
"""
Prometheus Metrics Exporter
===========================
Standardized metrics export for L0-L5 pipeline observability.

Contract: CTR-L0-METRICS-001

This module provides:
1. Extraction metrics (success, failure, latency)
2. Pipeline stage metrics (L0-L5)
3. Data quality metrics
4. Circuit breaker states
5. DLQ queue depth

Usage:
    from services.metrics_exporter import MetricsExporter, get_metrics

    metrics = get_metrics()
    metrics.record_extraction_success('fred', 'FEDFUNDS', 150, 0.5)
    metrics.record_extraction_failure('investing', 'VIX', 'ConnectionTimeout')

    # Push to Prometheus gateway
    metrics.push_to_gateway()

Version: 1.0.0
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import prometheus_client, fall back to mock if not available
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Summary,
        CollectorRegistry,
        push_to_gateway,
        generate_latest,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available. Metrics will be logged only.")


# =============================================================================
# METRIC DEFINITIONS
# =============================================================================

# Default buckets for latency histograms (in seconds)
LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)

# Default buckets for record counts
RECORD_BUCKETS = (1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000, 10000)


@dataclass
class MetricDefinition:
    """Definition of a metric."""
    name: str
    description: str
    labels: List[str]
    metric_type: str  # counter, gauge, histogram, summary


# Define all metrics
METRICS = {
    # Extraction metrics
    'extraction_total': MetricDefinition(
        name='l0_extraction_total',
        description='Total extraction attempts',
        labels=['source', 'variable', 'status'],
        metric_type='counter',
    ),
    'extraction_duration_seconds': MetricDefinition(
        name='l0_extraction_duration_seconds',
        description='Extraction duration in seconds',
        labels=['source', 'variable'],
        metric_type='histogram',
    ),
    'extraction_records': MetricDefinition(
        name='l0_extraction_records',
        description='Number of records extracted',
        labels=['source', 'variable'],
        metric_type='histogram',
    ),

    # Pipeline stage metrics
    'pipeline_stage_duration_seconds': MetricDefinition(
        name='pipeline_stage_duration_seconds',
        description='Duration of pipeline stage execution',
        labels=['stage', 'status'],
        metric_type='histogram',
    ),
    'pipeline_stage_last_success': MetricDefinition(
        name='pipeline_stage_last_success_timestamp',
        description='Timestamp of last successful stage execution',
        labels=['stage'],
        metric_type='gauge',
    ),

    # Data quality metrics
    'data_completeness': MetricDefinition(
        name='l0_data_completeness_ratio',
        description='Ratio of non-null values in extracted data',
        labels=['source', 'variable'],
        metric_type='gauge',
    ),
    'data_freshness_days': MetricDefinition(
        name='l0_data_freshness_days',
        description='Age of most recent data in days',
        labels=['source', 'variable'],
        metric_type='gauge',
    ),
    'validation_failures': MetricDefinition(
        name='l0_validation_failures_total',
        description='Total validation failures',
        labels=['validator', 'variable', 'severity'],
        metric_type='counter',
    ),

    # Circuit breaker metrics
    'circuit_breaker_state': MetricDefinition(
        name='l0_circuit_breaker_state',
        description='Circuit breaker state (0=closed, 1=open, 2=half-open)',
        labels=['source'],
        metric_type='gauge',
    ),
    'circuit_breaker_failures': MetricDefinition(
        name='l0_circuit_breaker_failures_total',
        description='Total circuit breaker failures',
        labels=['source'],
        metric_type='counter',
    ),

    # DLQ metrics
    'dlq_entries_total': MetricDefinition(
        name='l0_dlq_entries_total',
        description='Total entries in DLQ by status',
        labels=['status'],
        metric_type='gauge',
    ),
    'dlq_retries_total': MetricDefinition(
        name='l0_dlq_retries_total',
        description='Total DLQ retry attempts',
        labels=['source', 'status'],
        metric_type='counter',
    ),

    # Inference metrics (L5)
    'inference_latency_seconds': MetricDefinition(
        name='l5_inference_latency_seconds',
        description='Model inference latency',
        labels=['model_version'],
        metric_type='histogram',
    ),
    'inference_total': MetricDefinition(
        name='l5_inference_total',
        description='Total inference requests',
        labels=['model_version', 'action'],
        metric_type='counter',
    ),
}


# =============================================================================
# METRICS EXPORTER CLASS
# =============================================================================

class MetricsExporter:
    """
    Centralized Prometheus metrics exporter.

    Provides a consistent interface for recording metrics across
    all pipeline stages (L0-L5).
    """

    def __init__(
        self,
        registry: Optional['CollectorRegistry'] = None,
        pushgateway_url: Optional[str] = None,
        job_name: str = 'usdcop_pipeline',
    ):
        """
        Initialize metrics exporter.

        Args:
            registry: Prometheus CollectorRegistry (creates new if None)
            pushgateway_url: URL of Prometheus Pushgateway
            job_name: Job name for Pushgateway
        """
        self.pushgateway_url = pushgateway_url or os.environ.get(
            'PROMETHEUS_PUSHGATEWAY_URL', 'localhost:9091'
        )
        self.job_name = job_name

        if PROMETHEUS_AVAILABLE:
            self.registry = registry or CollectorRegistry()
            self._metrics: Dict[str, Any] = {}
            self._init_metrics()
        else:
            self.registry = None
            self._metrics = {}

        # In-memory fallback for when Prometheus is not available
        self._memory_metrics: Dict[str, List[Dict[str, Any]]] = {}

    def _init_metrics(self):
        """Initialize all Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            return

        for key, defn in METRICS.items():
            if defn.metric_type == 'counter':
                self._metrics[key] = Counter(
                    defn.name,
                    defn.description,
                    defn.labels,
                    registry=self.registry,
                )
            elif defn.metric_type == 'gauge':
                self._metrics[key] = Gauge(
                    defn.name,
                    defn.description,
                    defn.labels,
                    registry=self.registry,
                )
            elif defn.metric_type == 'histogram':
                buckets = LATENCY_BUCKETS if 'duration' in key or 'latency' in key else RECORD_BUCKETS
                self._metrics[key] = Histogram(
                    defn.name,
                    defn.description,
                    defn.labels,
                    buckets=buckets,
                    registry=self.registry,
                )
            elif defn.metric_type == 'summary':
                self._metrics[key] = Summary(
                    defn.name,
                    defn.description,
                    defn.labels,
                    registry=self.registry,
                )

    def _log_metric(self, metric_name: str, labels: Dict[str, str], value: float):
        """Log metric to memory (fallback when Prometheus unavailable)."""
        if metric_name not in self._memory_metrics:
            self._memory_metrics[metric_name] = []

        self._memory_metrics[metric_name].append({
            'timestamp': datetime.utcnow().isoformat(),
            'labels': labels,
            'value': value,
        })

        # Keep only last 1000 entries per metric
        if len(self._memory_metrics[metric_name]) > 1000:
            self._memory_metrics[metric_name] = self._memory_metrics[metric_name][-1000:]

    # =========================================================================
    # EXTRACTION METRICS
    # =========================================================================

    def record_extraction_success(
        self,
        source: str,
        variable: str,
        records: int,
        duration_seconds: float,
    ):
        """Record successful extraction."""
        labels = {'source': source, 'variable': variable, 'status': 'success'}

        if PROMETHEUS_AVAILABLE:
            self._metrics['extraction_total'].labels(**labels).inc()
            self._metrics['extraction_duration_seconds'].labels(
                source=source, variable=variable
            ).observe(duration_seconds)
            self._metrics['extraction_records'].labels(
                source=source, variable=variable
            ).observe(records)
        else:
            self._log_metric('extraction_success', labels, 1)

        logger.debug(
            "[METRICS] Extraction success: source=%s, variable=%s, records=%d, duration=%.2fs",
            source, variable, records, duration_seconds
        )

    def record_extraction_failure(
        self,
        source: str,
        variable: str,
        error: str,
        duration_seconds: float = 0,
    ):
        """Record failed extraction."""
        labels = {'source': source, 'variable': variable, 'status': 'failure'}

        if PROMETHEUS_AVAILABLE:
            self._metrics['extraction_total'].labels(**labels).inc()
            if duration_seconds > 0:
                self._metrics['extraction_duration_seconds'].labels(
                    source=source, variable=variable
                ).observe(duration_seconds)
        else:
            self._log_metric('extraction_failure', labels, 1)

        logger.warning(
            "[METRICS] Extraction failure: source=%s, variable=%s, error=%s",
            source, variable, error[:100]
        )

    # =========================================================================
    # PIPELINE METRICS
    # =========================================================================

    def record_pipeline_stage(
        self,
        stage: str,
        duration_seconds: float,
        success: bool = True,
    ):
        """Record pipeline stage execution."""
        status = 'success' if success else 'failure'

        if PROMETHEUS_AVAILABLE:
            self._metrics['pipeline_stage_duration_seconds'].labels(
                stage=stage, status=status
            ).observe(duration_seconds)
            if success:
                self._metrics['pipeline_stage_last_success'].labels(
                    stage=stage
                ).set_to_current_time()
        else:
            self._log_metric('pipeline_stage', {'stage': stage, 'status': status}, duration_seconds)

        logger.info(
            "[METRICS] Pipeline stage %s: status=%s, duration=%.2fs",
            stage, status, duration_seconds
        )

    # =========================================================================
    # DATA QUALITY METRICS
    # =========================================================================

    def record_data_quality(
        self,
        source: str,
        variable: str,
        completeness: float,
        freshness_days: float,
    ):
        """Record data quality metrics."""
        if PROMETHEUS_AVAILABLE:
            self._metrics['data_completeness'].labels(
                source=source, variable=variable
            ).set(completeness)
            self._metrics['data_freshness_days'].labels(
                source=source, variable=variable
            ).set(freshness_days)
        else:
            self._log_metric('data_quality', {
                'source': source, 'variable': variable
            }, completeness)

    def record_validation_failure(
        self,
        validator: str,
        variable: str,
        severity: str,
    ):
        """Record validation failure."""
        if PROMETHEUS_AVAILABLE:
            self._metrics['validation_failures'].labels(
                validator=validator, variable=variable, severity=severity
            ).inc()
        else:
            self._log_metric('validation_failure', {
                'validator': validator, 'variable': variable, 'severity': severity
            }, 1)

    # =========================================================================
    # CIRCUIT BREAKER METRICS
    # =========================================================================

    def record_circuit_breaker_state(self, source: str, state: str):
        """Record circuit breaker state change."""
        state_value = {'closed': 0, 'open': 1, 'half-open': 2}.get(state.lower(), -1)

        if PROMETHEUS_AVAILABLE:
            self._metrics['circuit_breaker_state'].labels(source=source).set(state_value)
        else:
            self._log_metric('circuit_breaker_state', {'source': source}, state_value)

        logger.info("[METRICS] Circuit breaker %s: state=%s", source, state)

    def record_circuit_breaker_failure(self, source: str):
        """Record circuit breaker failure."""
        if PROMETHEUS_AVAILABLE:
            self._metrics['circuit_breaker_failures'].labels(source=source).inc()
        else:
            self._log_metric('circuit_breaker_failure', {'source': source}, 1)

    # =========================================================================
    # DLQ METRICS
    # =========================================================================

    def record_dlq_entries(self, pending: int, failed: int, resolved: int):
        """Record DLQ entry counts."""
        if PROMETHEUS_AVAILABLE:
            self._metrics['dlq_entries_total'].labels(status='pending').set(pending)
            self._metrics['dlq_entries_total'].labels(status='failed_permanent').set(failed)
            self._metrics['dlq_entries_total'].labels(status='resolved').set(resolved)
        else:
            self._log_metric('dlq_entries', {'status': 'pending'}, pending)

    def record_dlq_retry(self, source: str, success: bool):
        """Record DLQ retry attempt."""
        status = 'success' if success else 'failure'

        if PROMETHEUS_AVAILABLE:
            self._metrics['dlq_retries_total'].labels(source=source, status=status).inc()
        else:
            self._log_metric('dlq_retry', {'source': source, 'status': status}, 1)

    # =========================================================================
    # INFERENCE METRICS (L5)
    # =========================================================================

    def record_inference(
        self,
        model_version: str,
        action: str,
        latency_seconds: float,
    ):
        """Record model inference."""
        if PROMETHEUS_AVAILABLE:
            self._metrics['inference_latency_seconds'].labels(
                model_version=model_version
            ).observe(latency_seconds)
            self._metrics['inference_total'].labels(
                model_version=model_version, action=action
            ).inc()
        else:
            self._log_metric('inference', {
                'model_version': model_version, 'action': action
            }, latency_seconds)

    # =========================================================================
    # EXPORT METHODS
    # =========================================================================

    def push_to_gateway(self) -> bool:
        """Push metrics to Prometheus Pushgateway."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("[METRICS] Cannot push - prometheus_client not available")
            return False

        try:
            push_to_gateway(
                self.pushgateway_url,
                job=self.job_name,
                registry=self.registry,
            )
            logger.info("[METRICS] Pushed metrics to gateway: %s", self.pushgateway_url)
            return True
        except Exception as e:
            logger.error("[METRICS] Failed to push to gateway: %s", e)
            return False

    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format."""
        if PROMETHEUS_AVAILABLE:
            return generate_latest(self.registry).decode('utf-8')
        else:
            return str(self._memory_metrics)

    def get_memory_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get in-memory metrics (for debugging)."""
        return self._memory_metrics


# =============================================================================
# DECORATORS
# =============================================================================

def track_extraction(source: str, variable: str):
    """
    Decorator to track extraction metrics.

    Usage:
        @track_extraction('fred', 'FEDFUNDS')
        def extract_fedfunds():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            metrics = get_metrics()
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Assume result has records_count attribute
                records = getattr(result, 'records_count', 0)
                metrics.record_extraction_success(source, variable, records, duration)

                return result

            except Exception as e:
                duration = time.time() - start_time
                metrics.record_extraction_failure(source, variable, str(e), duration)
                raise

        return wrapper
    return decorator


def track_pipeline_stage(stage: str):
    """
    Decorator to track pipeline stage metrics.

    Usage:
        @track_pipeline_stage('l0_macro_extract')
        def extract_macro_data():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            metrics = get_metrics()
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics.record_pipeline_stage(stage, duration, success=True)
                return result

            except Exception as e:
                duration = time.time() - start_time
                metrics.record_pipeline_stage(stage, duration, success=False)
                raise

        return wrapper
    return decorator


# =============================================================================
# SINGLETON
# =============================================================================

_metrics_instance: Optional[MetricsExporter] = None


def get_metrics() -> MetricsExporter:
    """Get the global MetricsExporter instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = MetricsExporter()
    return _metrics_instance


def reset_metrics():
    """Reset the global metrics instance (for testing)."""
    global _metrics_instance
    _metrics_instance = None
