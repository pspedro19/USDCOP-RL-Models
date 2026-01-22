"""
Prometheus Business Metrics for Inference API
==============================================
P2-1 Remediation: Add business-level metrics for monitoring

Metrics exposed:
- inference_requests_total: Total inference requests by model and signal
- inference_latency_seconds: Inference latency histogram
- active_models_count: Number of currently loaded models
- inference_errors_total: Total inference errors by type
- feature_nan_ratio: Ratio of NaN values in observations
- model_agreement_ratio: Agreement between champion and shadow models
- feast_hit_ratio: Ratio of Feast hits vs fallbacks

Author: Trading Team
Version: 1.0.0
Date: 2026-01-17
"""

import logging
from typing import Optional

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Info,
        generate_latest,
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# METRICS REGISTRY
# =============================================================================

if PROMETHEUS_AVAILABLE:
    # Request counter with labels
    INFERENCE_REQUESTS_TOTAL = Counter(
        'inference_requests_total',
        'Total number of inference requests',
        ['model_id', 'signal', 'status']
    )

    # Latency histogram (buckets in seconds)
    INFERENCE_LATENCY_SECONDS = Histogram(
        'inference_latency_seconds',
        'Inference request latency in seconds',
        ['model_id'],
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
    )

    # Active models gauge
    ACTIVE_MODELS_COUNT = Gauge(
        'active_models_count',
        'Number of currently loaded models'
    )

    # Error counter by type
    INFERENCE_ERRORS_TOTAL = Counter(
        'inference_errors_total',
        'Total inference errors by type',
        ['error_type', 'model_id']
    )

    # Feature quality gauge
    FEATURE_NAN_RATIO = Gauge(
        'feature_nan_ratio',
        'Ratio of NaN values in the last observation',
        ['model_id']
    )

    # Model agreement (shadow mode)
    MODEL_AGREEMENT_RATIO = Gauge(
        'model_agreement_ratio',
        'Agreement ratio between champion and shadow models (last 100)'
    )

    # Feast metrics
    FEAST_HIT_RATIO = Gauge(
        'feast_hit_ratio',
        'Ratio of Feast hits vs fallbacks'
    )

    FEAST_REQUESTS_TOTAL = Counter(
        'feast_requests_total',
        'Total Feast feature requests',
        ['source']  # 'feast' or 'fallback'
    )

    # Observation dimension validation
    OBSERVATION_DIM_MISMATCH = Counter(
        'observation_dim_mismatch_total',
        'Observations with unexpected dimensions',
        ['expected', 'actual']
    )

    # Trading signals histogram by action value
    SIGNAL_ACTION_HISTOGRAM = Histogram(
        'signal_action_value',
        'Distribution of raw action values from model',
        ['model_id'],
        buckets=[-1.0, -0.5, -0.33, 0.0, 0.33, 0.5, 1.0]
    )

    # System info
    INFERENCE_API_INFO = Info(
        'inference_api',
        'Inference API service information'
    )

else:
    # Mock classes when prometheus_client not available
    class MockMetric:
        def labels(self, *args, **kwargs):
            return self
        def inc(self, *args, **kwargs):
            pass
        def dec(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass
        def info(self, *args, **kwargs):
            pass

    INFERENCE_REQUESTS_TOTAL = MockMetric()
    INFERENCE_LATENCY_SECONDS = MockMetric()
    ACTIVE_MODELS_COUNT = MockMetric()
    INFERENCE_ERRORS_TOTAL = MockMetric()
    FEATURE_NAN_RATIO = MockMetric()
    MODEL_AGREEMENT_RATIO = MockMetric()
    FEAST_HIT_RATIO = MockMetric()
    FEAST_REQUESTS_TOTAL = MockMetric()
    OBSERVATION_DIM_MISMATCH = MockMetric()
    SIGNAL_ACTION_HISTOGRAM = MockMetric()
    INFERENCE_API_INFO = MockMetric()


# =============================================================================
# METRICS HELPER FUNCTIONS
# =============================================================================

def record_inference_request(
    model_id: str,
    signal: str,
    latency_seconds: float,
    success: bool = True
) -> None:
    """Record a completed inference request."""
    status = "success" if success else "error"
    INFERENCE_REQUESTS_TOTAL.labels(
        model_id=model_id,
        signal=signal,
        status=status
    ).inc()
    INFERENCE_LATENCY_SECONDS.labels(model_id=model_id).observe(latency_seconds)
    SIGNAL_ACTION_HISTOGRAM.labels(model_id=model_id).observe(0.0)  # Will be updated with actual


def record_inference_error(error_type: str, model_id: str = "unknown") -> None:
    """Record an inference error."""
    INFERENCE_ERRORS_TOTAL.labels(
        error_type=error_type,
        model_id=model_id
    ).inc()


def record_feature_quality(model_id: str, nan_ratio: float) -> None:
    """Record feature quality (NaN ratio)."""
    FEATURE_NAN_RATIO.labels(model_id=model_id).set(nan_ratio)


def record_feast_request(source: str) -> None:
    """Record a Feast feature request (source: 'feast' or 'fallback')."""
    FEAST_REQUESTS_TOTAL.labels(source=source).inc()


def update_active_models(count: int) -> None:
    """Update the count of active models."""
    ACTIVE_MODELS_COUNT.set(count)


def update_model_agreement(ratio: float) -> None:
    """Update model agreement ratio (shadow mode)."""
    MODEL_AGREEMENT_RATIO.set(ratio)


def update_feast_hit_ratio(ratio: float) -> None:
    """Update Feast hit ratio."""
    FEAST_HIT_RATIO.set(ratio)


def set_api_info(version: str, model_versions: dict = None) -> None:
    """Set API info labels."""
    if PROMETHEUS_AVAILABLE:
        info_dict = {"version": version}
        if model_versions:
            for model_id, model_version in model_versions.items():
                info_dict[f"model_{model_id}_version"] = str(model_version)
        INFERENCE_API_INFO.info(info_dict)


def get_metrics() -> bytes:
    """Get current metrics in Prometheus format."""
    if PROMETHEUS_AVAILABLE:
        return generate_latest(REGISTRY)
    return b"# prometheus_client not available\n"


def get_content_type() -> str:
    """Get content type for metrics endpoint."""
    if PROMETHEUS_AVAILABLE:
        return CONTENT_TYPE_LATEST
    return "text/plain"
