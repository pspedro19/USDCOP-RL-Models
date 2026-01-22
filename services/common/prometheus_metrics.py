"""
Prometheus Metrics for Observability.

This module provides Prometheus metrics for monitoring
all USDCOP services. Distinct from metrics.py which handles
financial calculations.

Design Patterns:
- Singleton Pattern: Single metrics registry per service
- Factory Pattern: Metrics creation via setup function

Usage:
    from common.prometheus_metrics import (
        setup_prometheus_metrics,
        inference_requests_total,
        inference_latency_seconds,
    )

    # Setup in main.py
    setup_prometheus_metrics(app, "inference-api")

    # Use in routes
    with inference_latency_seconds.labels(model_id="ppo_v20").time():
        result = model.predict(obs)
    inference_requests_total.labels(model_id="ppo_v20", status="success").inc()

Contract: CTR-OBS-001
"""

__all__ = [
    # Setup and configuration
    "setup_prometheus_metrics",
    "get_metrics_app",
    "PROMETHEUS_AVAILABLE",
    # Counters
    "inference_requests_total",
    "trade_signals_total",
    "model_load_total",
    "feature_calculation_errors_total",
    "circuit_breaker_activations_total",
    # Histograms
    "inference_latency_seconds",
    "feature_calculation_seconds",
    "db_query_seconds",
    "model_prediction_distribution",
    # Gauges
    "current_position_gauge",
    "model_confidence_gauge",
    "feature_drift_gauge",
    "data_freshness_gauge",
    "active_models_gauge",
    "consecutive_losses_gauge",
    # Macro ingestion metrics (P0-10)
    "macro_ingestion_success",
    "macro_ingestion_errors",
    "macro_data_staleness",
    "macro_ingestion_latency",
    "macro_indicators_available",
    # Info
    "service_info",
    # Decorators
    "track_latency",
    "track_latency_async",
    "count_requests",
    # Utility functions
    "record_inference",
    "record_trade_signal",
    "record_feature_drift",
    "record_data_freshness",
    # Macro ingestion helper functions (P0-10)
    "record_macro_ingestion_success",
    "record_macro_ingestion_error",
    "update_macro_staleness",
    "record_macro_ingestion_latency",
    "update_macro_indicators_available",
]

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Track initialization state
_initialized = False
_metrics_app = None

# =============================================================================
# TRY TO IMPORT PROMETHEUS CLIENT
# =============================================================================

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Info,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
        multiprocess,
        REGISTRY,
    )
    from prometheus_client.exposition import make_asgi_app
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning(
        "prometheus_client not installed. Install with: "
        "pip install prometheus-client"
    )


# =============================================================================
# NO-OP FALLBACKS (when prometheus_client not available)
# =============================================================================

class NoOpMetric:
    """No-operation metric for when prometheus_client is not available."""

    def __init__(self, *args, **kwargs):
        pass

    def labels(self, **kwargs):
        return self

    def inc(self, amount: float = 1):
        pass

    def dec(self, amount: float = 1):
        pass

    def set(self, value: float):
        pass

    def observe(self, value: float):
        pass

    @contextmanager
    def time(self):
        yield


class NoOpInfo:
    """No-operation info metric."""

    def __init__(self, *args, **kwargs):
        pass

    def info(self, val: Dict[str, str]):
        pass


# =============================================================================
# METRIC DEFINITIONS
# =============================================================================

if PROMETHEUS_AVAILABLE:
    # -------------------------------------------------------------------------
    # COUNTERS - Monotonically increasing values
    # -------------------------------------------------------------------------

    inference_requests_total = Counter(
        'usdcop_inference_requests_total',
        'Total number of inference requests',
        ['model_id', 'status']
    )

    trade_signals_total = Counter(
        'usdcop_trade_signals_total',
        'Total number of trade signals generated',
        ['model_id', 'signal_type']  # signal_type: LONG, SHORT, HOLD
    )

    model_load_total = Counter(
        'usdcop_model_load_total',
        'Total number of model loads',
        ['model_id', 'status']
    )

    feature_calculation_errors_total = Counter(
        'usdcop_feature_calculation_errors_total',
        'Total number of feature calculation errors',
        ['feature_name', 'error_type']
    )

    circuit_breaker_activations_total = Counter(
        'usdcop_circuit_breaker_activations_total',
        'Total number of circuit breaker activations',
        ['model_id', 'reason']
    )

    # -------------------------------------------------------------------------
    # HISTOGRAMS - Distribution of values
    # -------------------------------------------------------------------------

    inference_latency_seconds = Histogram(
        'usdcop_inference_latency_seconds',
        'Inference latency in seconds',
        ['model_id'],
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
    )

    feature_calculation_seconds = Histogram(
        'usdcop_feature_calculation_seconds',
        'Feature calculation time in seconds',
        ['feature_set'],  # feature_set: market, macro, all
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
    )

    db_query_seconds = Histogram(
        'usdcop_db_query_seconds',
        'Database query time in seconds',
        ['query_type'],  # query_type: ohlcv, macro, trades, etc.
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    )

    model_prediction_distribution = Histogram(
        'usdcop_model_prediction_distribution',
        'Distribution of model prediction values',
        ['model_id'],
        buckets=[-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )

    # -------------------------------------------------------------------------
    # GAUGES - Values that can go up and down
    # -------------------------------------------------------------------------

    current_position_gauge = Gauge(
        'usdcop_current_position',
        'Current trading position (-1=SHORT, 0=FLAT, 1=LONG)',
        ['model_id']
    )

    model_confidence_gauge = Gauge(
        'usdcop_model_confidence',
        'Model confidence score (0-1)',
        ['model_id']
    )

    feature_drift_gauge = Gauge(
        'usdcop_feature_drift',
        'Feature drift from training distribution (z-score)',
        ['feature_name']
    )

    data_freshness_gauge = Gauge(
        'usdcop_data_freshness_seconds',
        'Age of latest data in seconds',
        ['data_type']  # data_type: ohlcv, macro
    )

    active_models_gauge = Gauge(
        'usdcop_active_models',
        'Number of currently loaded models',
        []
    )

    consecutive_losses_gauge = Gauge(
        'usdcop_consecutive_losses',
        'Number of consecutive losing trades',
        ['model_id']
    )

    # -------------------------------------------------------------------------
    # MACRO INGESTION METRICS (P0-10)
    # -------------------------------------------------------------------------

    macro_ingestion_success = Counter(
        'usdcop_macro_ingestion_success_total',
        'Total successful macro data ingestions',
        ['source', 'indicator']
    )

    macro_ingestion_errors = Counter(
        'usdcop_macro_ingestion_errors_total',
        'Total failed macro data ingestions',
        ['source', 'indicator', 'error_type']
    )

    macro_data_staleness = Gauge(
        'usdcop_macro_data_staleness_seconds',
        'Age of latest macro data in seconds',
        ['source', 'indicator']
    )

    macro_ingestion_latency = Histogram(
        'usdcop_macro_ingestion_latency_seconds',
        'Latency of macro data ingestion',
        ['source'],
        buckets=[0.5, 1, 2, 5, 10, 30, 60, 120, 300]
    )

    macro_indicators_available = Gauge(
        'usdcop_macro_indicators_available',
        'Number of macro indicators with fresh data',
        ['source']
    )

    # -------------------------------------------------------------------------
    # INFO - Static information about the service
    # -------------------------------------------------------------------------

    service_info = Info(
        'usdcop_service',
        'Information about the USDCOP service'
    )

else:
    # Use no-op fallbacks
    inference_requests_total = NoOpMetric()
    trade_signals_total = NoOpMetric()
    model_load_total = NoOpMetric()
    feature_calculation_errors_total = NoOpMetric()
    circuit_breaker_activations_total = NoOpMetric()
    inference_latency_seconds = NoOpMetric()
    feature_calculation_seconds = NoOpMetric()
    db_query_seconds = NoOpMetric()
    model_prediction_distribution = NoOpMetric()
    current_position_gauge = NoOpMetric()
    model_confidence_gauge = NoOpMetric()
    feature_drift_gauge = NoOpMetric()
    data_freshness_gauge = NoOpMetric()
    active_models_gauge = NoOpMetric()
    consecutive_losses_gauge = NoOpMetric()
    macro_ingestion_success = NoOpMetric()
    macro_ingestion_errors = NoOpMetric()
    macro_data_staleness = NoOpMetric()
    macro_ingestion_latency = NoOpMetric()
    macro_indicators_available = NoOpMetric()
    service_info = NoOpInfo()


# =============================================================================
# SETUP AND INTEGRATION
# =============================================================================

def setup_prometheus_metrics(
    app: Any,
    service_name: str,
    service_version: str = "1.0.0",
    expose_endpoint: bool = True,
) -> None:
    """
    Setup Prometheus metrics for a FastAPI application.

    Args:
        app: FastAPI application instance
        service_name: Name of the service (e.g., "inference-api")
        service_version: Service version string
        expose_endpoint: Whether to expose /metrics endpoint

    Example:
        app = FastAPI()
        setup_prometheus_metrics(app, "inference-api", "1.0.0")
    """
    global _initialized, _metrics_app

    if not PROMETHEUS_AVAILABLE:
        logger.warning("Prometheus client not available, metrics disabled")
        return

    if _initialized:
        logger.debug("Prometheus metrics already initialized")
        return

    # Set service info
    service_info.info({
        'service_name': service_name,
        'version': service_version,
    })

    # Create metrics ASGI app
    if expose_endpoint:
        _metrics_app = make_asgi_app()

        # Mount metrics endpoint
        try:
            from starlette.routing import Mount
            app.mount("/metrics", _metrics_app)
            logger.info(f"Prometheus metrics exposed at /metrics for {service_name}")
        except Exception as e:
            logger.warning(f"Could not mount /metrics endpoint: {e}")

    _initialized = True
    logger.info(f"Prometheus metrics initialized for {service_name} v{service_version}")


def get_metrics_app() -> Optional[Any]:
    """Get the Prometheus metrics ASGI app."""
    return _metrics_app


# =============================================================================
# CONVENIENCE DECORATORS
# =============================================================================

def track_latency(histogram: Any, labels: Optional[Dict[str, str]] = None):
    """
    Decorator to track function execution time.

    Args:
        histogram: Prometheus Histogram metric
        labels: Optional labels for the metric

    Example:
        @track_latency(inference_latency_seconds, {"model_id": "ppo_v20"})
        def predict(obs):
            return model.predict(obs)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if labels:
                metric = histogram.labels(**labels)
            else:
                metric = histogram

            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start_time
                metric.observe(elapsed)

        return wrapper
    return decorator


def track_latency_async(histogram: Any, labels: Optional[Dict[str, str]] = None):
    """
    Decorator to track async function execution time.

    Example:
        @track_latency_async(inference_latency_seconds, {"model_id": "ppo_v20"})
        async def predict_async(obs):
            return await model.predict_async(obs)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if labels:
                metric = histogram.labels(**labels)
            else:
                metric = histogram

            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start_time
                metric.observe(elapsed)

        return wrapper
    return decorator


def count_requests(counter: Any, labels_fn: Optional[Callable] = None):
    """
    Decorator to count function calls.

    Args:
        counter: Prometheus Counter metric
        labels_fn: Optional function to generate labels from args/kwargs

    Example:
        @count_requests(inference_requests_total, lambda model_id, **kw: {"model_id": model_id, "status": "success"})
        def predict(model_id, obs):
            return model.predict(obs)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                if labels_fn:
                    labels = labels_fn(*args, **kwargs)
                    if "status" not in labels:
                        labels["status"] = "success"
                    counter.labels(**labels).inc()
                else:
                    counter.inc()
                return result
            except Exception as e:
                if labels_fn:
                    labels = labels_fn(*args, **kwargs)
                    labels["status"] = "error"
                    counter.labels(**labels).inc()
                raise

        return wrapper
    return decorator


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def record_inference(
    model_id: str,
    latency: float,
    prediction: float,
    success: bool = True,
) -> None:
    """
    Record a complete inference request with all relevant metrics.

    Args:
        model_id: Model identifier
        latency: Inference latency in seconds
        prediction: Model prediction value (-1 to 1)
        success: Whether inference was successful
    """
    status = "success" if success else "error"
    inference_requests_total.labels(model_id=model_id, status=status).inc()
    inference_latency_seconds.labels(model_id=model_id).observe(latency)

    if success:
        model_prediction_distribution.labels(model_id=model_id).observe(prediction)


def record_trade_signal(
    model_id: str,
    signal_type: str,
    confidence: float,
) -> None:
    """
    Record a trade signal generation.

    Args:
        model_id: Model identifier
        signal_type: Signal type (LONG, SHORT, HOLD)
        confidence: Model confidence (0-1)
    """
    trade_signals_total.labels(model_id=model_id, signal_type=signal_type).inc()
    model_confidence_gauge.labels(model_id=model_id).set(confidence)


def record_feature_drift(feature_name: str, zscore: float) -> None:
    """
    Record feature drift for monitoring.

    Args:
        feature_name: Name of the feature
        zscore: Z-score distance from training distribution
    """
    feature_drift_gauge.labels(feature_name=feature_name).set(zscore)


def record_data_freshness(data_type: str, age_seconds: float) -> None:
    """
    Record data freshness.

    Args:
        data_type: Type of data (ohlcv, macro)
        age_seconds: Age of latest data in seconds
    """
    data_freshness_gauge.labels(data_type=data_type).set(age_seconds)


# =============================================================================
# MACRO INGESTION HELPER FUNCTIONS (P0-10)
# =============================================================================

def record_macro_ingestion_success(source: str, indicator: str) -> None:
    """
    Record a successful macro data ingestion.

    This function increments the success counter for macro data ingestion,
    allowing tracking of successful data pulls from various sources.

    Args:
        source: Data source identifier (e.g., 'banrep', 'dane', 'fred')
        indicator: Macro indicator name (e.g., 'cpi', 'gdp', 'interest_rate')

    Example:
        record_macro_ingestion_success('banrep', 'interest_rate')
    """
    macro_ingestion_success.labels(source=source, indicator=indicator).inc()


def record_macro_ingestion_error(
    source: str,
    indicator: str,
    error_type: str
) -> None:
    """
    Record a failed macro data ingestion.

    This function increments the error counter for macro data ingestion,
    categorized by error type for debugging and alerting purposes.

    Args:
        source: Data source identifier (e.g., 'banrep', 'dane', 'fred')
        indicator: Macro indicator name (e.g., 'cpi', 'gdp', 'interest_rate')
        error_type: Type of error encountered (e.g., 'timeout', 'parse_error',
                    'api_error', 'validation_error', 'network_error')

    Example:
        record_macro_ingestion_error('fred', 'gdp', 'timeout')
    """
    macro_ingestion_errors.labels(
        source=source,
        indicator=indicator,
        error_type=error_type
    ).inc()


def update_macro_staleness(source: str, indicator: str, age_seconds: float) -> None:
    """
    Update the staleness gauge for a macro indicator.

    This function sets the current age (in seconds) of the latest data point
    for a specific macro indicator, enabling staleness-based alerting.

    Args:
        source: Data source identifier (e.g., 'banrep', 'dane', 'fred')
        indicator: Macro indicator name (e.g., 'cpi', 'gdp', 'interest_rate')
        age_seconds: Age of the latest data point in seconds

    Example:
        # Data is 2 hours old
        update_macro_staleness('banrep', 'interest_rate', 7200.0)
    """
    macro_data_staleness.labels(source=source, indicator=indicator).set(age_seconds)


def record_macro_ingestion_latency(source: str, latency_seconds: float) -> None:
    """
    Record the latency of a macro data ingestion operation.

    This function observes the time taken to fetch and process macro data
    from a specific source, useful for performance monitoring.

    Args:
        source: Data source identifier (e.g., 'banrep', 'dane', 'fred')
        latency_seconds: Time taken for the ingestion in seconds

    Example:
        start = time.perf_counter()
        fetch_macro_data('banrep')
        record_macro_ingestion_latency('banrep', time.perf_counter() - start)
    """
    macro_ingestion_latency.labels(source=source).observe(latency_seconds)


def update_macro_indicators_available(source: str, count: int) -> None:
    """
    Update the count of available macro indicators for a source.

    This function sets the number of macro indicators that have fresh
    (non-stale) data available from a specific source.

    Args:
        source: Data source identifier (e.g., 'banrep', 'dane', 'fred')
        count: Number of indicators with fresh data

    Example:
        # 5 out of 8 indicators have fresh data
        update_macro_indicators_available('banrep', 5)
    """
    macro_indicators_available.labels(source=source).set(count)
