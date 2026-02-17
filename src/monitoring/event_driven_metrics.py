# -*- coding: utf-8 -*-
"""
Event-Driven Metrics - V7.1 Prometheus Metrics
===============================================
Exports Prometheus metrics for the event-driven architecture.

Metrics Exported:
- event_notify_latency_seconds: Histogram of NOTIFY latencies
- event_notify_total: Counter of NOTIFY events
- event_circuit_breaker_state: Gauge of circuit breaker state
- event_dlq_size: Gauge of Dead Letter Queue size
- event_heartbeat_health: Gauge of heartbeat health (1=healthy, 0=unhealthy)
- feature_retrieval_seconds: Histogram of feature retrieval latencies
- feature_retrieval_backend: Counter by backend (postgresql, redis, fallback)

Grafana Dashboard JSON exported via /grafana/event_driven_dashboard.json

Author: Trading Team
Version: 1.0.0
Created: 2026-01-31
Contract: CTR-V7-METRICS
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import prometheus_client
try:
    from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed. Metrics will be logged only.")


# =============================================================================
# METRIC DEFINITIONS
# =============================================================================

# Custom registry for event-driven metrics
REGISTRY = CollectorRegistry() if PROMETHEUS_AVAILABLE else None

if PROMETHEUS_AVAILABLE:
    # NOTIFY latency histogram
    EVENT_NOTIFY_LATENCY = Histogram(
        'event_notify_latency_seconds',
        'Latency of PostgreSQL NOTIFY events',
        ['channel', 'event_type'],
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        registry=REGISTRY,
    )

    # NOTIFY event counter
    EVENT_NOTIFY_TOTAL = Counter(
        'event_notify_total',
        'Total number of NOTIFY events',
        ['channel', 'event_type', 'status'],  # status: received, processed, failed
        registry=REGISTRY,
    )

    # Circuit breaker state (0=CLOSED, 1=OPEN, 2=HALF_OPEN)
    CIRCUIT_BREAKER_STATE = Gauge(
        'event_circuit_breaker_state',
        'Circuit breaker state (0=CLOSED, 1=OPEN, 2=HALF_OPEN)',
        ['sensor_id'],
        registry=REGISTRY,
    )

    # Dead Letter Queue size
    DLQ_SIZE = Gauge(
        'event_dlq_size',
        'Current size of Dead Letter Queue',
        ['status'],  # pending, processing, dead
        registry=REGISTRY,
    )

    # Heartbeat health
    HEARTBEAT_HEALTH = Gauge(
        'event_heartbeat_health',
        'Heartbeat system health (1=healthy, 0=unhealthy)',
        registry=REGISTRY,
    )

    # Feature retrieval latency
    FEATURE_RETRIEVAL_LATENCY = Histogram(
        'feature_retrieval_seconds',
        'Latency of feature retrieval operations',
        ['backend'],  # postgresql, redis, fallback
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        registry=REGISTRY,
    )

    # Feature retrieval counter by backend
    FEATURE_RETRIEVAL_TOTAL = Counter(
        'feature_retrieval_total',
        'Total feature retrievals by backend',
        ['backend', 'status'],  # status: success, failure
        registry=REGISTRY,
    )

    # Market hours indicator
    MARKET_HOURS_ACTIVE = Gauge(
        'market_hours_active',
        'Whether currently in market hours (1=yes, 0=no)',
        registry=REGISTRY,
    )


# =============================================================================
# METRICS RECORDER
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states for metrics."""
    CLOSED = 0
    OPEN = 1
    HALF_OPEN = 2


@dataclass
class MetricsSnapshot:
    """Snapshot of current metrics for logging/export."""
    timestamp: str
    notify_events_total: int = 0
    notify_events_failed: int = 0
    notify_latency_avg_ms: float = 0.0
    notify_latency_p99_ms: float = 0.0
    circuit_breakers_open: int = 0
    dlq_pending: int = 0
    dlq_dead: int = 0
    heartbeat_healthy: bool = True
    feature_retrievals_total: int = 0
    feature_pg_pct: float = 0.0
    feature_redis_pct: float = 0.0
    feature_fallback_pct: float = 0.0


class EventDrivenMetrics:
    """
    Centralized metrics recorder for V7.1 event-driven architecture.

    Provides both Prometheus export and in-memory tracking for
    environments where Prometheus is not available.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._notify_latencies: List[float] = []
        self._notify_count = 0
        self._notify_failed = 0
        self._feature_counts = {'postgresql': 0, 'redis': 0, 'fallback': 0}
        self._circuit_states: Dict[str, CircuitState] = {}
        self._dlq_counts = {'pending': 0, 'processing': 0, 'dead': 0}
        self._heartbeat_healthy = True

    # -------------------------------------------------------------------------
    # NOTIFY METRICS
    # -------------------------------------------------------------------------

    def record_notify_event(
        self,
        channel: str,
        event_type: str,
        latency_seconds: float,
        success: bool = True
    ) -> None:
        """Record a NOTIFY event."""
        with self._lock:
            self._notify_count += 1
            if not success:
                self._notify_failed += 1
            self._notify_latencies.append(latency_seconds)

            # Keep only last 1000 latencies
            if len(self._notify_latencies) > 1000:
                self._notify_latencies = self._notify_latencies[-1000:]

        if PROMETHEUS_AVAILABLE:
            EVENT_NOTIFY_LATENCY.labels(channel=channel, event_type=event_type).observe(latency_seconds)
            status = 'processed' if success else 'failed'
            EVENT_NOTIFY_TOTAL.labels(channel=channel, event_type=event_type, status=status).inc()

        logger.debug(
            f"NOTIFY: channel={channel}, type={event_type}, "
            f"latency={latency_seconds*1000:.2f}ms, success={success}"
        )

    # -------------------------------------------------------------------------
    # CIRCUIT BREAKER METRICS
    # -------------------------------------------------------------------------

    def record_circuit_state(self, sensor_id: str, state: CircuitState) -> None:
        """Record circuit breaker state change."""
        with self._lock:
            self._circuit_states[sensor_id] = state

        if PROMETHEUS_AVAILABLE:
            CIRCUIT_BREAKER_STATE.labels(sensor_id=sensor_id).set(state.value)

        logger.info(f"Circuit breaker [{sensor_id}]: {state.name}")

    def get_open_circuits(self) -> List[str]:
        """Get list of sensors with open circuits."""
        with self._lock:
            return [
                sensor_id for sensor_id, state in self._circuit_states.items()
                if state == CircuitState.OPEN
            ]

    # -------------------------------------------------------------------------
    # DLQ METRICS
    # -------------------------------------------------------------------------

    def update_dlq_size(self, pending: int, processing: int, dead: int) -> None:
        """Update DLQ size metrics."""
        with self._lock:
            self._dlq_counts = {'pending': pending, 'processing': processing, 'dead': dead}

        if PROMETHEUS_AVAILABLE:
            DLQ_SIZE.labels(status='pending').set(pending)
            DLQ_SIZE.labels(status='processing').set(processing)
            DLQ_SIZE.labels(status='dead').set(dead)

        if dead > 0:
            logger.warning(f"DLQ: {dead} events marked as dead")

    # -------------------------------------------------------------------------
    # HEARTBEAT METRICS
    # -------------------------------------------------------------------------

    def record_heartbeat_status(self, healthy: bool) -> None:
        """Record heartbeat health status."""
        with self._lock:
            self._heartbeat_healthy = healthy

        if PROMETHEUS_AVAILABLE:
            HEARTBEAT_HEALTH.set(1.0 if healthy else 0.0)

        if not healthy:
            logger.error("Heartbeat: UNHEALTHY - NOTIFY system may be degraded")

    # -------------------------------------------------------------------------
    # FEATURE RETRIEVAL METRICS
    # -------------------------------------------------------------------------

    def record_feature_retrieval(
        self,
        backend: str,
        latency_seconds: float,
        success: bool = True
    ) -> None:
        """Record a feature retrieval operation."""
        with self._lock:
            if backend in self._feature_counts:
                self._feature_counts[backend] += 1

        if PROMETHEUS_AVAILABLE:
            FEATURE_RETRIEVAL_LATENCY.labels(backend=backend).observe(latency_seconds)
            status = 'success' if success else 'failure'
            FEATURE_RETRIEVAL_TOTAL.labels(backend=backend, status=status).inc()

    def record_market_hours(self, is_market: bool) -> None:
        """Record current market hours state."""
        if PROMETHEUS_AVAILABLE:
            MARKET_HOURS_ACTIVE.set(1.0 if is_market else 0.0)

    # -------------------------------------------------------------------------
    # SNAPSHOT & EXPORT
    # -------------------------------------------------------------------------

    def get_snapshot(self) -> MetricsSnapshot:
        """Get current metrics snapshot."""
        with self._lock:
            # Calculate latency stats
            latencies = self._notify_latencies
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
            p99_latency = sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 10 else 0.0

            # Calculate feature backend percentages
            total_features = sum(self._feature_counts.values())
            pg_pct = (self._feature_counts['postgresql'] / total_features * 100) if total_features else 0.0
            redis_pct = (self._feature_counts['redis'] / total_features * 100) if total_features else 0.0
            fallback_pct = (self._feature_counts['fallback'] / total_features * 100) if total_features else 0.0

            return MetricsSnapshot(
                timestamp=datetime.utcnow().isoformat(),
                notify_events_total=self._notify_count,
                notify_events_failed=self._notify_failed,
                notify_latency_avg_ms=avg_latency * 1000,
                notify_latency_p99_ms=p99_latency * 1000,
                circuit_breakers_open=sum(1 for s in self._circuit_states.values() if s == CircuitState.OPEN),
                dlq_pending=self._dlq_counts['pending'],
                dlq_dead=self._dlq_counts['dead'],
                heartbeat_healthy=self._heartbeat_healthy,
                feature_retrievals_total=total_features,
                feature_pg_pct=pg_pct,
                feature_redis_pct=redis_pct,
                feature_fallback_pct=fallback_pct,
            )

    def export_prometheus(self) -> bytes:
        """Export metrics in Prometheus format."""
        if not PROMETHEUS_AVAILABLE:
            return b"# Prometheus client not installed\n"
        return generate_latest(REGISTRY)


# =============================================================================
# GLOBAL METRICS INSTANCE
# =============================================================================

_metrics_instance: Optional[EventDrivenMetrics] = None


def get_metrics() -> EventDrivenMetrics:
    """Get global metrics instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = EventDrivenMetrics()
    return _metrics_instance


# =============================================================================
# GRAFANA DASHBOARD
# =============================================================================

GRAFANA_DASHBOARD_JSON = '''
{
  "annotations": {
    "list": []
  },
  "editable": true,
  "fiscalYearStartMonth": 0,
  "graphTooltip": 0,
  "id": null,
  "links": [],
  "liveNow": false,
  "panels": [
    {
      "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
      "fieldConfig": {
        "defaults": {"color": {"mode": "palette-classic"}, "unit": "ms"},
        "overrides": []
      },
      "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
      "id": 1,
      "options": {"legend": {"displayMode": "list"}, "tooltip": {"mode": "single"}},
      "targets": [
        {
          "expr": "histogram_quantile(0.99, rate(event_notify_latency_seconds_bucket[5m])) * 1000",
          "legendFormat": "p99",
          "refId": "A"
        },
        {
          "expr": "histogram_quantile(0.95, rate(event_notify_latency_seconds_bucket[5m])) * 1000",
          "legendFormat": "p95",
          "refId": "B"
        },
        {
          "expr": "histogram_quantile(0.50, rate(event_notify_latency_seconds_bucket[5m])) * 1000",
          "legendFormat": "p50",
          "refId": "C"
        }
      ],
      "title": "NOTIFY Latency",
      "type": "timeseries"
    },
    {
      "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
      "fieldConfig": {"defaults": {"unit": "short"}},
      "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
      "id": 2,
      "options": {"legend": {"displayMode": "list"}},
      "targets": [
        {
          "expr": "rate(event_notify_total{status='processed'}[1m])",
          "legendFormat": "Processed",
          "refId": "A"
        },
        {
          "expr": "rate(event_notify_total{status='failed'}[1m])",
          "legendFormat": "Failed",
          "refId": "B"
        }
      ],
      "title": "NOTIFY Events Rate",
      "type": "timeseries"
    },
    {
      "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
      "fieldConfig": {"defaults": {"mappings": [{"options": {"0": {"text": "CLOSED"}, "1": {"text": "OPEN"}, "2": {"text": "HALF-OPEN"}}, "type": "value"}]}},
      "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8},
      "id": 3,
      "options": {"colorMode": "value", "graphMode": "none"},
      "targets": [{"expr": "event_circuit_breaker_state", "legendFormat": "{{sensor_id}}", "refId": "A"}],
      "title": "Circuit Breaker State",
      "type": "stat"
    },
    {
      "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
      "fieldConfig": {"defaults": {"thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": null}, {"color": "yellow", "value": 1}, {"color": "red", "value": 5}]}}},
      "gridPos": {"h": 4, "w": 6, "x": 6, "y": 8},
      "id": 4,
      "options": {"colorMode": "value"},
      "targets": [{"expr": "event_dlq_size{status='pending'}", "legendFormat": "Pending", "refId": "A"}],
      "title": "DLQ Pending",
      "type": "stat"
    },
    {
      "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
      "fieldConfig": {"defaults": {"mappings": [{"options": {"0": {"color": "red", "text": "UNHEALTHY"}, "1": {"color": "green", "text": "HEALTHY"}}, "type": "value"}]}},
      "gridPos": {"h": 4, "w": 6, "x": 12, "y": 8},
      "id": 5,
      "options": {"colorMode": "background"},
      "targets": [{"expr": "event_heartbeat_health", "refId": "A"}],
      "title": "Heartbeat Health",
      "type": "stat"
    },
    {
      "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
      "fieldConfig": {"defaults": {"mappings": [{"options": {"0": {"text": "OFF-MARKET"}, "1": {"text": "MARKET HOURS"}}, "type": "value"}]}},
      "gridPos": {"h": 4, "w": 6, "x": 18, "y": 8},
      "id": 6,
      "options": {"colorMode": "value"},
      "targets": [{"expr": "market_hours_active", "refId": "A"}],
      "title": "Market Status",
      "type": "stat"
    },
    {
      "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
      "fieldConfig": {"defaults": {"unit": "ms"}},
      "gridPos": {"h": 8, "w": 12, "x": 0, "y": 12},
      "id": 7,
      "options": {"legend": {"displayMode": "list"}},
      "targets": [
        {
          "expr": "histogram_quantile(0.99, rate(feature_retrieval_seconds_bucket[5m])) * 1000",
          "legendFormat": "{{backend}} p99",
          "refId": "A"
        }
      ],
      "title": "Feature Retrieval Latency by Backend",
      "type": "timeseries"
    },
    {
      "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
      "fieldConfig": {"defaults": {"unit": "short"}},
      "gridPos": {"h": 8, "w": 12, "x": 12, "y": 12},
      "id": 8,
      "options": {"legend": {"displayMode": "list"}},
      "targets": [
        {
          "expr": "sum by(backend) (rate(feature_retrieval_total[5m]))",
          "legendFormat": "{{backend}}",
          "refId": "A"
        }
      ],
      "title": "Feature Retrievals by Backend",
      "type": "timeseries"
    }
  ],
  "schemaVersion": 38,
  "style": "dark",
  "tags": ["usdcop", "v7", "event-driven"],
  "templating": {"list": []},
  "time": {"from": "now-1h", "to": "now"},
  "timepicker": {},
  "timezone": "America/Bogota",
  "title": "USDCOP V7.1 Event-Driven Dashboard",
  "uid": "usdcop-v7-events",
  "version": 1,
  "weekStart": ""
}
'''


def export_grafana_dashboard(path: str = "grafana/event_driven_dashboard.json") -> None:
    """Export Grafana dashboard JSON to file."""
    import json
    from pathlib import Path

    dashboard = json.loads(GRAFANA_DASHBOARD_JSON)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(dashboard, f, indent=2)

    logger.info(f"Grafana dashboard exported to {output_path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Event-Driven Metrics")
    parser.add_argument("--export-dashboard", type=str, help="Export Grafana dashboard JSON")
    parser.add_argument("--snapshot", action="store_true", help="Print current metrics snapshot")
    args = parser.parse_args()

    if args.export_dashboard:
        export_grafana_dashboard(args.export_dashboard)
        print(f"Dashboard exported to {args.export_dashboard}")

    elif args.snapshot:
        import json
        metrics = get_metrics()
        snapshot = metrics.get_snapshot()
        print(json.dumps(snapshot.__dict__, indent=2))

    else:
        print("Event-Driven Metrics V7.1")
        print(f"Prometheus available: {PROMETHEUS_AVAILABLE}")
