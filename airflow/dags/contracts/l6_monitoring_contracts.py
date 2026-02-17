# -*- coding: utf-8 -*-
"""
L6 Production Monitoring Contracts
==================================

Pydantic contracts for Layer 6 (Production Monitoring) operations.
Ensures type safety and validation for monitoring, drift detection,
and alerting operations.

Contract: CTR-L6-001

Architecture:
    L5 (Inference) → L6 (Monitoring) → Alerts/Dashboards

Monitoring Domains:
    - Data Quality: Completeness, freshness, drift
    - Model Performance: Latency, accuracy, drift
    - System Health: Uptime, errors, resources
    - SLA Compliance: Latency P99, success rate

Version: 1.0.0
"""

from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# =============================================================================
# ENUMS
# =============================================================================

class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertChannel(str, Enum):
    """Alert notification channels."""
    SLACK = "slack"
    EMAIL = "email"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    LOG = "log"


class DriftType(str, Enum):
    """Types of drift detection."""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    FEATURE_DRIFT = "feature_drift"


class MonitoringStatus(str, Enum):
    """Status of monitoring check."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# =============================================================================
# SLA CONTRACTS
# =============================================================================

class SLAConfig(BaseModel):
    """SLA configuration for a service or pipeline stage."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="SLA name")
    latency_p50_ms: float = Field(default=100.0, description="P50 latency threshold (ms)")
    latency_p95_ms: float = Field(default=500.0, description="P95 latency threshold (ms)")
    latency_p99_ms: float = Field(default=1000.0, description="P99 latency threshold (ms)")
    success_rate_threshold: float = Field(default=0.99, description="Minimum success rate (0-1)")
    availability_threshold: float = Field(default=0.999, description="Minimum availability (0-1)")
    max_error_rate: float = Field(default=0.01, description="Maximum error rate (0-1)")


class SLAResult(BaseModel):
    """Result of SLA compliance check."""

    model_config = ConfigDict(frozen=True)

    sla_name: str
    timestamp: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    window_minutes: int = Field(default=60, description="Measurement window")

    # Measured values
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    success_rate: float
    availability: float
    error_rate: float

    # Compliance
    latency_compliant: bool
    success_rate_compliant: bool
    availability_compliant: bool
    overall_compliant: bool

    # Metadata
    sample_count: int = 0
    violations: List[str] = Field(default_factory=list)


# =============================================================================
# DRIFT DETECTION CONTRACTS
# =============================================================================

class DriftDetectionConfig(BaseModel):
    """Configuration for drift detection."""

    model_config = ConfigDict(frozen=True)

    detection_method: str = Field(default="ks_test", description="Statistical test method")
    threshold: float = Field(default=0.05, description="P-value threshold for significance")
    window_size: int = Field(default=1000, description="Number of samples in detection window")
    reference_window_size: int = Field(default=10000, description="Number of samples in reference")
    features_to_monitor: List[str] = Field(default_factory=list)
    check_interval_minutes: int = Field(default=60)


class DriftDetectionResult(BaseModel):
    """Result of drift detection analysis."""

    model_config = ConfigDict(frozen=True)

    timestamp: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    drift_type: DriftType
    drift_detected: bool
    drift_score: float = Field(..., ge=0.0, le=1.0, description="Drift magnitude (0-1)")
    p_value: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # Feature-level details
    features_drifted: List[str] = Field(default_factory=list)
    feature_scores: Dict[str, float] = Field(default_factory=dict)

    # Metadata
    reference_period: Optional[str] = None
    detection_period: Optional[str] = None
    sample_count: int = 0
    method_used: str = "ks_test"


# =============================================================================
# ALERT CONTRACTS
# =============================================================================

class AlertConfig(BaseModel):
    """Configuration for alerting rules."""

    model_config = ConfigDict(frozen=True)

    name: str
    severity: AlertSeverity
    channels: List[AlertChannel] = Field(default_factory=lambda: [AlertChannel.LOG])
    condition: str = Field(..., description="Alert condition expression")
    cooldown_minutes: int = Field(default=30, description="Minimum time between alerts")
    enabled: bool = True

    # Escalation
    escalate_after_minutes: Optional[int] = None
    escalation_severity: Optional[AlertSeverity] = None
    escalation_channels: List[AlertChannel] = Field(default_factory=list)


class Alert(BaseModel):
    """A triggered alert."""

    model_config = ConfigDict(frozen=True)

    alert_id: str = Field(..., description="Unique alert ID")
    alert_name: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: dt.datetime = Field(default_factory=dt.datetime.utcnow)

    # Context
    source: str = Field(..., description="Source system/component")
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold_value: Optional[float] = None

    # State
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: Optional[dt.datetime] = None

    # Metadata
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)


# =============================================================================
# HEALTH CHECK CONTRACTS
# =============================================================================

class ComponentHealth(BaseModel):
    """Health status of a single component."""

    model_config = ConfigDict(frozen=True)

    component: str
    status: MonitoringStatus
    message: Optional[str] = None
    last_check: dt.datetime = Field(default_factory=dt.datetime.utcnow)

    # Metrics
    latency_ms: Optional[float] = None
    error_count: int = 0
    success_count: int = 0

    # Details
    details: Dict[str, Any] = Field(default_factory=dict)


class SystemHealth(BaseModel):
    """Overall system health report."""

    model_config = ConfigDict(frozen=True)

    timestamp: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    overall_status: MonitoringStatus
    components: List[ComponentHealth] = Field(default_factory=list)

    # Summary
    healthy_count: int = 0
    degraded_count: int = 0
    unhealthy_count: int = 0

    # Metadata
    version: str = "1.0.0"
    environment: str = "production"

    @model_validator(mode='after')
    def calculate_overall_status(self) -> 'SystemHealth':
        """Calculate overall status from components."""
        if not self.components:
            return self

        statuses = [c.status for c in self.components]

        if MonitoringStatus.UNHEALTHY in statuses:
            object.__setattr__(self, 'overall_status', MonitoringStatus.UNHEALTHY)
        elif MonitoringStatus.DEGRADED in statuses:
            object.__setattr__(self, 'overall_status', MonitoringStatus.DEGRADED)
        elif all(s == MonitoringStatus.HEALTHY for s in statuses):
            object.__setattr__(self, 'overall_status', MonitoringStatus.HEALTHY)
        else:
            object.__setattr__(self, 'overall_status', MonitoringStatus.UNKNOWN)

        return self


# =============================================================================
# CIRCUIT BREAKER CONTRACTS
# =============================================================================

class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker."""

    model_config = ConfigDict(frozen=True)

    name: str
    failure_threshold: int = Field(default=5, description="Failures before opening")
    success_threshold: int = Field(default=3, description="Successes to close from half-open")
    timeout_seconds: int = Field(default=60, description="Time in open state before half-open")
    half_open_max_calls: int = Field(default=3, description="Max calls in half-open state")


class CircuitBreakerStatus(BaseModel):
    """Current status of a circuit breaker."""

    model_config = ConfigDict(frozen=True)

    name: str
    state: CircuitBreakerState
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[dt.datetime] = None
    last_success_time: Optional[dt.datetime] = None
    state_changed_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)

    # Stats
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0


# =============================================================================
# L6 MONITORING OUTPUT CONTRACT
# =============================================================================

class L6MonitoringOutput(BaseModel):
    """
    Output contract for L6 monitoring DAG.

    Contract: CTR-L6-OUTPUT-001

    This is the main output of the L6 monitoring pipeline,
    aggregating all monitoring results.
    """

    model_config = ConfigDict(frozen=True)

    # Identification
    run_id: str
    timestamp: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    pipeline_version: str = "1.0.0"

    # Drift Detection
    drift_detected: bool = False
    drift_results: List[DriftDetectionResult] = Field(default_factory=list)

    # SLA Compliance
    sla_compliant: bool = True
    sla_results: List[SLAResult] = Field(default_factory=list)

    # Performance Metrics
    latency_p99_ms: float = 0.0
    success_rate: float = 1.0
    error_rate: float = 0.0

    # Alerts
    alerts_triggered: List[Alert] = Field(default_factory=list)
    critical_alerts_count: int = 0

    # Health
    system_health: Optional[SystemHealth] = None

    # Circuit Breakers
    circuit_breakers: List[CircuitBreakerStatus] = Field(default_factory=list)
    open_breakers_count: int = 0

    # Actions Taken
    rollback_triggered: bool = False
    rollback_reason: Optional[str] = None
    automatic_actions: List[str] = Field(default_factory=list)

    # Metadata
    duration_seconds: float = 0.0
    checks_performed: int = 0
    errors: List[str] = Field(default_factory=list)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Get summary for logging/alerting."""
        return {
            'run_id': self.run_id,
            'timestamp': self.timestamp.isoformat(),
            'drift_detected': self.drift_detected,
            'sla_compliant': self.sla_compliant,
            'latency_p99_ms': self.latency_p99_ms,
            'success_rate': self.success_rate,
            'critical_alerts': self.critical_alerts_count,
            'open_breakers': self.open_breakers_count,
            'rollback_triggered': self.rollback_triggered,
            'checks_performed': self.checks_performed,
        }


# =============================================================================
# DEFAULT SLA CONFIGURATIONS
# =============================================================================

DEFAULT_SLAS = {
    'l0_extraction': SLAConfig(
        name='l0_extraction',
        latency_p50_ms=5000,
        latency_p95_ms=15000,
        latency_p99_ms=30000,
        success_rate_threshold=0.95,
    ),
    'l1_feature_calculation': SLAConfig(
        name='l1_feature_calculation',
        latency_p50_ms=500,
        latency_p95_ms=2000,
        latency_p99_ms=5000,
        success_rate_threshold=0.99,
    ),
    'l5_inference': SLAConfig(
        name='l5_inference',
        latency_p50_ms=100,
        latency_p95_ms=500,
        latency_p99_ms=1000,
        success_rate_threshold=0.999,
    ),
}


# =============================================================================
# DEFAULT ALERT CONFIGURATIONS
# =============================================================================

DEFAULT_ALERTS = {
    'high_latency': AlertConfig(
        name='high_latency',
        severity=AlertSeverity.WARNING,
        channels=[AlertChannel.SLACK, AlertChannel.LOG],
        condition='latency_p99_ms > 5000',
        cooldown_minutes=15,
    ),
    'low_success_rate': AlertConfig(
        name='low_success_rate',
        severity=AlertSeverity.CRITICAL,
        channels=[AlertChannel.SLACK, AlertChannel.PAGERDUTY],
        condition='success_rate < 0.95',
        cooldown_minutes=5,
        escalate_after_minutes=15,
        escalation_severity=AlertSeverity.EMERGENCY,
    ),
    'drift_detected': AlertConfig(
        name='drift_detected',
        severity=AlertSeverity.WARNING,
        channels=[AlertChannel.SLACK],
        condition='drift_detected == true',
        cooldown_minutes=60,
    ),
    'circuit_breaker_open': AlertConfig(
        name='circuit_breaker_open',
        severity=AlertSeverity.CRITICAL,
        channels=[AlertChannel.SLACK, AlertChannel.PAGERDUTY],
        condition='circuit_breaker_state == open',
        cooldown_minutes=5,
    ),
}
