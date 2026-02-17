"""
Airflow Custom Sensors for USD/COP Trading System
==================================================

Event-driven sensors to replace fixed schedule patterns.
Instead of running every 5 minutes blindly, sensors wait for actual new data.

V7.1 Architecture:
- PostgreSQL NOTIFY for near real-time event notification
- Circuit Breaker pattern with automatic fallback to polling
- Idempotent event processing via hash-based deduplication
- Dead Letter Queue for failed event retry
- Heartbeat monitoring for system health

Available Sensors:
- OHLCVBarSensor: PostgreSQL NOTIFY sensor for new OHLCV bars (V7.1)
- FeatureReadySensor: PostgreSQL NOTIFY sensor for feature readiness (V7.1)
- NewOHLCVBarSensor: Legacy alias for OHLCVBarSensor
- NewFeatureBarSensor: Legacy alias for FeatureReadySensor
- L1FeaturesSensor: Waits for L1 feature pipeline output with hash validation

Author: Pedro @ Lean Tech Solutions
Version: 2.0.0 (V7.1 Event-Driven)
Created: 2025-01-12
Updated: 2026-01-31
"""

# V7.1 PostgreSQL NOTIFY sensors (primary)
from sensors.postgres_notify_sensor import (
    OHLCVBarSensor,
    FeatureReadySensor,
    HeartbeatMonitor,
    CircuitBreaker,
    IdempotentProcessor,
    DeadLetterQueue,
    # Backward compatibility aliases
    NewOHLCVBarSensor,
    NewFeatureBarSensor,
)

# Legacy polling-based sensors (kept for fallback)
from sensors.new_bar_sensor import (
    NewOHLCVBarSensor as LegacyOHLCVBarSensor,
    NewFeatureBarSensor as LegacyFeatureBarSensor,
    DataFreshnessGuard,
)

from sensors.feature_sensor import (
    L1FeaturesSensor,
    L1FeaturesAvailableSensor,
)

__all__ = [
    # V7.1 Primary sensors
    'OHLCVBarSensor',
    'FeatureReadySensor',
    'HeartbeatMonitor',
    'CircuitBreaker',
    'IdempotentProcessor',
    'DeadLetterQueue',
    # Backward compatibility
    'NewOHLCVBarSensor',
    'NewFeatureBarSensor',
    # Legacy (for explicit fallback)
    'LegacyOHLCVBarSensor',
    'LegacyFeatureBarSensor',
    'DataFreshnessGuard',
    # L1 sensors
    'L1FeaturesSensor',
    'L1FeaturesAvailableSensor',
]
