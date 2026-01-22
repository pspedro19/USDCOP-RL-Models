"""
Airflow Custom Sensors for USD/COP Trading System
==================================================

Event-driven sensors to replace fixed schedule patterns.
Instead of running every 5 minutes blindly, sensors wait for actual new data.

Benefits:
- Prevents schedule drift
- Avoids overlapping jobs
- More event-driven architecture
- No extra infrastructure needed (uses existing PostgreSQL)

Available Sensors:
- NewOHLCVBarSensor: Waits for new OHLCV bars in usdcop_m5_ohlcv
- NewFeatureBarSensor: Waits for new feature data in inference_features_5m
- L1FeaturesSensor: Waits for L1 feature pipeline output with hash validation

Author: Pedro @ Lean Tech Solutions
Version: 1.1.0
Created: 2025-01-12
Updated: 2026-01-17
"""

from sensors.new_bar_sensor import (
    NewOHLCVBarSensor,
    NewFeatureBarSensor,
)

from sensors.feature_sensor import (
    L1FeaturesSensor,
    L1FeaturesAvailableSensor,
)

__all__ = [
    'NewOHLCVBarSensor',
    'NewFeatureBarSensor',
    'L1FeaturesSensor',
    'L1FeaturesAvailableSensor',
]
