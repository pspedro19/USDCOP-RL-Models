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

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Created: 2025-01-12
"""

from sensors.new_bar_sensor import (
    NewOHLCVBarSensor,
    NewFeatureBarSensor,
)

__all__ = [
    'NewOHLCVBarSensor',
    'NewFeatureBarSensor',
]
