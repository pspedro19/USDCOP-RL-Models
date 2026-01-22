# data-engineering/dags/sensors/__init__.py
"""
Custom Airflow Sensors.

This module provides custom sensors for data pipeline orchestration:

- DataFreshnessSensor: Validates data freshness before processing
"""

from .data_freshness_sensor import DataFreshnessSensor

__all__ = ["DataFreshnessSensor"]
