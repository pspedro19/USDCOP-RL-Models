"""
Services Package
Contract: CTR-SVC

Business logic services following Single Responsibility Principle.

Services:
- BacktestFeatureBuilder: Builds features for historical backtest periods

NOTE (audit A2-02): the former L1NRTDataService / L5NRTInferenceService were
DEAD — superseded by the Airflow DAG path (see `airflow/dags/l1_model_promotion.py`,
which replaced `L1NRTDataService.on_model_approved()`). They were removed 2026-07.
"""

from src.services.backtest_feature_builder import (
    BacktestFeatureBuilder,
    FeatureBuildConfig,
)

__all__ = [
    "BacktestFeatureBuilder",
    "FeatureBuildConfig",
]
