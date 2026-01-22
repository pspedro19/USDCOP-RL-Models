"""
Model Monitoring Module
=======================

Provides tools for detecting model drift, degradation, and anomalous behavior
in RL trading models.

Main classes:
- ModelMonitor: Detect action drift and performance degradation
- FeatureDriftDetector: Detect feature distribution drift using KS test
- MultivariateDriftDetector: Detect multivariate drift using MMD, Wasserstein, PCA
- DriftResult: Result dataclass for drift detection
- DriftReport: Complete drift report dataclass

MLOps-4: Feature Drift Detection
P1: Multivariate Drift Detection
"""

from .model_monitor import ModelMonitor, create_model_monitor
from .drift_detector import (
    FeatureDriftDetector,
    DriftResult,
    DriftReport,
    FeatureStats,
    ReferenceStatsManager,
    create_drift_detector,
    compute_reference_stats_from_dataframe,
)

# P1: Multivariate drift detection
from .multivariate_drift import (
    MultivariateDriftDetector,
    MultivariateDriftResult,
    MultivariateDriftReport,
    create_multivariate_detector,
    quick_multivariate_check,
)

# P1: Data Readiness Score
from .readiness_score import (
    ReadinessLevel,
    FeatureReadiness,
    PipelineReadiness,
    DailyDataReadinessReport,
    DataReadinessScorer,
    compute_readiness_score,
    get_readiness_metrics,
    update_prometheus_metrics,
)

__all__ = [
    # Action/Model drift monitoring
    "ModelMonitor",
    "create_model_monitor",
    # Feature drift detection (univariate)
    "FeatureDriftDetector",
    "DriftResult",
    "DriftReport",
    "FeatureStats",
    "ReferenceStatsManager",
    "create_drift_detector",
    "compute_reference_stats_from_dataframe",
    # Multivariate drift detection (P1)
    "MultivariateDriftDetector",
    "MultivariateDriftResult",
    "MultivariateDriftReport",
    "create_multivariate_detector",
    "quick_multivariate_check",
    # Readiness Score (P1)
    "ReadinessLevel",
    "FeatureReadiness",
    "PipelineReadiness",
    "DailyDataReadinessReport",
    "DataReadinessScorer",
    "compute_readiness_score",
    "get_readiness_metrics",
    "update_prometheus_metrics",
]
