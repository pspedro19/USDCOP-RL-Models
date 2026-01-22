"""
DAG Registry - Single Source of Truth for all DAG IDs
======================================================

This module defines all DAG IDs used in the USDCOP Trading System.
All DAGs must import their IDs from this file to ensure consistency.

Naming Convention:
    {layer}_{sublayer}_{purpose}

Layers (Data Lineage Order):
    L0 - Data Acquisition: Raw data ingestion (OHLCV, Macro)
    L1 - Feature Engineering: Calculate features from raw data
    L2 - Dataset Preparation: Create training datasets
    L3 - Model Training: Train RL models
    L4 - Validation/Backtest: Validate trained models
    L5 - Inference: Production inference
    L6 - Monitoring: Production monitoring and alerting

Author: Trading Team
Version: 1.0.0
Date: 2026-01-19
Contract: CTR-DAG-001
"""

from enum import Enum
from typing import Dict, List


class DagLayer(str, Enum):
    """DAG execution layers in data lineage order."""
    L0_DATA_ACQUISITION = "L0"
    L1_FEATURE_ENGINEERING = "L1"
    L2_DATASET_PREPARATION = "L2"
    L3_MODEL_TRAINING = "L3"
    L4_VALIDATION = "L4"
    L5_INFERENCE = "L5"
    L6_MONITORING = "L6"


# =============================================================================
# L0 - DATA ACQUISITION DAGS
# =============================================================================
# Order: init -> realtime -> backfill -> macro -> backup

L0_INIT_RESTORE = "l0_01_init_restore"
L0_OHLCV_REALTIME = "l0_02_ohlcv_realtime"
L0_OHLCV_BACKFILL = "l0_03_ohlcv_backfill"
L0_MACRO_DAILY = "l0_04_macro_daily"
L0_WEEKLY_BACKUP = "l0_05_weekly_backup"

# =============================================================================
# L1 - FEATURE ENGINEERING DAGS
# =============================================================================
# Order: feature_refresh -> feast_materialize

L1_FEATURE_REFRESH = "l1_01_feature_refresh"
L1_FEAST_MATERIALIZE = "l1_02_feast_materialize"

# =============================================================================
# L2 - DATASET PREPARATION DAGS
# =============================================================================
# Order: dataset_build -> drift_retrain (triggered by drift)

L2_DATASET_BUILD = "l2_01_dataset_build"
L2_DRIFT_RETRAIN = "l2_02_drift_retrain"

# =============================================================================
# L3 - MODEL TRAINING DAGS
# =============================================================================

L3_MODEL_TRAINING = "l3_01_model_training"

# =============================================================================
# L4 - VALIDATION DAGS
# =============================================================================
# Order: experiment_runner -> backtest_validation

L4_EXPERIMENT_RUNNER = "l4_01_experiment_runner"
L4_BACKTEST_VALIDATION = "l4_02_backtest_validation"
L4_SCHEDULED_RETRAINING = "l4_03_scheduled_retraining"

# =============================================================================
# L5 - INFERENCE DAGS
# =============================================================================

L5_PRODUCTION_INFERENCE = "l5_01_production_inference"

# =============================================================================
# L6 - MONITORING DAGS
# =============================================================================
# Order: production_monitor -> drift_monitor -> alert_monitor -> weekly_report

L6_PRODUCTION_MONITOR = "l6_01_production_monitor"
L6_DRIFT_MONITOR = "l6_02_drift_monitor"
L6_ALERT_MONITOR = "l6_03_alert_monitor"
L6_WEEKLY_REPORT = "l6_04_weekly_report"


# =============================================================================
# LEGACY IDs (for backward compatibility during migration)
# =============================================================================
# These are the old DAG IDs that should be migrated to new naming

LEGACY_DAG_MAPPING: Dict[str, str] = {
    # Old ID -> New ID
    "v3.l0_data_initialization": L0_INIT_RESTORE,
    "v3.l0_ohlcv_realtime": L0_OHLCV_REALTIME,
    "v3.l0_ohlcv_backfill": L0_OHLCV_BACKFILL,
    "v3.l0_macro_unified": L0_MACRO_DAILY,
    "v3.l0_weekly_backup": L0_WEEKLY_BACKUP,
    "v3.l1_feature_refresh": L1_FEATURE_REFRESH,
    "v3.l1b_feast_materialize": L1_FEAST_MATERIALIZE,
    "v3.l2_preprocessing_pipeline": L2_DATASET_BUILD,
    "l2b_drift_retraining": L2_DRIFT_RETRAIN,
    "l3_model_training": L3_MODEL_TRAINING,
    "l4_experiment_runner": L4_EXPERIMENT_RUNNER,
    "l4_scheduled_retraining": L4_SCHEDULED_RETRAINING,
    "v3.l4_backtest_validation": L4_BACKTEST_VALIDATION,
    "v3.l5_multi_model_inference": L5_PRODUCTION_INFERENCE,
    "l6_production_monitoring": L6_PRODUCTION_MONITOR,
    "l6_weekly_performance_report": L6_WEEKLY_REPORT,
    "mlops_drift_monitor": L6_DRIFT_MONITOR,
    "v3.alert_monitor": L6_ALERT_MONITOR,
}


# =============================================================================
# DAG DEPENDENCIES GRAPH
# =============================================================================
# Defines which DAGs trigger or depend on other DAGs

DAG_DEPENDENCIES: Dict[str, List[str]] = {
    L0_INIT_RESTORE: [L0_OHLCV_BACKFILL],  # init triggers backfill if needed
    L0_OHLCV_REALTIME: [L1_FEATURE_REFRESH],  # new bar triggers feature refresh
    L0_MACRO_DAILY: [],  # standalone
    L1_FEATURE_REFRESH: [L1_FEAST_MATERIALIZE, L5_PRODUCTION_INFERENCE],
    L2_DATASET_BUILD: [L3_MODEL_TRAINING],  # dataset triggers training
    L3_MODEL_TRAINING: [L4_BACKTEST_VALIDATION],  # model triggers backtest
    L6_DRIFT_MONITOR: [L2_DRIFT_RETRAIN],  # drift triggers retraining
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_layer(dag_id: str) -> DagLayer:
    """Get the layer for a DAG ID."""
    prefix = dag_id.split("_")[0]
    mapping = {
        "l0": DagLayer.L0_DATA_ACQUISITION,
        "l1": DagLayer.L1_FEATURE_ENGINEERING,
        "l2": DagLayer.L2_DATASET_PREPARATION,
        "l3": DagLayer.L3_MODEL_TRAINING,
        "l4": DagLayer.L4_VALIDATION,
        "l5": DagLayer.L5_INFERENCE,
        "l6": DagLayer.L6_MONITORING,
    }
    return mapping.get(prefix, DagLayer.L6_MONITORING)


def get_all_dag_ids() -> List[str]:
    """Get all registered DAG IDs in execution order."""
    return [
        # L0
        L0_INIT_RESTORE,
        L0_OHLCV_REALTIME,
        L0_OHLCV_BACKFILL,
        L0_MACRO_DAILY,
        L0_WEEKLY_BACKUP,
        # L1
        L1_FEATURE_REFRESH,
        L1_FEAST_MATERIALIZE,
        # L2
        L2_DATASET_BUILD,
        L2_DRIFT_RETRAIN,
        # L3
        L3_MODEL_TRAINING,
        # L4
        L4_EXPERIMENT_RUNNER,
        L4_BACKTEST_VALIDATION,
        L4_SCHEDULED_RETRAINING,
        # L5
        L5_PRODUCTION_INFERENCE,
        # L6
        L6_PRODUCTION_MONITOR,
        L6_DRIFT_MONITOR,
        L6_ALERT_MONITOR,
        L6_WEEKLY_REPORT,
    ]


def migrate_legacy_id(old_id: str) -> str:
    """Convert legacy DAG ID to new naming convention."""
    return LEGACY_DAG_MAPPING.get(old_id, old_id)


__all__ = [
    # Layer enum
    "DagLayer",
    # L0 IDs
    "L0_INIT_RESTORE",
    "L0_OHLCV_REALTIME",
    "L0_OHLCV_BACKFILL",
    "L0_MACRO_DAILY",
    "L0_WEEKLY_BACKUP",
    # L1 IDs
    "L1_FEATURE_REFRESH",
    "L1_FEAST_MATERIALIZE",
    # L2 IDs
    "L2_DATASET_BUILD",
    "L2_DRIFT_RETRAIN",
    # L3 IDs
    "L3_MODEL_TRAINING",
    # L4 IDs
    "L4_EXPERIMENT_RUNNER",
    "L4_BACKTEST_VALIDATION",
    "L4_SCHEDULED_RETRAINING",
    # L5 IDs
    "L5_PRODUCTION_INFERENCE",
    # L6 IDs
    "L6_PRODUCTION_MONITOR",
    "L6_DRIFT_MONITOR",
    "L6_ALERT_MONITOR",
    "L6_WEEKLY_REPORT",
    # Legacy
    "LEGACY_DAG_MAPPING",
    # Helpers
    "get_layer",
    "get_all_dag_ids",
    "migrate_legacy_id",
]
