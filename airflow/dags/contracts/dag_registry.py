"""
DAG Registry - Single Source of Truth for all DAG IDs
======================================================

This module defines all DAG IDs used in the USDCOP Trading System.
All DAGs must import their IDs from this file to ensure consistency.

IMPORTANT: This registry implements clear pipeline separation:
    - CORE: Shared infrastructure (data ingestion)
    - RL: Reinforcement Learning pipeline (5-minute data)
    - FORECAST: ML Forecasting pipeline (DAILY data)

Naming Convention:
    {pipeline}_{layer}_{sublayer}_{purpose}

    Pipelines:
        core_    → Shared infrastructure
        rl_      → Reinforcement Learning pipeline
        forecast_→ ML Forecasting pipeline

Layers (Data Lineage Order):
    L0 - Data Acquisition: Raw data ingestion (OHLCV, Macro)
    L1 - Feature Engineering: Calculate features from raw data
    L2 - Dataset Preparation: Create training datasets
    L3 - Model Training: Train models
    L4 - Validation/Backtest: Validate trained models
    L5 - Inference: Production inference
    L6 - Monitoring: Production monitoring and alerting

Author: Trading Team
Version: 2.0.0
Date: 2026-01-22
Contract: CTR-DAG-002
"""

from enum import Enum
from typing import Dict, List


class DagPipeline(str, Enum):
    """Pipeline categories for DAG organization."""
    CORE = "core"            # Shared infrastructure
    RL = "rl"                # Reinforcement Learning
    FORECAST = "forecast"    # ML Forecasting


class DagLayer(str, Enum):
    """DAG execution layers in data lineage order."""
    L0_DATA_ACQUISITION = "L0"
    L1_FEATURE_ENGINEERING = "L1"
    L2_DATASET_PREPARATION = "L2"
    L3_MODEL_TRAINING = "L3"
    L4_VALIDATION = "L4"
    L5_INFERENCE = "L5"
    L6_MONITORING = "L6"


class DagType(str, Enum):
    """Task type categories."""
    DATA = "data"
    FEATURE = "feature"
    TRAINING = "training"
    INFERENCE = "inference"
    BACKTEST = "backtest"
    ALERT = "alert"
    REPORT = "report"


# =============================================================================
# CORE INFRASTRUCTURE DAGS (Shared by RL and Forecasting)
# =============================================================================
# These DAGs handle data ingestion that serves BOTH pipelines

# L0 - Data Acquisition (5-minute OHLCV - used by RL)
CORE_L0_INIT_RESTORE = "core_l0_01_init_restore"
CORE_L0_OHLCV_REALTIME = "core_l0_02_ohlcv_realtime"
CORE_L0_OHLCV_BACKFILL = "core_l0_03_ohlcv_backfill"
CORE_L0_MACRO_DAILY = "core_l0_04_macro_daily"
CORE_L0_WEEKLY_BACKUP = "core_l0_05_weekly_backup"


# =============================================================================
# RL PIPELINE DAGS (Reinforcement Learning - 5-minute data)
# =============================================================================
# Data lineage: OHLCV_5min → inference_features_5m → PPO model → signals

# L1 - Feature Engineering (from 5-min OHLCV to inference_features_5m)
RL_L1_FEATURE_REFRESH = "rl_l1_01_feature_refresh"
RL_L1_FEAST_MATERIALIZE = "rl_l1_02_feast_materialize"

# L2 - Dataset Preparation (create RL training datasets)
RL_L2_DATASET_BUILD = "rl_l2_01_dataset_build"
RL_L2_DRIFT_RETRAIN = "rl_l2_02_drift_retrain"

# L3 - Model Training (PPO training)
RL_L3_MODEL_TRAINING = "rl_l3_01_model_training"

# L4 - Validation (RL model validation)
RL_L4_EXPERIMENT_RUNNER = "rl_l4_01_experiment_runner"
RL_L4_BACKTEST_VALIDATION = "rl_l4_02_backtest_validation"
RL_L4_SCHEDULED_RETRAINING = "rl_l4_03_scheduled_retraining"

# L5 - Inference (RL production inference every 5 minutes)
RL_L5_PRODUCTION_INFERENCE = "rl_l5_01_production_inference"

# L6 - Monitoring (RL-specific monitoring)
RL_L6_PRODUCTION_MONITOR = "rl_l6_01_production_monitor"
RL_L6_DRIFT_MONITOR = "rl_l6_02_drift_monitor"


# =============================================================================
# FORECASTING PIPELINE DAGS (ML Forecasting - DAILY data)
# =============================================================================
# Data lineage: USDCOP_daily → forecast_features → 9 models × 7 horizons → predictions

# L0 - Data Acquisition (DAILY USDCOP price from TwelveData/Investing)
FORECAST_L0_DAILY_DATA = "forecast_l0_01_daily_data"

# L1 - Feature Engineering (daily features for forecasting)
FORECAST_L1_DAILY_FEATURES = "forecast_l1_01_daily_features"

# L2 - Dataset Preparation (forecasting training datasets)
FORECAST_L2_DATASET_BUILD = "forecast_l2_01_dataset_build"

# L3 - Model Training (9 models × 7 horizons = 63 combinations)
FORECAST_L3_MODEL_TRAINING = "forecast_l3_01_model_training"

# L4 - Validation (walk-forward backtest and A/B experiments)
FORECAST_L4_BACKTEST_VALIDATION = "forecast_l4_01_backtest_validation"
FORECAST_L4_EXPERIMENT_RUNNER = "forecast_l4_02_experiment_runner"

# L5 - Inference (weekly predictions)
FORECAST_L5_WEEKLY_INFERENCE = "forecast_l5_01_weekly_inference"

# L6 - Monitoring (forecasting-specific monitoring)
FORECAST_L6_DRIFT_MONITOR = "forecast_l6_01_drift_monitor"
FORECAST_L6_ACCURACY_REPORT = "forecast_l6_02_accuracy_report"


# =============================================================================
# SHARED MONITORING DAGS
# =============================================================================
CORE_L6_ALERT_MONITOR = "core_l6_01_alert_monitor"
CORE_L6_WEEKLY_REPORT = "core_l6_02_weekly_report"


# =============================================================================
# LEGACY IDs (for backward compatibility during migration)
# =============================================================================

LEGACY_DAG_MAPPING: Dict[str, str] = {
    # Old ID → New ID
    # Core Infrastructure
    "v3.l0_data_initialization": CORE_L0_INIT_RESTORE,
    "l0_data_initialization": CORE_L0_INIT_RESTORE,
    "v3.l0_ohlcv_realtime": CORE_L0_OHLCV_REALTIME,
    "l0_ohlcv_realtime": CORE_L0_OHLCV_REALTIME,
    "v3.l0_ohlcv_backfill": CORE_L0_OHLCV_BACKFILL,
    "l0_ohlcv_backfill": CORE_L0_OHLCV_BACKFILL,
    "v3.l0_macro_unified": CORE_L0_MACRO_DAILY,
    "l0_macro_unified": CORE_L0_MACRO_DAILY,
    "v3.l0_weekly_backup": CORE_L0_WEEKLY_BACKUP,
    "l0_weekly_backup": CORE_L0_WEEKLY_BACKUP,

    # RL Pipeline
    "v3.l1_feature_refresh": RL_L1_FEATURE_REFRESH,
    "l1_feature_refresh": RL_L1_FEATURE_REFRESH,
    "v3.l1b_feast_materialize": RL_L1_FEAST_MATERIALIZE,
    "l1b_feast_materialize": RL_L1_FEAST_MATERIALIZE,
    "v3.l2_preprocessing_pipeline": RL_L2_DATASET_BUILD,
    "l2_preprocessing_pipeline": RL_L2_DATASET_BUILD,
    "l2b_drift_retraining": RL_L2_DRIFT_RETRAIN,
    "l3_model_training": RL_L3_MODEL_TRAINING,
    "l4_experiment_runner": RL_L4_EXPERIMENT_RUNNER,
    "l4_scheduled_retraining": RL_L4_SCHEDULED_RETRAINING,
    "v3.l4_backtest_validation": RL_L4_BACKTEST_VALIDATION,
    "l4_backtest_validation": RL_L4_BACKTEST_VALIDATION,
    "v3.l5_multi_model_inference": RL_L5_PRODUCTION_INFERENCE,
    "l5_multi_model_inference": RL_L5_PRODUCTION_INFERENCE,
    "l6_production_monitoring": RL_L6_PRODUCTION_MONITOR,
    "mlops_drift_monitor": RL_L6_DRIFT_MONITOR,

    # Forecasting Pipeline
    "l3b_forecasting_training": FORECAST_L3_MODEL_TRAINING,
    "l3_02_forecasting_training": FORECAST_L3_MODEL_TRAINING,
    "l5b_forecasting_inference": FORECAST_L5_WEEKLY_INFERENCE,
    "l5_02_forecasting_inference": FORECAST_L5_WEEKLY_INFERENCE,

    # Shared Monitoring
    "v3.alert_monitor": CORE_L6_ALERT_MONITOR,
    "alert_monitor": CORE_L6_ALERT_MONITOR,
    "l6_weekly_performance_report": CORE_L6_WEEKLY_REPORT,
}


# =============================================================================
# DAG DEPENDENCIES GRAPH
# =============================================================================

DAG_DEPENDENCIES: Dict[str, List[str]] = {
    # Core Infrastructure
    CORE_L0_INIT_RESTORE: [CORE_L0_OHLCV_BACKFILL],
    CORE_L0_OHLCV_REALTIME: [RL_L1_FEATURE_REFRESH],
    CORE_L0_MACRO_DAILY: [],

    # RL Pipeline
    RL_L1_FEATURE_REFRESH: [RL_L1_FEAST_MATERIALIZE, RL_L5_PRODUCTION_INFERENCE],
    RL_L2_DATASET_BUILD: [RL_L3_MODEL_TRAINING],
    RL_L3_MODEL_TRAINING: [RL_L4_BACKTEST_VALIDATION],
    RL_L6_DRIFT_MONITOR: [RL_L2_DRIFT_RETRAIN],

    # Forecasting Pipeline
    FORECAST_L0_DAILY_DATA: [FORECAST_L1_DAILY_FEATURES],
    FORECAST_L1_DAILY_FEATURES: [FORECAST_L2_DATASET_BUILD],
    FORECAST_L2_DATASET_BUILD: [FORECAST_L3_MODEL_TRAINING],
    FORECAST_L3_MODEL_TRAINING: [FORECAST_L4_BACKTEST_VALIDATION],
    FORECAST_L6_DRIFT_MONITOR: [],
}


# =============================================================================
# DAG TAGS - SSOT for categorization
# =============================================================================

DAG_TAGS: Dict[str, List[str]] = {
    # Core Infrastructure (shared)
    CORE_L0_INIT_RESTORE: ["core", "l0", "data", "init", "ohlcv"],
    CORE_L0_OHLCV_REALTIME: ["core", "l0", "data", "ohlcv", "realtime"],
    CORE_L0_OHLCV_BACKFILL: ["core", "l0", "data", "ohlcv", "backfill"],
    CORE_L0_MACRO_DAILY: ["core", "l0", "data", "macro", "daily"],
    CORE_L0_WEEKLY_BACKUP: ["core", "l0", "data", "backup"],

    # RL Pipeline
    RL_L1_FEATURE_REFRESH: ["rl", "l1", "feature", "5min"],
    RL_L1_FEAST_MATERIALIZE: ["rl", "l1", "feature", "feast"],
    RL_L2_DATASET_BUILD: ["rl", "l2", "dataset", "training"],
    RL_L2_DRIFT_RETRAIN: ["rl", "l2", "dataset", "drift"],
    RL_L3_MODEL_TRAINING: ["rl", "l3", "training", "ppo"],
    RL_L4_EXPERIMENT_RUNNER: ["rl", "l4", "experiment", "backtest"],
    RL_L4_BACKTEST_VALIDATION: ["rl", "l4", "backtest", "validation"],
    RL_L4_SCHEDULED_RETRAINING: ["rl", "l4", "training", "scheduled"],
    RL_L5_PRODUCTION_INFERENCE: ["rl", "l5", "inference", "signal", "5min"],
    RL_L6_PRODUCTION_MONITOR: ["rl", "l6", "monitoring", "production"],
    RL_L6_DRIFT_MONITOR: ["rl", "l6", "monitoring", "drift"],

    # Forecasting Pipeline
    FORECAST_L0_DAILY_DATA: ["forecast", "l0", "data", "daily", "usdcop"],
    FORECAST_L1_DAILY_FEATURES: ["forecast", "l1", "feature", "daily"],
    FORECAST_L2_DATASET_BUILD: ["forecast", "l2", "dataset", "daily"],
    FORECAST_L3_MODEL_TRAINING: ["forecast", "l3", "training", "ml", "monthly"],
    FORECAST_L4_BACKTEST_VALIDATION: ["forecast", "l4", "backtest", "walkforward"],
    FORECAST_L4_EXPERIMENT_RUNNER: ["forecast", "l4", "experiment", "ab_testing"],
    FORECAST_L5_WEEKLY_INFERENCE: ["forecast", "l5", "inference", "weekly"],
    FORECAST_L6_DRIFT_MONITOR: ["forecast", "l6", "monitoring", "drift"],
    FORECAST_L6_ACCURACY_REPORT: ["forecast", "l6", "report", "accuracy"],

    # Shared Monitoring
    CORE_L6_ALERT_MONITOR: ["core", "l6", "alert", "system"],
    CORE_L6_WEEKLY_REPORT: ["core", "l6", "report", "weekly"],
}


# =============================================================================
# DATA GRANULARITY CONTRACT
# =============================================================================
# Critical: RL and Forecasting use DIFFERENT data granularity

DATA_GRANULARITY: Dict[str, Dict[str, str]] = {
    "rl": {
        "ohlcv_table": "usdcop_m5_ohlcv",
        "features_view": "inference_features_5m",
        "granularity": "5min",
        "inference_frequency": "5min",
    },
    "forecast": {
        "ohlcv_table": "bi.dim_daily_usdcop",
        "features_view": "bi.v_forecasting_features",
        "granularity": "daily",
        "inference_frequency": "weekly",
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_pipeline(dag_id: str) -> DagPipeline:
    """Get the pipeline for a DAG ID."""
    if dag_id.startswith("core_"):
        return DagPipeline.CORE
    elif dag_id.startswith("rl_"):
        return DagPipeline.RL
    elif dag_id.startswith("forecast_"):
        return DagPipeline.FORECAST
    # Legacy fallback
    return DagPipeline.CORE


def get_layer(dag_id: str) -> DagLayer:
    """Get the layer for a DAG ID."""
    # Extract layer from new format: {pipeline}_l{N}_...
    parts = dag_id.split("_")
    for part in parts:
        if part.startswith("l") and len(part) == 2 and part[1].isdigit():
            layer_mapping = {
                "l0": DagLayer.L0_DATA_ACQUISITION,
                "l1": DagLayer.L1_FEATURE_ENGINEERING,
                "l2": DagLayer.L2_DATASET_PREPARATION,
                "l3": DagLayer.L3_MODEL_TRAINING,
                "l4": DagLayer.L4_VALIDATION,
                "l5": DagLayer.L5_INFERENCE,
                "l6": DagLayer.L6_MONITORING,
            }
            return layer_mapping.get(part, DagLayer.L6_MONITORING)
    return DagLayer.L6_MONITORING


def get_all_dag_ids() -> List[str]:
    """Get all registered DAG IDs in execution order."""
    return [
        # Core L0
        CORE_L0_INIT_RESTORE,
        CORE_L0_OHLCV_REALTIME,
        CORE_L0_OHLCV_BACKFILL,
        CORE_L0_MACRO_DAILY,
        CORE_L0_WEEKLY_BACKUP,
        # RL Pipeline
        RL_L1_FEATURE_REFRESH,
        RL_L1_FEAST_MATERIALIZE,
        RL_L2_DATASET_BUILD,
        RL_L2_DRIFT_RETRAIN,
        RL_L3_MODEL_TRAINING,
        RL_L4_EXPERIMENT_RUNNER,
        RL_L4_BACKTEST_VALIDATION,
        RL_L4_SCHEDULED_RETRAINING,
        RL_L5_PRODUCTION_INFERENCE,
        RL_L6_PRODUCTION_MONITOR,
        RL_L6_DRIFT_MONITOR,
        # Forecasting Pipeline
        FORECAST_L0_DAILY_DATA,
        FORECAST_L1_DAILY_FEATURES,
        FORECAST_L2_DATASET_BUILD,
        FORECAST_L3_MODEL_TRAINING,
        FORECAST_L4_BACKTEST_VALIDATION,
        FORECAST_L4_EXPERIMENT_RUNNER,
        FORECAST_L5_WEEKLY_INFERENCE,
        FORECAST_L6_DRIFT_MONITOR,
        FORECAST_L6_ACCURACY_REPORT,
        # Shared Monitoring
        CORE_L6_ALERT_MONITOR,
        CORE_L6_WEEKLY_REPORT,
    ]


def get_dags_by_pipeline(pipeline: DagPipeline) -> List[str]:
    """Get all DAG IDs for a specific pipeline."""
    return [
        dag_id for dag_id, tags in DAG_TAGS.items()
        if pipeline.value in tags
    ]


def get_dag_tags(dag_id: str) -> List[str]:
    """Get tags for a DAG ID from SSOT."""
    return DAG_TAGS.get(dag_id, ["untagged"])


def migrate_legacy_id(old_id: str) -> str:
    """Convert legacy DAG ID to new naming convention."""
    return LEGACY_DAG_MAPPING.get(old_id, old_id)


def get_data_granularity(pipeline: str) -> Dict[str, str]:
    """Get data granularity config for a pipeline."""
    return DATA_GRANULARITY.get(pipeline, DATA_GRANULARITY["rl"])


__all__ = [
    # Enums
    "DagPipeline",
    "DagLayer",
    "DagType",
    # Core IDs
    "CORE_L0_INIT_RESTORE",
    "CORE_L0_OHLCV_REALTIME",
    "CORE_L0_OHLCV_BACKFILL",
    "CORE_L0_MACRO_DAILY",
    "CORE_L0_WEEKLY_BACKUP",
    "CORE_L6_ALERT_MONITOR",
    "CORE_L6_WEEKLY_REPORT",
    # RL Pipeline IDs
    "RL_L1_FEATURE_REFRESH",
    "RL_L1_FEAST_MATERIALIZE",
    "RL_L2_DATASET_BUILD",
    "RL_L2_DRIFT_RETRAIN",
    "RL_L3_MODEL_TRAINING",
    "RL_L4_EXPERIMENT_RUNNER",
    "RL_L4_BACKTEST_VALIDATION",
    "RL_L4_SCHEDULED_RETRAINING",
    "RL_L5_PRODUCTION_INFERENCE",
    "RL_L6_PRODUCTION_MONITOR",
    "RL_L6_DRIFT_MONITOR",
    # Forecasting Pipeline IDs
    "FORECAST_L0_DAILY_DATA",
    "FORECAST_L1_DAILY_FEATURES",
    "FORECAST_L2_DATASET_BUILD",
    "FORECAST_L3_MODEL_TRAINING",
    "FORECAST_L4_BACKTEST_VALIDATION",
    "FORECAST_L4_EXPERIMENT_RUNNER",
    "FORECAST_L5_WEEKLY_INFERENCE",
    "FORECAST_L6_DRIFT_MONITOR",
    "FORECAST_L6_ACCURACY_REPORT",
    # Tags & Config
    "DAG_TAGS",
    "DATA_GRANULARITY",
    # Legacy
    "LEGACY_DAG_MAPPING",
    "DAG_DEPENDENCIES",
    # Helpers
    "get_pipeline",
    "get_layer",
    "get_all_dag_ids",
    "get_dags_by_pipeline",
    "get_dag_tags",
    "migrate_legacy_id",
    "get_data_granularity",
]
