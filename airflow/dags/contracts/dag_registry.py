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
    L7 - Execution: Smart order execution (trailing stops, broker)

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
    L7_EXECUTION = "L7"


class DagType(str, Enum):
    """Task type categories."""
    DATA = "data"
    FEATURE = "feature"
    TRAINING = "training"
    INFERENCE = "inference"
    BACKTEST = "backtest"
    ALERT = "alert"
    REPORT = "report"
    EXECUTION = "execution"


# =============================================================================
# CORE INFRASTRUCTURE DAGS (Shared by RL and Forecasting)
# =============================================================================
# These DAGs handle data ingestion that serves BOTH pipelines

# L0 - Data Acquisition (OHLCV + Macro)
CORE_L0_OHLCV_BACKFILL = "core_l0_01_ohlcv_backfill"
CORE_L0_OHLCV_REALTIME = "core_l0_02_ohlcv_realtime"
CORE_L0_MACRO_BACKFILL = "core_l0_03_macro_backfill"
CORE_L0_MACRO_UPDATE = "core_l0_04_macro_update"
CORE_L0_SEED_BACKUP = "core_l0_05_seed_backup"
CORE_L0_USDCOP_FORWARD_MACRO_PIT = "core_l0_04_usdcop_forward_macro_pit"


# =============================================================================
# RL PIPELINE DAGS (Reinforcement Learning - 5-minute data)
# =============================================================================
# Data lineage: OHLCV_5min → inference_features_5m → PPO model → signals

# L1 - Feature Engineering (from 5-min OHLCV to inference_features_5m + inference_ready_nrt)
RL_L1_FEATURE_REFRESH = "rl_l1_01_feature_refresh"
RL_L1_MODEL_PROMOTION = "rl_l1_03_model_promotion"  # Populate inference_ready_nrt on model approval

# L2 - Dataset Preparation (create RL training datasets)
RL_L2_DATASET_BUILD = "rl_l2_01_dataset_build"
RL_L2_DRIFT_RETRAIN = "rl_l2_02_drift_retrain"

# L3 - Model Training (PPO training)
RL_L3_MODEL_TRAINING = "rl_l3_01_model_training"

# L4 - Validation (RL model validation)
RL_L4_EXPERIMENT_RUNNER = "rl_l4_01_experiment_runner"  # DEPRECATED: Use RL_L4_BACKTEST_PROMOTION
RL_L4_BACKTEST_VALIDATION = "rl_l4_02_backtest_validation"  # DEPRECATED: Merged into RL_L4_BACKTEST_PROMOTION
RL_L4_SCHEDULED_RETRAINING = "rl_l4_03_scheduled_retraining"  # DEPRECATED: fold into RL_L4_BACKTEST_PROMOTION

# DEPRECATED DAG ids — kept as constants for back-references but excluded from
# get_all_dag_ids() so nothing schedules/treats them as active (audit A2-01/07/09).
DEPRECATED_DAGS = frozenset({
    RL_L4_EXPERIMENT_RUNNER,
    RL_L4_BACKTEST_VALIDATION,
    RL_L4_SCHEDULED_RETRAINING,
})
RL_L4_BACKTEST_PROMOTION = "rl_l4_04_backtest_promotion"  # NEW: Unified backtest + promotion (primer voto)

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

# --- H1 Daily Pipeline (renamed from FORECAST_L5_*/L6_*/L7_* → correct layer assignments) ---
# L3 - Training (weekly retraining of 9 models, H=1)
FORECAST_H1_L3_WEEKLY_TRAINING = "forecast_h1_l3_weekly_training"
# L4 - Backtest Promotion (OOS backtest + dashboard export + approval gates)
FORECAST_H1_L4_BACKTEST_PROMOTION = "forecast_h1_l4_backtest_promotion"
# L5 - Inference (daily predictions + vol-targeting)
FORECAST_H1_L5_DAILY_INFERENCE = "forecast_h1_l5_daily_inference"
FORECAST_H1_L5_VOL_TARGETING = "forecast_h1_l5_vol_targeting"
# L6 - Monitoring (paper trading monitor)
FORECAST_H1_L6_PAPER_MONITOR = "forecast_h1_l6_paper_monitor"
# L7 - Execution (smart executor with trailing stops)
FORECAST_H1_L7_SMART_EXECUTOR = "forecast_h1_l7_smart_executor"

# --- H1 prospective shadow ledgers (research-only, never authorize signal or capital) ---
# These are the pre-registered FORWARD judges the quant constitution §1/§5 relies on: they
# append-only commit predictions and outcomes, and they are NOT part of the H1 production track.
# v2 supersedes v1 — see below and `.claude/specs/assets/usdcop/HYPOTHESIS-REGISTRY.md`.
FORECAST_H1_REGIME_SHADOW_V2 = "forecast_h1_regime_shadow_v2"
FORECAST_H1_DAILY_SHADOW_V1 = "forecast_h1_daily_shadow_v1"
# WITHDRAWN experiment. HYPOTHESIS-REGISTRY records "V1 se retiró con 0 predicciones y 0
# outcomes" (class_weight divergence, mutable history, unpinned hashes). Its module is still on
# disk and its generator writes the SAME ledger index that v2 now owns
# (`h1_regime_shadow_index.json`, whose own `supersedes` field names this experiment), so it is
# kept DEPRECATED here and fenced fail-closed in the module itself.
FORECAST_H1_REGIME_SHADOW_V1 = "forecast_h1_regime_shadow"
# DEPRECATED_DAGS is bound above with the RL-L4 ids, before this constant exists, so the
# withdrawn shadow is folded in here rather than duplicating the literal string up there.
DEPRECATED_DAGS = DEPRECATED_DAGS | frozenset({FORECAST_H1_REGIME_SHADOW_V1})

# --- H5 Weekly Pipeline (renamed from FORECAST_H5_L5A_*/L5B_*/L5C_* → correct layer assignments) ---
# L3 - Training (weekly retraining of Ridge + BR, H=5)
FORECAST_H5_L3_WEEKLY_TRAINING = "forecast_h5_l3_weekly_training"
# L4 - Backtest Promotion (OOS backtest + dashboard export + approval gates)
FORECAST_H5_L4_BACKTEST_PROMOTION = "forecast_h5_l4_backtest_promotion"
# L4b - Production Deploy (post-Vote-2: retrain full window + export production + seed DB;
#        triggered by the dashboard deploy API via Airflow REST, or manually)
FORECAST_H5_L4B_PRODUCTION_DEPLOY = "forecast_h5_l4b_production_deploy"
# L5 - Inference (weekly signal + vol-targeting)
FORECAST_H5_L5_WEEKLY_SIGNAL = "forecast_h5_l5_weekly_signal"
FORECAST_H5_L5_VOL_TARGETING = "forecast_h5_l5_vol_targeting"
# L6 - Monitoring (weekly paper trading monitor) — unchanged
FORECAST_H5_L6_WEEKLY_MONITOR = "forecast_h5_l6_weekly_monitor"
# L7 - Execution (multiday executor with TP/HS) — unchanged
FORECAST_H5_L7_MULTIDAY_EXECUTOR = "forecast_h5_l7_multiday_executor"

# Legacy forecasting constants (kept for backward compatibility)
FORECAST_L5_WEEKLY_INFERENCE = "forecast_l5_weekly_inference_legacy"
FORECAST_L5_VOL_TARGETING = FORECAST_H1_L5_VOL_TARGETING
FORECAST_L5_DAILY_INFERENCE = FORECAST_H1_L5_DAILY_INFERENCE
FORECAST_L6_DRIFT_MONITOR = "forecast_l6_01_drift_monitor"
FORECAST_L6_ACCURACY_REPORT = "forecast_l6_02_accuracy_report"
FORECAST_L6_PAPER_TRADING = FORECAST_H1_L6_PAPER_MONITOR
FORECAST_L7_SMART_EXECUTOR = FORECAST_H1_L7_SMART_EXECUTOR
FORECAST_H5_L5A_WEEKLY_TRAINING = FORECAST_H5_L3_WEEKLY_TRAINING
FORECAST_H5_L5B_WEEKLY_SIGNAL = FORECAST_H5_L5_WEEKLY_SIGNAL
FORECAST_H5_L5C_VOL_TARGETING = FORECAST_H5_L5_VOL_TARGETING


# =============================================================================
# NEWS ENGINE & ANALYSIS DAGS
# =============================================================================
NEWS_DAILY_PIPELINE = "news_daily_pipeline"
NEWS_ALERT_MONITOR = "news_alert_monitor"
NEWS_WEEKLY_DIGEST = "news_weekly_digest"
NEWS_MAINTENANCE = "news_maintenance"
ANALYSIS_L8_DAILY_GENERATION = "analysis_l8_daily_generation"


# =============================================================================
# SHARED MONITORING DAGS
# =============================================================================
CORE_L6_ALERT_MONITOR = "core_l6_01_alert_monitor"
CORE_L6_WEEKLY_REPORT = "core_l6_02_weekly_report"


# =============================================================================
# PLATFORM / OPS DAGS
# =============================================================================
# These DAGs shipped without registry constants, so `get_all_dag_ids()` under-
# reported the fleet and every doc that quoted it inherited the error
# (CTR-KNOWLEDGE-INVENTORY-001, reconciled 2026-07-20 against the AST inventory).
CORE_WATCHDOG = "core_watchdog"
RECONCILIATION_DAILY = "reconciliation_daily"
RBAC_ENTITLEMENTS_DAILY = "rbac_entitlements_daily"
FORECAST_WEEKLY_GENERATION = "forecast_weekly_generation"
FORECAST_ASSET_ANALYSIS_WEEKLY = "forecast_asset_analysis_weekly"
FORWARD_LEDGER_WEEKLY = "forward_ledger_weekly"  # Fri 19:00 UTC forward-evidence ledger (2026-07-21)
L0_MULTIFRAME_CATCHUP = "l0_multiframe_catchup"  # hourly M5/1h/4h/daily/monthly catch-up + matview refresh (2026-07-22)
CONTROL_SYSTEM_HEALTH = "control_system_health"  # BL-25 tres relojes (datos/modelo/PnL), :30 13-18 UTC Mon-Fri (2026-07-27)

# Config-driven, emitted per enabled asset by airflow/dags/asset_pipeline_factory.py
# from config/assets/pipelines.yaml — never hardcode; derive with get_asset_dag_ids().
ASSET_PIPELINE_DAG_PATTERN = "asset_{asset}_pipeline_weekly"

# Declared historically but NO module on disk emits them. Kept as constants so any
# stale back-reference still resolves, yet excluded from the active fleet: counting
# them is how "40 DAGs" got into the specs while 45 real ones existed.
ABSENT_DAGS = frozenset({
    FORECAST_L0_DAILY_DATA,
    FORECAST_L1_DAILY_FEATURES,
    FORECAST_L2_DATASET_BUILD,
    FORECAST_L4_BACKTEST_VALIDATION,
    FORECAST_L4_EXPERIMENT_RUNNER,
    FORECAST_L6_DRIFT_MONITOR,
    FORECAST_L6_ACCURACY_REPORT,
})


# =============================================================================
# LEGACY IDs (for backward compatibility during migration)
# =============================================================================

LEGACY_DAG_MAPPING: Dict[str, str] = {
    # Old ID → New ID
    # Core Infrastructure (L0)
    "v3.l0_data_initialization": CORE_L0_OHLCV_BACKFILL,
    "l0_data_initialization": CORE_L0_OHLCV_BACKFILL,
    "v3.l0_ohlcv_realtime": CORE_L0_OHLCV_REALTIME,
    "l0_ohlcv_realtime": CORE_L0_OHLCV_REALTIME,
    "v3.l0_ohlcv_realtime_multi": CORE_L0_OHLCV_REALTIME,
    "v3.l0_ohlcv_backfill": CORE_L0_OHLCV_BACKFILL,
    "l0_ohlcv_backfill": CORE_L0_OHLCV_BACKFILL,
    "l0_03_ohlcv_backfill": CORE_L0_OHLCV_BACKFILL,
    "l0_ohlcv_historical_backfill": CORE_L0_OHLCV_BACKFILL,
    "v3.l0_macro_unified": CORE_L0_MACRO_UPDATE,
    "l0_macro_unified": CORE_L0_MACRO_UPDATE,
    "l0_macro_update": CORE_L0_MACRO_UPDATE,
    "l0_macro_backfill": CORE_L0_MACRO_BACKFILL,
    "v3.l0_weekly_backup": CORE_L0_OHLCV_BACKFILL,
    "l0_weekly_backup": CORE_L0_OHLCV_BACKFILL,
    "l0_backup_restore": CORE_L0_OHLCV_BACKFILL,
    "l0_restore_manual": CORE_L0_OHLCV_BACKFILL,

    # RL Pipeline
    "v3.l1_feature_refresh": RL_L1_FEATURE_REFRESH,
    "l1_feature_refresh": RL_L1_FEATURE_REFRESH,
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
    "l5b_forecasting_inference": FORECAST_H1_L3_WEEKLY_TRAINING,
    "l5_02_forecasting_inference": FORECAST_H1_L3_WEEKLY_TRAINING,

    # ML Forecasting rename (2026-02-17): L5a/L5b/L5c → L3/L5, correct layer assignments
    "forecast_l5_01_weekly_inference": FORECAST_H1_L3_WEEKLY_TRAINING,
    "forecast_l5_03_daily_inference": FORECAST_H1_L5_DAILY_INFERENCE,
    "forecast_l5_02_vol_targeting": FORECAST_H1_L5_VOL_TARGETING,
    "forecast_l7_01_smart_executor": FORECAST_H1_L7_SMART_EXECUTOR,
    "forecast_l6_03_paper_trading": FORECAST_H1_L6_PAPER_MONITOR,
    "forecast_h5_l5a_weekly_training": FORECAST_H5_L3_WEEKLY_TRAINING,
    "forecast_h5_l5b_weekly_signal": FORECAST_H5_L5_WEEKLY_SIGNAL,
    "forecast_h5_l5c_vol_targeting": FORECAST_H5_L5_VOL_TARGETING,

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
    CORE_L0_OHLCV_BACKFILL: [],
    CORE_L0_OHLCV_REALTIME: [RL_L1_FEATURE_REFRESH],
    CORE_L0_MACRO_BACKFILL: [],
    CORE_L0_MACRO_UPDATE: [],
    CORE_L0_SEED_BACKUP: [],  # Runs after OHLCV realtime + macro update complete
    CORE_L0_USDCOP_FORWARD_MACRO_PIT: [],

    # RL Pipeline
    RL_L1_FEATURE_REFRESH: [RL_L5_PRODUCTION_INFERENCE],
    RL_L1_MODEL_PROMOTION: [],  # Manual trigger or event-driven (model_approved)
    RL_L2_DATASET_BUILD: [RL_L3_MODEL_TRAINING],
    RL_L3_MODEL_TRAINING: [RL_L4_BACKTEST_PROMOTION],  # L3 triggers L4 backtest+promotion
    RL_L4_BACKTEST_PROMOTION: [],  # L4 saves proposal for Dashboard approval
    RL_L6_DRIFT_MONITOR: [RL_L2_DRIFT_RETRAIN],

    # Forecasting Pipeline (generic)
    FORECAST_L0_DAILY_DATA: [FORECAST_L1_DAILY_FEATURES],
    FORECAST_L1_DAILY_FEATURES: [FORECAST_L2_DATASET_BUILD],
    FORECAST_L2_DATASET_BUILD: [FORECAST_L3_MODEL_TRAINING],
    FORECAST_L3_MODEL_TRAINING: [FORECAST_L4_BACKTEST_VALIDATION],
    FORECAST_L6_DRIFT_MONITOR: [],

    # H1 Daily Pipeline
    FORECAST_H1_L3_WEEKLY_TRAINING: [FORECAST_H1_L5_DAILY_INFERENCE],  # L3 trains models for L5
    FORECAST_H1_L4_BACKTEST_PROMOTION: [],  # Manual trigger, exports to dashboard
    FORECAST_H1_L5_DAILY_INFERENCE: [FORECAST_H1_L5_VOL_TARGETING],  # L5 inference feeds vol-targeting
    FORECAST_H1_L5_VOL_TARGETING: [FORECAST_H1_L7_SMART_EXECUTOR, FORECAST_H1_L6_PAPER_MONITOR],
    FORECAST_H1_L7_SMART_EXECUTOR: [],
    FORECAST_H1_L6_PAPER_MONITOR: [],

    # H5 Weekly Pipeline
    FORECAST_H5_L3_WEEKLY_TRAINING: [FORECAST_H5_L5_WEEKLY_SIGNAL],  # L3 trains models for L5
    FORECAST_H5_L4_BACKTEST_PROMOTION: [],  # Manual trigger, exports to dashboard
    FORECAST_H5_L4B_PRODUCTION_DEPLOY: [],  # Triggered by dashboard deploy API (post-Vote-2)
    FORECAST_H5_L5_WEEKLY_SIGNAL: [FORECAST_H5_L5_VOL_TARGETING],
    FORECAST_H5_L5_VOL_TARGETING: [FORECAST_H5_L7_MULTIDAY_EXECUTOR],
    FORECAST_H5_L7_MULTIDAY_EXECUTOR: [],
    FORECAST_H5_L6_WEEKLY_MONITOR: [FORECAST_H5_L7_MULTIDAY_EXECUTOR],

    # News Engine & Analysis
    NEWS_DAILY_PIPELINE: [],  # Runs on schedule (3x daily)
    NEWS_ALERT_MONITOR: [],  # Runs on schedule (every 30 min)
    NEWS_WEEKLY_DIGEST: [NEWS_DAILY_PIPELINE],  # Depends on daily ingestion
    NEWS_MAINTENANCE: [],  # Runs on schedule (Sunday)
    ANALYSIS_L8_DAILY_GENERATION: [NEWS_DAILY_PIPELINE],  # Must run AFTER last news ingestion

    # Platform / Ops
    CONTROL_SYSTEM_HEALTH: [],  # BL-25: reads facts/snapshots; promotion DAGs read its snapshot
}


# =============================================================================
# DAG TAGS - SSOT for categorization
# =============================================================================

DAG_TAGS: Dict[str, List[str]] = {
    # Core Infrastructure (shared) — 6 L0 DAGs
    CORE_L0_OHLCV_BACKFILL: ["core", "l0", "data", "ohlcv", "backfill"],
    CORE_L0_OHLCV_REALTIME: ["core", "l0", "data", "ohlcv", "realtime"],
    CORE_L0_MACRO_BACKFILL: ["core", "l0", "data", "macro", "backfill"],
    CORE_L0_MACRO_UPDATE: ["core", "l0", "data", "macro", "update"],
    CORE_L0_SEED_BACKUP: ["core", "l0", "backup", "seed"],
    CORE_L0_USDCOP_FORWARD_MACRO_PIT: ["core", "l0", "data", "macro", "pit", "forward"],

    # RL Pipeline
    RL_L1_FEATURE_REFRESH: ["rl", "l1", "feature", "5min"],
    RL_L1_MODEL_PROMOTION: ["rl", "l1", "model-promotion", "inference-ready-nrt"],
    RL_L2_DATASET_BUILD: ["rl", "l2", "dataset", "training"],
    RL_L2_DRIFT_RETRAIN: ["rl", "l2", "dataset", "drift"],
    RL_L3_MODEL_TRAINING: ["rl", "l3", "training", "ppo"],
    RL_L4_EXPERIMENT_RUNNER: ["rl", "l4", "experiment", "backtest", "deprecated"],
    RL_L4_BACKTEST_VALIDATION: ["rl", "l4", "backtest", "validation", "deprecated"],
    RL_L4_SCHEDULED_RETRAINING: ["rl", "l4", "training", "scheduled"],
    RL_L4_BACKTEST_PROMOTION: ["rl", "l4", "backtest", "promotion", "two-vote"],
    RL_L5_PRODUCTION_INFERENCE: ["rl", "l5", "inference", "signal", "5min"],
    RL_L6_PRODUCTION_MONITOR: ["rl", "l6", "monitoring", "production"],
    RL_L6_DRIFT_MONITOR: ["rl", "l6", "monitoring", "drift"],

    # Forecasting Pipeline (generic)
    FORECAST_L0_DAILY_DATA: ["forecast", "l0", "data", "daily", "usdcop"],
    FORECAST_L1_DAILY_FEATURES: ["forecast", "l1", "feature", "daily"],
    FORECAST_L2_DATASET_BUILD: ["forecast", "l2", "dataset", "daily"],
    FORECAST_L3_MODEL_TRAINING: ["forecast", "l3", "training", "ml", "monthly"],
    FORECAST_L4_BACKTEST_VALIDATION: ["forecast", "l4", "backtest", "walkforward"],
    FORECAST_L4_EXPERIMENT_RUNNER: ["forecast", "l4", "experiment", "ab_testing"],
    FORECAST_L6_DRIFT_MONITOR: ["forecast", "l6", "monitoring", "drift"],
    FORECAST_L6_ACCURACY_REPORT: ["forecast", "l6", "report", "accuracy"],

    # H1 Daily Pipeline
    FORECAST_H1_L3_WEEKLY_TRAINING: ["forecast", "h1", "l3", "training", "weekly"],
    FORECAST_H1_L4_BACKTEST_PROMOTION: ["forecast", "h1", "l4", "backtest", "promotion", "two-vote"],
    FORECAST_H1_L5_DAILY_INFERENCE: ["forecast", "h1", "l5", "inference", "daily"],
    FORECAST_H1_L5_VOL_TARGETING: ["forecast", "h1", "l5", "vol-targeting", "daily"],
    FORECAST_H1_L6_PAPER_MONITOR: ["forecast", "h1", "l6", "paper-trading", "daily"],
    FORECAST_H1_L7_SMART_EXECUTOR: ["forecast", "h1", "l7", "execution", "trailing-stop"],
    FORECAST_H1_REGIME_SHADOW_V2: ["forecast", "h1", "research", "shadow", "usdcop", "pit", "forward-ledger"],
    FORECAST_H1_DAILY_SHADOW_V1: ["forecast", "h1", "research", "shadow", "usdcop", "pit", "forward-ledger"],
    FORECAST_H1_REGIME_SHADOW_V1: ["forecast", "h1", "research", "shadow", "usdcop", "deprecated"],

    # H5 Weekly Pipeline
    FORECAST_H5_L3_WEEKLY_TRAINING: ["forecast", "h5", "l3", "training", "weekly", "linear"],
    FORECAST_H5_L4_BACKTEST_PROMOTION: ["forecast", "h5", "l4", "backtest", "promotion", "two-vote"],
    FORECAST_H5_L4B_PRODUCTION_DEPLOY: ["forecast", "h5", "l4b", "deploy", "production", "two-vote"],
    FORECAST_H5_L5_WEEKLY_SIGNAL: ["forecast", "h5", "l5", "signal", "weekly"],
    FORECAST_H5_L5_VOL_TARGETING: ["forecast", "h5", "l5", "vol-targeting", "weekly"],
    FORECAST_H5_L7_MULTIDAY_EXECUTOR: ["forecast", "h5", "l7", "execution", "multiday"],
    FORECAST_H5_L6_WEEKLY_MONITOR: ["forecast", "h5", "l6", "monitoring", "paper-trading", "weekly"],

    # News Engine & Analysis
    NEWS_DAILY_PIPELINE: ["news", "l0", "ingestion", "enrichment", "daily"],
    NEWS_ALERT_MONITOR: ["news", "l0", "alert", "gdelt", "crisis"],
    NEWS_WEEKLY_DIGEST: ["news", "l1", "digest", "weekly"],
    NEWS_MAINTENANCE: ["news", "l0", "maintenance", "cleanup"],
    ANALYSIS_L8_DAILY_GENERATION: ["analysis", "l8", "llm", "daily", "weekly"],

    # Shared Monitoring
    CORE_L6_ALERT_MONITOR: ["core", "l6", "alert", "system"],
    CORE_L6_WEEKLY_REPORT: ["core", "l6", "report", "weekly"],

    # Platform / Ops
    CONTROL_SYSTEM_HEALTH: ["control", "monitoring", "three-clocks", "system-health", "bl-25"],
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
        "inference_frequency": "daily",  # L5b runs daily (models trained weekly by L5a)
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
                "l7": DagLayer.L7_EXECUTION,
            }
            return layer_mapping.get(part, DagLayer.L6_MONITORING)
    return DagLayer.L6_MONITORING


def get_all_dag_ids() -> List[str]:
    """Get all registered DAG IDs in execution order."""
    return [
        # Core L0 (6 DAGs)
        CORE_L0_OHLCV_BACKFILL,
        CORE_L0_OHLCV_REALTIME,
        CORE_L0_MACRO_BACKFILL,
        CORE_L0_MACRO_UPDATE,
        CORE_L0_SEED_BACKUP,
        CORE_L0_USDCOP_FORWARD_MACRO_PIT,
        # RL Pipeline
        RL_L1_FEATURE_REFRESH,
        RL_L1_MODEL_PROMOTION,
        RL_L2_DATASET_BUILD,
        RL_L2_DRIFT_RETRAIN,
        RL_L3_MODEL_TRAINING,
        # RL_L4_EXPERIMENT_RUNNER / RL_L4_BACKTEST_VALIDATION / RL_L4_SCHEDULED_RETRAINING
        # are DEPRECATED (see DEPRECATED_DAGS) — excluded from the active order so
        # consumers don't treat them as live (audit A2-09).
        RL_L4_BACKTEST_PROMOTION,
        RL_L5_PRODUCTION_INFERENCE,
        RL_L6_PRODUCTION_MONITOR,
        RL_L6_DRIFT_MONITOR,
        # Forecasting Pipeline (generic)
        # NOTE: FORECAST_L0/L1/L2/L4/L6_* are in ABSENT_DAGS — declared here
        # historically but no module on disk emits them.
        FORECAST_L3_MODEL_TRAINING,
        # H1 Daily Pipeline
        FORECAST_H1_L3_WEEKLY_TRAINING,
        FORECAST_H1_L4_BACKTEST_PROMOTION,
        FORECAST_H1_L5_DAILY_INFERENCE,
        FORECAST_H1_L5_VOL_TARGETING,
        FORECAST_H1_L6_PAPER_MONITOR,
        FORECAST_H1_L7_SMART_EXECUTOR,
        # H1 prospective shadow ledgers (research-only). FORECAST_H1_REGIME_SHADOW_V1 is
        # DEPRECATED (withdrawn experiment) and therefore absent from this order.
        FORECAST_H1_REGIME_SHADOW_V2,
        FORECAST_H1_DAILY_SHADOW_V1,
        # H5 Weekly Pipeline
        FORECAST_H5_L3_WEEKLY_TRAINING,
        FORECAST_H5_L4_BACKTEST_PROMOTION,
        FORECAST_H5_L4B_PRODUCTION_DEPLOY,
        FORECAST_H5_L5_WEEKLY_SIGNAL,
        FORECAST_H5_L5_VOL_TARGETING,
        FORECAST_H5_L7_MULTIDAY_EXECUTOR,
        FORECAST_H5_L6_WEEKLY_MONITOR,
        # News Engine & Analysis
        NEWS_DAILY_PIPELINE,
        NEWS_ALERT_MONITOR,
        NEWS_WEEKLY_DIGEST,
        NEWS_MAINTENANCE,
        ANALYSIS_L8_DAILY_GENERATION,
        # Shared Monitoring
        CORE_L6_ALERT_MONITOR,
        CORE_L6_WEEKLY_REPORT,
        # Platform / Ops
        CORE_WATCHDOG,
        RECONCILIATION_DAILY,
        RBAC_ENTITLEMENTS_DAILY,
        FORECAST_WEEKLY_GENERATION,
        FORECAST_ASSET_ANALYSIS_WEEKLY,
        FORWARD_LEDGER_WEEKLY,
        L0_MULTIFRAME_CATCHUP,
        CONTROL_SYSTEM_HEALTH,
    ]


def get_asset_dag_ids() -> List[str]:
    """DAG ids emitted by the asset pipeline factory, one per ENABLED asset.

    Config-driven (`config/assets/pipelines.yaml`), so they never appear as literals
    in any module. Any inventory that scans source alone under-counts the fleet by
    exactly the number of enabled assets.
    """
    import os

    import yaml

    here = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(here, "..", "..", "..", "config", "assets", "pipelines.yaml")
    cfg_path = os.path.normpath(cfg_path)
    if not os.path.isfile(cfg_path):
        return []
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    pattern = cfg.get("dag_id_pattern", ASSET_PIPELINE_DAG_PATTERN)
    return [
        pattern.format(asset=aid)
        for aid, spec in (cfg.get("assets") or {}).items()
        if spec.get("enabled")
    ]


def get_active_dag_ids() -> List[str]:
    """Every DAG that should exist in a healthy Airflow deployment.

    This is the executable SSOT the knowledge inventory reconciles against —
    static registrations plus config-driven asset pipelines, minus deprecated
    and absent ids.
    """
    ids = [d for d in get_all_dag_ids() if d not in DEPRECATED_DAGS and d not in ABSENT_DAGS]
    return ids + get_asset_dag_ids()


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


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================
# These aliases allow old import patterns to work during migration

# L0 aliases (used by DAGs and xcom_contracts)
L0_OHLCV_REALTIME = CORE_L0_OHLCV_REALTIME
L0_OHLCV_BACKFILL = CORE_L0_OHLCV_BACKFILL
L0_MACRO_UPDATE = CORE_L0_MACRO_UPDATE
L0_MACRO_BACKFILL = CORE_L0_MACRO_BACKFILL
L0_MACRO_DAILY = CORE_L0_MACRO_UPDATE  # Legacy name (was CORE_L0_MACRO_DAILY)
L0_SEED_BACKUP = CORE_L0_SEED_BACKUP
L0_USDCOP_FORWARD_MACRO_PIT = CORE_L0_USDCOP_FORWARD_MACRO_PIT

# L1 aliases
L1_FEATURE_REFRESH = RL_L1_FEATURE_REFRESH
L1_MODEL_PROMOTION = RL_L1_MODEL_PROMOTION

# L2 aliases
L2_DATASET_BUILD = RL_L2_DATASET_BUILD
L2_PREPROCESSING_PIPELINE = RL_L2_DATASET_BUILD  # Legacy name

# L3 aliases
L3_MODEL_TRAINING = RL_L3_MODEL_TRAINING

# L4 aliases
L4_EXPERIMENT_RUNNER = RL_L4_EXPERIMENT_RUNNER
L4_SCHEDULED_RETRAINING = RL_L4_SCHEDULED_RETRAINING
L4_BACKTEST_PROMOTION = RL_L4_BACKTEST_PROMOTION

# L5 aliases
L5_PRODUCTION_INFERENCE = RL_L5_PRODUCTION_INFERENCE

# L6 aliases
L6_PRODUCTION_MONITOR = RL_L6_PRODUCTION_MONITOR
L6_DRIFT_MONITOR = RL_L6_DRIFT_MONITOR
L6_ALERT_MONITOR = CORE_L6_ALERT_MONITOR
L6_WEEKLY_REPORT = CORE_L6_WEEKLY_REPORT
L2_DRIFT_RETRAIN = RL_L2_DRIFT_RETRAIN


__all__ = [
    # Enums
    "DagPipeline",
    "DagLayer",
    "DagType",
    # Core L0 IDs (6 DAGs)
    "CORE_L0_OHLCV_BACKFILL",
    "CORE_L0_OHLCV_REALTIME",
    "CORE_L0_MACRO_BACKFILL",
    "CORE_L0_MACRO_UPDATE",
    "CORE_L0_SEED_BACKUP",
    "CORE_L0_USDCOP_FORWARD_MACRO_PIT",
    # News Engine & Analysis IDs
    "NEWS_DAILY_PIPELINE",
    "NEWS_ALERT_MONITOR",
    "NEWS_WEEKLY_DIGEST",
    "NEWS_MAINTENANCE",
    "ANALYSIS_L8_DAILY_GENERATION",
    # Core L6 IDs
    "CORE_L6_ALERT_MONITOR",
    "CORE_L6_WEEKLY_REPORT",
    # RL Pipeline IDs
    "RL_L1_FEATURE_REFRESH",
    "RL_L1_MODEL_PROMOTION",
    "RL_L2_DATASET_BUILD",
    "RL_L2_DRIFT_RETRAIN",
    "RL_L3_MODEL_TRAINING",
    "RL_L4_EXPERIMENT_RUNNER",
    "RL_L4_BACKTEST_VALIDATION",
    "RL_L4_SCHEDULED_RETRAINING",
    "RL_L4_BACKTEST_PROMOTION",
    "RL_L5_PRODUCTION_INFERENCE",
    "RL_L6_PRODUCTION_MONITOR",
    "RL_L6_DRIFT_MONITOR",
    # Forecasting Pipeline IDs (generic)
    "FORECAST_L0_DAILY_DATA",
    "FORECAST_L1_DAILY_FEATURES",
    "FORECAST_L2_DATASET_BUILD",
    "FORECAST_L3_MODEL_TRAINING",
    "FORECAST_L4_BACKTEST_VALIDATION",
    "FORECAST_L4_EXPERIMENT_RUNNER",
    "FORECAST_L6_DRIFT_MONITOR",
    "FORECAST_L6_ACCURACY_REPORT",
    # H1 Daily Pipeline IDs
    "FORECAST_H1_L3_WEEKLY_TRAINING",
    "FORECAST_H1_L4_BACKTEST_PROMOTION",
    "FORECAST_H1_L5_DAILY_INFERENCE",
    "FORECAST_H1_L5_VOL_TARGETING",
    "FORECAST_H1_L6_PAPER_MONITOR",
    "FORECAST_H1_L7_SMART_EXECUTOR",
    # H5 Weekly Pipeline IDs
    "FORECAST_H5_L3_WEEKLY_TRAINING",
    "FORECAST_H5_L4_BACKTEST_PROMOTION",
    "FORECAST_H5_L5_WEEKLY_SIGNAL",
    "FORECAST_H5_L5_VOL_TARGETING",
    "FORECAST_H5_L7_MULTIDAY_EXECUTOR",
    "FORECAST_H5_L6_WEEKLY_MONITOR",
    # Legacy forecasting aliases (backward compat)
    "FORECAST_L5_WEEKLY_INFERENCE",
    "FORECAST_L5_DAILY_INFERENCE",
    "FORECAST_L5_VOL_TARGETING",
    "FORECAST_L6_PAPER_TRADING",
    "FORECAST_L7_SMART_EXECUTOR",
    "FORECAST_H5_L5A_WEEKLY_TRAINING",
    "FORECAST_H5_L5B_WEEKLY_SIGNAL",
    "FORECAST_H5_L5C_VOL_TARGETING",
    # Backward compatibility aliases (L0)
    "L0_OHLCV_REALTIME",
    "L0_OHLCV_BACKFILL",
    "L0_MACRO_UPDATE",
    "L0_MACRO_BACKFILL",
    "L0_MACRO_DAILY",
    "L0_SEED_BACKUP",
    # Backward compatibility aliases (L1-L6)
    "L1_FEATURE_REFRESH",
    "L1_MODEL_PROMOTION",
    "L2_DATASET_BUILD",
    "L2_PREPROCESSING_PIPELINE",
    "L2_DRIFT_RETRAIN",
    "L3_MODEL_TRAINING",
    "L4_EXPERIMENT_RUNNER",
    "L4_SCHEDULED_RETRAINING",
    "L4_BACKTEST_PROMOTION",
    "L5_PRODUCTION_INFERENCE",
    "L6_PRODUCTION_MONITOR",
    "L6_DRIFT_MONITOR",
    "L6_ALERT_MONITOR",
    "L6_WEEKLY_REPORT",
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
