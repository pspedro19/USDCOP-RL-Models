"""
XCom Contracts - SSOT para comunicación inter-DAG.

Este módulo define los contratos de comunicación entre DAGs L2, L3, y L4.
TODOS los DAGs DEBEN usar estas clases para XCom push/pull.

Contract: CTR-XCOM-001

Usage:
    from airflow.dags.contracts.xcom_contracts import L2XComKeys, L3XComKeys, L2Output

    # Push con contratos
    output = L2Output(
        dataset_path="/data/experiments/baseline/train.parquet",
        dataset_hash="abc123",
        ...
    )
    output.push_to_xcom(ti)

    # Pull con helper
    l2_data = pull_l2_output(ti, dag_id='v3.l2_preprocessing_pipeline')

Author: Trading Team
Version: 1.0.0
Created: 2026-01-18
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path for SSOT imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# SSOT imports for hash utilities
from src.utils.hash_utils import (
    compute_file_hash as _compute_file_hash_canonical,
    compute_json_hash as _compute_json_hash_canonical,
    HashResult,
)


# =============================================================================
# XCOM KEY ENUMS - SSOT for inter-DAG communication keys
# =============================================================================


class L0XComKeysEnum(str, Enum):
    """
    XCom keys for L0 Macro Data Acquisition output.

    These keys define the contract for L0 macro output.
    L1 MUST use these keys for pull operations.
    """
    FRED_DATA = "l0_fred_data"
    TWELVEDATA_DATA = "l0_twelvedata_data"
    INVESTING_DATA = "l0_investing_data"
    BANREP_DATA = "l0_banrep_data"
    EMBI_DATA = "l0_embi_data"
    FFILL_RESULT = "l0_ffill_result"
    READINESS_REPORT = "l0_readiness_report"
    IS_READY_FOR_INFERENCE = "l0_is_ready_for_inference"
    INDICATORS_UPDATED = "l0_indicators_updated"
    FULL_OUTPUT = "l0_full_output"


class L1XComKeysEnum(str, Enum):
    """
    XCom keys for L1 Feature Engineering output.

    These keys define the contract for L1 feature output.
    L2 and L5 MUST use these keys for pull operations.
    """
    FEATURES_COUNT = "l1_features_count"
    MACRO_ROWS_USED = "l1_macro_rows_used"
    DETECTED_OHLCV_TIME = "l1_detected_ohlcv_time"
    LAST_PROCESSED_OHLCV_TIME = "l1_last_processed_ohlcv_time"
    BUILDER_VERSION = "l1_builder_version"
    FEATURE_ORDER_HASH = "l1_feature_order_hash"
    NORM_STATS_HASH = "l1_norm_stats_hash"
    INFERENCE_FEATURES_WRITTEN = "l1_inference_features_written"
    FULL_OUTPUT = "l1_full_output"


class L2XComKeysEnum(str, Enum):
    """
    XCom keys for L2 Preprocessing output.

    These keys define the contract for L2 output.
    L3 and L4 MUST use these keys for pull operations.
    """
    DATASET_PATH = "l2_dataset_path"
    DATASET_HASH = "l2_dataset_hash"
    DATE_RANGE_START = "l2_date_range_start"
    DATE_RANGE_END = "l2_date_range_end"
    FEATURE_ORDER_HASH = "l2_feature_order_hash"
    FEATURE_COLUMNS = "l2_feature_columns"
    ROW_COUNT = "l2_row_count"
    EXPERIMENT_NAME = "l2_experiment_name"
    NORM_STATS_PATH = "l2_norm_stats_path"
    MANIFEST_PATH = "l2_manifest_path"
    FULL_OUTPUT = "l2_full_output"


class L3XComKeysEnum(str, Enum):
    """
    XCom keys for L3 Training output.

    These keys define the contract for L3 output.
    L4 MUST use these keys for pull operations.
    """
    MODEL_PATH = "l3_model_path"
    MODEL_HASH = "l3_model_hash"
    MLFLOW_RUN_ID = "l3_mlflow_run_id"
    MLFLOW_EXPERIMENT_ID = "l3_mlflow_experiment_id"
    TRAINING_DURATION = "l3_training_duration"
    BEST_REWARD = "l3_best_reward"
    FINAL_METRICS = "l3_final_metrics"
    NORM_STATS_HASH = "l3_norm_stats_hash"
    CONFIG_HASH = "l3_config_hash"
    DVC_TAG = "l3_dvc_tag"
    FULL_OUTPUT = "l3_full_output"

    # Reward system fields (CTR-REWARD-SNAPSHOT-001)
    REWARD_CONTRACT_ID = "l3_reward_contract_id"
    REWARD_CONFIG_HASH = "l3_reward_config_hash"
    REWARD_CONFIG_URI = "l3_reward_config_uri"
    CURRICULUM_FINAL_PHASE = "l3_curriculum_final_phase"
    TOTAL_REWARD_COMPONENTS = "l3_total_reward_components"
    REWARD_WEIGHTS = "l3_reward_weights"  # Full weights dict


class L4XComKeysEnum(str, Enum):
    """
    XCom keys for L4 Experiment Runner output.
    """
    EXPERIMENT_RESULT = "l4_experiment_result"
    COMPARISON_RESULT = "l4_comparison_result"
    REGISTRY_PATH = "l4_registry_path"
    LINEAGE_RECORD = "l4_lineage_record"
    NOTIFICATION_SENT = "l4_notification_sent"
    FULL_OUTPUT = "l4_full_output"


class L5XComKeysEnum(str, Enum):
    """
    XCom keys for L5 Multi-Model Inference output.

    These keys define the contract for L5 inference output.
    L6 monitoring MUST use these keys for pull operations.
    """
    INFERENCE_RESULT = "l5_inference_result"
    MODEL_ID = "l5_model_id"
    SIGNAL = "l5_signal"
    CONFIDENCE = "l5_confidence"
    EXECUTION_PRICE = "l5_execution_price"
    POSITION = "l5_position"
    INFERENCE_LATENCY_MS = "l5_inference_latency_ms"
    FEATURE_HASH = "l5_feature_hash"
    IS_CHALLENGER = "l5_is_challenger"
    CANARY_STAGE = "l5_canary_stage"
    FULL_OUTPUT = "l5_full_output"


class L6XComKeysEnum(str, Enum):
    """
    XCom keys for L6 Production Monitoring output.

    These keys define the contract for L6 monitoring output.
    """
    MONITORING_REPORT = "l6_monitoring_report"
    DRIFT_DETECTED = "l6_drift_detected"
    DRIFT_SCORE = "l6_drift_score"
    ALERT_TRIGGERED = "l6_alert_triggered"
    MODEL_PERFORMANCE = "l6_model_performance"
    RETRAINING_RECOMMENDED = "l6_retraining_recommended"
    METRICS_SUMMARY = "l6_metrics_summary"
    FULL_OUTPUT = "l6_full_output"


# =============================================================================
# DAG IDs - Centralized references (SSOT for inter-DAG communication)
# =============================================================================

# Layer 0: Data Acquisition
L0_MACRO_DAG_ID = "v3.l0_macro_unified"
L0_OHLCV_REALTIME_DAG_ID = "v3.l0_ohlcv_realtime"
L0_OHLCV_BACKFILL_DAG_ID = "v3.l0_ohlcv_backfill"
L0_WEEKLY_BACKUP_DAG_ID = "v3.l0_weekly_backup"
L0_DATA_INIT_DAG_ID = "v3.l0_data_initialization"

# Layer 1: Feature Engineering
L1_DAG_ID = "v3.l1_feature_refresh"
L1_FEAST_DAG_ID = "v3.l1b_feast_materialize"

# Layer 2: Data Preprocessing
L2_DAG_ID = "v3.l2_preprocessing_pipeline"
L2_DRIFT_DAG_ID = "l2b_drift_retraining"

# Layer 3: Model Training
L3_DAG_ID = "v3.l3_model_training"

# Layer 4: Experiment/Validation
L4_DAG_ID = "l4_experiment_runner"
L4_BACKTEST_DAG_ID = "v3.l4_backtest_validation"
L4_SCHEDULED_DAG_ID = "l4_scheduled_retraining"

# Layer 5: Inference
L5_DAG_ID = "v3.l5_multi_model_inference"

# Layer 6: Production Monitoring
L6_DAG_ID = "l6_production_monitoring"


# =============================================================================
# OUTPUT DATACLASSES - Structured output for XCom
# =============================================================================


@dataclass
class L2Output:
    """
    Structured output from L2 Preprocessing.

    This dataclass encapsulates all data produced by L2.
    Use for consistent serialization/deserialization between DAGs.

    MinIO-First Architecture:
    - Primary storage: S3 URIs (s3://experiments/{exp}/datasets/{ver}/...)
    - Local paths deprecated but supported for backward compatibility

    Example:
        output = L2Output(
            dataset_uri="s3://experiments/baseline/datasets/v1/train.parquet",
            norm_stats_uri="s3://experiments/baseline/datasets/v1/norm_stats.json",
            manifest_uri="s3://experiments/baseline/datasets/v1/manifest.json",
            experiment_id="baseline",
            version="v1",
            dataset_hash="abc123",
            date_range_start="2023-01-01",
            date_range_end="2024-12-31",
            feature_order_hash="def456",
            feature_columns=["log_ret_5m", "rsi_9"],
            row_count=1000,
        )

        # Push to XCom
        output.push_to_xcom(ti)

        # Pull and reconstruct
        data = ti.xcom_pull(key=L2XComKeysEnum.FULL_OUTPUT.value, dag_id=L2_DAG_ID)
        restored = L2Output.from_dict(data)
    """
    # S3 URIs (PRIMARY - MinIO-first architecture)
    dataset_uri: Optional[str] = None          # s3://experiments/{exp}/datasets/{ver}/train.parquet
    norm_stats_uri: Optional[str] = None       # s3://experiments/{exp}/datasets/{ver}/norm_stats.json
    manifest_uri: Optional[str] = None         # s3://experiments/{exp}/datasets/{ver}/manifest.json

    # Experiment identification
    experiment_id: Optional[str] = None
    version: Optional[str] = None

    # Required metadata
    dataset_hash: str = ""
    date_range_start: str = ""
    date_range_end: str = ""
    feature_order_hash: str = ""
    feature_columns: List[str] = field(default_factory=list)
    row_count: int = 0

    # DEPRECATED - backward compatibility (use S3 URIs instead)
    dataset_path: Optional[str] = None         # Legacy local path
    experiment_name: Optional[str] = None      # Use experiment_id instead
    norm_stats_path: Optional[str] = None      # Use norm_stats_uri instead
    manifest_path: Optional[str] = None        # Use manifest_uri instead

    def is_minio_first(self) -> bool:
        """Check if this output uses MinIO-first storage."""
        return self.dataset_uri is not None and self.dataset_uri.startswith("s3://")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for XCom push."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "L2Output":
        """Create from XCom pull dictionary."""
        if data is None:
            raise ValueError("Cannot create L2Output from None")
        return cls(**data)

    def push_to_xcom(self, ti) -> None:
        """
        Push all fields to XCom using contract keys.

        Args:
            ti: Airflow TaskInstance
        """
        # S3 URIs (MinIO-first)
        if self.dataset_uri:
            ti.xcom_push(key="l2_dataset_uri", value=self.dataset_uri)
        if self.norm_stats_uri:
            ti.xcom_push(key="l2_norm_stats_uri", value=self.norm_stats_uri)
        if self.manifest_uri:
            ti.xcom_push(key="l2_manifest_uri", value=self.manifest_uri)
        if self.experiment_id:
            ti.xcom_push(key="l2_experiment_id", value=self.experiment_id)
        if self.version:
            ti.xcom_push(key="l2_version", value=self.version)

        # Legacy fields (backward compatibility)
        ti.xcom_push(key=L2XComKeysEnum.DATASET_PATH.value, value=self.dataset_path or self.dataset_uri)
        ti.xcom_push(key=L2XComKeysEnum.DATASET_HASH.value, value=self.dataset_hash)
        ti.xcom_push(key=L2XComKeysEnum.DATE_RANGE_START.value, value=self.date_range_start)
        ti.xcom_push(key=L2XComKeysEnum.DATE_RANGE_END.value, value=self.date_range_end)
        ti.xcom_push(key=L2XComKeysEnum.FEATURE_ORDER_HASH.value, value=self.feature_order_hash)
        ti.xcom_push(key=L2XComKeysEnum.FEATURE_COLUMNS.value, value=self.feature_columns)
        ti.xcom_push(key=L2XComKeysEnum.ROW_COUNT.value, value=self.row_count)
        ti.xcom_push(key=L2XComKeysEnum.EXPERIMENT_NAME.value, value=self.experiment_name or self.experiment_id)

        if self.norm_stats_path or self.norm_stats_uri:
            ti.xcom_push(key=L2XComKeysEnum.NORM_STATS_PATH.value, value=self.norm_stats_path or self.norm_stats_uri)
        if self.manifest_path or self.manifest_uri:
            ti.xcom_push(key=L2XComKeysEnum.MANIFEST_PATH.value, value=self.manifest_path or self.manifest_uri)

        # Also push full output for easy retrieval
        ti.xcom_push(key=L2XComKeysEnum.FULL_OUTPUT.value, value=self.to_dict())


@dataclass
class L3Output:
    """
    Structured output from L3 Training.

    MinIO-First Architecture:
    - Primary storage: S3 URIs (s3://experiments/{exp}/models/{ver}/...)
    - Local paths deprecated but supported for backward compatibility

    Reward System Integration:
    - reward_config_uri: S3 URI to reward configuration JSON
    - reward_contract_id: Version of reward contract (SSOT)
    - reward_config_hash: Hash for lineage tracking
    """
    # S3 URIs (PRIMARY - MinIO-first architecture)
    model_uri: Optional[str] = None            # s3://experiments/{exp}/models/{ver}/policy.onnx
    norm_stats_uri: Optional[str] = None       # s3://experiments/{exp}/models/{ver}/norm_stats.json
    config_uri: Optional[str] = None           # s3://experiments/{exp}/models/{ver}/config.yaml
    lineage_uri: Optional[str] = None          # s3://experiments/{exp}/models/{ver}/lineage.json

    # Experiment identification
    experiment_id: Optional[str] = None
    version: Optional[str] = None

    # Integrity hashes
    model_hash: str = ""
    norm_stats_hash: str = ""
    config_hash: Optional[str] = None
    feature_order_hash: Optional[str] = None

    # Contract compliance
    observation_dim: int = 15

    # Training results
    mlflow_run_id: str = ""
    mlflow_experiment_id: Optional[str] = None
    training_duration: float = 0.0
    best_reward: float = 0.0
    final_metrics: Optional[Dict[str, float]] = None

    # DVC tracking
    dvc_tag: Optional[str] = None

    # Link to dataset (MinIO-first)
    dataset_snapshot: Optional[Dict[str, Any]] = None

    # Reward system (CTR-REWARD-SNAPSHOT-001)
    reward_config_uri: Optional[str] = None    # s3://experiments/{exp}/reward_configs/{ver}/reward_config.json
    reward_contract_id: Optional[str] = None   # e.g., "v1.0.0"
    reward_config_hash: Optional[str] = None   # SHA-256 hash for lineage
    curriculum_final_phase: Optional[str] = None  # "phase_1", "phase_2", or "phase_3"
    reward_weights: Optional[Dict[str, float]] = None  # Component weights used

    # Additional metadata for L3 DAG
    dataset_hash: Optional[str] = None         # Hash of training dataset
    experiment_name: Optional[str] = None      # Alias for experiment_id (backward compat)

    # DEPRECATED - backward compatibility (use S3 URIs instead)
    model_path: Optional[str] = None           # Legacy local path

    def is_minio_first(self) -> bool:
        """Check if this output uses MinIO-first storage."""
        return self.model_uri is not None and self.model_uri.startswith("s3://")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for XCom push."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "L3Output":
        """Create from XCom pull dictionary."""
        if data is None:
            raise ValueError("Cannot create L3Output from None")
        return cls(**data)

    def push_to_xcom(self, ti) -> None:
        """Push all fields to XCom using contract keys."""
        # S3 URIs (MinIO-first)
        if self.model_uri:
            ti.xcom_push(key="l3_model_uri", value=self.model_uri)
        if self.norm_stats_uri:
            ti.xcom_push(key="l3_norm_stats_uri", value=self.norm_stats_uri)
        if self.config_uri:
            ti.xcom_push(key="l3_config_uri", value=self.config_uri)
        if self.lineage_uri:
            ti.xcom_push(key="l3_lineage_uri", value=self.lineage_uri)
        if self.experiment_id:
            ti.xcom_push(key="l3_experiment_id", value=self.experiment_id)
        if self.version:
            ti.xcom_push(key="l3_version", value=self.version)
        if self.feature_order_hash:
            ti.xcom_push(key="l3_feature_order_hash", value=self.feature_order_hash)
        if self.observation_dim:
            ti.xcom_push(key="l3_observation_dim", value=self.observation_dim)

        # Legacy fields (backward compatibility)
        ti.xcom_push(key=L3XComKeysEnum.MODEL_PATH.value, value=self.model_path or self.model_uri)
        ti.xcom_push(key=L3XComKeysEnum.MODEL_HASH.value, value=self.model_hash)
        ti.xcom_push(key=L3XComKeysEnum.MLFLOW_RUN_ID.value, value=self.mlflow_run_id)
        ti.xcom_push(key=L3XComKeysEnum.TRAINING_DURATION.value, value=self.training_duration)
        ti.xcom_push(key=L3XComKeysEnum.BEST_REWARD.value, value=self.best_reward)
        ti.xcom_push(key=L3XComKeysEnum.NORM_STATS_HASH.value, value=self.norm_stats_hash)

        if self.mlflow_experiment_id:
            ti.xcom_push(key=L3XComKeysEnum.MLFLOW_EXPERIMENT_ID.value, value=self.mlflow_experiment_id)
        if self.final_metrics:
            ti.xcom_push(key=L3XComKeysEnum.FINAL_METRICS.value, value=self.final_metrics)
        if self.config_hash:
            ti.xcom_push(key=L3XComKeysEnum.CONFIG_HASH.value, value=self.config_hash)
        if self.dvc_tag:
            ti.xcom_push(key=L3XComKeysEnum.DVC_TAG.value, value=self.dvc_tag)

        # Reward system fields (CTR-REWARD-SNAPSHOT-001)
        if self.reward_config_uri:
            ti.xcom_push(key=L3XComKeysEnum.REWARD_CONFIG_URI.value, value=self.reward_config_uri)
        if self.reward_contract_id:
            ti.xcom_push(key=L3XComKeysEnum.REWARD_CONTRACT_ID.value, value=self.reward_contract_id)
        if self.reward_config_hash:
            ti.xcom_push(key=L3XComKeysEnum.REWARD_CONFIG_HASH.value, value=self.reward_config_hash)
        if self.curriculum_final_phase:
            ti.xcom_push(key=L3XComKeysEnum.CURRICULUM_FINAL_PHASE.value, value=self.curriculum_final_phase)
        if self.reward_weights:
            ti.xcom_push(key=L3XComKeysEnum.TOTAL_REWARD_COMPONENTS.value, value=len(self.reward_weights))

        # Also push full output for easy retrieval
        ti.xcom_push(key=L3XComKeysEnum.FULL_OUTPUT.value, value=self.to_dict())


@dataclass
class L0Output:
    """
    Structured output from L0 Macro Data Acquisition.

    This dataclass encapsulates all data produced by L0 macro scraping.
    """
    # Readiness status
    is_ready_for_inference: bool = False
    readiness_report: Optional[Dict[str, Any]] = None

    # Data counts per source
    fred_indicators_updated: int = 0
    twelvedata_indicators_updated: int = 0
    investing_indicators_updated: int = 0
    banrep_indicators_updated: int = 0
    embi_indicators_updated: int = 0

    # Forward fill results
    ffill_applied: bool = False
    ffill_rows_affected: int = 0

    # Metadata
    execution_date: Optional[str] = None
    total_indicators: int = 37

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for XCom push."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "L0Output":
        """Create from XCom pull dictionary."""
        if data is None:
            raise ValueError("Cannot create L0Output from None")
        return cls(**data)

    def push_to_xcom(self, ti) -> None:
        """Push all fields to XCom using contract keys."""
        ti.xcom_push(key=L0XComKeysEnum.IS_READY_FOR_INFERENCE.value, value=self.is_ready_for_inference)
        if self.readiness_report:
            ti.xcom_push(key=L0XComKeysEnum.READINESS_REPORT.value, value=self.readiness_report)
        ti.xcom_push(key=L0XComKeysEnum.INDICATORS_UPDATED.value, value={
            "fred": self.fred_indicators_updated,
            "twelvedata": self.twelvedata_indicators_updated,
            "investing": self.investing_indicators_updated,
            "banrep": self.banrep_indicators_updated,
            "embi": self.embi_indicators_updated,
        })
        ti.xcom_push(key=L0XComKeysEnum.FFILL_RESULT.value, value={
            "applied": self.ffill_applied,
            "rows_affected": self.ffill_rows_affected,
        })
        ti.xcom_push(key=L0XComKeysEnum.FULL_OUTPUT.value, value=self.to_dict())


@dataclass
class L1Output:
    """
    Structured output from L1 Feature Engineering.

    This dataclass encapsulates all data produced by L1 feature refresh.
    """
    # Feature counts
    features_count: int = 0
    macro_rows_used: int = 0
    inference_features_written: int = 0

    # Timing
    detected_ohlcv_time: Optional[str] = None
    last_processed_ohlcv_time: Optional[str] = None

    # Hashes for validation
    builder_version: str = ""
    feature_order_hash: str = ""
    norm_stats_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for XCom push."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "L1Output":
        """Create from XCom pull dictionary."""
        if data is None:
            raise ValueError("Cannot create L1Output from None")
        return cls(**data)

    def push_to_xcom(self, ti) -> None:
        """Push all fields to XCom using contract keys."""
        ti.xcom_push(key=L1XComKeysEnum.FEATURES_COUNT.value, value=self.features_count)
        ti.xcom_push(key=L1XComKeysEnum.MACRO_ROWS_USED.value, value=self.macro_rows_used)
        ti.xcom_push(key=L1XComKeysEnum.INFERENCE_FEATURES_WRITTEN.value, value=self.inference_features_written)
        if self.detected_ohlcv_time:
            ti.xcom_push(key=L1XComKeysEnum.DETECTED_OHLCV_TIME.value, value=self.detected_ohlcv_time)
        if self.last_processed_ohlcv_time:
            ti.xcom_push(key=L1XComKeysEnum.LAST_PROCESSED_OHLCV_TIME.value, value=self.last_processed_ohlcv_time)
        ti.xcom_push(key=L1XComKeysEnum.BUILDER_VERSION.value, value=self.builder_version)
        ti.xcom_push(key=L1XComKeysEnum.FEATURE_ORDER_HASH.value, value=self.feature_order_hash)
        ti.xcom_push(key=L1XComKeysEnum.NORM_STATS_HASH.value, value=self.norm_stats_hash)
        ti.xcom_push(key=L1XComKeysEnum.FULL_OUTPUT.value, value=self.to_dict())


@dataclass
class L4Output:
    """
    Structured output from L4 Experiment Runner.
    """
    experiment_result: Dict[str, Any]
    comparison_result: Optional[Dict[str, Any]] = None
    registry_path: Optional[str] = None
    lineage_record: Optional[Dict[str, str]] = None
    notification_sent: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for XCom push."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "L4Output":
        """Create from XCom pull dictionary."""
        if data is None:
            raise ValueError("Cannot create L4Output from None")
        return cls(**data)

    def push_to_xcom(self, ti) -> None:
        """Push all fields to XCom using contract keys."""
        ti.xcom_push(key=L4XComKeysEnum.EXPERIMENT_RESULT.value, value=self.experiment_result)
        if self.comparison_result:
            ti.xcom_push(key=L4XComKeysEnum.COMPARISON_RESULT.value, value=self.comparison_result)
        if self.registry_path:
            ti.xcom_push(key=L4XComKeysEnum.REGISTRY_PATH.value, value=self.registry_path)
        if self.lineage_record:
            ti.xcom_push(key=L4XComKeysEnum.LINEAGE_RECORD.value, value=self.lineage_record)
        ti.xcom_push(key=L4XComKeysEnum.NOTIFICATION_SENT.value, value=self.notification_sent)

        # Also push full output for easy retrieval
        ti.xcom_push(key=L4XComKeysEnum.FULL_OUTPUT.value, value=self.to_dict())


@dataclass
class L5Output:
    """
    Structured output from L5 Multi-Model Inference.

    This dataclass encapsulates all data produced by L5 inference.
    """
    # Inference result
    model_id: str = ""
    signal: float = 0.0  # -1 (short), 0 (flat), 1 (long)
    confidence: float = 0.0
    execution_price: float = 0.0
    position: float = 0.0

    # Performance
    inference_latency_ms: float = 0.0
    feature_hash: str = ""

    # Canary deployment
    is_challenger: bool = False
    canary_stage: str = "production"

    # Full inference result
    inference_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for XCom push."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "L5Output":
        """Create from XCom pull dictionary."""
        if data is None:
            raise ValueError("Cannot create L5Output from None")
        return cls(**data)

    def push_to_xcom(self, ti) -> None:
        """Push all fields to XCom using contract keys."""
        ti.xcom_push(key=L5XComKeysEnum.MODEL_ID.value, value=self.model_id)
        ti.xcom_push(key=L5XComKeysEnum.SIGNAL.value, value=self.signal)
        ti.xcom_push(key=L5XComKeysEnum.CONFIDENCE.value, value=self.confidence)
        ti.xcom_push(key=L5XComKeysEnum.EXECUTION_PRICE.value, value=self.execution_price)
        ti.xcom_push(key=L5XComKeysEnum.POSITION.value, value=self.position)
        ti.xcom_push(key=L5XComKeysEnum.INFERENCE_LATENCY_MS.value, value=self.inference_latency_ms)
        ti.xcom_push(key=L5XComKeysEnum.FEATURE_HASH.value, value=self.feature_hash)
        ti.xcom_push(key=L5XComKeysEnum.IS_CHALLENGER.value, value=self.is_challenger)
        ti.xcom_push(key=L5XComKeysEnum.CANARY_STAGE.value, value=self.canary_stage)
        if self.inference_result:
            ti.xcom_push(key=L5XComKeysEnum.INFERENCE_RESULT.value, value=self.inference_result)
        ti.xcom_push(key=L5XComKeysEnum.FULL_OUTPUT.value, value=self.to_dict())


@dataclass
class L6Output:
    """
    Structured output from L6 Production Monitoring.

    This dataclass encapsulates all data produced by L6 monitoring.
    """
    # Drift detection
    drift_detected: bool = False
    drift_score: float = 0.0

    # Alert status
    alert_triggered: bool = False
    retraining_recommended: bool = False

    # Model performance
    model_performance: Optional[Dict[str, float]] = None
    metrics_summary: Optional[Dict[str, Any]] = None

    # Full monitoring report
    monitoring_report: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for XCom push."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "L6Output":
        """Create from XCom pull dictionary."""
        if data is None:
            raise ValueError("Cannot create L6Output from None")
        return cls(**data)

    def push_to_xcom(self, ti) -> None:
        """Push all fields to XCom using contract keys."""
        ti.xcom_push(key=L6XComKeysEnum.DRIFT_DETECTED.value, value=self.drift_detected)
        ti.xcom_push(key=L6XComKeysEnum.DRIFT_SCORE.value, value=self.drift_score)
        ti.xcom_push(key=L6XComKeysEnum.ALERT_TRIGGERED.value, value=self.alert_triggered)
        ti.xcom_push(key=L6XComKeysEnum.RETRAINING_RECOMMENDED.value, value=self.retraining_recommended)
        if self.model_performance:
            ti.xcom_push(key=L6XComKeysEnum.MODEL_PERFORMANCE.value, value=self.model_performance)
        if self.metrics_summary:
            ti.xcom_push(key=L6XComKeysEnum.METRICS_SUMMARY.value, value=self.metrics_summary)
        if self.monitoring_report:
            ti.xcom_push(key=L6XComKeysEnum.MONITORING_REPORT.value, value=self.monitoring_report)
        ti.xcom_push(key=L6XComKeysEnum.FULL_OUTPUT.value, value=self.to_dict())


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def pull_l2_output(
    ti,
    dag_id: str = L2_DAG_ID,
    include_prior_dates: bool = True
) -> Optional[L2Output]:
    """
    Pull L2 output from XCom.

    Args:
        ti: TaskInstance
        dag_id: DAG ID to pull from
        include_prior_dates: Whether to include prior dates in search

    Returns:
        L2Output dataclass or None if not found
    """
    # Try to get full output first
    full_output = ti.xcom_pull(
        key=L2XComKeysEnum.FULL_OUTPUT.value,
        dag_id=dag_id,
        include_prior_dates=include_prior_dates,
    )

    if full_output:
        return L2Output.from_dict(full_output)

    # Fallback: Try individual keys
    dataset_path = ti.xcom_pull(
        key=L2XComKeysEnum.DATASET_PATH.value,
        dag_id=dag_id,
        include_prior_dates=include_prior_dates,
    )

    if not dataset_path:
        return None

    return L2Output(
        dataset_path=dataset_path,
        dataset_hash=ti.xcom_pull(key=L2XComKeysEnum.DATASET_HASH.value, dag_id=dag_id, include_prior_dates=include_prior_dates) or "",
        date_range_start=ti.xcom_pull(key=L2XComKeysEnum.DATE_RANGE_START.value, dag_id=dag_id, include_prior_dates=include_prior_dates) or "",
        date_range_end=ti.xcom_pull(key=L2XComKeysEnum.DATE_RANGE_END.value, dag_id=dag_id, include_prior_dates=include_prior_dates) or "",
        feature_order_hash=ti.xcom_pull(key=L2XComKeysEnum.FEATURE_ORDER_HASH.value, dag_id=dag_id, include_prior_dates=include_prior_dates) or "",
        feature_columns=ti.xcom_pull(key=L2XComKeysEnum.FEATURE_COLUMNS.value, dag_id=dag_id, include_prior_dates=include_prior_dates) or [],
        row_count=ti.xcom_pull(key=L2XComKeysEnum.ROW_COUNT.value, dag_id=dag_id, include_prior_dates=include_prior_dates) or 0,
        experiment_name=ti.xcom_pull(key=L2XComKeysEnum.EXPERIMENT_NAME.value, dag_id=dag_id, include_prior_dates=include_prior_dates),
        norm_stats_path=ti.xcom_pull(key=L2XComKeysEnum.NORM_STATS_PATH.value, dag_id=dag_id, include_prior_dates=include_prior_dates),
        manifest_path=ti.xcom_pull(key=L2XComKeysEnum.MANIFEST_PATH.value, dag_id=dag_id, include_prior_dates=include_prior_dates),
    )


def pull_l3_output(
    ti,
    dag_id: str = L3_DAG_ID,
    include_prior_dates: bool = True
) -> Optional[L3Output]:
    """
    Pull L3 output from XCom.

    Args:
        ti: TaskInstance
        dag_id: DAG ID to pull from
        include_prior_dates: Whether to include prior dates in search

    Returns:
        L3Output dataclass or None if not found
    """
    # Try to get full output first
    full_output = ti.xcom_pull(
        key=L3XComKeysEnum.FULL_OUTPUT.value,
        dag_id=dag_id,
        include_prior_dates=include_prior_dates,
    )

    if full_output:
        return L3Output.from_dict(full_output)

    # Fallback: Try individual keys
    model_path = ti.xcom_pull(
        key=L3XComKeysEnum.MODEL_PATH.value,
        dag_id=dag_id,
        include_prior_dates=include_prior_dates,
    )

    if not model_path:
        return None

    return L3Output(
        model_path=model_path,
        model_hash=ti.xcom_pull(key=L3XComKeysEnum.MODEL_HASH.value, dag_id=dag_id, include_prior_dates=include_prior_dates) or "",
        mlflow_run_id=ti.xcom_pull(key=L3XComKeysEnum.MLFLOW_RUN_ID.value, dag_id=dag_id, include_prior_dates=include_prior_dates) or "",
        training_duration=ti.xcom_pull(key=L3XComKeysEnum.TRAINING_DURATION.value, dag_id=dag_id, include_prior_dates=include_prior_dates) or 0.0,
        best_reward=ti.xcom_pull(key=L3XComKeysEnum.BEST_REWARD.value, dag_id=dag_id, include_prior_dates=include_prior_dates) or 0.0,
        norm_stats_hash=ti.xcom_pull(key=L3XComKeysEnum.NORM_STATS_HASH.value, dag_id=dag_id, include_prior_dates=include_prior_dates) or "",
        mlflow_experiment_id=ti.xcom_pull(key=L3XComKeysEnum.MLFLOW_EXPERIMENT_ID.value, dag_id=dag_id, include_prior_dates=include_prior_dates),
        final_metrics=ti.xcom_pull(key=L3XComKeysEnum.FINAL_METRICS.value, dag_id=dag_id, include_prior_dates=include_prior_dates),
        config_hash=ti.xcom_pull(key=L3XComKeysEnum.CONFIG_HASH.value, dag_id=dag_id, include_prior_dates=include_prior_dates),
        dvc_tag=ti.xcom_pull(key=L3XComKeysEnum.DVC_TAG.value, dag_id=dag_id, include_prior_dates=include_prior_dates),
        # Reward system fields
        reward_config_uri=ti.xcom_pull(key=L3XComKeysEnum.REWARD_CONFIG_URI.value, dag_id=dag_id, include_prior_dates=include_prior_dates),
        reward_contract_id=ti.xcom_pull(key=L3XComKeysEnum.REWARD_CONTRACT_ID.value, dag_id=dag_id, include_prior_dates=include_prior_dates),
        reward_config_hash=ti.xcom_pull(key=L3XComKeysEnum.REWARD_CONFIG_HASH.value, dag_id=dag_id, include_prior_dates=include_prior_dates),
        curriculum_final_phase=ti.xcom_pull(key=L3XComKeysEnum.CURRICULUM_FINAL_PHASE.value, dag_id=dag_id, include_prior_dates=include_prior_dates),
        reward_weights=ti.xcom_pull(key=L3XComKeysEnum.REWARD_WEIGHTS.value, dag_id=dag_id, include_prior_dates=include_prior_dates),
    )


def pull_l0_output(
    ti,
    dag_id: str = L0_MACRO_DAG_ID,
    include_prior_dates: bool = True
) -> Optional[L0Output]:
    """
    Pull L0 output from XCom.

    Args:
        ti: TaskInstance
        dag_id: DAG ID to pull from
        include_prior_dates: Whether to include prior dates in search

    Returns:
        L0Output dataclass or None if not found
    """
    full_output = ti.xcom_pull(
        key=L0XComKeysEnum.FULL_OUTPUT.value,
        dag_id=dag_id,
        include_prior_dates=include_prior_dates,
    )

    if full_output:
        return L0Output.from_dict(full_output)

    # Fallback: Try individual keys
    is_ready = ti.xcom_pull(
        key=L0XComKeysEnum.IS_READY_FOR_INFERENCE.value,
        dag_id=dag_id,
        include_prior_dates=include_prior_dates,
    )

    if is_ready is None:
        return None

    return L0Output(
        is_ready_for_inference=is_ready,
        readiness_report=ti.xcom_pull(key=L0XComKeysEnum.READINESS_REPORT.value, dag_id=dag_id, include_prior_dates=include_prior_dates),
    )


def pull_l1_output(
    ti,
    dag_id: str = L1_DAG_ID,
    include_prior_dates: bool = True
) -> Optional[L1Output]:
    """
    Pull L1 output from XCom.

    Args:
        ti: TaskInstance
        dag_id: DAG ID to pull from
        include_prior_dates: Whether to include prior dates in search

    Returns:
        L1Output dataclass or None if not found
    """
    full_output = ti.xcom_pull(
        key=L1XComKeysEnum.FULL_OUTPUT.value,
        dag_id=dag_id,
        include_prior_dates=include_prior_dates,
    )

    if full_output:
        return L1Output.from_dict(full_output)

    # Fallback: Try individual keys
    features_count = ti.xcom_pull(
        key=L1XComKeysEnum.FEATURES_COUNT.value,
        dag_id=dag_id,
        include_prior_dates=include_prior_dates,
    )

    if features_count is None:
        return None

    return L1Output(
        features_count=features_count,
        macro_rows_used=ti.xcom_pull(key=L1XComKeysEnum.MACRO_ROWS_USED.value, dag_id=dag_id, include_prior_dates=include_prior_dates) or 0,
        builder_version=ti.xcom_pull(key=L1XComKeysEnum.BUILDER_VERSION.value, dag_id=dag_id, include_prior_dates=include_prior_dates) or "",
        feature_order_hash=ti.xcom_pull(key=L1XComKeysEnum.FEATURE_ORDER_HASH.value, dag_id=dag_id, include_prior_dates=include_prior_dates) or "",
        norm_stats_hash=ti.xcom_pull(key=L1XComKeysEnum.NORM_STATS_HASH.value, dag_id=dag_id, include_prior_dates=include_prior_dates) or "",
    )


def pull_l4_output(
    ti,
    dag_id: str = L4_DAG_ID,
    include_prior_dates: bool = True
) -> Optional[L4Output]:
    """
    Pull L4 output from XCom.

    Args:
        ti: TaskInstance
        dag_id: DAG ID to pull from
        include_prior_dates: Whether to include prior dates in search

    Returns:
        L4Output dataclass or None if not found
    """
    full_output = ti.xcom_pull(
        key=L4XComKeysEnum.FULL_OUTPUT.value,
        dag_id=dag_id,
        include_prior_dates=include_prior_dates,
    )

    if full_output:
        return L4Output.from_dict(full_output)

    # Fallback: Try individual keys
    experiment_result = ti.xcom_pull(
        key=L4XComKeysEnum.EXPERIMENT_RESULT.value,
        dag_id=dag_id,
        include_prior_dates=include_prior_dates,
    )

    if experiment_result is None:
        return None

    return L4Output(
        experiment_result=experiment_result,
        comparison_result=ti.xcom_pull(key=L4XComKeysEnum.COMPARISON_RESULT.value, dag_id=dag_id, include_prior_dates=include_prior_dates),
        registry_path=ti.xcom_pull(key=L4XComKeysEnum.REGISTRY_PATH.value, dag_id=dag_id, include_prior_dates=include_prior_dates),
        lineage_record=ti.xcom_pull(key=L4XComKeysEnum.LINEAGE_RECORD.value, dag_id=dag_id, include_prior_dates=include_prior_dates),
        notification_sent=ti.xcom_pull(key=L4XComKeysEnum.NOTIFICATION_SENT.value, dag_id=dag_id, include_prior_dates=include_prior_dates) or False,
    )


def pull_l5_output(
    ti,
    dag_id: str = L5_DAG_ID,
    include_prior_dates: bool = True
) -> Optional[L5Output]:
    """
    Pull L5 output from XCom.

    Args:
        ti: TaskInstance
        dag_id: DAG ID to pull from
        include_prior_dates: Whether to include prior dates in search

    Returns:
        L5Output dataclass or None if not found
    """
    full_output = ti.xcom_pull(
        key=L5XComKeysEnum.FULL_OUTPUT.value,
        dag_id=dag_id,
        include_prior_dates=include_prior_dates,
    )

    if full_output:
        return L5Output.from_dict(full_output)

    # Fallback: Try individual keys
    model_id = ti.xcom_pull(
        key=L5XComKeysEnum.MODEL_ID.value,
        dag_id=dag_id,
        include_prior_dates=include_prior_dates,
    )

    if model_id is None:
        return None

    return L5Output(
        model_id=model_id,
        signal=ti.xcom_pull(key=L5XComKeysEnum.SIGNAL.value, dag_id=dag_id, include_prior_dates=include_prior_dates) or 0.0,
        confidence=ti.xcom_pull(key=L5XComKeysEnum.CONFIDENCE.value, dag_id=dag_id, include_prior_dates=include_prior_dates) or 0.0,
        execution_price=ti.xcom_pull(key=L5XComKeysEnum.EXECUTION_PRICE.value, dag_id=dag_id, include_prior_dates=include_prior_dates) or 0.0,
        position=ti.xcom_pull(key=L5XComKeysEnum.POSITION.value, dag_id=dag_id, include_prior_dates=include_prior_dates) or 0.0,
        inference_latency_ms=ti.xcom_pull(key=L5XComKeysEnum.INFERENCE_LATENCY_MS.value, dag_id=dag_id, include_prior_dates=include_prior_dates) or 0.0,
        feature_hash=ti.xcom_pull(key=L5XComKeysEnum.FEATURE_HASH.value, dag_id=dag_id, include_prior_dates=include_prior_dates) or "",
        is_challenger=ti.xcom_pull(key=L5XComKeysEnum.IS_CHALLENGER.value, dag_id=dag_id, include_prior_dates=include_prior_dates) or False,
        canary_stage=ti.xcom_pull(key=L5XComKeysEnum.CANARY_STAGE.value, dag_id=dag_id, include_prior_dates=include_prior_dates) or "production",
        inference_result=ti.xcom_pull(key=L5XComKeysEnum.INFERENCE_RESULT.value, dag_id=dag_id, include_prior_dates=include_prior_dates),
    )


def pull_l6_output(
    ti,
    dag_id: str = L6_DAG_ID,
    include_prior_dates: bool = True
) -> Optional[L6Output]:
    """
    Pull L6 output from XCom.

    Args:
        ti: TaskInstance
        dag_id: DAG ID to pull from
        include_prior_dates: Whether to include prior dates in search

    Returns:
        L6Output dataclass or None if not found
    """
    full_output = ti.xcom_pull(
        key=L6XComKeysEnum.FULL_OUTPUT.value,
        dag_id=dag_id,
        include_prior_dates=include_prior_dates,
    )

    if full_output:
        return L6Output.from_dict(full_output)

    # Fallback: Try individual keys
    drift_detected = ti.xcom_pull(
        key=L6XComKeysEnum.DRIFT_DETECTED.value,
        dag_id=dag_id,
        include_prior_dates=include_prior_dates,
    )

    if drift_detected is None:
        return None

    return L6Output(
        drift_detected=drift_detected,
        drift_score=ti.xcom_pull(key=L6XComKeysEnum.DRIFT_SCORE.value, dag_id=dag_id, include_prior_dates=include_prior_dates) or 0.0,
        alert_triggered=ti.xcom_pull(key=L6XComKeysEnum.ALERT_TRIGGERED.value, dag_id=dag_id, include_prior_dates=include_prior_dates) or False,
        retraining_recommended=ti.xcom_pull(key=L6XComKeysEnum.RETRAINING_RECOMMENDED.value, dag_id=dag_id, include_prior_dates=include_prior_dates) or False,
        model_performance=ti.xcom_pull(key=L6XComKeysEnum.MODEL_PERFORMANCE.value, dag_id=dag_id, include_prior_dates=include_prior_dates),
        metrics_summary=ti.xcom_pull(key=L6XComKeysEnum.METRICS_SUMMARY.value, dag_id=dag_id, include_prior_dates=include_prior_dates),
        monitoring_report=ti.xcom_pull(key=L6XComKeysEnum.MONITORING_REPORT.value, dag_id=dag_id, include_prior_dates=include_prior_dates),
    )


def compute_file_hash(file_path: str, algorithm: str = 'sha256', length: int = 16) -> str:
    """
    Compute hash of a file for lineage tracking.

    SSOT: Delegates to src.utils.hash_utils.compute_file_hash

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (sha256, md5)
        length: Length of hash to return

    Returns:
        Truncated hash string
    """
    result = _compute_file_hash_canonical(file_path, algorithm=algorithm)
    return result.full_hash[:length]


def compute_config_hash(config: Dict[str, Any], length: int = 16) -> str:
    """
    Compute hash of a configuration dictionary.

    SSOT: Uses canonical JSON hashing pattern

    Args:
        config: Configuration dictionary
        length: Length of hash to return

    Returns:
        Truncated hash string
    """
    import hashlib
    config_str = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(config_str.encode()).hexdigest()[:length]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enum keys
    "L0XComKeysEnum",
    "L1XComKeysEnum",
    "L2XComKeysEnum",
    "L3XComKeysEnum",
    "L4XComKeysEnum",
    "L5XComKeysEnum",
    "L6XComKeysEnum",
    # Layer 0 DAG IDs
    "L0_MACRO_DAG_ID",
    "L0_OHLCV_REALTIME_DAG_ID",
    "L0_OHLCV_BACKFILL_DAG_ID",
    "L0_WEEKLY_BACKUP_DAG_ID",
    "L0_DATA_INIT_DAG_ID",
    # Layer 1 DAG IDs
    "L1_DAG_ID",
    "L1_FEAST_DAG_ID",
    # Layer 2 DAG IDs
    "L2_DAG_ID",
    "L2_DRIFT_DAG_ID",
    # Layer 3 DAG IDs
    "L3_DAG_ID",
    # Layer 4 DAG IDs
    "L4_DAG_ID",
    "L4_BACKTEST_DAG_ID",
    "L4_SCHEDULED_DAG_ID",
    # Layer 5 DAG IDs
    "L5_DAG_ID",
    # Layer 6 DAG IDs
    "L6_DAG_ID",
    # Output dataclasses
    "L0Output",
    "L1Output",
    "L2Output",
    "L3Output",
    "L4Output",
    "L5Output",
    "L6Output",
    # Helper functions
    "pull_l0_output",
    "pull_l1_output",
    "pull_l2_output",
    "pull_l3_output",
    "pull_l4_output",
    "pull_l5_output",
    "pull_l6_output",
    "compute_file_hash",
    "compute_config_hash",
    # SSOT re-exports
    "HashResult",
]
