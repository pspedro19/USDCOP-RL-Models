"""
Storage Contracts for MinIO-First ML Pipeline
==============================================

Defines immutable dataclasses for dataset, model, backtest, and A/B comparison snapshots.
These serve as SSOT for artifact state and enable lineage tracking.

Key Design Principles:
1. All dataclasses are frozen=True (immutable)
2. All URIs use s3:// scheme
3. All hashes are content-based (SHA256, truncated to 16 chars)
4. Snapshots are XCom-serializable

Contract: CTR-STORAGE-CONTRACTS-001

Author: Trading Team
Version: 1.0.0
Created: 2026-01-18
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union
import hashlib
import json


# =============================================================================
# CONSTANTS
# =============================================================================

STORAGE_CONTRACT_VERSION = "1.0.0"
EXPERIMENTS_BUCKET = "experiments"
PRODUCTION_BUCKET = "production"
DVC_BUCKET = "dvc-storage"


# =============================================================================
# LINEAGE RECORD
# =============================================================================


@dataclass(frozen=True)
class LineageRecord:
    """
    Immutable record of artifact lineage.

    Tracks the chain of transformations from raw data to deployed model.
    Used for reproducibility and audit trail.
    """
    artifact_type: str  # 'dataset', 'model', 'backtest', 'comparison'
    artifact_id: str
    parent_id: Optional[str]  # Previous artifact in chain
    parent_type: Optional[str]

    # Source information
    source_uri: Optional[str]  # s3:// URI of source
    source_hash: Optional[str]

    # Transformation metadata
    transform_name: Optional[str]  # e.g., 'l2_preprocessing', 'l3_training'
    transform_params: Optional[Tuple[Tuple[str, str], ...]]  # Frozen dict as tuples

    # Temporal context
    created_at: datetime
    created_by: str  # User or DAG that created this

    # Git/DVC tracking
    git_commit: Optional[str] = None
    dvc_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "artifact_type": self.artifact_type,
            "artifact_id": self.artifact_id,
            "parent_id": self.parent_id,
            "parent_type": self.parent_type,
            "source_uri": self.source_uri,
            "source_hash": self.source_hash,
            "transform_name": self.transform_name,
            "transform_params": dict(self.transform_params) if self.transform_params else None,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "git_commit": self.git_commit,
            "dvc_version": self.dvc_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LineageRecord":
        """Create from dictionary."""
        transform_params = data.get("transform_params")
        if transform_params and isinstance(transform_params, dict):
            transform_params = tuple(transform_params.items())

        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            artifact_type=data["artifact_type"],
            artifact_id=data["artifact_id"],
            parent_id=data.get("parent_id"),
            parent_type=data.get("parent_type"),
            source_uri=data.get("source_uri"),
            source_hash=data.get("source_hash"),
            transform_name=data.get("transform_name"),
            transform_params=transform_params,
            created_at=created_at or datetime.utcnow(),
            created_by=data.get("created_by", "unknown"),
            git_commit=data.get("git_commit"),
            dvc_version=data.get("dvc_version"),
        )


# =============================================================================
# DATASET SNAPSHOT
# =============================================================================


@dataclass(frozen=True)
class DatasetSnapshot:
    """
    Immutable dataset version - SSOT for dataset state.

    Represents a complete dataset at a point in time, including:
    - Storage URIs
    - Integrity hashes
    - Schema information
    - Lineage tracking

    Contract: CTR-DATASET-SNAPSHOT-001
    """
    experiment_id: str
    version: str

    # Storage URIs (s3://experiments/{exp}/datasets/{ver}/...)
    storage_uri: str           # train.parquet
    norm_stats_uri: str        # norm_stats.json
    manifest_uri: str          # manifest.json

    # Integrity hashes (SHA256, 16 chars)
    data_hash: str
    schema_hash: str
    norm_stats_hash: str

    # Dataset dimensions
    row_count: int
    size_bytes: int

    # Feature schema
    feature_columns: Tuple[str, ...]  # Frozen list as tuple
    feature_order_hash: str

    # Temporal range
    date_range_start: str
    date_range_end: str

    # Lineage
    parent_version: Optional[str]
    lineage: Optional[LineageRecord]
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary (for XCom)."""
        return {
            "experiment_id": self.experiment_id,
            "version": self.version,
            "storage_uri": self.storage_uri,
            "norm_stats_uri": self.norm_stats_uri,
            "manifest_uri": self.manifest_uri,
            "data_hash": self.data_hash,
            "schema_hash": self.schema_hash,
            "norm_stats_hash": self.norm_stats_hash,
            "row_count": self.row_count,
            "size_bytes": self.size_bytes,
            "feature_columns": list(self.feature_columns),
            "feature_order_hash": self.feature_order_hash,
            "date_range_start": self.date_range_start,
            "date_range_end": self.date_range_end,
            "parent_version": self.parent_version,
            "lineage": self.lineage.to_dict() if self.lineage else None,
            "created_at": self.created_at.isoformat(),
        }

    def to_xcom_dict(self) -> Dict[str, Any]:
        """
        Serialize for Airflow XCom (minimal for quick access).

        Only includes essential fields needed by downstream DAGs.
        """
        return {
            "experiment_id": self.experiment_id,
            "version": self.version,
            "storage_uri": self.storage_uri,
            "data_hash": self.data_hash,
            "row_count": self.row_count,
            "feature_order_hash": self.feature_order_hash,
            "norm_stats_uri": self.norm_stats_uri,
            "date_range_start": self.date_range_start,
            "date_range_end": self.date_range_end,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetSnapshot":
        """Create from dictionary."""
        feature_columns = data.get("feature_columns", [])
        if isinstance(feature_columns, list):
            feature_columns = tuple(feature_columns)

        lineage_data = data.get("lineage")
        lineage = LineageRecord.from_dict(lineage_data) if lineage_data else None

        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            experiment_id=data["experiment_id"],
            version=data["version"],
            storage_uri=data["storage_uri"],
            norm_stats_uri=data["norm_stats_uri"],
            manifest_uri=data["manifest_uri"],
            data_hash=data["data_hash"],
            schema_hash=data["schema_hash"],
            norm_stats_hash=data.get("norm_stats_hash", ""),
            row_count=data["row_count"],
            size_bytes=data.get("size_bytes", 0),
            feature_columns=feature_columns,
            feature_order_hash=data["feature_order_hash"],
            date_range_start=data["date_range_start"],
            date_range_end=data["date_range_end"],
            parent_version=data.get("parent_version"),
            lineage=lineage,
            created_at=created_at or datetime.utcnow(),
        )


# =============================================================================
# MODEL SNAPSHOT
# =============================================================================


@dataclass(frozen=True)
class ModelSnapshot:
    """
    Immutable model version - SSOT for model state.

    Represents a trained model at a point in time, including:
    - Storage URIs
    - Integrity hashes
    - Contract compliance
    - Performance metrics

    Contract: CTR-MODEL-SNAPSHOT-001
    """
    experiment_id: str
    version: str

    # Storage URIs (s3://experiments/{exp}/models/{ver}/...)
    storage_uri: str           # policy.onnx or policy.zip
    norm_stats_uri: str        # norm_stats.json
    config_uri: str            # config.yaml
    lineage_uri: str           # lineage.json

    # Integrity hashes (SHA256, 16 chars)
    model_hash: str
    norm_stats_hash: str
    config_hash: str

    # Contract compliance
    observation_dim: int
    action_space: int
    feature_order_hash: str
    feature_order: Tuple[str, ...]  # Actual feature order for validation

    # Performance metrics (from backtest)
    test_sharpe: Optional[float]
    test_max_drawdown: Optional[float]
    test_win_rate: Optional[float]
    test_total_return: Optional[float]

    # Training metadata
    training_duration_seconds: float
    mlflow_run_id: Optional[str]
    best_reward: Optional[float]

    # Lineage - link to dataset
    dataset_snapshot: Optional[DatasetSnapshot]
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "version": self.version,
            "storage_uri": self.storage_uri,
            "norm_stats_uri": self.norm_stats_uri,
            "config_uri": self.config_uri,
            "lineage_uri": self.lineage_uri,
            "model_hash": self.model_hash,
            "norm_stats_hash": self.norm_stats_hash,
            "config_hash": self.config_hash,
            "observation_dim": self.observation_dim,
            "action_space": self.action_space,
            "feature_order_hash": self.feature_order_hash,
            "feature_order": list(self.feature_order),
            "test_sharpe": self.test_sharpe,
            "test_max_drawdown": self.test_max_drawdown,
            "test_win_rate": self.test_win_rate,
            "test_total_return": self.test_total_return,
            "training_duration_seconds": self.training_duration_seconds,
            "mlflow_run_id": self.mlflow_run_id,
            "best_reward": self.best_reward,
            "dataset_snapshot": self.dataset_snapshot.to_dict() if self.dataset_snapshot else None,
            "created_at": self.created_at.isoformat(),
        }

    def to_xcom_dict(self) -> Dict[str, Any]:
        """
        Serialize for Airflow XCom (minimal for quick access).
        """
        return {
            "experiment_id": self.experiment_id,
            "version": self.version,
            "storage_uri": self.storage_uri,
            "model_hash": self.model_hash,
            "norm_stats_hash": self.norm_stats_hash,
            "feature_order_hash": self.feature_order_hash,
            "observation_dim": self.observation_dim,
            "mlflow_run_id": self.mlflow_run_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelSnapshot":
        """Create from dictionary."""
        feature_order = data.get("feature_order", [])
        if isinstance(feature_order, list):
            feature_order = tuple(feature_order)

        dataset_data = data.get("dataset_snapshot")
        dataset_snapshot = DatasetSnapshot.from_dict(dataset_data) if dataset_data else None

        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            experiment_id=data["experiment_id"],
            version=data["version"],
            storage_uri=data["storage_uri"],
            norm_stats_uri=data["norm_stats_uri"],
            config_uri=data["config_uri"],
            lineage_uri=data["lineage_uri"],
            model_hash=data["model_hash"],
            norm_stats_hash=data["norm_stats_hash"],
            config_hash=data["config_hash"],
            observation_dim=data["observation_dim"],
            action_space=data["action_space"],
            feature_order_hash=data["feature_order_hash"],
            feature_order=feature_order,
            test_sharpe=data.get("test_sharpe"),
            test_max_drawdown=data.get("test_max_drawdown"),
            test_win_rate=data.get("test_win_rate"),
            test_total_return=data.get("test_total_return"),
            training_duration_seconds=data.get("training_duration_seconds", 0.0),
            mlflow_run_id=data.get("mlflow_run_id"),
            best_reward=data.get("best_reward"),
            dataset_snapshot=dataset_snapshot,
            created_at=created_at or datetime.utcnow(),
        )


# =============================================================================
# BACKTEST SNAPSHOT
# =============================================================================


@dataclass(frozen=True)
class BacktestSnapshot:
    """
    Immutable backtest result.

    Represents a complete backtest run, including:
    - Performance metrics
    - Trade log
    - Equity curve

    Contract: CTR-BACKTEST-SNAPSHOT-001
    """
    experiment_id: str
    model_version: str
    backtest_id: str

    # Storage URIs (s3://experiments/{exp}/backtests/{id}/...)
    storage_uri: str           # result.json
    trades_uri: str            # trades.parquet
    equity_curve_uri: str      # equity_curve.parquet

    # Performance metrics
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: Optional[float]
    avg_trade_return: Optional[float]

    # Integrity
    result_hash: str

    # Temporal context
    backtest_start: str
    backtest_end: str
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "model_version": self.model_version,
            "backtest_id": self.backtest_id,
            "storage_uri": self.storage_uri,
            "trades_uri": self.trades_uri,
            "equity_curve_uri": self.equity_curve_uri,
            "sharpe_ratio": self.sharpe_ratio,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "profit_factor": self.profit_factor,
            "avg_trade_return": self.avg_trade_return,
            "result_hash": self.result_hash,
            "backtest_start": self.backtest_start,
            "backtest_end": self.backtest_end,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BacktestSnapshot":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            experiment_id=data["experiment_id"],
            model_version=data["model_version"],
            backtest_id=data["backtest_id"],
            storage_uri=data["storage_uri"],
            trades_uri=data["trades_uri"],
            equity_curve_uri=data["equity_curve_uri"],
            sharpe_ratio=data["sharpe_ratio"],
            total_return=data["total_return"],
            max_drawdown=data["max_drawdown"],
            win_rate=data["win_rate"],
            total_trades=data["total_trades"],
            profit_factor=data.get("profit_factor"),
            avg_trade_return=data.get("avg_trade_return"),
            result_hash=data["result_hash"],
            backtest_start=data["backtest_start"],
            backtest_end=data["backtest_end"],
            created_at=created_at or datetime.utcnow(),
        )


# =============================================================================
# REWARD CONFIG SNAPSHOT
# =============================================================================


@dataclass(frozen=True)
class RewardConfigSnapshot:
    """
    Immutable reward configuration snapshot.

    Tracks the exact reward configuration used during training.
    Essential for reproducibility and comparing experiments.

    Contract: CTR-REWARD-SNAPSHOT-001
    """
    experiment_id: str
    version: str

    # Storage URI
    storage_uri: str  # s3://experiments/{exp}/reward_config.json

    # Contract reference
    reward_contract_id: str
    reward_config_hash: str

    # Component weights
    weight_pnl: float
    weight_dsr: float
    weight_sortino: float
    weight_regime_penalty: float
    weight_holding_decay: float
    weight_anti_gaming: float

    # Enabled flags
    enable_normalization: bool
    enable_curriculum: bool
    enable_banrep_detection: bool
    enable_oil_tracking: bool

    # Curriculum settings (if enabled)
    curriculum_phase_1_steps: int
    curriculum_phase_2_steps: int
    curriculum_phase_3_steps: int

    # Full config (frozen as tuple of tuples for immutability)
    component_configs: Tuple[Tuple[str, str], ...]  # Name -> JSON serialized config

    # Metadata
    created_at: datetime

    @property
    def enabled_components(self) -> Tuple[str, ...]:
        """Get names of enabled components."""
        enabled = []
        if self.weight_pnl > 0:
            enabled.append("pnl")
        if self.weight_dsr > 0:
            enabled.append("dsr")
        if self.weight_sortino > 0:
            enabled.append("sortino")
        if self.weight_regime_penalty > 0:
            enabled.append("regime_penalty")
        if self.weight_holding_decay > 0:
            enabled.append("holding_decay")
        if self.weight_anti_gaming > 0:
            enabled.append("anti_gaming")
        if self.enable_banrep_detection:
            enabled.append("banrep")
        if self.enable_oil_tracking:
            enabled.append("oil")
        if self.enable_normalization:
            enabled.append("normalizer")
        return tuple(enabled)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "version": self.version,
            "storage_uri": self.storage_uri,
            "reward_contract_id": self.reward_contract_id,
            "reward_config_hash": self.reward_config_hash,
            "weight_pnl": self.weight_pnl,
            "weight_dsr": self.weight_dsr,
            "weight_sortino": self.weight_sortino,
            "weight_regime_penalty": self.weight_regime_penalty,
            "weight_holding_decay": self.weight_holding_decay,
            "weight_anti_gaming": self.weight_anti_gaming,
            "enable_normalization": self.enable_normalization,
            "enable_curriculum": self.enable_curriculum,
            "enable_banrep_detection": self.enable_banrep_detection,
            "enable_oil_tracking": self.enable_oil_tracking,
            "curriculum_phase_1_steps": self.curriculum_phase_1_steps,
            "curriculum_phase_2_steps": self.curriculum_phase_2_steps,
            "curriculum_phase_3_steps": self.curriculum_phase_3_steps,
            "component_configs": dict(self.component_configs),
            "enabled_components": list(self.enabled_components),
            "created_at": self.created_at.isoformat(),
        }

    def to_xcom_dict(self) -> Dict[str, Any]:
        """Minimal serialization for Airflow XCom."""
        return {
            "experiment_id": self.experiment_id,
            "version": self.version,
            "reward_contract_id": self.reward_contract_id,
            "reward_config_hash": self.reward_config_hash,
            "storage_uri": self.storage_uri,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RewardConfigSnapshot":
        """Create from dictionary."""
        component_configs = data.get("component_configs", {})
        if isinstance(component_configs, dict):
            component_configs = tuple(component_configs.items())

        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            experiment_id=data["experiment_id"],
            version=data["version"],
            storage_uri=data["storage_uri"],
            reward_contract_id=data["reward_contract_id"],
            reward_config_hash=data["reward_config_hash"],
            weight_pnl=data["weight_pnl"],
            weight_dsr=data["weight_dsr"],
            weight_sortino=data["weight_sortino"],
            weight_regime_penalty=data["weight_regime_penalty"],
            weight_holding_decay=data["weight_holding_decay"],
            weight_anti_gaming=data["weight_anti_gaming"],
            enable_normalization=data["enable_normalization"],
            enable_curriculum=data["enable_curriculum"],
            enable_banrep_detection=data["enable_banrep_detection"],
            enable_oil_tracking=data["enable_oil_tracking"],
            curriculum_phase_1_steps=data.get("curriculum_phase_1_steps", 100000),
            curriculum_phase_2_steps=data.get("curriculum_phase_2_steps", 200000),
            curriculum_phase_3_steps=data.get("curriculum_phase_3_steps", 300000),
            component_configs=component_configs,
            created_at=created_at or datetime.utcnow(),
        )

    @classmethod
    def from_reward_config(
        cls,
        experiment_id: str,
        version: str,
        storage_uri: str,
        reward_contract_id: str,
        reward_config: "RewardConfig",  # type: ignore
    ) -> "RewardConfigSnapshot":
        """
        Create snapshot from a RewardConfig instance.

        Args:
            experiment_id: Experiment identifier
            version: Version string
            storage_uri: S3 URI where config is stored
            reward_contract_id: Contract ID being used
            reward_config: RewardConfig instance

        Returns:
            Immutable RewardConfigSnapshot
        """
        # Serialize component configs
        component_configs = tuple([
            ("dsr", json.dumps(reward_config.dsr.to_dict())),
            ("sortino", json.dumps(reward_config.sortino.to_dict())),
            ("regime", json.dumps(reward_config.regime.to_dict())),
            ("market_impact", json.dumps(reward_config.market_impact.to_dict())),
            ("holding_decay", json.dumps(reward_config.holding_decay.to_dict())),
            ("anti_gaming", json.dumps(reward_config.anti_gaming.to_dict())),
            ("normalizer", json.dumps(reward_config.normalizer.to_dict())),
            ("banrep", json.dumps(reward_config.banrep.to_dict())),
            ("oil_correlation", json.dumps(reward_config.oil_correlation.to_dict())),
            ("pnl_transform", json.dumps(reward_config.pnl_transform.to_dict())),
            ("curriculum", json.dumps(reward_config.curriculum.to_dict())),
        ])

        return cls(
            experiment_id=experiment_id,
            version=version,
            storage_uri=storage_uri,
            reward_contract_id=reward_contract_id,
            reward_config_hash=reward_config.to_hash(),
            weight_pnl=reward_config.weight_pnl,
            weight_dsr=reward_config.weight_dsr,
            weight_sortino=reward_config.weight_sortino,
            weight_regime_penalty=reward_config.weight_regime_penalty,
            weight_holding_decay=reward_config.weight_holding_decay,
            weight_anti_gaming=reward_config.weight_anti_gaming,
            enable_normalization=reward_config.enable_normalization,
            enable_curriculum=reward_config.enable_curriculum,
            enable_banrep_detection=reward_config.enable_banrep_detection,
            enable_oil_tracking=reward_config.enable_oil_tracking,
            curriculum_phase_1_steps=reward_config.curriculum.phase_1_steps,
            curriculum_phase_2_steps=reward_config.curriculum.phase_2_steps,
            curriculum_phase_3_steps=reward_config.curriculum.phase_3_steps,
            component_configs=component_configs,
            created_at=datetime.utcnow(),
        )


# =============================================================================
# A/B COMPARISON SNAPSHOT
# =============================================================================


@dataclass(frozen=True)
class ABComparisonSnapshot:
    """
    Immutable A/B comparison result.

    Represents a statistical comparison between two models.

    Contract: CTR-AB-SNAPSHOT-001
    """
    comparison_id: str
    experiment_id: str

    # Storage URI
    storage_uri: str           # ab_result.json
    shadow_trades_uri: Optional[str]  # shadow_trades.parquet

    # Models compared
    baseline_experiment_id: str
    baseline_version: str
    baseline_model_hash: str

    treatment_experiment_id: str
    treatment_version: str
    treatment_model_hash: str

    # Statistical results
    primary_metric: str        # e.g., 'sharpe_ratio'
    baseline_value: float
    treatment_value: float
    p_value: float
    effect_size: float
    confidence_interval_low: float
    confidence_interval_high: float

    # Decision
    is_significant: bool
    confidence_level: float    # e.g., 0.95
    recommendation: str        # 'deploy_treatment', 'keep_baseline', 'inconclusive'

    # Sample info
    baseline_trades: int
    treatment_trades: int
    comparison_duration_hours: float

    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "comparison_id": self.comparison_id,
            "experiment_id": self.experiment_id,
            "storage_uri": self.storage_uri,
            "shadow_trades_uri": self.shadow_trades_uri,
            "baseline_experiment_id": self.baseline_experiment_id,
            "baseline_version": self.baseline_version,
            "baseline_model_hash": self.baseline_model_hash,
            "treatment_experiment_id": self.treatment_experiment_id,
            "treatment_version": self.treatment_version,
            "treatment_model_hash": self.treatment_model_hash,
            "primary_metric": self.primary_metric,
            "baseline_value": self.baseline_value,
            "treatment_value": self.treatment_value,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "confidence_interval_low": self.confidence_interval_low,
            "confidence_interval_high": self.confidence_interval_high,
            "is_significant": self.is_significant,
            "confidence_level": self.confidence_level,
            "recommendation": self.recommendation,
            "baseline_trades": self.baseline_trades,
            "treatment_trades": self.treatment_trades,
            "comparison_duration_hours": self.comparison_duration_hours,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ABComparisonSnapshot":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            comparison_id=data["comparison_id"],
            experiment_id=data["experiment_id"],
            storage_uri=data["storage_uri"],
            shadow_trades_uri=data.get("shadow_trades_uri"),
            baseline_experiment_id=data["baseline_experiment_id"],
            baseline_version=data["baseline_version"],
            baseline_model_hash=data["baseline_model_hash"],
            treatment_experiment_id=data["treatment_experiment_id"],
            treatment_version=data["treatment_version"],
            treatment_model_hash=data["treatment_model_hash"],
            primary_metric=data["primary_metric"],
            baseline_value=data["baseline_value"],
            treatment_value=data["treatment_value"],
            p_value=data["p_value"],
            effect_size=data["effect_size"],
            confidence_interval_low=data["confidence_interval_low"],
            confidence_interval_high=data["confidence_interval_high"],
            is_significant=data["is_significant"],
            confidence_level=data["confidence_level"],
            recommendation=data["recommendation"],
            baseline_trades=data["baseline_trades"],
            treatment_trades=data["treatment_trades"],
            comparison_duration_hours=data["comparison_duration_hours"],
            created_at=created_at or datetime.utcnow(),
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def compute_content_hash(content: bytes) -> str:
    """
    Compute SHA256 hash of content, truncated to 16 chars.

    Args:
        content: Raw bytes to hash

    Returns:
        Truncated SHA256 hash (16 characters)
    """
    return hashlib.sha256(content).hexdigest()[:16]


def compute_schema_hash(columns: List[str]) -> str:
    """
    Compute hash of column schema.

    Args:
        columns: List of column names in order

    Returns:
        Truncated SHA256 hash (16 characters)
    """
    schema_str = ",".join(columns)
    return hashlib.sha256(schema_str.encode()).hexdigest()[:16]


def compute_json_hash(data: Dict[str, Any]) -> str:
    """
    Compute hash of JSON-serializable data.

    Uses canonical JSON (sorted keys, minimal separators) for consistency.

    Args:
        data: Dictionary to hash

    Returns:
        Truncated SHA256 hash (16 characters)
    """
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """
    Parse S3 URI into bucket and key.

    Args:
        uri: S3 URI (s3://bucket/path/to/key)

    Returns:
        Tuple of (bucket, key)

    Raises:
        ValueError: If URI is not valid S3 format
    """
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")

    path = uri[5:]  # Remove 's3://'
    parts = path.split("/", 1)

    if len(parts) < 2:
        raise ValueError(f"Invalid S3 URI (no key): {uri}")

    return parts[0], parts[1]


def build_s3_uri(bucket: str, key: str) -> str:
    """
    Build S3 URI from bucket and key.

    Args:
        bucket: Bucket name
        key: Object key

    Returns:
        S3 URI (s3://bucket/key)
    """
    return f"s3://{bucket}/{key}"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "STORAGE_CONTRACT_VERSION",
    "EXPERIMENTS_BUCKET",
    "PRODUCTION_BUCKET",
    "DVC_BUCKET",
    # Dataclasses
    "LineageRecord",
    "DatasetSnapshot",
    "ModelSnapshot",
    "BacktestSnapshot",
    "RewardConfigSnapshot",
    "ABComparisonSnapshot",
    # Helper functions
    "compute_content_hash",
    "compute_schema_hash",
    "compute_json_hash",
    "parse_s3_uri",
    "build_s3_uri",
]
