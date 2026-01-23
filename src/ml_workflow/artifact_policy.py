"""
Artifact Storage Policy (SSOT)
==============================

Defines the canonical storage locations for all ML pipeline artifacts.
This is the Single Source of Truth for artifact management.

Principle: "MLflow-First + DVC-Tracked"
- Models: MLflow Model Registry (SSOT) + MinIO backend
- Datasets: DVC versioned + MinIO storage
- Metrics: MLflow + PostgreSQL facts
- Lineage: Unified in PostgreSQL with MLflow/DVC references

@version 1.0.0
@principle MLflow-First + DVC-Tracked
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import hashlib
import json
import os
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class ArtifactType(str, Enum):
    """Types of artifacts in the ML pipeline."""
    DATASET = "dataset"
    MODEL = "model"
    METRICS = "metrics"
    CONFIG = "config"
    CHECKPOINT = "checkpoint"
    BACKTEST = "backtest"
    FORECAST = "forecast"
    LINEAGE = "lineage"


class StorageBackend(str, Enum):
    """Available storage backends."""
    MLFLOW = "mlflow"           # MLflow tracking server + artifacts
    MINIO = "minio"             # S3-compatible object storage
    POSTGRESQL = "postgresql"   # Relational database
    DVC = "dvc"                 # Data Version Control
    LOCAL = "local"             # Local filesystem (cache only)


class ModelStage(str, Enum):
    """MLflow Model Registry stages."""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


# =============================================================================
# ARTIFACT POLICY CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class ArtifactStorageRule:
    """Rule for storing a specific artifact type."""
    artifact_type: ArtifactType
    primary_backend: StorageBackend
    secondary_backend: Optional[StorageBackend] = None
    metadata_backend: StorageBackend = StorageBackend.POSTGRESQL
    versioning_backend: Optional[StorageBackend] = None
    is_mandatory: bool = True
    retention_days: int = 365


# Define the canonical storage rules
ARTIFACT_STORAGE_RULES: Dict[ArtifactType, ArtifactStorageRule] = {
    ArtifactType.DATASET: ArtifactStorageRule(
        artifact_type=ArtifactType.DATASET,
        primary_backend=StorageBackend.DVC,        # DVC is primary for datasets
        secondary_backend=StorageBackend.MINIO,    # MinIO as DVC remote
        metadata_backend=StorageBackend.POSTGRESQL,
        versioning_backend=StorageBackend.DVC,
        is_mandatory=True,
        retention_days=730,  # 2 years
    ),
    ArtifactType.MODEL: ArtifactStorageRule(
        artifact_type=ArtifactType.MODEL,
        primary_backend=StorageBackend.MLFLOW,     # MLflow is primary for models
        secondary_backend=StorageBackend.MINIO,    # MLflow uses MinIO as backend
        metadata_backend=StorageBackend.MLFLOW,    # MLflow Model Registry
        versioning_backend=StorageBackend.MLFLOW,
        is_mandatory=True,
        retention_days=365,
    ),
    ArtifactType.METRICS: ArtifactStorageRule(
        artifact_type=ArtifactType.METRICS,
        primary_backend=StorageBackend.MLFLOW,     # MLflow for tracking
        secondary_backend=StorageBackend.POSTGRESQL,  # DB for facts
        metadata_backend=StorageBackend.POSTGRESQL,
        versioning_backend=None,  # Metrics are immutable per run
        is_mandatory=True,
        retention_days=365,
    ),
    ArtifactType.CONFIG: ArtifactStorageRule(
        artifact_type=ArtifactType.CONFIG,
        primary_backend=StorageBackend.MLFLOW,     # Log as MLflow artifact
        secondary_backend=StorageBackend.DVC,      # Version with code
        metadata_backend=StorageBackend.MLFLOW,
        versioning_backend=StorageBackend.DVC,
        is_mandatory=True,
        retention_days=365,
    ),
    ArtifactType.CHECKPOINT: ArtifactStorageRule(
        artifact_type=ArtifactType.CHECKPOINT,
        primary_backend=StorageBackend.MINIO,      # Direct to MinIO
        secondary_backend=StorageBackend.LOCAL,    # Local cache
        metadata_backend=StorageBackend.MLFLOW,
        versioning_backend=None,
        is_mandatory=False,  # Checkpoints are optional
        retention_days=30,   # Short retention
    ),
    ArtifactType.BACKTEST: ArtifactStorageRule(
        artifact_type=ArtifactType.BACKTEST,
        primary_backend=StorageBackend.MINIO,
        secondary_backend=StorageBackend.POSTGRESQL,
        metadata_backend=StorageBackend.POSTGRESQL,
        versioning_backend=StorageBackend.MLFLOW,
        is_mandatory=True,
        retention_days=365,
    ),
    ArtifactType.FORECAST: ArtifactStorageRule(
        artifact_type=ArtifactType.FORECAST,
        primary_backend=StorageBackend.POSTGRESQL,  # Facts table
        secondary_backend=StorageBackend.MINIO,     # Images/charts
        metadata_backend=StorageBackend.POSTGRESQL,
        versioning_backend=None,
        is_mandatory=True,
        retention_days=365,
    ),
    ArtifactType.LINEAGE: ArtifactStorageRule(
        artifact_type=ArtifactType.LINEAGE,
        primary_backend=StorageBackend.POSTGRESQL,  # Lineage audit table
        secondary_backend=StorageBackend.MLFLOW,    # MLflow tags
        metadata_backend=StorageBackend.POSTGRESQL,
        versioning_backend=None,
        is_mandatory=True,
        retention_days=730,  # 2 years
    ),
}


# =============================================================================
# MINIO BUCKET STRUCTURE
# =============================================================================

MINIO_BUCKET_STRUCTURE = {
    "datasets": {
        "description": "DVC-managed datasets",
        "retention_days": 730,
        "versioning": True,
        "paths": {
            "raw": "datasets/raw/{symbol}/{timeframe}/",
            "processed": "datasets/processed/{version}/",
            "forecasting": "datasets/forecasting/{version}/",
        }
    },
    "models": {
        "description": "MLflow model artifacts backend",
        "retention_days": 365,
        "versioning": True,
        "paths": {
            "rl": "models/rl/{model_name}/{version}/",
            "forecasting": "models/forecasting/{model_id}/h{horizon}/{version}/",
        }
    },
    "mlflow-artifacts": {
        "description": "MLflow tracking artifacts",
        "retention_days": 365,
        "versioning": True,
        "paths": {
            "runs": "mlflow-artifacts/{experiment_id}/{run_id}/",
        }
    },
    "backtest-results": {
        "description": "Backtest reports and trades",
        "retention_days": 365,
        "versioning": False,
        "paths": {
            "reports": "backtest-results/{model_name}/{date}/",
        }
    },
    "forecasts": {
        "description": "Forecast images and reports",
        "retention_days": 365,
        "versioning": False,
        "paths": {
            "weekly": "forecasts/{year}/week{week}/",
        }
    },
}


# =============================================================================
# ARTIFACT POLICY CLASS
# =============================================================================

@dataclass
class ArtifactLocation:
    """Resolved location for an artifact."""
    artifact_type: ArtifactType
    primary_uri: str
    secondary_uri: Optional[str] = None
    metadata_uri: Optional[str] = None
    version: Optional[str] = None
    hash: Optional[str] = None


class ArtifactPolicy:
    """
    Enforces artifact storage policy across the ML pipeline.

    This class is the gatekeeper for all artifact storage operations.
    It ensures artifacts are stored according to the MLflow-First + DVC-Tracked principle.

    Usage:
        policy = ArtifactPolicy()

        # Get storage location for a model
        location = policy.get_artifact_location(
            ArtifactType.MODEL,
            model_name="ppo-usdcop",
            version="v1.0.0"
        )

        # Validate an artifact exists where it should
        is_valid = policy.validate_artifact(ArtifactType.MODEL, location)
    """

    def __init__(
        self,
        mlflow_tracking_uri: Optional[str] = None,
        minio_endpoint: Optional[str] = None,
        db_connection_string: Optional[str] = None,
    ):
        self.mlflow_tracking_uri = mlflow_tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        self.minio_endpoint = minio_endpoint or os.environ.get(
            "MINIO_ENDPOINT", "localhost:9000"
        )
        self.db_connection_string = db_connection_string or os.environ.get(
            "DATABASE_URL"
        )

        self._rules = ARTIFACT_STORAGE_RULES
        self._bucket_structure = MINIO_BUCKET_STRUCTURE

    def get_rule(self, artifact_type: ArtifactType) -> ArtifactStorageRule:
        """Get the storage rule for an artifact type."""
        return self._rules[artifact_type]

    def is_mandatory(self, artifact_type: ArtifactType) -> bool:
        """Check if storing this artifact type is mandatory."""
        return self._rules[artifact_type].is_mandatory

    def get_primary_backend(self, artifact_type: ArtifactType) -> StorageBackend:
        """Get the primary storage backend for an artifact type."""
        return self._rules[artifact_type].primary_backend

    def get_artifact_location(
        self,
        artifact_type: ArtifactType,
        **kwargs
    ) -> ArtifactLocation:
        """
        Resolve the storage location for an artifact.

        Args:
            artifact_type: Type of artifact
            **kwargs: Context-specific parameters (model_name, version, etc.)

        Returns:
            ArtifactLocation with resolved URIs
        """
        rule = self._rules[artifact_type]

        if artifact_type == ArtifactType.MODEL:
            return self._resolve_model_location(rule, **kwargs)
        elif artifact_type == ArtifactType.DATASET:
            return self._resolve_dataset_location(rule, **kwargs)
        elif artifact_type == ArtifactType.METRICS:
            return self._resolve_metrics_location(rule, **kwargs)
        else:
            return self._resolve_generic_location(rule, artifact_type, **kwargs)

    def _resolve_model_location(
        self,
        rule: ArtifactStorageRule,
        model_name: str = "model",
        version: str = "latest",
        pipeline: str = "rl",
        horizon: Optional[int] = None,
        **kwargs
    ) -> ArtifactLocation:
        """Resolve model storage location."""

        # MLflow Model Registry URI
        primary_uri = f"models:/{model_name}/{version}"

        # MinIO backend path
        if pipeline == "forecasting" and horizon:
            secondary_path = f"models/forecasting/{model_name}/h{horizon}/{version}/"
        else:
            secondary_path = f"models/{pipeline}/{model_name}/{version}/"

        secondary_uri = f"s3://models/{secondary_path}"

        # Metadata in MLflow
        metadata_uri = f"{self.mlflow_tracking_uri}/api/2.0/mlflow/registered-models/get?name={model_name}"

        return ArtifactLocation(
            artifact_type=ArtifactType.MODEL,
            primary_uri=primary_uri,
            secondary_uri=secondary_uri,
            metadata_uri=metadata_uri,
            version=version,
        )

    def _resolve_dataset_location(
        self,
        rule: ArtifactStorageRule,
        dataset_name: str = "dataset",
        version: str = "latest",
        pipeline: str = "rl",
        **kwargs
    ) -> ArtifactLocation:
        """Resolve dataset storage location."""

        # DVC path (primary)
        if pipeline == "forecasting":
            primary_uri = f"dvc://datasets/forecasting/{version}/{dataset_name}"
            secondary_path = f"datasets/forecasting/{version}/"
        else:
            primary_uri = f"dvc://datasets/processed/{version}/{dataset_name}"
            secondary_path = f"datasets/processed/{version}/"

        secondary_uri = f"s3://datasets/{secondary_path}"

        return ArtifactLocation(
            artifact_type=ArtifactType.DATASET,
            primary_uri=primary_uri,
            secondary_uri=secondary_uri,
            version=version,
        )

    def _resolve_metrics_location(
        self,
        rule: ArtifactStorageRule,
        experiment_name: str = "default",
        run_id: Optional[str] = None,
        **kwargs
    ) -> ArtifactLocation:
        """Resolve metrics storage location."""

        # MLflow run URI
        if run_id:
            primary_uri = f"{self.mlflow_tracking_uri}/#/experiments/{experiment_name}/runs/{run_id}"
        else:
            primary_uri = f"{self.mlflow_tracking_uri}/#/experiments/{experiment_name}"

        # PostgreSQL table
        secondary_uri = "postgresql://bi.fact_model_metrics"

        return ArtifactLocation(
            artifact_type=ArtifactType.METRICS,
            primary_uri=primary_uri,
            secondary_uri=secondary_uri,
        )

    def _resolve_generic_location(
        self,
        rule: ArtifactStorageRule,
        artifact_type: ArtifactType,
        name: str = "artifact",
        version: str = "latest",
        **kwargs
    ) -> ArtifactLocation:
        """Resolve generic artifact location."""

        if rule.primary_backend == StorageBackend.MINIO:
            primary_uri = f"s3://{artifact_type.value}/{name}/{version}/"
        elif rule.primary_backend == StorageBackend.MLFLOW:
            primary_uri = f"mlflow://{artifact_type.value}/{name}"
        elif rule.primary_backend == StorageBackend.POSTGRESQL:
            primary_uri = f"postgresql://bi.{artifact_type.value}"
        else:
            primary_uri = f"file://{artifact_type.value}/{name}"

        return ArtifactLocation(
            artifact_type=artifact_type,
            primary_uri=primary_uri,
            version=version,
        )

    def validate_policy_compliance(
        self,
        artifact_type: ArtifactType,
        actual_location: str,
    ) -> Tuple[bool, List[str]]:
        """
        Validate that an artifact is stored according to policy.

        Args:
            artifact_type: Type of artifact
            actual_location: Where the artifact is actually stored

        Returns:
            Tuple of (is_compliant, list of violations)
        """
        violations = []
        rule = self._rules[artifact_type]

        # Check primary backend
        if rule.primary_backend == StorageBackend.MLFLOW:
            if not (actual_location.startswith("models:/") or
                    actual_location.startswith("runs:/")):
                violations.append(
                    f"{artifact_type.value} must be stored in MLflow, got: {actual_location}"
                )

        elif rule.primary_backend == StorageBackend.DVC:
            if not (actual_location.startswith("dvc://") or
                    actual_location.endswith(".dvc")):
                violations.append(
                    f"{artifact_type.value} must be DVC-tracked, got: {actual_location}"
                )

        elif rule.primary_backend == StorageBackend.MINIO:
            if not actual_location.startswith("s3://"):
                violations.append(
                    f"{artifact_type.value} must be in MinIO (s3://), got: {actual_location}"
                )

        # Check if mandatory
        if rule.is_mandatory and not actual_location:
            violations.append(f"{artifact_type.value} storage is mandatory but location is empty")

        return len(violations) == 0, violations

    def to_dict(self) -> Dict[str, Any]:
        """Export policy as dictionary."""
        return {
            "principle": "MLflow-First + DVC-Tracked",
            "version": "1.0.0",
            "rules": {
                k.value: {
                    "primary": v.primary_backend.value,
                    "secondary": v.secondary_backend.value if v.secondary_backend else None,
                    "mandatory": v.is_mandatory,
                    "retention_days": v.retention_days,
                }
                for k, v in self._rules.items()
            },
            "buckets": self._bucket_structure,
        }


# =============================================================================
# POLICY ENFORCEMENT DECORATORS
# =============================================================================

def enforce_mlflow_first(func):
    """
    Decorator to enforce MLflow-First principle for model storage.

    Ensures that any function saving a model also registers it to MLflow.
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        # Check if result contains model path but no MLflow run_id
        if hasattr(result, 'model_path') and not hasattr(result, 'mlflow_run_id'):
            logger.warning(
                f"POLICY VIOLATION: Model saved without MLflow tracking. "
                f"Path: {getattr(result, 'model_path', 'unknown')}"
            )

        return result

    return wrapper


def enforce_dvc_tracking(func):
    """
    Decorator to enforce DVC tracking for dataset operations.

    Ensures datasets are properly versioned with DVC.
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        # Check if result contains dataset path but no DVC tag
        if hasattr(result, 'dataset_path') and not hasattr(result, 'dvc_tag'):
            logger.warning(
                f"POLICY VIOLATION: Dataset saved without DVC tracking. "
                f"Path: {getattr(result, 'dataset_path', 'unknown')}"
            )

        return result

    return wrapper


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_policy_instance: Optional[ArtifactPolicy] = None


def get_artifact_policy() -> ArtifactPolicy:
    """Get the artifact policy singleton."""
    global _policy_instance
    if _policy_instance is None:
        _policy_instance = ArtifactPolicy()
    return _policy_instance


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ArtifactType",
    "StorageBackend",
    "ModelStage",
    # Classes
    "ArtifactStorageRule",
    "ArtifactLocation",
    "ArtifactPolicy",
    # Constants
    "ARTIFACT_STORAGE_RULES",
    "MINIO_BUCKET_STRUCTURE",
    # Functions
    "get_artifact_policy",
    # Decorators
    "enforce_mlflow_first",
    "enforce_dvc_tracking",
]
