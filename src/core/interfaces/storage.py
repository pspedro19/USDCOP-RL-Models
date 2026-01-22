"""
Storage Interfaces for USD/COP Trading System
==============================================

Defines abstract interfaces for object storage operations following DIP.
Enables MinIO-first architecture with PostgreSQL for production only.

Interface Segregation Principle (ISP):
- IObjectStorageRepository: Low-level object storage (put/get/list)
- IDatasetRepository: High-level dataset operations
- IModelRepository: High-level model operations
- IBacktestRepository: High-level backtest result operations

Contract: CTR-STORAGE-001
- All experiment artifacts stored in MinIO (s3://experiments/)
- Production models promoted to PostgreSQL model_registry + MinIO (s3://production/)
- Storage URIs use s3:// scheme for both MinIO and AWS S3

Author: Trading Team
Version: 1.0.0
Created: 2026-01-18
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    import pandas as pd
    from .storage_contracts import (
        DatasetSnapshot,
        ModelSnapshot,
        BacktestSnapshot,
        ABComparisonSnapshot,
        LineageRecord,
    )


# =============================================================================
# ARTIFACT METADATA
# =============================================================================


@dataclass(frozen=True)
class ArtifactMetadata:
    """
    Immutable artifact metadata returned from storage operations.

    Attributes:
        artifact_id: Unique identifier for the artifact (typically the key)
        version: Content-based version (hash)
        content_hash: SHA256 hash of content (first 16 chars)
        created_at: When the artifact was created
        size_bytes: Size of the artifact in bytes
        storage_uri: Full S3 URI (s3://bucket/key)
        metadata: Custom metadata dictionary
    """
    artifact_id: str
    version: str
    content_hash: str
    created_at: datetime
    size_bytes: int
    storage_uri: str  # s3://bucket/path
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "artifact_id": self.artifact_id,
            "version": self.version,
            "content_hash": self.content_hash,
            "created_at": self.created_at.isoformat(),
            "size_bytes": self.size_bytes,
            "storage_uri": self.storage_uri,
            "metadata": self.metadata,
        }


# =============================================================================
# LOW-LEVEL STORAGE INTERFACE
# =============================================================================


class IObjectStorageRepository(ABC):
    """
    Abstract interface for low-level object storage operations.

    Dependency Inversion: High-level modules depend on this interface,
    not on concrete MinIO/S3/local implementations.

    Contract: CTR-STORAGE-002
    - put_object returns ArtifactMetadata with computed hash
    - get_object returns raw bytes
    - All operations are bucket-aware
    """

    @abstractmethod
    def put_object(
        self,
        bucket: str,
        key: str,
        data: bytes,
        metadata: Optional[Dict[str, Any]] = None,
        content_type: Optional[str] = None,
    ) -> ArtifactMetadata:
        """
        Store an object and return its metadata.

        Args:
            bucket: Target bucket name
            key: Object key (path within bucket)
            data: Raw bytes to store
            metadata: Custom metadata dictionary
            content_type: MIME type of the content

        Returns:
            ArtifactMetadata with computed hash and storage URI
        """
        pass

    @abstractmethod
    def get_object(self, bucket: str, key: str) -> bytes:
        """
        Retrieve object data.

        Args:
            bucket: Bucket name
            key: Object key

        Returns:
            Raw bytes of the object

        Raises:
            ObjectNotFoundError: If object doesn't exist
        """
        pass

    @abstractmethod
    def get_metadata(self, bucket: str, key: str) -> ArtifactMetadata:
        """
        Get object metadata without downloading content.

        Args:
            bucket: Bucket name
            key: Object key

        Returns:
            ArtifactMetadata for the object

        Raises:
            ObjectNotFoundError: If object doesn't exist
        """
        pass

    @abstractmethod
    def list_objects(
        self,
        bucket: str,
        prefix: str,
        recursive: bool = True,
    ) -> List[ArtifactMetadata]:
        """
        List objects matching prefix.

        Args:
            bucket: Bucket name
            prefix: Key prefix to filter
            recursive: Whether to list recursively

        Returns:
            List of ArtifactMetadata for matching objects
        """
        pass

    @abstractmethod
    def object_exists(self, bucket: str, key: str) -> bool:
        """
        Check if object exists.

        Args:
            bucket: Bucket name
            key: Object key

        Returns:
            True if object exists
        """
        pass

    @abstractmethod
    def delete_object(self, bucket: str, key: str) -> bool:
        """
        Delete an object.

        Args:
            bucket: Bucket name
            key: Object key

        Returns:
            True if deleted (or didn't exist)
        """
        pass

    @abstractmethod
    def copy_object(
        self,
        source_bucket: str,
        source_key: str,
        dest_bucket: str,
        dest_key: str,
    ) -> ArtifactMetadata:
        """
        Copy object between buckets or keys.

        Args:
            source_bucket: Source bucket
            source_key: Source key
            dest_bucket: Destination bucket
            dest_key: Destination key

        Returns:
            ArtifactMetadata of the copied object
        """
        pass

    @abstractmethod
    def ensure_bucket_exists(self, bucket: str) -> bool:
        """
        Create bucket if it doesn't exist.

        Args:
            bucket: Bucket name

        Returns:
            True if bucket exists or was created
        """
        pass


# =============================================================================
# HIGH-LEVEL DATASET INTERFACE
# =============================================================================


class IDatasetRepository(ABC):
    """
    High-level interface for dataset operations.

    Single Responsibility: Manage dataset lifecycle in object storage.

    Contract: CTR-DATASET-001
    - Datasets stored as parquet in s3://experiments/{exp_id}/datasets/{version}/
    - Each dataset has train.parquet, norm_stats.json, manifest.json
    - Returns DatasetSnapshot for lineage tracking
    """

    @abstractmethod
    def save_dataset(
        self,
        experiment_id: str,
        data: "pd.DataFrame",
        version: str,
        metadata: Dict[str, Any],
    ) -> "DatasetSnapshot":
        """
        Save dataset to object storage.

        Creates:
        - {exp_id}/datasets/{version}/train.parquet
        - {exp_id}/datasets/{version}/norm_stats.json
        - {exp_id}/datasets/{version}/manifest.json

        Args:
            experiment_id: Unique experiment identifier
            data: DataFrame to save
            version: Version string (e.g., "v1", "20260118_123456")
            metadata: Additional metadata (date_range, parent_version, etc.)

        Returns:
            DatasetSnapshot with all URIs and hashes
        """
        pass

    @abstractmethod
    def load_dataset(
        self,
        experiment_id: str,
        version: Optional[str] = None,
    ) -> "pd.DataFrame":
        """
        Load dataset from object storage.

        Args:
            experiment_id: Experiment identifier
            version: Specific version or None for latest

        Returns:
            DataFrame loaded from parquet
        """
        pass

    @abstractmethod
    def get_snapshot(
        self,
        experiment_id: str,
        version: Optional[str] = None,
    ) -> "DatasetSnapshot":
        """
        Get dataset snapshot (metadata without loading data).

        Args:
            experiment_id: Experiment identifier
            version: Specific version or None for latest

        Returns:
            DatasetSnapshot with URIs and metadata
        """
        pass

    @abstractmethod
    def list_versions(self, experiment_id: str) -> List["DatasetSnapshot"]:
        """
        List all dataset versions for an experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            List of DatasetSnapshot, newest first
        """
        pass

    @abstractmethod
    def get_norm_stats(
        self,
        experiment_id: str,
        version: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Get normalization statistics for a dataset.

        Args:
            experiment_id: Experiment identifier
            version: Specific version or None for latest

        Returns:
            Normalization stats dictionary
        """
        pass


# =============================================================================
# HIGH-LEVEL MODEL INTERFACE
# =============================================================================


class IModelRepository(ABC):
    """
    High-level interface for model operations.

    Single Responsibility: Manage model lifecycle in object storage.

    Contract: CTR-MODEL-001
    - Models stored in s3://experiments/{exp_id}/models/{version}/
    - Each model has policy.onnx, policy.zip, norm_stats.json, config.yaml, lineage.json
    - promote_to_production copies to s3://production/models/{model_id}/
    """

    @abstractmethod
    def save_model(
        self,
        experiment_id: str,
        model_path: Union[str, Path],
        norm_stats: Dict[str, Any],
        config: Dict[str, Any],
        lineage: "LineageRecord",
        version: Optional[str] = None,
    ) -> "ModelSnapshot":
        """
        Save trained model to object storage.

        Creates:
        - {exp_id}/models/{version}/policy.onnx (if exists)
        - {exp_id}/models/{version}/policy.zip
        - {exp_id}/models/{version}/norm_stats.json
        - {exp_id}/models/{version}/config.yaml
        - {exp_id}/models/{version}/lineage.json

        Args:
            experiment_id: Experiment identifier
            model_path: Path to model file(s)
            norm_stats: Normalization statistics
            config: Training configuration
            lineage: Lineage record linking to dataset
            version: Version string or auto-generated

        Returns:
            ModelSnapshot with all URIs and hashes
        """
        pass

    @abstractmethod
    def load_model(
        self,
        experiment_id: str,
        version: Optional[str] = None,
    ) -> bytes:
        """
        Load model bytes from object storage.

        Args:
            experiment_id: Experiment identifier
            version: Specific version or None for latest

        Returns:
            Raw bytes of model file
        """
        pass

    @abstractmethod
    def get_snapshot(
        self,
        experiment_id: str,
        version: Optional[str] = None,
    ) -> "ModelSnapshot":
        """
        Get model snapshot (metadata without loading model).

        Args:
            experiment_id: Experiment identifier
            version: Specific version or None for latest

        Returns:
            ModelSnapshot with URIs and metadata
        """
        pass

    @abstractmethod
    def list_versions(self, experiment_id: str) -> List["ModelSnapshot"]:
        """
        List all model versions for an experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            List of ModelSnapshot, newest first
        """
        pass

    @abstractmethod
    def promote_to_production(
        self,
        experiment_id: str,
        version: str,
        model_id: Optional[str] = None,
    ) -> str:
        """
        Promote model to production bucket.

        1. Copies model to s3://production/models/{model_id}/
        2. Returns model_id for PostgreSQL registration

        Args:
            experiment_id: Source experiment identifier
            version: Model version to promote
            model_id: Optional custom model ID

        Returns:
            Generated or provided model_id
        """
        pass


# =============================================================================
# HIGH-LEVEL BACKTEST INTERFACE
# =============================================================================


class IBacktestRepository(ABC):
    """
    High-level interface for backtest results.

    Single Responsibility: Manage backtest results in object storage.

    Contract: CTR-BACKTEST-001
    - Results stored in s3://experiments/{exp_id}/backtests/{backtest_id}/
    - Each backtest has result.json, trades.parquet, equity_curve.parquet
    """

    @abstractmethod
    def save_backtest(
        self,
        experiment_id: str,
        model_version: str,
        result: Dict[str, Any],
        trades: "pd.DataFrame",
        equity_curve: "pd.DataFrame",
        backtest_id: Optional[str] = None,
    ) -> "BacktestSnapshot":
        """
        Save backtest results to object storage.

        Args:
            experiment_id: Experiment identifier
            model_version: Version of model used
            result: Backtest metrics dictionary
            trades: DataFrame of trades
            equity_curve: DataFrame of equity curve
            backtest_id: Optional custom ID

        Returns:
            BacktestSnapshot with all URIs
        """
        pass

    @abstractmethod
    def load_backtest(
        self,
        experiment_id: str,
        backtest_id: str,
    ) -> Dict[str, Any]:
        """
        Load backtest result.

        Args:
            experiment_id: Experiment identifier
            backtest_id: Backtest identifier

        Returns:
            Backtest result dictionary
        """
        pass

    @abstractmethod
    def list_backtests(self, experiment_id: str) -> List["BacktestSnapshot"]:
        """
        List all backtests for an experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            List of BacktestSnapshot, newest first
        """
        pass


# =============================================================================
# HIGH-LEVEL A/B COMPARISON INTERFACE
# =============================================================================


class IABComparisonRepository(ABC):
    """
    High-level interface for A/B comparison results.

    Single Responsibility: Manage A/B test results in object storage.

    Contract: CTR-AB-001
    - Results stored in s3://experiments/{exp_id}/comparisons/{comparison_id}/
    - Each comparison has ab_result.json, shadow_trades.parquet
    """

    @abstractmethod
    def save_comparison(
        self,
        experiment_id: str,
        baseline_model: "ModelSnapshot",
        treatment_model: "ModelSnapshot",
        result: Dict[str, Any],
        shadow_trades: Optional["pd.DataFrame"] = None,
    ) -> "ABComparisonSnapshot":
        """
        Save A/B comparison results.

        Args:
            experiment_id: Experiment identifier
            baseline_model: Baseline model snapshot
            treatment_model: Treatment model snapshot
            result: Comparison metrics
            shadow_trades: Optional shadow trades DataFrame

        Returns:
            ABComparisonSnapshot with URIs
        """
        pass

    @abstractmethod
    def load_comparison(
        self,
        experiment_id: str,
        comparison_id: str,
    ) -> Dict[str, Any]:
        """
        Load comparison result.

        Args:
            experiment_id: Experiment identifier
            comparison_id: Comparison identifier

        Returns:
            Comparison result dictionary
        """
        pass


# =============================================================================
# EXCEPTIONS
# =============================================================================


class ObjectNotFoundError(Exception):
    """Object not found in storage."""
    pass


class StorageError(Exception):
    """Generic storage error."""
    pass


class IntegrityError(Exception):
    """Hash/integrity verification failed."""
    pass


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Metadata
    "ArtifactMetadata",
    # Interfaces
    "IObjectStorageRepository",
    "IDatasetRepository",
    "IModelRepository",
    "IBacktestRepository",
    "IABComparisonRepository",
    # Exceptions
    "ObjectNotFoundError",
    "StorageError",
    "IntegrityError",
]
