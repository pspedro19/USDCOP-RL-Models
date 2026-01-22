"""
MinIO Repository Implementation
===============================

Concrete implementations of storage interfaces using MinIO.

This module provides:
- MinIORepository: Low-level object storage operations
- MinIODatasetRepository: High-level dataset operations
- MinIOModelRepository: High-level model operations
- MinIOBacktestRepository: High-level backtest operations
- MinIOABComparisonRepository: High-level A/B comparison operations

Contract: CTR-MINIO-001
- All operations use S3-compatible API
- Buckets auto-created if they don't exist
- Content hashes computed on upload

Author: Trading Team
Version: 1.0.0
Created: 2026-01-18
"""

import hashlib
import json
import logging
import os
import yaml
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from minio import Minio
from minio.error import S3Error

from src.core.interfaces.storage import (
    ArtifactMetadata,
    IObjectStorageRepository,
    IDatasetRepository,
    IModelRepository,
    IBacktestRepository,
    IABComparisonRepository,
    ObjectNotFoundError,
    StorageError,
)
from src.core.contracts.storage_contracts import (
    EXPERIMENTS_BUCKET,
    PRODUCTION_BUCKET,
    DatasetSnapshot,
    ModelSnapshot,
    BacktestSnapshot,
    ABComparisonSnapshot,
    LineageRecord,
    compute_content_hash,
    compute_schema_hash,
    compute_json_hash,
    build_s3_uri,
)

logger = logging.getLogger(__name__)


# =============================================================================
# LOW-LEVEL MINIO REPOSITORY
# =============================================================================


class MinIORepository(IObjectStorageRepository):
    """
    MinIO implementation of object storage.

    Provides S3-compatible operations for storing and retrieving artifacts.

    Example:
        >>> repo = MinIORepository(
        ...     endpoint="localhost:9000",
        ...     access_key="minioadmin",
        ...     secret_key="minioadmin",
        ... )
        >>> metadata = repo.put_object("experiments", "test/data.json", b'{"test": 1}')
        >>> print(metadata.storage_uri)
        's3://experiments/test/data.json'
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool = False,
        region: Optional[str] = None,
    ):
        """
        Initialize MinIO client.

        Args:
            endpoint: MinIO server endpoint (e.g., "localhost:9000")
            access_key: Access key (username)
            secret_key: Secret key (password)
            secure: Use HTTPS if True
            region: Optional region name
        """
        self._client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            region=region,
        )
        self._endpoint = endpoint
        logger.info(f"MinIO client initialized: {endpoint}")

    @classmethod
    def from_env(cls) -> "MinIORepository":
        """
        Create MinIO repository from environment variables.

        Environment Variables:
            MINIO_ENDPOINT: Server endpoint (default: localhost:9000)
            MINIO_ACCESS_KEY: Access key
            MINIO_SECRET_KEY: Secret key
            MINIO_SECURE: Use HTTPS (default: false)

        Returns:
            Configured MinIORepository
        """
        return cls(
            endpoint=os.environ.get("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
            secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
        )

    def put_object(
        self,
        bucket: str,
        key: str,
        data: bytes,
        metadata: Optional[Dict[str, Any]] = None,
        content_type: Optional[str] = None,
    ) -> ArtifactMetadata:
        """Store object and return metadata with computed hash."""
        try:
            # Ensure bucket exists
            self.ensure_bucket_exists(bucket)

            # Compute hash
            content_hash = compute_content_hash(data)

            # Prepare metadata
            minio_metadata = {
                "x-amz-meta-content-hash": content_hash,
                "x-amz-meta-created-at": datetime.utcnow().isoformat(),
            }
            if metadata:
                for k, v in metadata.items():
                    # MinIO metadata keys must be lowercase
                    minio_metadata[f"x-amz-meta-{k.lower()}"] = str(v)

            # Upload
            self._client.put_object(
                bucket,
                key,
                BytesIO(data),
                len(data),
                content_type=content_type or "application/octet-stream",
                metadata=minio_metadata,
            )

            logger.info(f"Uploaded {bucket}/{key} ({len(data)} bytes, hash: {content_hash})")

            return ArtifactMetadata(
                artifact_id=key,
                version=content_hash,
                content_hash=content_hash,
                created_at=datetime.utcnow(),
                size_bytes=len(data),
                storage_uri=build_s3_uri(bucket, key),
                metadata=metadata or {},
            )

        except S3Error as e:
            logger.error(f"Failed to upload {bucket}/{key}: {e}")
            raise StorageError(f"Failed to upload object: {e}")

    def get_object(self, bucket: str, key: str) -> bytes:
        """Retrieve object data."""
        try:
            response = self._client.get_object(bucket, key)
            data = response.read()
            response.close()
            response.release_conn()
            logger.debug(f"Downloaded {bucket}/{key} ({len(data)} bytes)")
            return data

        except S3Error as e:
            if e.code == "NoSuchKey":
                raise ObjectNotFoundError(f"Object not found: {bucket}/{key}")
            logger.error(f"Failed to download {bucket}/{key}: {e}")
            raise StorageError(f"Failed to download object: {e}")

    def get_metadata(self, bucket: str, key: str) -> ArtifactMetadata:
        """Get object metadata without downloading content."""
        try:
            stat = self._client.stat_object(bucket, key)

            # Extract custom metadata
            metadata = {}
            content_hash = ""
            created_at = datetime.utcnow()

            if stat.metadata:
                for k, v in stat.metadata.items():
                    k_lower = k.lower()
                    if k_lower == "x-amz-meta-content-hash":
                        content_hash = v
                    elif k_lower == "x-amz-meta-created-at":
                        created_at = datetime.fromisoformat(v)
                    elif k_lower.startswith("x-amz-meta-"):
                        metadata[k_lower[11:]] = v

            return ArtifactMetadata(
                artifact_id=key,
                version=content_hash or stat.etag.strip('"'),
                content_hash=content_hash or stat.etag.strip('"'),
                created_at=created_at,
                size_bytes=stat.size,
                storage_uri=build_s3_uri(bucket, key),
                metadata=metadata,
            )

        except S3Error as e:
            if e.code == "NoSuchKey":
                raise ObjectNotFoundError(f"Object not found: {bucket}/{key}")
            raise StorageError(f"Failed to get metadata: {e}")

    def list_objects(
        self,
        bucket: str,
        prefix: str,
        recursive: bool = True,
    ) -> List[ArtifactMetadata]:
        """List objects matching prefix."""
        try:
            objects = self._client.list_objects(bucket, prefix=prefix, recursive=recursive)
            results = []

            for obj in objects:
                results.append(ArtifactMetadata(
                    artifact_id=obj.object_name,
                    version=obj.etag.strip('"') if obj.etag else "",
                    content_hash=obj.etag.strip('"') if obj.etag else "",
                    created_at=obj.last_modified or datetime.utcnow(),
                    size_bytes=obj.size or 0,
                    storage_uri=build_s3_uri(bucket, obj.object_name),
                    metadata={},
                ))

            return results

        except S3Error as e:
            logger.error(f"Failed to list objects {bucket}/{prefix}: {e}")
            raise StorageError(f"Failed to list objects: {e}")

    def object_exists(self, bucket: str, key: str) -> bool:
        """Check if object exists."""
        try:
            self._client.stat_object(bucket, key)
            return True
        except S3Error:
            return False

    def delete_object(self, bucket: str, key: str) -> bool:
        """Delete an object."""
        try:
            self._client.remove_object(bucket, key)
            logger.info(f"Deleted {bucket}/{key}")
            return True
        except S3Error as e:
            logger.error(f"Failed to delete {bucket}/{key}: {e}")
            return False

    def copy_object(
        self,
        source_bucket: str,
        source_key: str,
        dest_bucket: str,
        dest_key: str,
    ) -> ArtifactMetadata:
        """Copy object between buckets or keys."""
        try:
            from minio.commonconfig import CopySource

            self.ensure_bucket_exists(dest_bucket)

            source = CopySource(source_bucket, source_key)
            self._client.copy_object(dest_bucket, dest_key, source)

            logger.info(f"Copied {source_bucket}/{source_key} -> {dest_bucket}/{dest_key}")
            return self.get_metadata(dest_bucket, dest_key)

        except S3Error as e:
            logger.error(f"Failed to copy object: {e}")
            raise StorageError(f"Failed to copy object: {e}")

    def ensure_bucket_exists(self, bucket: str) -> bool:
        """Create bucket if it doesn't exist."""
        try:
            if not self._client.bucket_exists(bucket):
                self._client.make_bucket(bucket)
                logger.info(f"Created bucket: {bucket}")
            return True
        except S3Error as e:
            logger.error(f"Failed to ensure bucket {bucket}: {e}")
            raise StorageError(f"Failed to ensure bucket: {e}")


# =============================================================================
# HIGH-LEVEL DATASET REPOSITORY
# =============================================================================


class MinIODatasetRepository(IDatasetRepository):
    """
    High-level dataset operations for MinIO.

    Manages datasets as structured artifacts in:
    s3://experiments/{experiment_id}/datasets/{version}/

    Files per version:
    - train.parquet: The actual dataset
    - norm_stats.json: Normalization statistics
    - manifest.json: Metadata and hashes
    """

    BUCKET = EXPERIMENTS_BUCKET
    DATASET_PREFIX = "datasets"

    def __init__(self, storage: IObjectStorageRepository):
        """
        Initialize with low-level storage.

        Args:
            storage: IObjectStorageRepository implementation
        """
        self._storage = storage

    @classmethod
    def from_env(cls) -> "MinIODatasetRepository":
        """Create from environment variables."""
        return cls(MinIORepository.from_env())

    def save_dataset(
        self,
        experiment_id: str,
        data: pd.DataFrame,
        version: str,
        metadata: Dict[str, Any],
    ) -> DatasetSnapshot:
        """Save dataset to MinIO with all supporting files."""
        base_path = f"{experiment_id}/{self.DATASET_PREFIX}/{version}"

        # 1. Save parquet
        parquet_buffer = BytesIO()
        data.to_parquet(parquet_buffer, index=False, compression="snappy")
        parquet_bytes = parquet_buffer.getvalue()
        data_hash = compute_content_hash(parquet_bytes)

        data_key = f"{base_path}/train.parquet"
        self._storage.put_object(
            self.BUCKET,
            data_key,
            parquet_bytes,
            metadata={"data_hash": data_hash, "row_count": str(len(data))},
            content_type="application/octet-stream",
        )

        # 2. Compute and save norm_stats
        norm_stats = self._compute_norm_stats(data)
        norm_stats_bytes = json.dumps(norm_stats, indent=2, sort_keys=True).encode()
        norm_stats_hash = compute_content_hash(norm_stats_bytes)

        norm_stats_key = f"{base_path}/norm_stats.json"
        self._storage.put_object(
            self.BUCKET,
            norm_stats_key,
            norm_stats_bytes,
            content_type="application/json",
        )

        # 3. Compute schema hash
        schema_hash = compute_schema_hash(list(data.columns))
        feature_order_hash = metadata.get("feature_order_hash", schema_hash)

        # 4. Build and save manifest
        manifest = {
            "experiment_id": experiment_id,
            "version": version,
            "data_hash": data_hash,
            "schema_hash": schema_hash,
            "norm_stats_hash": norm_stats_hash,
            "row_count": len(data),
            "size_bytes": len(parquet_bytes),
            "columns": list(data.columns),
            "feature_order_hash": feature_order_hash,
            "date_range_start": metadata.get("date_range_start", ""),
            "date_range_end": metadata.get("date_range_end", ""),
            "parent_version": metadata.get("parent_version"),
            "created_at": datetime.utcnow().isoformat(),
            **{k: v for k, v in metadata.items() if k not in [
                "date_range_start", "date_range_end", "parent_version", "feature_order_hash"
            ]},
        }
        manifest_bytes = json.dumps(manifest, indent=2).encode()

        manifest_key = f"{base_path}/manifest.json"
        self._storage.put_object(
            self.BUCKET,
            manifest_key,
            manifest_bytes,
            content_type="application/json",
        )

        logger.info(
            f"Saved dataset {experiment_id}/{version}: "
            f"{len(data)} rows, {len(parquet_bytes)} bytes"
        )

        # Build lineage record if provided
        lineage = None
        if metadata.get("lineage"):
            lineage = LineageRecord.from_dict(metadata["lineage"])

        return DatasetSnapshot(
            experiment_id=experiment_id,
            version=version,
            storage_uri=build_s3_uri(self.BUCKET, data_key),
            norm_stats_uri=build_s3_uri(self.BUCKET, norm_stats_key),
            manifest_uri=build_s3_uri(self.BUCKET, manifest_key),
            data_hash=data_hash,
            schema_hash=schema_hash,
            norm_stats_hash=norm_stats_hash,
            row_count=len(data),
            size_bytes=len(parquet_bytes),
            feature_columns=tuple(data.columns),
            feature_order_hash=feature_order_hash,
            date_range_start=metadata.get("date_range_start", ""),
            date_range_end=metadata.get("date_range_end", ""),
            parent_version=metadata.get("parent_version"),
            lineage=lineage,
            created_at=datetime.utcnow(),
        )

    def load_dataset(
        self,
        experiment_id: str,
        version: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load dataset from MinIO."""
        if version is None:
            version = self._get_latest_version(experiment_id)

        data_key = f"{experiment_id}/{self.DATASET_PREFIX}/{version}/train.parquet"
        parquet_bytes = self._storage.get_object(self.BUCKET, data_key)

        df = pd.read_parquet(BytesIO(parquet_bytes))
        logger.info(f"Loaded dataset {experiment_id}/{version}: {len(df)} rows")
        return df

    def get_snapshot(
        self,
        experiment_id: str,
        version: Optional[str] = None,
    ) -> DatasetSnapshot:
        """Get dataset snapshot from manifest."""
        if version is None:
            version = self._get_latest_version(experiment_id)

        manifest_key = f"{experiment_id}/{self.DATASET_PREFIX}/{version}/manifest.json"
        manifest_bytes = self._storage.get_object(self.BUCKET, manifest_key)
        manifest = json.loads(manifest_bytes)

        base_path = f"{experiment_id}/{self.DATASET_PREFIX}/{version}"

        return DatasetSnapshot(
            experiment_id=experiment_id,
            version=version,
            storage_uri=build_s3_uri(self.BUCKET, f"{base_path}/train.parquet"),
            norm_stats_uri=build_s3_uri(self.BUCKET, f"{base_path}/norm_stats.json"),
            manifest_uri=build_s3_uri(self.BUCKET, manifest_key),
            data_hash=manifest["data_hash"],
            schema_hash=manifest["schema_hash"],
            norm_stats_hash=manifest.get("norm_stats_hash", ""),
            row_count=manifest["row_count"],
            size_bytes=manifest.get("size_bytes", 0),
            feature_columns=tuple(manifest["columns"]),
            feature_order_hash=manifest["feature_order_hash"],
            date_range_start=manifest.get("date_range_start", ""),
            date_range_end=manifest.get("date_range_end", ""),
            parent_version=manifest.get("parent_version"),
            lineage=None,
            created_at=datetime.fromisoformat(manifest["created_at"]),
        )

    def list_versions(self, experiment_id: str) -> List[DatasetSnapshot]:
        """List all dataset versions for an experiment."""
        prefix = f"{experiment_id}/{self.DATASET_PREFIX}/"
        objects = self._storage.list_objects(self.BUCKET, prefix, recursive=False)

        versions = []
        for obj in objects:
            # Extract version from path
            parts = obj.artifact_id.rstrip("/").split("/")
            if len(parts) >= 3:
                version = parts[-1]
                try:
                    snapshot = self.get_snapshot(experiment_id, version)
                    versions.append(snapshot)
                except (ObjectNotFoundError, StorageError):
                    logger.warning(f"Could not load snapshot for {experiment_id}/{version}")

        # Sort by created_at descending
        versions.sort(key=lambda s: s.created_at, reverse=True)
        return versions

    def get_norm_stats(
        self,
        experiment_id: str,
        version: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Get normalization statistics."""
        if version is None:
            version = self._get_latest_version(experiment_id)

        norm_stats_key = f"{experiment_id}/{self.DATASET_PREFIX}/{version}/norm_stats.json"
        norm_stats_bytes = self._storage.get_object(self.BUCKET, norm_stats_key)
        return json.loads(norm_stats_bytes)

    def _get_latest_version(self, experiment_id: str) -> str:
        """Get the latest version for an experiment."""
        versions = self.list_versions(experiment_id)
        if not versions:
            raise ObjectNotFoundError(f"No datasets found for {experiment_id}")
        return versions[0].version

    def _compute_norm_stats(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute normalization statistics for numeric columns."""
        stats = {}
        for col in data.columns:
            if data[col].dtype in ["float64", "float32", "int64", "int32"]:
                stats[col] = {
                    "mean": float(data[col].mean()),
                    "std": float(data[col].std()),
                    "min": float(data[col].min()),
                    "max": float(data[col].max()),
                }
        return stats


# =============================================================================
# HIGH-LEVEL MODEL REPOSITORY
# =============================================================================


class MinIOModelRepository(IModelRepository):
    """
    High-level model operations for MinIO.

    Manages models as structured artifacts in:
    s3://experiments/{experiment_id}/models/{version}/

    Files per version:
    - policy.onnx: ONNX model (if available)
    - policy.zip: Stable Baselines3 model
    - norm_stats.json: Normalization statistics
    - config.yaml: Training configuration
    - lineage.json: Lineage record
    """

    BUCKET = EXPERIMENTS_BUCKET
    PRODUCTION_BUCKET = PRODUCTION_BUCKET
    MODELS_PREFIX = "models"

    def __init__(self, storage: IObjectStorageRepository):
        """Initialize with low-level storage."""
        self._storage = storage

    @classmethod
    def from_env(cls) -> "MinIOModelRepository":
        """Create from environment variables."""
        return cls(MinIORepository.from_env())

    def save_model(
        self,
        experiment_id: str,
        model_path: Union[str, Path],
        norm_stats: Dict[str, Any],
        config: Dict[str, Any],
        lineage: LineageRecord,
        version: Optional[str] = None,
    ) -> ModelSnapshot:
        """Save trained model to MinIO."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Auto-generate version if not provided
        if version is None:
            version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        base_path = f"{experiment_id}/{self.MODELS_PREFIX}/{version}"

        # 1. Upload model file
        with open(model_path, "rb") as f:
            model_bytes = f.read()
        model_hash = compute_content_hash(model_bytes)

        model_key = f"{base_path}/policy.zip"
        self._storage.put_object(
            self.BUCKET,
            model_key,
            model_bytes,
            content_type="application/octet-stream",
        )

        # 2. Upload ONNX if exists
        onnx_path = model_path.with_suffix(".onnx")
        if onnx_path.exists():
            with open(onnx_path, "rb") as f:
                onnx_bytes = f.read()
            onnx_key = f"{base_path}/policy.onnx"
            self._storage.put_object(self.BUCKET, onnx_key, onnx_bytes)

        # 3. Upload norm_stats
        norm_stats_bytes = json.dumps(norm_stats, indent=2, sort_keys=True).encode()
        norm_stats_hash = compute_content_hash(norm_stats_bytes)
        norm_stats_key = f"{base_path}/norm_stats.json"
        self._storage.put_object(
            self.BUCKET,
            norm_stats_key,
            norm_stats_bytes,
            content_type="application/json",
        )

        # 4. Upload config
        config_bytes = yaml.dump(config, default_flow_style=False).encode()
        config_hash = compute_content_hash(config_bytes)
        config_key = f"{base_path}/config.yaml"
        self._storage.put_object(
            self.BUCKET,
            config_key,
            config_bytes,
            content_type="application/x-yaml",
        )

        # 5. Upload lineage
        lineage_bytes = json.dumps(lineage.to_dict(), indent=2).encode()
        lineage_key = f"{base_path}/lineage.json"
        self._storage.put_object(
            self.BUCKET,
            lineage_key,
            lineage_bytes,
            content_type="application/json",
        )

        # Extract feature order from config or lineage
        feature_order = config.get("feature_order", [])
        if isinstance(feature_order, list):
            feature_order = tuple(feature_order)

        logger.info(f"Saved model {experiment_id}/{version} (hash: {model_hash})")

        # Load dataset snapshot if available
        dataset_snapshot = None
        if lineage.parent_id and lineage.parent_type == "dataset":
            try:
                dataset_repo = MinIODatasetRepository(self._storage)
                dataset_snapshot = dataset_repo.get_snapshot(
                    lineage.parent_id.split("/")[0],  # experiment_id
                    lineage.parent_id.split("/")[-1],  # version
                )
            except Exception as e:
                logger.warning(f"Could not load parent dataset snapshot: {e}")

        return ModelSnapshot(
            experiment_id=experiment_id,
            version=version,
            storage_uri=build_s3_uri(self.BUCKET, model_key),
            norm_stats_uri=build_s3_uri(self.BUCKET, norm_stats_key),
            config_uri=build_s3_uri(self.BUCKET, config_key),
            lineage_uri=build_s3_uri(self.BUCKET, lineage_key),
            model_hash=model_hash,
            norm_stats_hash=norm_stats_hash,
            config_hash=config_hash,
            observation_dim=config.get("observation_dim", 15),
            action_space=config.get("action_space", 3),
            feature_order_hash=config.get("feature_order_hash", ""),
            feature_order=feature_order,
            test_sharpe=None,
            test_max_drawdown=None,
            test_win_rate=None,
            test_total_return=None,
            training_duration_seconds=config.get("training_duration_seconds", 0.0),
            mlflow_run_id=config.get("mlflow_run_id"),
            best_reward=config.get("best_reward"),
            dataset_snapshot=dataset_snapshot,
            created_at=datetime.utcnow(),
        )

    def load_model(
        self,
        experiment_id: str,
        version: Optional[str] = None,
    ) -> bytes:
        """Load model bytes from MinIO."""
        if version is None:
            version = self._get_latest_version(experiment_id)

        model_key = f"{experiment_id}/{self.MODELS_PREFIX}/{version}/policy.zip"
        return self._storage.get_object(self.BUCKET, model_key)

    def get_snapshot(
        self,
        experiment_id: str,
        version: Optional[str] = None,
    ) -> ModelSnapshot:
        """Get model snapshot."""
        if version is None:
            version = self._get_latest_version(experiment_id)

        base_path = f"{experiment_id}/{self.MODELS_PREFIX}/{version}"

        # Load config
        config_key = f"{base_path}/config.yaml"
        config_bytes = self._storage.get_object(self.BUCKET, config_key)
        config = yaml.safe_load(config_bytes)

        # Load lineage
        lineage_key = f"{base_path}/lineage.json"
        lineage_bytes = self._storage.get_object(self.BUCKET, lineage_key)
        lineage = LineageRecord.from_dict(json.loads(lineage_bytes))

        # Get model metadata
        model_key = f"{base_path}/policy.zip"
        model_meta = self._storage.get_metadata(self.BUCKET, model_key)

        # Get norm_stats metadata
        norm_stats_key = f"{base_path}/norm_stats.json"
        norm_stats_meta = self._storage.get_metadata(self.BUCKET, norm_stats_key)

        feature_order = config.get("feature_order", [])
        if isinstance(feature_order, list):
            feature_order = tuple(feature_order)

        return ModelSnapshot(
            experiment_id=experiment_id,
            version=version,
            storage_uri=build_s3_uri(self.BUCKET, model_key),
            norm_stats_uri=build_s3_uri(self.BUCKET, norm_stats_key),
            config_uri=build_s3_uri(self.BUCKET, config_key),
            lineage_uri=build_s3_uri(self.BUCKET, lineage_key),
            model_hash=model_meta.content_hash,
            norm_stats_hash=norm_stats_meta.content_hash,
            config_hash=compute_content_hash(config_bytes),
            observation_dim=config.get("observation_dim", 15),
            action_space=config.get("action_space", 3),
            feature_order_hash=config.get("feature_order_hash", ""),
            feature_order=feature_order,
            test_sharpe=config.get("test_sharpe"),
            test_max_drawdown=config.get("test_max_drawdown"),
            test_win_rate=config.get("test_win_rate"),
            test_total_return=config.get("test_total_return"),
            training_duration_seconds=config.get("training_duration_seconds", 0.0),
            mlflow_run_id=config.get("mlflow_run_id"),
            best_reward=config.get("best_reward"),
            dataset_snapshot=None,
            created_at=model_meta.created_at,
        )

    def list_versions(self, experiment_id: str) -> List[ModelSnapshot]:
        """List all model versions for an experiment."""
        prefix = f"{experiment_id}/{self.MODELS_PREFIX}/"
        objects = self._storage.list_objects(self.BUCKET, prefix, recursive=False)

        versions = []
        for obj in objects:
            parts = obj.artifact_id.rstrip("/").split("/")
            if len(parts) >= 3:
                version = parts[-1]
                try:
                    snapshot = self.get_snapshot(experiment_id, version)
                    versions.append(snapshot)
                except (ObjectNotFoundError, StorageError):
                    logger.warning(f"Could not load model snapshot for {experiment_id}/{version}")

        versions.sort(key=lambda s: s.created_at, reverse=True)
        return versions

    def promote_to_production(
        self,
        experiment_id: str,
        version: str,
        model_id: Optional[str] = None,
    ) -> str:
        """Copy model to production bucket."""
        # Generate model_id if not provided
        snapshot = self.get_snapshot(experiment_id, version)
        if model_id is None:
            model_id = f"ppo_{version}_{snapshot.model_hash[:8]}"

        source_base = f"{experiment_id}/{self.MODELS_PREFIX}/{version}"
        dest_base = f"models/{model_id}"

        # Copy all files
        files = ["policy.zip", "policy.onnx", "norm_stats.json", "config.yaml", "lineage.json"]
        for filename in files:
            source_key = f"{source_base}/{filename}"
            dest_key = f"{dest_base}/{filename}"

            if self._storage.object_exists(self.BUCKET, source_key):
                self._storage.copy_object(
                    self.BUCKET, source_key,
                    self.PRODUCTION_BUCKET, dest_key,
                )

        logger.info(f"Promoted model {experiment_id}/{version} to production as {model_id}")
        return model_id

    def _get_latest_version(self, experiment_id: str) -> str:
        """Get the latest version for an experiment."""
        versions = self.list_versions(experiment_id)
        if not versions:
            raise ObjectNotFoundError(f"No models found for {experiment_id}")
        return versions[0].version


# =============================================================================
# HIGH-LEVEL BACKTEST REPOSITORY
# =============================================================================


class MinIOBacktestRepository(IBacktestRepository):
    """
    High-level backtest operations for MinIO.

    Manages backtests as structured artifacts in:
    s3://experiments/{experiment_id}/backtests/{backtest_id}/
    """

    BUCKET = EXPERIMENTS_BUCKET
    BACKTESTS_PREFIX = "backtests"

    def __init__(self, storage: IObjectStorageRepository):
        """Initialize with low-level storage."""
        self._storage = storage

    @classmethod
    def from_env(cls) -> "MinIOBacktestRepository":
        """Create from environment variables."""
        return cls(MinIORepository.from_env())

    def save_backtest(
        self,
        experiment_id: str,
        model_version: str,
        result: Dict[str, Any],
        trades: pd.DataFrame,
        equity_curve: pd.DataFrame,
        backtest_id: Optional[str] = None,
    ) -> BacktestSnapshot:
        """Save backtest results to MinIO."""
        if backtest_id is None:
            backtest_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        base_path = f"{experiment_id}/{self.BACKTESTS_PREFIX}/{backtest_id}"

        # 1. Save result JSON
        result_bytes = json.dumps(result, indent=2, default=str).encode()
        result_hash = compute_content_hash(result_bytes)
        result_key = f"{base_path}/result.json"
        self._storage.put_object(
            self.BUCKET,
            result_key,
            result_bytes,
            content_type="application/json",
        )

        # 2. Save trades parquet
        trades_buffer = BytesIO()
        trades.to_parquet(trades_buffer, index=False)
        trades_key = f"{base_path}/trades.parquet"
        self._storage.put_object(
            self.BUCKET,
            trades_key,
            trades_buffer.getvalue(),
        )

        # 3. Save equity curve parquet
        equity_buffer = BytesIO()
        equity_curve.to_parquet(equity_buffer, index=False)
        equity_key = f"{base_path}/equity_curve.parquet"
        self._storage.put_object(
            self.BUCKET,
            equity_key,
            equity_buffer.getvalue(),
        )

        logger.info(f"Saved backtest {experiment_id}/{backtest_id}")

        return BacktestSnapshot(
            experiment_id=experiment_id,
            model_version=model_version,
            backtest_id=backtest_id,
            storage_uri=build_s3_uri(self.BUCKET, result_key),
            trades_uri=build_s3_uri(self.BUCKET, trades_key),
            equity_curve_uri=build_s3_uri(self.BUCKET, equity_key),
            sharpe_ratio=result.get("sharpe_ratio", 0.0),
            total_return=result.get("total_return", 0.0),
            max_drawdown=result.get("max_drawdown", 0.0),
            win_rate=result.get("win_rate", 0.0),
            total_trades=result.get("total_trades", 0),
            profit_factor=result.get("profit_factor"),
            avg_trade_return=result.get("avg_trade_return"),
            result_hash=result_hash,
            backtest_start=result.get("backtest_start", ""),
            backtest_end=result.get("backtest_end", ""),
            created_at=datetime.utcnow(),
        )

    def load_backtest(
        self,
        experiment_id: str,
        backtest_id: str,
    ) -> Dict[str, Any]:
        """Load backtest result."""
        result_key = f"{experiment_id}/{self.BACKTESTS_PREFIX}/{backtest_id}/result.json"
        result_bytes = self._storage.get_object(self.BUCKET, result_key)
        return json.loads(result_bytes)

    def list_backtests(self, experiment_id: str) -> List[BacktestSnapshot]:
        """List all backtests for an experiment."""
        prefix = f"{experiment_id}/{self.BACKTESTS_PREFIX}/"
        objects = self._storage.list_objects(self.BUCKET, prefix, recursive=False)

        backtests = []
        for obj in objects:
            parts = obj.artifact_id.rstrip("/").split("/")
            if len(parts) >= 3:
                backtest_id = parts[-1]
                try:
                    result = self.load_backtest(experiment_id, backtest_id)
                    base_path = f"{experiment_id}/{self.BACKTESTS_PREFIX}/{backtest_id}"

                    backtests.append(BacktestSnapshot(
                        experiment_id=experiment_id,
                        model_version=result.get("model_version", ""),
                        backtest_id=backtest_id,
                        storage_uri=build_s3_uri(self.BUCKET, f"{base_path}/result.json"),
                        trades_uri=build_s3_uri(self.BUCKET, f"{base_path}/trades.parquet"),
                        equity_curve_uri=build_s3_uri(self.BUCKET, f"{base_path}/equity_curve.parquet"),
                        sharpe_ratio=result.get("sharpe_ratio", 0.0),
                        total_return=result.get("total_return", 0.0),
                        max_drawdown=result.get("max_drawdown", 0.0),
                        win_rate=result.get("win_rate", 0.0),
                        total_trades=result.get("total_trades", 0),
                        profit_factor=result.get("profit_factor"),
                        avg_trade_return=result.get("avg_trade_return"),
                        result_hash=result.get("result_hash", ""),
                        backtest_start=result.get("backtest_start", ""),
                        backtest_end=result.get("backtest_end", ""),
                        created_at=obj.created_at,
                    ))
                except Exception as e:
                    logger.warning(f"Could not load backtest {backtest_id}: {e}")

        backtests.sort(key=lambda b: b.created_at, reverse=True)
        return backtests


# =============================================================================
# HIGH-LEVEL A/B COMPARISON REPOSITORY
# =============================================================================


class MinIOABComparisonRepository(IABComparisonRepository):
    """
    High-level A/B comparison operations for MinIO.

    Manages comparisons as structured artifacts in:
    s3://experiments/{experiment_id}/comparisons/{comparison_id}/
    """

    BUCKET = EXPERIMENTS_BUCKET
    COMPARISONS_PREFIX = "comparisons"

    def __init__(self, storage: IObjectStorageRepository):
        """Initialize with low-level storage."""
        self._storage = storage

    @classmethod
    def from_env(cls) -> "MinIOABComparisonRepository":
        """Create from environment variables."""
        return cls(MinIORepository.from_env())

    def save_comparison(
        self,
        experiment_id: str,
        baseline_model: ModelSnapshot,
        treatment_model: ModelSnapshot,
        result: Dict[str, Any],
        shadow_trades: Optional[pd.DataFrame] = None,
    ) -> ABComparisonSnapshot:
        """Save A/B comparison results."""
        comparison_id = (
            f"{baseline_model.version}_vs_{treatment_model.version}_"
            f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )

        base_path = f"{experiment_id}/{self.COMPARISONS_PREFIX}/{comparison_id}"

        # 1. Save result JSON
        result_bytes = json.dumps(result, indent=2, default=str).encode()
        result_key = f"{base_path}/ab_result.json"
        self._storage.put_object(
            self.BUCKET,
            result_key,
            result_bytes,
            content_type="application/json",
        )

        # 2. Save shadow trades if provided
        shadow_trades_uri = None
        if shadow_trades is not None and len(shadow_trades) > 0:
            trades_buffer = BytesIO()
            shadow_trades.to_parquet(trades_buffer, index=False)
            trades_key = f"{base_path}/shadow_trades.parquet"
            self._storage.put_object(self.BUCKET, trades_key, trades_buffer.getvalue())
            shadow_trades_uri = build_s3_uri(self.BUCKET, trades_key)

        logger.info(f"Saved A/B comparison {experiment_id}/{comparison_id}")

        return ABComparisonSnapshot(
            comparison_id=comparison_id,
            experiment_id=experiment_id,
            storage_uri=build_s3_uri(self.BUCKET, result_key),
            shadow_trades_uri=shadow_trades_uri,
            baseline_experiment_id=baseline_model.experiment_id,
            baseline_version=baseline_model.version,
            baseline_model_hash=baseline_model.model_hash,
            treatment_experiment_id=treatment_model.experiment_id,
            treatment_version=treatment_model.version,
            treatment_model_hash=treatment_model.model_hash,
            primary_metric=result.get("primary_metric", "sharpe_ratio"),
            baseline_value=result.get("baseline_value", 0.0),
            treatment_value=result.get("treatment_value", 0.0),
            p_value=result.get("p_value", 1.0),
            effect_size=result.get("effect_size", 0.0),
            confidence_interval_low=result.get("confidence_interval_low", 0.0),
            confidence_interval_high=result.get("confidence_interval_high", 0.0),
            is_significant=result.get("is_significant", False),
            confidence_level=result.get("confidence_level", 0.95),
            recommendation=result.get("recommendation", "inconclusive"),
            baseline_trades=result.get("baseline_trades", 0),
            treatment_trades=result.get("treatment_trades", 0),
            comparison_duration_hours=result.get("comparison_duration_hours", 0.0),
            created_at=datetime.utcnow(),
        )

    def load_comparison(
        self,
        experiment_id: str,
        comparison_id: str,
    ) -> Dict[str, Any]:
        """Load comparison result."""
        result_key = f"{experiment_id}/{self.COMPARISONS_PREFIX}/{comparison_id}/ab_result.json"
        result_bytes = self._storage.get_object(self.BUCKET, result_key)
        return json.loads(result_bytes)
