"""
Local Storage Repository Implementation
=======================================

Local filesystem implementation of storage interfaces for testing.

This module provides:
- LocalStorageRepository: Low-level file operations
- LocalDatasetRepository: High-level dataset operations
- LocalModelRepository: High-level model operations

Used for:
- Unit testing without MinIO
- Local development
- Offline mode

Author: Trading Team
Version: 1.0.0
Created: 2026-01-18
"""

import hashlib
import json
import logging
import os
import shutil
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import yaml

from src.core.interfaces.storage import (
    ArtifactMetadata,
    IObjectStorageRepository,
    IDatasetRepository,
    IModelRepository,
    ObjectNotFoundError,
    StorageError,
)
from src.core.contracts.storage_contracts import (
    EXPERIMENTS_BUCKET,
    PRODUCTION_BUCKET,
    DatasetSnapshot,
    ModelSnapshot,
    LineageRecord,
    compute_content_hash,
    compute_schema_hash,
    build_s3_uri,
)

logger = logging.getLogger(__name__)


# =============================================================================
# LOW-LEVEL LOCAL STORAGE REPOSITORY
# =============================================================================


class LocalStorageRepository(IObjectStorageRepository):
    """
    Local filesystem implementation of object storage.

    Mimics S3/MinIO API for testing purposes.
    """

    def __init__(self, base_path: Union[str, Path]):
        """
        Initialize with base storage path.

        Args:
            base_path: Root directory for storage
        """
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalStorageRepository initialized: {self._base_path}")

    def _get_path(self, bucket: str, key: str) -> Path:
        """Get full filesystem path for bucket/key."""
        return self._base_path / bucket / key

    def put_object(
        self,
        bucket: str,
        key: str,
        data: bytes,
        metadata: Optional[Dict[str, Any]] = None,
        content_type: Optional[str] = None,
    ) -> ArtifactMetadata:
        """Store object to local filesystem."""
        try:
            path = self._get_path(bucket, key)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Write data
            with open(path, "wb") as f:
                f.write(data)

            # Write metadata sidecar
            content_hash = compute_content_hash(data)
            meta = {
                "content_hash": content_hash,
                "size_bytes": len(data),
                "created_at": datetime.utcnow().isoformat(),
                "content_type": content_type or "application/octet-stream",
                **(metadata or {}),
            }
            meta_path = path.with_suffix(path.suffix + ".meta.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            logger.debug(f"Saved {bucket}/{key} ({len(data)} bytes)")

            return ArtifactMetadata(
                artifact_id=key,
                version=content_hash,
                content_hash=content_hash,
                created_at=datetime.utcnow(),
                size_bytes=len(data),
                storage_uri=build_s3_uri(bucket, key),
                metadata=metadata or {},
            )

        except Exception as e:
            logger.error(f"Failed to save {bucket}/{key}: {e}")
            raise StorageError(f"Failed to save object: {e}")

    def get_object(self, bucket: str, key: str) -> bytes:
        """Retrieve object from local filesystem."""
        path = self._get_path(bucket, key)
        if not path.exists():
            raise ObjectNotFoundError(f"Object not found: {bucket}/{key}")

        with open(path, "rb") as f:
            return f.read()

    def get_metadata(self, bucket: str, key: str) -> ArtifactMetadata:
        """Get object metadata."""
        path = self._get_path(bucket, key)
        if not path.exists():
            raise ObjectNotFoundError(f"Object not found: {bucket}/{key}")

        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
        else:
            # Compute metadata from file
            with open(path, "rb") as f:
                data = f.read()
            meta = {
                "content_hash": compute_content_hash(data),
                "size_bytes": len(data),
                "created_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            }

        return ArtifactMetadata(
            artifact_id=key,
            version=meta["content_hash"],
            content_hash=meta["content_hash"],
            created_at=datetime.fromisoformat(meta["created_at"]),
            size_bytes=meta["size_bytes"],
            storage_uri=build_s3_uri(bucket, key),
            metadata={k: v for k, v in meta.items() if k not in ["content_hash", "size_bytes", "created_at"]},
        )

    def list_objects(
        self,
        bucket: str,
        prefix: str,
        recursive: bool = True,
    ) -> List[ArtifactMetadata]:
        """List objects matching prefix."""
        bucket_path = self._base_path / bucket
        if not bucket_path.exists():
            return []

        prefix_path = bucket_path / prefix
        if prefix_path.is_file():
            return [self.get_metadata(bucket, prefix)]

        results = []
        search_path = prefix_path if prefix_path.exists() else bucket_path

        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"

        for path in search_path.glob(pattern):
            if path.is_file() and not path.name.endswith(".meta.json"):
                relative = path.relative_to(bucket_path)
                key = str(relative).replace("\\", "/")
                if key.startswith(prefix) or prefix == "":
                    try:
                        results.append(self.get_metadata(bucket, key))
                    except ObjectNotFoundError:
                        pass

        return results

    def object_exists(self, bucket: str, key: str) -> bool:
        """Check if object exists."""
        return self._get_path(bucket, key).exists()

    def delete_object(self, bucket: str, key: str) -> bool:
        """Delete an object."""
        path = self._get_path(bucket, key)
        meta_path = path.with_suffix(path.suffix + ".meta.json")

        try:
            if path.exists():
                path.unlink()
            if meta_path.exists():
                meta_path.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to delete {bucket}/{key}: {e}")
            return False

    def copy_object(
        self,
        source_bucket: str,
        source_key: str,
        dest_bucket: str,
        dest_key: str,
    ) -> ArtifactMetadata:
        """Copy object."""
        data = self.get_object(source_bucket, source_key)

        try:
            source_meta = self.get_metadata(source_bucket, source_key)
            metadata = source_meta.metadata
        except ObjectNotFoundError:
            metadata = {}

        return self.put_object(dest_bucket, dest_key, data, metadata)

    def ensure_bucket_exists(self, bucket: str) -> bool:
        """Create bucket directory if needed."""
        bucket_path = self._base_path / bucket
        bucket_path.mkdir(parents=True, exist_ok=True)
        return True


# =============================================================================
# HIGH-LEVEL LOCAL DATASET REPOSITORY
# =============================================================================


class LocalDatasetRepository(IDatasetRepository):
    """
    Local filesystem implementation of dataset repository.

    Mirrors MinIODatasetRepository interface for testing.
    """

    BUCKET = EXPERIMENTS_BUCKET
    DATASET_PREFIX = "datasets"

    def __init__(self, storage: IObjectStorageRepository):
        """Initialize with low-level storage."""
        self._storage = storage

    def save_dataset(
        self,
        experiment_id: str,
        data: pd.DataFrame,
        version: str,
        metadata: Dict[str, Any],
    ) -> DatasetSnapshot:
        """Save dataset to local storage."""
        base_path = f"{experiment_id}/{self.DATASET_PREFIX}/{version}"

        # Save parquet
        parquet_buffer = BytesIO()
        data.to_parquet(parquet_buffer, index=False, compression="snappy")
        parquet_bytes = parquet_buffer.getvalue()
        data_hash = compute_content_hash(parquet_bytes)

        data_key = f"{base_path}/train.parquet"
        self._storage.put_object(self.BUCKET, data_key, parquet_bytes)

        # Save norm_stats
        norm_stats = self._compute_norm_stats(data)
        norm_stats_bytes = json.dumps(norm_stats, indent=2, sort_keys=True).encode()
        norm_stats_hash = compute_content_hash(norm_stats_bytes)
        norm_stats_key = f"{base_path}/norm_stats.json"
        self._storage.put_object(self.BUCKET, norm_stats_key, norm_stats_bytes)

        # Compute schema hash
        schema_hash = compute_schema_hash(list(data.columns))
        feature_order_hash = metadata.get("feature_order_hash", schema_hash)

        # Save manifest
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
        }
        manifest_bytes = json.dumps(manifest, indent=2).encode()
        manifest_key = f"{base_path}/manifest.json"
        self._storage.put_object(self.BUCKET, manifest_key, manifest_bytes)

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
            lineage=None,
            created_at=datetime.utcnow(),
        )

    def load_dataset(
        self,
        experiment_id: str,
        version: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load dataset from local storage."""
        if version is None:
            version = self._get_latest_version(experiment_id)

        data_key = f"{experiment_id}/{self.DATASET_PREFIX}/{version}/train.parquet"
        parquet_bytes = self._storage.get_object(self.BUCKET, data_key)
        return pd.read_parquet(BytesIO(parquet_bytes))

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
        """List all dataset versions."""
        prefix = f"{experiment_id}/{self.DATASET_PREFIX}/"
        objects = self._storage.list_objects(self.BUCKET, prefix, recursive=False)

        versions = []
        seen = set()

        for obj in objects:
            parts = obj.artifact_id.rstrip("/").split("/")
            if len(parts) >= 3:
                version = parts[2]
                if version not in seen:
                    seen.add(version)
                    try:
                        snapshot = self.get_snapshot(experiment_id, version)
                        versions.append(snapshot)
                    except (ObjectNotFoundError, StorageError):
                        pass

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
        """Compute normalization statistics."""
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
# HIGH-LEVEL LOCAL MODEL REPOSITORY
# =============================================================================


class LocalModelRepository(IModelRepository):
    """
    Local filesystem implementation of model repository.

    Mirrors MinIOModelRepository interface for testing.
    """

    BUCKET = EXPERIMENTS_BUCKET
    PRODUCTION_BUCKET = PRODUCTION_BUCKET
    MODELS_PREFIX = "models"

    def __init__(self, storage: IObjectStorageRepository):
        """Initialize with low-level storage."""
        self._storage = storage

    def save_model(
        self,
        experiment_id: str,
        model_path: Union[str, Path],
        norm_stats: Dict[str, Any],
        config: Dict[str, Any],
        lineage: LineageRecord,
        version: Optional[str] = None,
    ) -> ModelSnapshot:
        """Save trained model."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if version is None:
            version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        base_path = f"{experiment_id}/{self.MODELS_PREFIX}/{version}"

        # Upload model
        with open(model_path, "rb") as f:
            model_bytes = f.read()
        model_hash = compute_content_hash(model_bytes)
        model_key = f"{base_path}/policy.zip"
        self._storage.put_object(self.BUCKET, model_key, model_bytes)

        # Upload norm_stats
        norm_stats_bytes = json.dumps(norm_stats, indent=2, sort_keys=True).encode()
        norm_stats_hash = compute_content_hash(norm_stats_bytes)
        norm_stats_key = f"{base_path}/norm_stats.json"
        self._storage.put_object(self.BUCKET, norm_stats_key, norm_stats_bytes)

        # Upload config
        config_bytes = yaml.dump(config, default_flow_style=False).encode()
        config_hash = compute_content_hash(config_bytes)
        config_key = f"{base_path}/config.yaml"
        self._storage.put_object(self.BUCKET, config_key, config_bytes)

        # Upload lineage
        lineage_bytes = json.dumps(lineage.to_dict(), indent=2).encode()
        lineage_key = f"{base_path}/lineage.json"
        self._storage.put_object(self.BUCKET, lineage_key, lineage_bytes)

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
            dataset_snapshot=None,
            created_at=datetime.utcnow(),
        )

    def load_model(
        self,
        experiment_id: str,
        version: Optional[str] = None,
    ) -> bytes:
        """Load model bytes."""
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

        config_key = f"{base_path}/config.yaml"
        config_bytes = self._storage.get_object(self.BUCKET, config_key)
        config = yaml.safe_load(config_bytes)

        model_key = f"{base_path}/policy.zip"
        model_meta = self._storage.get_metadata(self.BUCKET, model_key)

        norm_stats_key = f"{base_path}/norm_stats.json"
        norm_stats_meta = self._storage.get_metadata(self.BUCKET, norm_stats_key)

        lineage_key = f"{base_path}/lineage.json"

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
        """List all model versions."""
        prefix = f"{experiment_id}/{self.MODELS_PREFIX}/"
        objects = self._storage.list_objects(self.BUCKET, prefix, recursive=False)

        versions = []
        seen = set()

        for obj in objects:
            parts = obj.artifact_id.rstrip("/").split("/")
            if len(parts) >= 3:
                version = parts[2]
                if version not in seen:
                    seen.add(version)
                    try:
                        snapshot = self.get_snapshot(experiment_id, version)
                        versions.append(snapshot)
                    except (ObjectNotFoundError, StorageError):
                        pass

        versions.sort(key=lambda s: s.created_at, reverse=True)
        return versions

    def promote_to_production(
        self,
        experiment_id: str,
        version: str,
        model_id: Optional[str] = None,
    ) -> str:
        """Copy model to production bucket."""
        snapshot = self.get_snapshot(experiment_id, version)
        if model_id is None:
            model_id = f"ppo_{version}_{snapshot.model_hash[:8]}"

        source_base = f"{experiment_id}/{self.MODELS_PREFIX}/{version}"
        dest_base = f"models/{model_id}"

        files = ["policy.zip", "norm_stats.json", "config.yaml", "lineage.json"]
        for filename in files:
            source_key = f"{source_base}/{filename}"
            dest_key = f"{dest_base}/{filename}"

            if self._storage.object_exists(self.BUCKET, source_key):
                self._storage.copy_object(
                    self.BUCKET, source_key,
                    self.PRODUCTION_BUCKET, dest_key,
                )

        return model_id

    def _get_latest_version(self, experiment_id: str) -> str:
        """Get the latest version for an experiment."""
        versions = self.list_versions(experiment_id)
        if not versions:
            raise ObjectNotFoundError(f"No models found for {experiment_id}")
        return versions[0].version


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "LocalStorageRepository",
    "LocalDatasetRepository",
    "LocalModelRepository",
]
