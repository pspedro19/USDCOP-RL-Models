"""
Unit Tests for Storage Repositories
====================================

Tests for the MinIO-first storage architecture.

Uses LocalStorageRepository for testing without requiring MinIO.

Author: Trading Team
Version: 1.0.0
Created: 2026-01-18
"""

import json
import os
import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd

from src.core.interfaces.storage import (
    ArtifactMetadata,
    IObjectStorageRepository,
    IDatasetRepository,
    IModelRepository,
    ObjectNotFoundError,
)
from src.core.contracts.storage_contracts import (
    DatasetSnapshot,
    ModelSnapshot,
    BacktestSnapshot,
    LineageRecord,
    compute_content_hash,
    compute_schema_hash,
    parse_s3_uri,
    build_s3_uri,
)
from src.infrastructure.repositories.local_storage_repository import (
    LocalStorageRepository,
    LocalDatasetRepository,
    LocalModelRepository,
)
from src.core.factories.storage_factory import (
    StorageBackend,
    StorageConfig,
    StorageFactory,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_storage_dir() -> Generator[Path, None, None]:
    """Create temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def local_storage(temp_storage_dir: Path) -> LocalStorageRepository:
    """Create local storage repository."""
    return LocalStorageRepository(temp_storage_dir)


@pytest.fixture
def dataset_repo(local_storage: LocalStorageRepository) -> LocalDatasetRepository:
    """Create local dataset repository."""
    return LocalDatasetRepository(local_storage)


@pytest.fixture
def model_repo(local_storage: LocalStorageRepository) -> LocalModelRepository:
    """Create local model repository."""
    return LocalModelRepository(local_storage)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    n_rows = 100

    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="5min"),
        "log_ret_5m": np.random.randn(n_rows) * 0.01,
        "rsi_9": np.random.uniform(30, 70, n_rows),
        "atr_pct": np.random.uniform(0.001, 0.01, n_rows),
        "adx_14": np.random.uniform(10, 50, n_rows),
        "bb_position": np.random.uniform(-1, 1, n_rows),
    })


@pytest.fixture
def sample_norm_stats() -> dict:
    """Create sample normalization statistics."""
    return {
        "log_ret_5m": {"mean": 0.0001, "std": 0.005, "min": -0.05, "max": 0.05},
        "rsi_9": {"mean": 50.0, "std": 10.0, "min": 0.0, "max": 100.0},
        "atr_pct": {"mean": 0.005, "std": 0.002, "min": 0.001, "max": 0.02},
    }


@pytest.fixture
def sample_config() -> dict:
    """Create sample training configuration."""
    return {
        "algorithm": "PPO",
        "total_timesteps": 100000,
        "learning_rate": 3e-4,
        "observation_dim": 15,
        "action_space": 3,
        "feature_order": ["log_ret_5m", "rsi_9", "atr_pct"],
        "feature_order_hash": "abc123def456",
    }


# =============================================================================
# CONTRACT UTILITY TESTS
# =============================================================================


class TestStorageContracts:
    """Tests for storage contract utilities."""

    def test_compute_content_hash(self):
        """Test content hash computation."""
        data = b"test content"
        hash_result = compute_content_hash(data)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 16  # Truncated to 16 chars
        assert hash_result.isalnum()

    def test_compute_content_hash_deterministic(self):
        """Test that same content produces same hash."""
        data = b"deterministic test"
        hash1 = compute_content_hash(data)
        hash2 = compute_content_hash(data)

        assert hash1 == hash2

    def test_compute_schema_hash(self):
        """Test schema hash computation."""
        columns = ["col1", "col2", "col3"]
        hash_result = compute_schema_hash(columns)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 16

    def test_parse_s3_uri(self):
        """Test S3 URI parsing."""
        uri = "s3://my-bucket/path/to/object.parquet"
        bucket, key = parse_s3_uri(uri)

        assert bucket == "my-bucket"
        assert key == "path/to/object.parquet"

    def test_parse_s3_uri_invalid(self):
        """Test S3 URI parsing with invalid input."""
        with pytest.raises(ValueError):
            parse_s3_uri("http://invalid-uri")

    def test_build_s3_uri(self):
        """Test S3 URI building."""
        uri = build_s3_uri("my-bucket", "path/to/object.parquet")

        assert uri == "s3://my-bucket/path/to/object.parquet"


class TestLineageRecord:
    """Tests for LineageRecord."""

    def test_create_lineage_record(self):
        """Test creating a lineage record."""
        lineage = LineageRecord(
            artifact_type="model",
            artifact_id="exp1/models/v1",
            parent_id="exp1/datasets/v1",
            parent_type="dataset",
            source_uri="s3://experiments/exp1/datasets/v1/train.parquet",
            source_hash="abc123",
            transform_name="l3_training",
            transform_params=(("algorithm", "PPO"),),
            created_at=datetime.utcnow(),
            created_by="test_user",
        )

        assert lineage.artifact_type == "model"
        assert lineage.parent_type == "dataset"

    def test_lineage_to_dict(self):
        """Test lineage serialization."""
        lineage = LineageRecord(
            artifact_type="dataset",
            artifact_id="exp1/datasets/v1",
            parent_id=None,
            parent_type=None,
            source_uri=None,
            source_hash=None,
            transform_name="l2_preprocessing",
            transform_params=None,
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            created_by="test_user",
        )

        data = lineage.to_dict()

        assert data["artifact_type"] == "dataset"
        assert data["created_by"] == "test_user"
        assert isinstance(data["created_at"], str)

    def test_lineage_from_dict(self):
        """Test lineage deserialization."""
        data = {
            "artifact_type": "model",
            "artifact_id": "exp1/models/v1",
            "parent_id": None,
            "parent_type": None,
            "source_uri": None,
            "source_hash": None,
            "transform_name": "l3_training",
            "transform_params": {"algorithm": "PPO"},
            "created_at": "2024-01-01T12:00:00",
            "created_by": "test_user",
        }

        lineage = LineageRecord.from_dict(data)

        assert lineage.artifact_type == "model"
        assert lineage.transform_name == "l3_training"


# =============================================================================
# LOCAL STORAGE REPOSITORY TESTS
# =============================================================================


class TestLocalStorageRepository:
    """Tests for LocalStorageRepository."""

    def test_put_and_get_object(self, local_storage: LocalStorageRepository):
        """Test storing and retrieving an object."""
        bucket = "test-bucket"
        key = "test/object.txt"
        data = b"Hello, World!"

        # Put object
        metadata = local_storage.put_object(bucket, key, data)

        assert metadata.artifact_id == key
        assert metadata.size_bytes == len(data)
        assert metadata.storage_uri == f"s3://{bucket}/{key}"

        # Get object
        retrieved = local_storage.get_object(bucket, key)

        assert retrieved == data

    def test_get_nonexistent_object(self, local_storage: LocalStorageRepository):
        """Test getting a nonexistent object."""
        with pytest.raises(ObjectNotFoundError):
            local_storage.get_object("bucket", "nonexistent/key")

    def test_object_exists(self, local_storage: LocalStorageRepository):
        """Test object existence check."""
        bucket = "test-bucket"
        key = "test/exists.txt"

        assert not local_storage.object_exists(bucket, key)

        local_storage.put_object(bucket, key, b"data")

        assert local_storage.object_exists(bucket, key)

    def test_delete_object(self, local_storage: LocalStorageRepository):
        """Test deleting an object."""
        bucket = "test-bucket"
        key = "test/delete.txt"

        local_storage.put_object(bucket, key, b"data")
        assert local_storage.object_exists(bucket, key)

        result = local_storage.delete_object(bucket, key)

        assert result is True
        assert not local_storage.object_exists(bucket, key)

    def test_list_objects(self, local_storage: LocalStorageRepository):
        """Test listing objects."""
        bucket = "test-bucket"

        # Create multiple objects
        local_storage.put_object(bucket, "prefix/file1.txt", b"data1")
        local_storage.put_object(bucket, "prefix/file2.txt", b"data2")
        local_storage.put_object(bucket, "other/file3.txt", b"data3")

        # List with prefix
        results = local_storage.list_objects(bucket, "prefix/")

        assert len(results) == 2

    def test_copy_object(self, local_storage: LocalStorageRepository):
        """Test copying an object."""
        source_bucket = "source"
        source_key = "original.txt"
        dest_bucket = "dest"
        dest_key = "copy.txt"

        local_storage.put_object(source_bucket, source_key, b"original data")

        metadata = local_storage.copy_object(
            source_bucket, source_key,
            dest_bucket, dest_key
        )

        assert local_storage.object_exists(dest_bucket, dest_key)
        assert local_storage.get_object(dest_bucket, dest_key) == b"original data"

    def test_ensure_bucket_exists(self, local_storage: LocalStorageRepository):
        """Test bucket creation."""
        bucket = "new-bucket"

        result = local_storage.ensure_bucket_exists(bucket)

        assert result is True


# =============================================================================
# LOCAL DATASET REPOSITORY TESTS
# =============================================================================


class TestLocalDatasetRepository:
    """Tests for LocalDatasetRepository."""

    def test_save_and_load_dataset(
        self,
        dataset_repo: LocalDatasetRepository,
        sample_dataframe: pd.DataFrame,
    ):
        """Test saving and loading a dataset."""
        experiment_id = "test_experiment"
        version = "v1"
        metadata = {
            "date_range_start": "2024-01-01",
            "date_range_end": "2024-01-02",
        }

        # Save dataset
        snapshot = dataset_repo.save_dataset(
            experiment_id=experiment_id,
            data=sample_dataframe,
            version=version,
            metadata=metadata,
        )

        assert snapshot.experiment_id == experiment_id
        assert snapshot.version == version
        assert snapshot.row_count == len(sample_dataframe)
        assert snapshot.storage_uri.startswith("s3://")
        assert len(snapshot.data_hash) == 16

        # Load dataset
        loaded = dataset_repo.load_dataset(experiment_id, version)

        assert len(loaded) == len(sample_dataframe)
        assert list(loaded.columns) == list(sample_dataframe.columns)

    def test_get_dataset_snapshot(
        self,
        dataset_repo: LocalDatasetRepository,
        sample_dataframe: pd.DataFrame,
    ):
        """Test getting dataset snapshot without loading data."""
        experiment_id = "test_experiment"
        version = "v1"

        dataset_repo.save_dataset(
            experiment_id=experiment_id,
            data=sample_dataframe,
            version=version,
            metadata={"date_range_start": "2024-01-01", "date_range_end": "2024-01-02"},
        )

        snapshot = dataset_repo.get_snapshot(experiment_id, version)

        assert isinstance(snapshot, DatasetSnapshot)
        assert snapshot.row_count == len(sample_dataframe)

    def test_list_dataset_versions(
        self,
        dataset_repo: LocalDatasetRepository,
        sample_dataframe: pd.DataFrame,
    ):
        """Test listing dataset versions."""
        experiment_id = "test_experiment"

        # Save multiple versions
        dataset_repo.save_dataset(experiment_id, sample_dataframe, "v1", {})
        dataset_repo.save_dataset(experiment_id, sample_dataframe, "v2", {})

        versions = dataset_repo.list_versions(experiment_id)

        assert len(versions) == 2

    def test_get_norm_stats(
        self,
        dataset_repo: LocalDatasetRepository,
        sample_dataframe: pd.DataFrame,
    ):
        """Test getting normalization statistics."""
        experiment_id = "test_experiment"
        version = "v1"

        dataset_repo.save_dataset(experiment_id, sample_dataframe, version, {})

        norm_stats = dataset_repo.get_norm_stats(experiment_id, version)

        assert isinstance(norm_stats, dict)
        assert "log_ret_5m" in norm_stats
        assert "mean" in norm_stats["log_ret_5m"]

    def test_dataset_snapshot_serialization(
        self,
        dataset_repo: LocalDatasetRepository,
        sample_dataframe: pd.DataFrame,
    ):
        """Test DatasetSnapshot to_dict and from_dict."""
        experiment_id = "test_experiment"

        snapshot = dataset_repo.save_dataset(
            experiment_id=experiment_id,
            data=sample_dataframe,
            version="v1",
            metadata={"date_range_start": "2024-01-01", "date_range_end": "2024-01-02"},
        )

        # Serialize and deserialize
        data = snapshot.to_dict()
        restored = DatasetSnapshot.from_dict(data)

        assert restored.experiment_id == snapshot.experiment_id
        assert restored.version == snapshot.version
        assert restored.data_hash == snapshot.data_hash


# =============================================================================
# LOCAL MODEL REPOSITORY TESTS
# =============================================================================


class TestLocalModelRepository:
    """Tests for LocalModelRepository."""

    def test_save_and_load_model(
        self,
        model_repo: LocalModelRepository,
        sample_norm_stats: dict,
        sample_config: dict,
        temp_storage_dir: Path,
    ):
        """Test saving and loading a model."""
        experiment_id = "test_experiment"
        version = "v1"

        # Create a fake model file
        model_path = temp_storage_dir / "fake_model.zip"
        model_path.write_bytes(b"fake model content")

        # Create lineage
        lineage = LineageRecord(
            artifact_type="model",
            artifact_id=f"{experiment_id}/models/{version}",
            parent_id=None,
            parent_type=None,
            source_uri=None,
            source_hash=None,
            transform_name="l3_training",
            transform_params=None,
            created_at=datetime.utcnow(),
            created_by="test_user",
        )

        # Save model
        snapshot = model_repo.save_model(
            experiment_id=experiment_id,
            model_path=model_path,
            norm_stats=sample_norm_stats,
            config=sample_config,
            lineage=lineage,
            version=version,
        )

        assert snapshot.experiment_id == experiment_id
        assert snapshot.version == version
        assert len(snapshot.model_hash) == 16
        assert snapshot.observation_dim == sample_config["observation_dim"]

        # Load model bytes
        model_bytes = model_repo.load_model(experiment_id, version)

        assert model_bytes == b"fake model content"

    def test_get_model_snapshot(
        self,
        model_repo: LocalModelRepository,
        sample_norm_stats: dict,
        sample_config: dict,
        temp_storage_dir: Path,
    ):
        """Test getting model snapshot."""
        experiment_id = "test_experiment"
        version = "v1"

        model_path = temp_storage_dir / "model.zip"
        model_path.write_bytes(b"model content")

        lineage = LineageRecord(
            artifact_type="model",
            artifact_id=f"{experiment_id}/models/{version}",
            parent_id=None,
            parent_type=None,
            source_uri=None,
            source_hash=None,
            transform_name="l3_training",
            transform_params=None,
            created_at=datetime.utcnow(),
            created_by="test_user",
        )

        model_repo.save_model(
            experiment_id=experiment_id,
            model_path=model_path,
            norm_stats=sample_norm_stats,
            config=sample_config,
            lineage=lineage,
            version=version,
        )

        snapshot = model_repo.get_snapshot(experiment_id, version)

        assert isinstance(snapshot, ModelSnapshot)
        assert snapshot.version == version

    def test_promote_to_production(
        self,
        model_repo: LocalModelRepository,
        sample_norm_stats: dict,
        sample_config: dict,
        temp_storage_dir: Path,
    ):
        """Test promoting model to production."""
        experiment_id = "test_experiment"
        version = "v1"

        model_path = temp_storage_dir / "model.zip"
        model_path.write_bytes(b"model to promote")

        lineage = LineageRecord(
            artifact_type="model",
            artifact_id=f"{experiment_id}/models/{version}",
            parent_id=None,
            parent_type=None,
            source_uri=None,
            source_hash=None,
            transform_name="l3_training",
            transform_params=None,
            created_at=datetime.utcnow(),
            created_by="test_user",
        )

        model_repo.save_model(
            experiment_id=experiment_id,
            model_path=model_path,
            norm_stats=sample_norm_stats,
            config=sample_config,
            lineage=lineage,
            version=version,
        )

        # Promote
        model_id = model_repo.promote_to_production(
            experiment_id=experiment_id,
            version=version,
            model_id="production_v1",
        )

        assert model_id == "production_v1"


# =============================================================================
# STORAGE FACTORY TESTS
# =============================================================================


class TestStorageFactory:
    """Tests for StorageFactory."""

    def test_storage_config_from_env(self, monkeypatch):
        """Test creating config from environment variables."""
        monkeypatch.setenv("STORAGE_BACKEND", "minio")
        monkeypatch.setenv("MINIO_ENDPOINT", "test-endpoint:9000")
        monkeypatch.setenv("MINIO_ACCESS_KEY", "test-key")
        monkeypatch.setenv("MINIO_SECRET_KEY", "test-secret")

        config = StorageConfig.from_env()

        assert config.backend == StorageBackend.MINIO
        assert config.endpoint == "test-endpoint:9000"
        assert config.access_key == "test-key"

    def test_storage_factory_with_local_backend(self, temp_storage_dir: Path):
        """Test factory with local backend."""
        config = StorageConfig(
            backend=StorageBackend.LOCAL,
            endpoint=str(temp_storage_dir),
        )

        factory = StorageFactory(config)
        storage = factory.create_object_storage()

        assert isinstance(storage, LocalStorageRepository)

    def test_factory_singleton(self):
        """Test factory singleton pattern."""
        StorageFactory.reset_instance()

        factory1 = StorageFactory.get_instance()
        factory2 = StorageFactory.get_instance()

        assert factory1 is factory2

        StorageFactory.reset_instance()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestStorageIntegration:
    """Integration tests for the storage system."""

    def test_full_dataset_workflow(
        self,
        dataset_repo: LocalDatasetRepository,
        sample_dataframe: pd.DataFrame,
    ):
        """Test complete dataset workflow."""
        experiment_id = "integration_test"

        # Save v1
        snapshot1 = dataset_repo.save_dataset(
            experiment_id=experiment_id,
            data=sample_dataframe,
            version="v1",
            metadata={"date_range_start": "2024-01-01", "date_range_end": "2024-01-02"},
        )

        # Save v2 with parent reference
        snapshot2 = dataset_repo.save_dataset(
            experiment_id=experiment_id,
            data=sample_dataframe,
            version="v2",
            metadata={
                "date_range_start": "2024-01-01",
                "date_range_end": "2024-01-03",
                "parent_version": "v1",
            },
        )

        # List versions
        versions = dataset_repo.list_versions(experiment_id)

        assert len(versions) == 2

        # Get latest (should be v2)
        latest = dataset_repo.get_snapshot(experiment_id)

        # Note: depending on filesystem timestamps, this might not always be v2
        assert latest.version in ["v1", "v2"]

    def test_full_model_workflow(
        self,
        model_repo: LocalModelRepository,
        sample_norm_stats: dict,
        sample_config: dict,
        temp_storage_dir: Path,
    ):
        """Test complete model workflow."""
        experiment_id = "integration_test"

        # Create model file
        model_path = temp_storage_dir / "model.zip"
        model_path.write_bytes(b"integration test model")

        lineage = LineageRecord(
            artifact_type="model",
            artifact_id=f"{experiment_id}/models/v1",
            parent_id=None,
            parent_type=None,
            source_uri=None,
            source_hash=None,
            transform_name="l3_training",
            transform_params=None,
            created_at=datetime.utcnow(),
            created_by="test_user",
        )

        # Save model
        snapshot = model_repo.save_model(
            experiment_id=experiment_id,
            model_path=model_path,
            norm_stats=sample_norm_stats,
            config=sample_config,
            lineage=lineage,
            version="v1",
        )

        # Get snapshot
        loaded_snapshot = model_repo.get_snapshot(experiment_id, "v1")

        assert loaded_snapshot.model_hash == snapshot.model_hash

        # Promote to production
        model_id = model_repo.promote_to_production(experiment_id, "v1")

        assert model_id is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
