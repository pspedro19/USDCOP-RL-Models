"""
Storage Factory
===============

Factory Pattern implementation for creating storage repositories.

Supports multiple backends:
- MinIO (default for local/docker)
- AWS S3 (production cloud)
- Local filesystem (testing)

Contract: CTR-STORAGE-FACTORY-001
- Factory creates repositories from config
- Environment variables override config
- All repositories implement storage interfaces

Author: Trading Team
Version: 1.0.0
Created: 2026-01-18
"""

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import yaml

from src.core.interfaces.storage import (
    IObjectStorageRepository,
    IDatasetRepository,
    IModelRepository,
    IBacktestRepository,
    IABComparisonRepository,
)

logger = logging.getLogger(__name__)


# =============================================================================
# STORAGE BACKEND ENUM
# =============================================================================


class StorageBackend(str, Enum):
    """Supported storage backends."""
    MINIO = "minio"
    AWS_S3 = "aws_s3"
    LOCAL = "local"


# =============================================================================
# STORAGE CONFIGURATION
# =============================================================================


class StorageConfig:
    """
    Storage configuration with environment variable support.

    Loads from config file with environment variable overrides.
    """

    DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "storage_config.yaml"

    def __init__(
        self,
        backend: StorageBackend = StorageBackend.MINIO,
        endpoint: str = "localhost:9000",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin",
        secure: bool = False,
        region: Optional[str] = None,
        experiments_bucket: str = "experiments",
        production_bucket: str = "production",
        dvc_bucket: str = "dvc-storage",
    ):
        self.backend = backend
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure
        self.region = region
        self.experiments_bucket = experiments_bucket
        self.production_bucket = production_bucket
        self.dvc_bucket = dvc_bucket

    @classmethod
    def from_env(cls) -> "StorageConfig":
        """
        Create config from environment variables.

        Environment Variables:
            STORAGE_BACKEND: minio, aws_s3, or local
            MINIO_ENDPOINT: Server endpoint
            MINIO_ACCESS_KEY: Access key
            MINIO_SECRET_KEY: Secret key
            MINIO_SECURE: Use HTTPS (true/false)
            AWS_REGION: AWS region (for S3)
            EXPERIMENTS_BUCKET: Experiments bucket name
            PRODUCTION_BUCKET: Production bucket name
        """
        backend_str = os.environ.get("STORAGE_BACKEND", "minio")
        try:
            backend = StorageBackend(backend_str.lower())
        except ValueError:
            logger.warning(f"Unknown backend '{backend_str}', defaulting to MinIO")
            backend = StorageBackend.MINIO

        return cls(
            backend=backend,
            endpoint=os.environ.get("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
            secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
            region=os.environ.get("AWS_REGION"),
            experiments_bucket=os.environ.get("EXPERIMENTS_BUCKET", "experiments"),
            production_bucket=os.environ.get("PRODUCTION_BUCKET", "production"),
            dvc_bucket=os.environ.get("DVC_BUCKET", "dvc-storage"),
        )

    @classmethod
    def from_file(cls, config_path: Optional[Union[str, Path]] = None) -> "StorageConfig":
        """
        Load config from YAML file with env overrides.

        Args:
            config_path: Path to config file (default: config/storage_config.yaml)
        """
        config_path = Path(config_path) if config_path else cls.DEFAULT_CONFIG_PATH

        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls.from_env()

        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}

        # Environment variables override file config
        backend_str = os.environ.get("STORAGE_BACKEND", data.get("backend", "minio"))
        try:
            backend = StorageBackend(backend_str.lower())
        except ValueError:
            backend = StorageBackend.MINIO

        return cls(
            backend=backend,
            endpoint=os.environ.get("MINIO_ENDPOINT", data.get("endpoint", "localhost:9000")),
            access_key=os.environ.get("MINIO_ACCESS_KEY", data.get("access_key", "minioadmin")),
            secret_key=os.environ.get("MINIO_SECRET_KEY", data.get("secret_key", "minioadmin")),
            secure=os.environ.get("MINIO_SECURE", str(data.get("secure", False))).lower() == "true",
            region=os.environ.get("AWS_REGION", data.get("region")),
            experiments_bucket=os.environ.get(
                "EXPERIMENTS_BUCKET",
                data.get("buckets", {}).get("experiments", "experiments")
            ),
            production_bucket=os.environ.get(
                "PRODUCTION_BUCKET",
                data.get("buckets", {}).get("production", "production")
            ),
            dvc_bucket=os.environ.get(
                "DVC_BUCKET",
                data.get("buckets", {}).get("dvc", "dvc-storage")
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backend": self.backend.value,
            "endpoint": self.endpoint,
            "access_key": self.access_key,
            "secret_key": "***",  # Don't log secrets
            "secure": self.secure,
            "region": self.region,
            "buckets": {
                "experiments": self.experiments_bucket,
                "production": self.production_bucket,
                "dvc": self.dvc_bucket,
            },
        }


# =============================================================================
# STORAGE FACTORY
# =============================================================================


class StorageFactory:
    """
    Factory for creating storage repositories.

    Supports:
    - Multiple backends (MinIO, S3, local)
    - Configuration from file or environment
    - Lazy initialization

    Example:
        >>> factory = StorageFactory.from_env()
        >>> dataset_repo = factory.create_dataset_repository()
        >>> model_repo = factory.create_model_repository()
    """

    _instance: Optional["StorageFactory"] = None
    _object_storage: Optional[IObjectStorageRepository] = None

    def __init__(self, config: StorageConfig):
        """
        Initialize factory with configuration.

        Args:
            config: StorageConfig instance
        """
        self._config = config
        logger.info(f"StorageFactory initialized with backend: {config.backend.value}")

    @classmethod
    def from_env(cls) -> "StorageFactory":
        """Create factory from environment variables."""
        return cls(StorageConfig.from_env())

    @classmethod
    def from_config(cls, config_path: Optional[Union[str, Path]] = None) -> "StorageFactory":
        """Create factory from config file."""
        return cls(StorageConfig.from_file(config_path))

    @classmethod
    def get_instance(cls) -> "StorageFactory":
        """
        Get singleton factory instance.

        Creates from environment if not exists.
        """
        if cls._instance is None:
            cls._instance = cls.from_env()
        return cls._instance

    @classmethod
    def set_instance(cls, factory: "StorageFactory") -> None:
        """Set singleton instance (for testing)."""
        cls._instance = factory

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None
        cls._object_storage = None

    @property
    def config(self) -> StorageConfig:
        """Get current configuration."""
        return self._config

    def create_object_storage(self) -> IObjectStorageRepository:
        """
        Create low-level object storage repository.

        Returns appropriate implementation based on backend config.
        """
        if self._object_storage is not None:
            return self._object_storage

        if self._config.backend == StorageBackend.MINIO:
            from src.infrastructure.repositories.minio_repository import MinIORepository
            self._object_storage = MinIORepository(
                endpoint=self._config.endpoint,
                access_key=self._config.access_key,
                secret_key=self._config.secret_key,
                secure=self._config.secure,
                region=self._config.region,
            )

        elif self._config.backend == StorageBackend.AWS_S3:
            # AWS S3 uses same MinIO client with S3 endpoint
            from src.infrastructure.repositories.minio_repository import MinIORepository
            self._object_storage = MinIORepository(
                endpoint="s3.amazonaws.com",
                access_key=self._config.access_key,
                secret_key=self._config.secret_key,
                secure=True,
                region=self._config.region,
            )

        elif self._config.backend == StorageBackend.LOCAL:
            from src.infrastructure.repositories.local_storage_repository import LocalStorageRepository
            self._object_storage = LocalStorageRepository(
                base_path=Path(self._config.endpoint)
            )

        else:
            raise ValueError(f"Unknown backend: {self._config.backend}")

        return self._object_storage

    def create_dataset_repository(self) -> IDatasetRepository:
        """Create high-level dataset repository."""
        if self._config.backend == StorageBackend.LOCAL:
            from src.infrastructure.repositories.local_storage_repository import LocalDatasetRepository
            return LocalDatasetRepository(self.create_object_storage())

        from src.infrastructure.repositories.minio_repository import MinIODatasetRepository
        return MinIODatasetRepository(self.create_object_storage())

    def create_model_repository(self) -> IModelRepository:
        """Create high-level model repository."""
        if self._config.backend == StorageBackend.LOCAL:
            from src.infrastructure.repositories.local_storage_repository import LocalModelRepository
            return LocalModelRepository(self.create_object_storage())

        from src.infrastructure.repositories.minio_repository import MinIOModelRepository
        return MinIOModelRepository(self.create_object_storage())

    def create_backtest_repository(self) -> IBacktestRepository:
        """Create high-level backtest repository."""
        from src.infrastructure.repositories.minio_repository import MinIOBacktestRepository
        return MinIOBacktestRepository(self.create_object_storage())

    def create_ab_comparison_repository(self) -> IABComparisonRepository:
        """Create high-level A/B comparison repository."""
        from src.infrastructure.repositories.minio_repository import MinIOABComparisonRepository
        return MinIOABComparisonRepository(self.create_object_storage())


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_storage_factory() -> StorageFactory:
    """Get singleton storage factory."""
    return StorageFactory.get_instance()


def get_dataset_repository() -> IDatasetRepository:
    """Get dataset repository from singleton factory."""
    return StorageFactory.get_instance().create_dataset_repository()


def get_model_repository() -> IModelRepository:
    """Get model repository from singleton factory."""
    return StorageFactory.get_instance().create_model_repository()


def get_backtest_repository() -> IBacktestRepository:
    """Get backtest repository from singleton factory."""
    return StorageFactory.get_instance().create_backtest_repository()


def get_ab_comparison_repository() -> IABComparisonRepository:
    """Get A/B comparison repository from singleton factory."""
    return StorageFactory.get_instance().create_ab_comparison_repository()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "StorageBackend",
    # Config
    "StorageConfig",
    # Factory
    "StorageFactory",
    # Convenience functions
    "get_storage_factory",
    "get_dataset_repository",
    "get_model_repository",
    "get_backtest_repository",
    "get_ab_comparison_repository",
]
