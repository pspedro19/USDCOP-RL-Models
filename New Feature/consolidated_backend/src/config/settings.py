# backend/src/config/settings.py
"""
Centralized Settings Configuration for USD/COP Forecasting Pipeline.

This module centralizes ALL environment variables and configuration settings
using pydantic-settings for validation and type safety.

Environment variables can be set in:
- .env file in project root
- System environment variables
- Docker/Kubernetes environment

Usage:
    from backend.src.config import get_settings, DATA_DIR

    settings = get_settings()
    print(settings.database_url)
    print(DATA_DIR / "features.csv")
"""

import os
from pathlib import Path
from functools import lru_cache
from typing import Optional, List
from dataclasses import dataclass, field

# Try to import pydantic-settings, fallback to dataclass if not available
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    try:
        from pydantic import BaseSettings, Field, validator
        PYDANTIC_AVAILABLE = True
    except ImportError:
        PYDANTIC_AVAILABLE = False


# =============================================================================
# PATH CONSTANTS - Relative to project
# =============================================================================

# Determine project root dynamically
def _find_project_root() -> Path:
    """Find project root by looking for marker files."""
    current = Path(__file__).resolve()

    # Walk up the directory tree
    for parent in [current] + list(current.parents):
        # Check for project markers
        if (parent / "backend").exists() and (parent / "api").exists():
            return parent
        if (parent / "requirements.txt").exists() and (parent / "backend").exists():
            return parent

    # Fallback: assume standard structure
    # config.py is at backend/src/config/settings.py
    return current.parent.parent.parent.parent


PROJECT_ROOT = _find_project_root()
BASE_DIR = PROJECT_ROOT  # Alias for compatibility
BACKEND_DIR = PROJECT_ROOT / "backend"
API_DIR = PROJECT_ROOT / "api"
DATA_ENGINEERING_DIR = PROJECT_ROOT / "data-engineering"

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"
LOGS_DIR = OUTPUT_DIR / "logs"

# Infrastructure
INFRASTRUCTURE_DIR = PROJECT_ROOT / "infrastructure"


# =============================================================================
# SETTINGS CLASS
# =============================================================================

if PYDANTIC_AVAILABLE:
    class Settings(BaseSettings):
        """
        Application settings loaded from environment variables.

        All settings can be overridden via environment variables.
        Prefix is optional - variables can be with or without USDCOP_ prefix.
        """

        # =================================================================
        # Application
        # =================================================================
        app_name: str = Field(default="USD/COP Forecasting Pipeline", alias="APP_NAME")
        app_env: str = Field(default="development", alias="APP_ENV")
        debug: bool = Field(default=False, alias="DEBUG")
        log_level: str = Field(default="INFO", alias="LOG_LEVEL")

        # =================================================================
        # Database (PostgreSQL)
        # =================================================================
        db_host: str = Field(default="localhost", alias="DB_HOST")
        db_port: int = Field(default=5432, alias="DB_PORT")
        db_name: str = Field(default="usdcop", alias="DB_NAME")
        db_user: str = Field(default="postgres", alias="DB_USER")
        db_password: str = Field(default="postgres", alias="DB_PASSWORD")
        db_schema: str = Field(default="core", alias="DB_SCHEMA")

        @property
        def database_url(self) -> str:
            """Construct database URL."""
            return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

        @property
        def async_database_url(self) -> str:
            """Construct async database URL for asyncpg."""
            return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

        # =================================================================
        # MinIO / S3 Object Storage
        # =================================================================
        minio_endpoint: str = Field(default="localhost:9000", alias="MINIO_ENDPOINT")
        minio_access_key: str = Field(default="minioadmin", alias="MINIO_ACCESS_KEY")
        minio_secret_key: str = Field(default="minioadmin", alias="MINIO_SECRET_KEY")
        minio_bucket: str = Field(default="ml-models", alias="MINIO_BUCKET")
        minio_secure: bool = Field(default=False, alias="MINIO_SECURE")

        # AWS S3 Compatible settings
        aws_access_key_id: Optional[str] = Field(default=None, alias="AWS_ACCESS_KEY_ID")
        aws_secret_access_key: Optional[str] = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
        aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
        s3_endpoint_url: Optional[str] = Field(default=None, alias="S3_ENDPOINT_URL")

        # =================================================================
        # MLflow
        # =================================================================
        mlflow_tracking_uri: str = Field(default="./mlruns", alias="MLFLOW_TRACKING_URI")
        mlflow_experiment_name: str = Field(default="usdcop-forecasting", alias="MLFLOW_EXPERIMENT_NAME")
        mlflow_artifact_location: Optional[str] = Field(default=None, alias="MLFLOW_ARTIFACT_LOCATION")
        mlflow_registry_uri: Optional[str] = Field(default=None, alias="MLFLOW_REGISTRY_URI")

        # =================================================================
        # API Settings
        # =================================================================
        api_host: str = Field(default="0.0.0.0", alias="API_HOST")
        api_port: int = Field(default=8000, alias="API_PORT")
        api_workers: int = Field(default=4, alias="API_WORKERS")
        api_reload: bool = Field(default=False, alias="API_RELOAD")

        # CORS
        cors_origins: str = Field(default="*", alias="CORS_ORIGINS")

        # JWT / Authentication
        jwt_secret_key: str = Field(default="your-secret-key-change-in-production", alias="JWT_SECRET_KEY")
        jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
        jwt_expiration_minutes: int = Field(default=30, alias="JWT_EXPIRATION_MINUTES")

        # =================================================================
        # Redis (Caching)
        # =================================================================
        redis_host: str = Field(default="localhost", alias="REDIS_HOST")
        redis_port: int = Field(default=6379, alias="REDIS_PORT")
        redis_db: int = Field(default=0, alias="REDIS_DB")
        redis_password: Optional[str] = Field(default=None, alias="REDIS_PASSWORD")

        @property
        def redis_url(self) -> str:
            """Construct Redis URL."""
            if self.redis_password:
                return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
            return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

        # =================================================================
        # Airflow
        # =================================================================
        airflow_home: str = Field(default="/opt/airflow", alias="AIRFLOW_HOME")
        airflow_dags_folder: str = Field(default="/opt/airflow/dags", alias="AIRFLOW_DAGS_FOLDER")

        # =================================================================
        # Data Sources / External APIs
        # =================================================================
        fred_api_key: Optional[str] = Field(default=None, alias="FRED_API_KEY")
        twelvedata_api_key: Optional[str] = Field(default=None, alias="TWELVEDATA_API_KEY")

        # =================================================================
        # Email / Notifications
        # =================================================================
        smtp_host: Optional[str] = Field(default=None, alias="SMTP_HOST")
        smtp_port: int = Field(default=587, alias="SMTP_PORT")
        smtp_user: Optional[str] = Field(default=None, alias="SMTP_USER")
        smtp_password: Optional[str] = Field(default=None, alias="SMTP_PASSWORD")
        alert_email: Optional[str] = Field(default=None, alias="ALERT_EMAIL")

        # =================================================================
        # ML Pipeline Settings
        # =================================================================
        ml_horizons: str = Field(default="1,5,10,15,20,25,30", alias="ML_HORIZONS")
        ml_models: str = Field(default="ridge,bayesian_ridge,xgboost,lightgbm,catboost", alias="ML_MODELS")
        ml_random_state: int = Field(default=42, alias="ML_RANDOM_STATE")
        ml_train_size: float = Field(default=0.8, alias="ML_TRAIN_SIZE")
        ml_n_features: int = Field(default=15, alias="ML_N_FEATURES")
        optuna_n_trials: int = Field(default=50, alias="OPTUNA_N_TRIALS")

        @property
        def horizons_list(self) -> List[int]:
            """Parse horizons string to list of integers."""
            return [int(h.strip()) for h in self.ml_horizons.split(",")]

        @property
        def models_list(self) -> List[str]:
            """Parse models string to list."""
            return [m.strip() for m in self.ml_models.split(",")]

        # =================================================================
        # Paths (can override defaults)
        # =================================================================
        data_path: Optional[str] = Field(default=None, alias="DATA_PATH")
        output_path: Optional[str] = Field(default=None, alias="OUTPUT_PATH")
        models_path: Optional[str] = Field(default=None, alias="MODELS_PATH")

        def get_data_dir(self) -> Path:
            """Get data directory, using env var if set."""
            if self.data_path:
                return Path(self.data_path)
            return DATA_DIR

        def get_output_dir(self) -> Path:
            """Get output directory, using env var if set."""
            if self.output_path:
                return Path(self.output_path)
            return OUTPUT_DIR

        def get_models_dir(self) -> Path:
            """Get models directory, using env var if set."""
            if self.models_path:
                return Path(self.models_path)
            return MODELS_DIR

        class Config:
            """Pydantic config."""
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = False
            extra = "ignore"  # Ignore extra env vars

else:
    # Fallback dataclass implementation if pydantic not available
    @dataclass
    class Settings:
        """Application settings (dataclass fallback)."""

        # Application
        app_name: str = "USD/COP Forecasting Pipeline"
        app_env: str = "development"
        debug: bool = False
        log_level: str = "INFO"

        # Database
        db_host: str = "localhost"
        db_port: int = 5432
        db_name: str = "usdcop"
        db_user: str = "postgres"
        db_password: str = "postgres"
        db_schema: str = "core"

        # MinIO
        minio_endpoint: str = "localhost:9000"
        minio_access_key: str = "minioadmin"
        minio_secret_key: str = "minioadmin"
        minio_bucket: str = "ml-models"
        minio_secure: bool = False

        # MLflow
        mlflow_tracking_uri: str = "./mlruns"
        mlflow_experiment_name: str = "usdcop-forecasting"

        # API
        api_host: str = "0.0.0.0"
        api_port: int = 8000
        jwt_secret_key: str = "your-secret-key-change-in-production"

        # Redis
        redis_host: str = "localhost"
        redis_port: int = 6379

        # ML
        ml_random_state: int = 42
        ml_train_size: float = 0.8
        ml_n_features: int = 15

        @property
        def database_url(self) -> str:
            return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

        @property
        def redis_url(self) -> str:
            return f"redis://{self.redis_host}:{self.redis_port}/0"

        def get_data_dir(self) -> Path:
            return DATA_DIR

        def get_output_dir(self) -> Path:
            return OUTPUT_DIR

        def get_models_dir(self) -> Path:
            return MODELS_DIR

        @classmethod
        def from_env(cls) -> "Settings":
            """Create settings from environment variables."""
            return cls(
                app_name=os.getenv("APP_NAME", cls.app_name),
                app_env=os.getenv("APP_ENV", cls.app_env),
                debug=os.getenv("DEBUG", "false").lower() == "true",
                log_level=os.getenv("LOG_LEVEL", cls.log_level),
                db_host=os.getenv("DB_HOST", cls.db_host),
                db_port=int(os.getenv("DB_PORT", cls.db_port)),
                db_name=os.getenv("DB_NAME", cls.db_name),
                db_user=os.getenv("DB_USER", cls.db_user),
                db_password=os.getenv("DB_PASSWORD", cls.db_password),
                minio_endpoint=os.getenv("MINIO_ENDPOINT", cls.minio_endpoint),
                minio_access_key=os.getenv("MINIO_ACCESS_KEY", cls.minio_access_key),
                minio_secret_key=os.getenv("MINIO_SECRET_KEY", cls.minio_secret_key),
                mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", cls.mlflow_tracking_uri),
                api_host=os.getenv("API_HOST", cls.api_host),
                api_port=int(os.getenv("API_PORT", cls.api_port)),
                redis_host=os.getenv("REDIS_HOST", cls.redis_host),
                redis_port=int(os.getenv("REDIS_PORT", cls.redis_port)),
            )


# =============================================================================
# SETTINGS SINGLETON
# =============================================================================

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.

    Returns:
        Settings instance

    Example:
        settings = get_settings()
        print(settings.database_url)
    """
    if PYDANTIC_AVAILABLE:
        return Settings()
    else:
        return Settings.from_env()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_directories():
    """Create all required directories if they don't exist."""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        OUTPUT_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        LOGS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_data_path(filename: str) -> Path:
    """Get full path for a data file."""
    return DATA_DIR / filename


def get_output_path(filename: str) -> Path:
    """Get full path for an output file."""
    return OUTPUT_DIR / filename


def get_model_path(filename: str) -> Path:
    """Get full path for a model file."""
    return MODELS_DIR / filename
