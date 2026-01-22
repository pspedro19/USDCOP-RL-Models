# api/src/config.py
"""
Centralized configuration for the USD/COP Forecasting API.

This module provides a single source of truth for environment variables
and service clients, eliminating duplication across routers.

Usage:
    from .config import settings, get_db_connection, get_minio_client

    # Access environment variables
    endpoint = settings.MINIO_ENDPOINT

    # Get database connection
    conn = get_db_connection()

    # Get MinIO client
    client = get_minio_client()
"""

import os
import logging
from typing import Dict, Optional
from dataclasses import dataclass

from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# Settings Dataclass
# =============================================================================

@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://pipeline:pipeline_secret@localhost:5432/pipeline_db"
    )

    # MinIO
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minio_secret")

    # MLflow
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    # Paths
    OUTPUTS_PATH: str = os.getenv("OUTPUTS_PATH", (Path(__file__).parent.parent.parent / "outputs").as_posix())

    # JWT
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))


# Singleton instance
settings = Settings()


# =============================================================================
# Database Utilities
# =============================================================================

def parse_database_url(url: str) -> Dict[str, str]:
    """
    Parse PostgreSQL connection URL into components.

    Args:
        url: PostgreSQL connection URL (format: postgresql://user:pass@host:port/db)

    Returns:
        Dictionary with host, port, database, user, password keys
    """
    url = url.replace("postgresql://", "")
    user_pass, host_db = url.split("@")
    user, password = user_pass.split(":")
    host_port, db = host_db.split("/")
    host, port = host_port.split(":") if ":" in host_port else (host_port, "5432")

    return {
        "host": host,
        "port": port,
        "database": db,
        "user": user,
        "password": password
    }


def get_db_connection(database_url: Optional[str] = None):
    """
    Establish a connection to PostgreSQL database.

    Args:
        database_url: Optional custom DATABASE_URL. Uses settings if not provided.

    Returns:
        psycopg2 connection object

    Raises:
        ImportError: If psycopg2 is not installed
        Exception: If connection fails
    """
    import psycopg2

    url = database_url or settings.DATABASE_URL
    config = parse_database_url(url)

    return psycopg2.connect(
        host=config["host"],
        port=config["port"],
        database=config["database"],
        user=config["user"],
        password=config["password"]
    )


# =============================================================================
# MinIO Utilities
# =============================================================================

def get_minio_client():
    """
    Create a MinIO client instance.

    Returns:
        Minio client object or None if minio package not installed
    """
    try:
        from minio import Minio

        return Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=False
        )
    except ImportError:
        logger.warning("minio package not installed")
        return None


# =============================================================================
# Health Check Utilities
# =============================================================================

def get_db_config() -> Dict[str, str]:
    """Get parsed database configuration."""
    return parse_database_url(settings.DATABASE_URL)


def get_minio_config() -> Dict[str, str]:
    """Get MinIO configuration."""
    return {
        "endpoint": settings.MINIO_ENDPOINT,
        "access_key": settings.MINIO_ACCESS_KEY,
        "secret_key": settings.MINIO_SECRET_KEY
    }
