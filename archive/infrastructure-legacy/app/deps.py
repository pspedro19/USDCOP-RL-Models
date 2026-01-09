#!/usr/bin/env python3
"""
API Dependencies - Storage Registry Pattern
============================================
Provides unified access to PostgreSQL and MinIO storage backends
using the Storage Registry configuration (config/storage.yaml).

Key Patterns:
- Repository Pattern: Abstract storage backend details
- Manifest System: Track dataset lineage with run.json + latest.json
- Contract Stability: API never exposes internal storage structure
"""

import json
import os
import hashlib
from functools import lru_cache
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

import yaml
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import s3fs
from sqlalchemy import create_engine, text, pool
from sqlalchemy.orm import sessionmaker
from fastapi import HTTPException

# ==========================================
# ENVIRONMENT CONFIGURATION
# ==========================================
# Construct PostgreSQL URL from environment variables
POSTGRES_USER = os.getenv("POSTGRES_USER", "admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "admin123")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "usdcop_trading")

POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:9000")
S3_KEY = os.getenv("S3_KEY", "minioadmin")
S3_SECRET = os.getenv("S3_SECRET", "minioadmin123")

# ==========================================
# DATABASE ENGINE (PostgreSQL + TimescaleDB)
# ==========================================
engine = create_engine(
    POSTGRES_URL,
    pool_size=5,              # Base connections
    max_overflow=10,          # Additional connections
    pool_pre_ping=True,       # Verify connection before use
    pool_recycle=3600,        # Recycle connections after 1 hour
    poolclass=pool.QueuePool,
    echo=False                # Set True for SQL debugging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ==========================================
# S3/MinIO FILE SYSTEM
# ==========================================
s3 = s3fs.S3FileSystem(
    client_kwargs={"endpoint_url": S3_ENDPOINT},
    key=S3_KEY,
    secret=S3_SECRET,
    use_ssl=False  # Set True for production AWS S3
)

# ==========================================
# STORAGE REGISTRY LOADER
# ==========================================
@lru_cache()
def load_storage_registry() -> Dict[str, Any]:
    """
    Load storage registry from config/storage.yaml

    Returns mapping of layers (l0, l1, ..., l6) to their storage backends
    """
    config_path = Path(__file__).parent.parent / "config" / "storage.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Storage registry not found: {config_path}\n"
            "Run setup to create config/storage.yaml"
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config["layers"]


@lru_cache()
def load_quality_gates() -> Dict[str, Dict[str, Any]]:
    """Load quality gate criteria for GO/NO-GO decisions"""
    config_path = Path(__file__).parent.parent / "config" / "storage.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config.get("quality_gates", {})


# ==========================================
# MANIFEST SYSTEM
# ==========================================
def read_latest_manifest(bucket: str, layer_prefix: str) -> Dict[str, Any]:
    """
    Read the latest manifest for a given layer

    Args:
        bucket: S3 bucket name (e.g., 'usdcop')
        layer_prefix: Layer prefix (e.g., 'l4', 'l5')

    Returns:
        {
            "run_id": "2025-10-20",
            "layer": "l4",
            "path": "l4/2025-10-20/",
            "dataset_hash": "sha256:abc123...",
            "started_at": "2025-10-20T08:00:00Z",
            "completed_at": "2025-10-20T08:15:00Z",
            "status": "success",
            "files": [...]
        }

    Raises:
        HTTPException(404): If no manifest found
    """
    path = f"{bucket}/_meta/{layer_prefix}_latest.json"

    try:
        with s3.open(path, "rb") as f:
            latest = json.load(f)

        return latest

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No manifest found for layer {layer_prefix}. "
                   f"Pipeline may not have run yet or path {path} is incorrect."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading manifest: {str(e)}"
        )


def read_run_manifest(bucket: str, layer_prefix: str, run_id: str) -> Dict[str, Any]:
    """
    Read a specific run manifest

    Args:
        bucket: S3 bucket name
        layer_prefix: Layer prefix
        run_id: Specific run ID (e.g., '2025-10-20')

    Returns:
        Full manifest for that run
    """
    path = f"{bucket}/_meta/{layer_prefix}_{run_id}_run.json"

    try:
        with s3.open(path, "rb") as f:
            return json.load(f)

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Run manifest not found: {run_id} for layer {layer_prefix}"
        )


def write_manifest(
    bucket: str,
    layer: str,
    run_id: str,
    files: List[Dict[str, Any]],
    status: str = "success",
    metadata: Optional[Dict] = None
) -> None:
    """
    Write run manifest and update latest pointer

    Args:
        bucket: S3 bucket name
        layer: Layer name (l0, l1, etc.)
        run_id: Run identifier
        files: List of file metadata dicts
        status: success|failed|running
        metadata: Additional metadata
    """
    now = datetime.utcnow().isoformat() + "Z"

    # Calculate dataset hash
    files_str = json.dumps(files, sort_keys=True)
    dataset_hash = "sha256:" + hashlib.sha256(files_str.encode()).hexdigest()[:16]

    manifest = {
        "run_id": run_id,
        "layer": layer,
        "path": f"{layer}/{run_id}/",
        "dataset_hash": dataset_hash,
        "started_at": metadata.get("started_at", now) if metadata else now,
        "completed_at": now,
        "status": status,
        "files": files,
        "metadata": metadata or {}
    }

    # Write run manifest
    run_path = f"{bucket}/_meta/{layer}_{run_id}_run.json"
    with s3.open(run_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Update latest pointer (only if success)
    if status == "success":
        latest_path = f"{bucket}/_meta/{layer}_latest.json"
        latest = {
            "run_id": run_id,
            "layer": layer,
            "path": f"{layer}/{run_id}/",
            "dataset_hash": dataset_hash,
            "updated_at": now
        }
        with s3.open(latest_path, "w") as f:
            json.dump(latest, f, indent=2)


# ==========================================
# DATA READERS (Repository Pattern)
# ==========================================
def read_parquet_dataset(
    bucket: str,
    path: str,
    columns: Optional[List[str]] = None,
    filters: Optional[Any] = None
) -> pd.DataFrame:
    """
    Read Parquet dataset from S3/MinIO with column projection and predicate pushdown

    Args:
        bucket: S3 bucket name
        path: Path to parquet file/directory
        columns: List of columns to read (None = all)
        filters: PyArrow filters for predicate pushdown

    Returns:
        Pandas DataFrame
    """
    url = f"s3://{bucket}/{path}"

    try:
        dataset = ds.dataset(url, filesystem=s3, format="parquet")
        table = dataset.to_table(columns=columns, filter=filters)
        return table.to_pandas()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading parquet from {url}: {str(e)}"
        )


def read_json_metadata(bucket: str, path: str) -> Dict[str, Any]:
    """
    Read JSON metadata file from S3/MinIO

    Args:
        bucket: S3 bucket name
        path: Path to JSON file

    Returns:
        Parsed JSON as dict
    """
    url = f"{bucket}/{path}"

    try:
        with s3.open(url, "rb") as f:
            return json.load(f)

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Metadata file not found: {path}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading metadata: {str(e)}"
        )


def execute_sql_query(query: str, params: Optional[Dict] = None) -> pd.DataFrame:
    """
    Execute SQL query against PostgreSQL and return DataFrame

    Args:
        query: SQL query string (use :param for parameters)
        params: Dict of parameters

    Returns:
        Query results as DataFrame
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query), params or {})

            # If SELECT query, return DataFrame
            if result.returns_rows:
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
            else:
                # For INSERT/UPDATE/DELETE, return row count
                return pd.DataFrame({"rows_affected": [result.rowcount]})

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )


def execute_sql_scalar(query: str, params: Optional[Dict] = None) -> Any:
    """
    Execute SQL query and return single scalar value

    Args:
        query: SQL query
        params: Parameters

    Returns:
        Single value from first row, first column
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            row = result.first()
            return row[0] if row else None

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )


# ==========================================
# LAYER-SPECIFIC READERS
# ==========================================
def get_layer_config(layer: str) -> Dict[str, Any]:
    """Get storage configuration for a specific layer"""
    registry = load_storage_registry()

    if layer not in registry:
        raise HTTPException(
            status_code=404,
            detail=f"Layer {layer} not found in storage registry"
        )

    return registry[layer]


def read_layer_data(
    layer: str,
    run_id: Optional[str] = None,
    columns: Optional[List[str]] = None,
    filters: Optional[Any] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Unified reader for any pipeline layer (L0-L6)

    Automatically selects correct backend (DB or S3) based on storage registry

    Args:
        layer: Layer name ('l0', 'l1', ..., 'l6')
        run_id: Specific run ID (None = latest)
        columns: Columns to select
        filters: Filter conditions
        limit: Row limit

    Returns:
        DataFrame with layer data
    """
    config = get_layer_config(layer)
    backend = config["backend"]

    if backend == "postgres":
        # Read from PostgreSQL
        table = config["table"]

        # Build query
        col_str = ", ".join(columns) if columns else "*"
        query = f"SELECT {col_str} FROM {table}"

        if limit:
            query += f" LIMIT {limit}"

        return execute_sql_query(query)

    elif backend == "s3":
        # Read from MinIO/S3
        bucket = config["bucket"]
        prefix = config["prefix"]

        # Get manifest to find data path
        if run_id is None:
            manifest = read_latest_manifest(bucket, prefix)
        else:
            manifest = read_run_manifest(bucket, prefix, run_id)

        # Read main data file (first .parquet file in manifest)
        data_file = next(
            (f for f in manifest["files"] if f["name"].endswith(".parquet")),
            None
        )

        if not data_file:
            raise HTTPException(
                status_code=404,
                detail=f"No parquet data file found for layer {layer}, run {run_id or 'latest'}"
            )

        df = read_parquet_dataset(bucket, data_file["path"], columns, filters)

        if limit:
            df = df.head(limit)

        return df

    else:
        raise HTTPException(
            status_code=500,
            detail=f"Unknown backend: {backend}"
        )


# ==========================================
# HEALTH CHECKS
# ==========================================
def check_db_health() -> bool:
    """Check if PostgreSQL is accessible"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except:
        return False


def check_s3_health() -> bool:
    """Check if MinIO/S3 is accessible"""
    try:
        s3.ls("usdcop")  # Try to list main bucket
        return True
    except:
        return False


def get_system_health() -> Dict[str, Any]:
    """Get overall system health status"""
    db_ok = check_db_health()
    s3_ok = check_s3_health()

    return {
        "status": "healthy" if (db_ok and s3_ok) else "degraded",
        "postgres": "up" if db_ok else "down",
        "s3": "up" if s3_ok else "down",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


# ==========================================
# CACHE HELPERS
# ==========================================
def calculate_etag(data: Any) -> str:
    """Calculate ETag for HTTP caching"""
    if isinstance(data, dict):
        data_str = json.dumps(data, sort_keys=True)
    elif isinstance(data, pd.DataFrame):
        data_str = data.to_json(orient="records")
    else:
        data_str = str(data)

    return hashlib.sha256(data_str.encode()).hexdigest()[:16]


# ==========================================
# EXPORTS
# ==========================================
__all__ = [
    # Storage Registry
    "load_storage_registry",
    "load_quality_gates",
    "get_layer_config",

    # Manifests
    "read_latest_manifest",
    "read_run_manifest",
    "write_manifest",

    # Data Readers
    "read_parquet_dataset",
    "read_json_metadata",
    "execute_sql_query",
    "execute_sql_scalar",
    "read_layer_data",

    # Health
    "check_db_health",
    "check_s3_health",
    "get_system_health",

    # Utils
    "calculate_etag",

    # Backends
    "engine",
    "SessionLocal",
    "s3"
]
