"""
Pytest Configuration and Fixtures

This module provides shared fixtures for all tests including:
- Database connections
- MinIO client
- API client
- Test data generators
"""

import os
import sys
import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Generator, Optional
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import numpy as np

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "api" / "src"))


# =============================================================================
# Environment Configuration
# =============================================================================

# Test environment variables
TEST_ENV = {
    "DATABASE_URL": os.getenv(
        "TEST_DATABASE_URL",
        "postgresql://pipeline:pipeline_secret@localhost:5432/pipeline_db_test"
    ),
    "MINIO_ENDPOINT": os.getenv("TEST_MINIO_ENDPOINT", "localhost:9000"),
    "MINIO_ACCESS_KEY": os.getenv("TEST_MINIO_ACCESS_KEY", "minioadmin"),
    "MINIO_SECRET_KEY": os.getenv("TEST_MINIO_SECRET_KEY", "minio_secret"),
    "MLFLOW_TRACKING_URI": os.getenv("TEST_MLFLOW_TRACKING_URI", "http://localhost:5000"),
    "API_BASE_URL": os.getenv("TEST_API_BASE_URL", "http://localhost:8000"),
    "OUTPUTS_PATH": os.getenv("TEST_OUTPUTS_PATH", str(PROJECT_ROOT / "outputs")),
}


# =============================================================================
# Pytest Markers
# =============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests that may take > 1 minute")
    config.addinivalue_line("markers", "requires_db: Tests that require PostgreSQL")
    config.addinivalue_line("markers", "requires_minio: Tests that require MinIO")
    config.addinivalue_line("markers", "requires_mlflow: Tests that require MLflow")
    config.addinivalue_line("markers", "requires_api: Tests that require API running")


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def db_connection():
    """
    Create a database connection for testing.

    Yields:
        Connection object or None if connection fails
    """
    try:
        import psycopg2

        url = TEST_ENV["DATABASE_URL"]
        url = url.replace("postgresql://", "")
        user_pass, host_db = url.split("@")
        user, password = user_pass.split(":")
        host_port, db = host_db.split("/")
        host, port = host_port.split(":") if ":" in host_port else (host_port, "5432")

        conn = psycopg2.connect(
            host=host,
            port=port,
            database=db,
            user=user,
            password=password,
            connect_timeout=5
        )
        yield conn
        conn.close()

    except Exception as e:
        pytest.skip(f"Database connection not available: {e}")
        yield None


@pytest.fixture
def db_cursor(db_connection):
    """Create a database cursor for testing."""
    if db_connection is None:
        pytest.skip("Database connection not available")

    cursor = db_connection.cursor()
    yield cursor
    cursor.close()


# =============================================================================
# MinIO Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def minio_client():
    """
    Create a MinIO client for testing.

    Yields:
        MinIO client or None if not available
    """
    try:
        from minio import Minio

        client = Minio(
            endpoint=TEST_ENV["MINIO_ENDPOINT"],
            access_key=TEST_ENV["MINIO_ACCESS_KEY"],
            secret_key=TEST_ENV["MINIO_SECRET_KEY"],
            secure=False
        )

        # Test connection
        client.list_buckets()
        yield client

    except Exception as e:
        pytest.skip(f"MinIO connection not available: {e}")
        yield None


@pytest.fixture
def test_bucket(minio_client):
    """
    Create a test bucket for testing.

    Yields:
        Bucket name
    """
    if minio_client is None:
        pytest.skip("MinIO not available")

    bucket_name = f"test-bucket-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    try:
        minio_client.make_bucket(bucket_name)
        yield bucket_name

        # Cleanup
        objects = minio_client.list_objects(bucket_name, recursive=True)
        for obj in objects:
            minio_client.remove_object(bucket_name, obj.object_name)
        minio_client.remove_bucket(bucket_name)

    except Exception as e:
        pytest.skip(f"Could not create test bucket: {e}")


# =============================================================================
# API Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def api_client():
    """
    Create an HTTP client for API testing.

    Yields:
        httpx.Client instance
    """
    try:
        import httpx

        client = httpx.Client(
            base_url=TEST_ENV["API_BASE_URL"],
            timeout=30.0
        )
        yield client
        client.close()

    except Exception as e:
        pytest.skip(f"API client not available: {e}")
        yield None


@pytest.fixture
def auth_token(api_client):
    """
    Get an authentication token for API testing.

    Yields:
        JWT token string
    """
    if api_client is None:
        pytest.skip("API client not available")

    try:
        response = api_client.post(
            "/auth/login",
            data={
                "username": "admin",
                "password": "admin123"
            }
        )

        if response.status_code == 200:
            token = response.json().get("access_token")
            yield token
        else:
            pytest.skip("Could not obtain auth token")

    except Exception as e:
        pytest.skip(f"Authentication failed: {e}")


@pytest.fixture
def auth_headers(auth_token):
    """Get authorization headers for API requests."""
    if auth_token is None:
        pytest.skip("Auth token not available")

    return {"Authorization": f"Bearer {auth_token}"}


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_features_df():
    """
    Generate sample features DataFrame for testing.

    Returns:
        DataFrame with sample features
    """
    np.random.seed(42)
    n_samples = 500

    dates = pd.date_range(
        start="2023-01-01",
        periods=n_samples,
        freq="B"  # Business days
    )

    # Generate realistic USDCOP data
    base_price = 4200
    returns = np.random.normal(0, 0.005, n_samples)
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "Date": dates,
        "Close": prices,
        "Open": prices * (1 + np.random.normal(0, 0.001, n_samples)),
        "High": prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
        "Low": prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
        "Volume": np.random.randint(1000000, 10000000, n_samples),
    })

    # Add some features
    for i in range(1, 11):
        df[f"feature_{i}"] = np.random.randn(n_samples)

    # Add technical indicators
    df["returns"] = df["Close"].pct_change()
    df["volatility_20"] = df["returns"].rolling(20).std()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["ma_50"] = df["Close"].rolling(50).mean()

    # Add targets
    for h in [1, 5, 10, 15, 20]:
        df[f"target_{h}d"] = np.log(df["Close"].shift(-h) / df["Close"])

    return df.dropna().reset_index(drop=True)


@pytest.fixture
def sample_forecasts():
    """
    Generate sample forecasts for testing.

    Returns:
        List of forecast dictionaries
    """
    models = ["ridge", "xgboost", "lightgbm", "catboost", "bayesian_ridge"]
    horizons = [5, 10, 20, 40]
    current_price = 4200
    base_date = datetime.now()

    forecasts = []

    for model in models:
        for horizon in horizons:
            pred_return = np.random.uniform(-0.02, 0.02)
            pred_price = current_price * np.exp(pred_return)

            forecasts.append({
                "model": model,
                "horizon": horizon,
                "inference_date": base_date.strftime("%Y-%m-%d"),
                "target_date": (base_date + timedelta(days=horizon)).strftime("%Y-%m-%d"),
                "current_price": current_price,
                "predicted_price": pred_price,
                "predicted_return_pct": pred_return * 100,
                "direction": "UP" if pred_return > 0 else "DOWN",
                "signal": "BUY" if pred_return > 0.005 else ("SELL" if pred_return < -0.005 else "HOLD"),
            })

    return forecasts


@pytest.fixture
def sample_model_metrics():
    """
    Generate sample model metrics for testing.

    Returns:
        DataFrame with model metrics
    """
    models = ["ridge", "xgboost", "lightgbm", "catboost", "bayesian_ridge"]
    horizons = [5, 10, 20, 40]

    records = []
    for model in models:
        for horizon in horizons:
            records.append({
                "model_name": model,
                "horizon": horizon,
                "direction_accuracy": np.random.uniform(55, 75),
                "rmse": np.random.uniform(0.01, 0.03),
                "mae": np.random.uniform(0.008, 0.025),
                "r2": np.random.uniform(0.1, 0.4),
            })

    return pd.DataFrame(records)


@pytest.fixture
def temp_output_dir():
    """
    Create a temporary output directory for testing.

    Yields:
        Path to temporary directory
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create subdirectories
        (output_dir / "runs").mkdir(parents=True)
        (output_dir / "weekly").mkdir(parents=True)
        (output_dir / "bi").mkdir(parents=True)
        (output_dir / "models").mkdir(parents=True)

        yield output_dir


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_minio_client():
    """Create a mock MinIO client for testing without actual MinIO."""
    mock = MagicMock()
    mock.bucket_exists.return_value = True
    mock.list_buckets.return_value = []
    mock.list_objects.return_value = []
    mock.fput_object.return_value = MagicMock(etag="mock-etag")
    mock.get_object.return_value = MagicMock(
        read=lambda: b'{"test": "data"}',
        close=lambda: None
    )
    return mock


@pytest.fixture
def mock_db_connection():
    """Create a mock database connection for testing without actual PostgreSQL."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    mock_cursor.fetchone.return_value = (1,)
    mock_cursor.fetchall.return_value = []
    mock_conn.cursor.return_value = mock_cursor

    return mock_conn


@pytest.fixture
def mock_mlflow_client():
    """Create a mock MLflow client for testing without actual MLflow."""
    mock = MagicMock()
    mock.start_run.return_value.__enter__ = MagicMock()
    mock.start_run.return_value.__exit__ = MagicMock()
    mock.log_param.return_value = None
    mock.log_metric.return_value = None
    mock.log_artifact.return_value = None
    return mock


# =============================================================================
# Helper Fixtures
# =============================================================================

@pytest.fixture
def project_root():
    """Get the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def backend_path(project_root):
    """Get the backend directory path."""
    return project_root / "backend"


@pytest.fixture
def api_path(project_root):
    """Get the API directory path."""
    return project_root / "api"


@pytest.fixture
def dags_path(project_root):
    """Get the DAGs directory path."""
    return project_root / "data-engineering" / "dags"


# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Cleanup test artifacts after each test."""
    yield

    # Cleanup temporary files if any
    import gc
    gc.collect()
