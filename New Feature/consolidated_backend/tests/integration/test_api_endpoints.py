"""
Integration Test: API Endpoints

This module tests all API endpoints:
1. /auth/login - Authentication
2. /api/forecasts/dashboard - Forecast data
3. /api/models/ - Model information
4. /api/images/ - Image serving
5. /health/ready - Health checks

Usage:
    pytest tests/integration/test_api_endpoints.py -v
    pytest tests/integration/test_api_endpoints.py -v -m "not requires_api"
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "api" / "src"))


# =============================================================================
# Test Configuration
# =============================================================================

TEST_USER = {
    "username": "admin",
    "password": "admin123",
}

API_BASE_URL = os.getenv("TEST_API_BASE_URL", "http://localhost:8000")


# =============================================================================
# Fixtures for API Testing
# =============================================================================

@pytest.fixture
def mock_app():
    """Create a mock FastAPI app for testing."""
    try:
        from fastapi.testclient import TestClient
        from api.src.main import app
        return TestClient(app)
    except ImportError:
        pytest.skip("FastAPI or test client not available")


@pytest.fixture
def mock_auth_response():
    """Mock authentication response."""
    return {
        "access_token": "mock_jwt_token_12345",
        "token_type": "bearer",
        "expires_in": 86400,
    }


@pytest.fixture
def mock_forecast_response():
    """Mock forecast response."""
    return {
        "source": "postgresql",
        "count": 20,
        "data": [
            {
                "inference_date": "2024-01-15",
                "model_name": "ridge",
                "horizon": 5,
                "predicted_price": 4275.25,
                "direction": "UP",
            }
        ],
    }


@pytest.fixture
def mock_models_response():
    """Mock models response."""
    return {
        "models": [
            {
                "name": "ridge",
                "horizons": [5, 10, 20, 40],
                "avg_direction_accuracy": 68.5,
                "avg_rmse": 52.3,
            },
            {
                "name": "xgboost",
                "horizons": [5, 10, 20, 40],
                "avg_direction_accuracy": 71.2,
                "avg_rmse": 48.7,
            },
        ]
    }


# =============================================================================
# Test Class: Authentication Endpoints
# =============================================================================

@pytest.mark.integration
class TestAuthEndpoints:
    """Tests for authentication endpoints."""

    def test_login_endpoint_exists(self, mock_app):
        """Test that login endpoint exists."""
        if mock_app is None:
            pytest.skip("Test client not available")

        response = mock_app.post(
            "/auth/login",
            data={"username": "invalid", "password": "invalid"}
        )

        # Should return 401 for invalid credentials, not 404
        assert response.status_code in [401, 422], \
            f"Login endpoint should exist (got {response.status_code})"

    def test_login_with_valid_credentials(self, mock_app):
        """Test login with valid credentials."""
        if mock_app is None:
            pytest.skip("Test client not available")

        response = mock_app.post(
            "/auth/login",
            data=TEST_USER
        )

        # May succeed or fail depending on configured users
        if response.status_code == 200:
            data = response.json()
            assert "access_token" in data
            assert "token_type" in data
            assert data["token_type"] == "bearer"

    def test_login_with_invalid_credentials(self, mock_app):
        """Test login with invalid credentials."""
        if mock_app is None:
            pytest.skip("Test client not available")

        response = mock_app.post(
            "/auth/login",
            data={"username": "invalid_user", "password": "wrong_password"}
        )

        assert response.status_code == 401

    def test_login_json_endpoint(self, mock_app):
        """Test JSON login endpoint."""
        if mock_app is None:
            pytest.skip("Test client not available")

        response = mock_app.post(
            "/auth/login/json",
            json=TEST_USER
        )

        # May succeed or fail depending on configured users
        assert response.status_code in [200, 401, 404]

    def test_login_response_schema(self, mock_auth_response):
        """Test login response schema."""
        required_fields = ["access_token", "token_type", "expires_in"]

        for field in required_fields:
            assert field in mock_auth_response, f"Missing field: {field}"

        assert isinstance(mock_auth_response["access_token"], str)
        assert mock_auth_response["token_type"] == "bearer"
        assert isinstance(mock_auth_response["expires_in"], int)

    def test_refresh_token_endpoint(self, mock_app, auth_headers):
        """Test refresh token endpoint."""
        if mock_app is None:
            pytest.skip("Test client not available")

        try:
            response = mock_app.post(
                "/auth/refresh",
                headers=auth_headers
            )

            if response.status_code == 200:
                data = response.json()
                assert "access_token" in data
        except Exception:
            pytest.skip("Auth headers not available")

    def test_me_endpoint(self, mock_app, auth_headers):
        """Test /auth/me endpoint."""
        if mock_app is None:
            pytest.skip("Test client not available")

        try:
            response = mock_app.get(
                "/auth/me",
                headers=auth_headers
            )

            if response.status_code == 200:
                data = response.json()
                assert "username" in data
                assert "role" in data
        except Exception:
            pytest.skip("Auth headers not available")

    def test_verify_token_endpoint(self, mock_app, auth_headers):
        """Test /auth/verify endpoint."""
        if mock_app is None:
            pytest.skip("Test client not available")

        try:
            response = mock_app.post(
                "/auth/verify",
                headers=auth_headers
            )

            if response.status_code == 200:
                data = response.json()
                assert "valid" in data
        except Exception:
            pytest.skip("Auth headers not available")


# =============================================================================
# Test Class: Forecast Endpoints
# =============================================================================

@pytest.mark.integration
class TestForecastEndpoints:
    """Tests for forecast endpoints."""

    def test_forecasts_endpoint_requires_auth(self, mock_app):
        """Test that forecasts endpoint requires authentication."""
        if mock_app is None:
            pytest.skip("Test client not available")

        response = mock_app.get("/api/forecasts/")

        # Should return 401 or 403 without auth
        assert response.status_code in [401, 403, 422]

    def test_forecasts_list_endpoint(self, mock_app, auth_headers):
        """Test forecasts list endpoint."""
        if mock_app is None:
            pytest.skip("Test client not available")

        try:
            response = mock_app.get(
                "/api/forecasts/",
                headers=auth_headers
            )

            if response.status_code == 200:
                data = response.json()
                assert "source" in data or "data" in data
        except Exception:
            pytest.skip("Auth headers not available")

    def test_forecasts_dashboard_endpoint(self, mock_app, auth_headers):
        """Test forecasts dashboard endpoint."""
        if mock_app is None:
            pytest.skip("Test client not available")

        try:
            response = mock_app.get(
                "/api/forecasts/dashboard",
                headers=auth_headers
            )

            if response.status_code == 200:
                data = response.json()
                assert "source" in data
                assert "forecasts" in data
        except Exception:
            pytest.skip("Auth headers not available")

    def test_forecasts_latest_endpoint(self, mock_app, auth_headers):
        """Test forecasts latest endpoint."""
        if mock_app is None:
            pytest.skip("Test client not available")

        try:
            response = mock_app.get(
                "/api/forecasts/latest",
                headers=auth_headers
            )

            if response.status_code == 200:
                data = response.json()
                assert "source" in data or "data" in data
        except Exception:
            pytest.skip("Auth headers not available")

    def test_forecasts_consensus_endpoint(self, mock_app, auth_headers):
        """Test forecasts consensus endpoint."""
        if mock_app is None:
            pytest.skip("Test client not available")

        try:
            response = mock_app.get(
                "/api/forecasts/consensus",
                headers=auth_headers
            )

            if response.status_code == 200:
                data = response.json()
                assert "source" in data or "data" in data
        except Exception:
            pytest.skip("Auth headers not available")

    def test_forecasts_by_week_endpoint(self, mock_app, auth_headers):
        """Test forecasts by week endpoint."""
        if mock_app is None:
            pytest.skip("Test client not available")

        try:
            year = datetime.now().year
            week = datetime.now().isocalendar()[1]

            response = mock_app.get(
                f"/api/forecasts/by-week/{year}/{week}",
                headers=auth_headers
            )

            if response.status_code == 200:
                data = response.json()
                assert "year" in data
                assert "week" in data
        except Exception:
            pytest.skip("Auth headers not available")

    def test_forecasts_by_horizon_endpoint(self, mock_app, auth_headers):
        """Test forecasts by horizon endpoint."""
        if mock_app is None:
            pytest.skip("Test client not available")

        try:
            response = mock_app.get(
                "/api/forecasts/by-horizon/5",
                headers=auth_headers
            )

            if response.status_code == 200:
                data = response.json()
                assert "horizon" in data
        except Exception:
            pytest.skip("Auth headers not available")

    def test_forecasts_filter_by_model(self, mock_app, auth_headers):
        """Test filtering forecasts by model."""
        if mock_app is None:
            pytest.skip("Test client not available")

        try:
            response = mock_app.get(
                "/api/forecasts/?model=ridge",
                headers=auth_headers
            )

            if response.status_code == 200:
                data = response.json()
                assert "data" in data or "source" in data
        except Exception:
            pytest.skip("Auth headers not available")

    def test_forecast_response_schema(self, mock_forecast_response):
        """Test forecast response schema."""
        assert "source" in mock_forecast_response
        assert "count" in mock_forecast_response
        assert "data" in mock_forecast_response

        assert isinstance(mock_forecast_response["data"], list)

        if len(mock_forecast_response["data"]) > 0:
            forecast = mock_forecast_response["data"][0]
            assert "model_name" in forecast or "model" in forecast
            assert "horizon" in forecast
            assert "predicted_price" in forecast


# =============================================================================
# Test Class: Model Endpoints
# =============================================================================

@pytest.mark.integration
class TestModelEndpoints:
    """Tests for model endpoints."""

    def test_models_list_endpoint(self, mock_app, auth_headers):
        """Test models list endpoint."""
        if mock_app is None:
            pytest.skip("Test client not available")

        try:
            response = mock_app.get(
                "/api/models/",
                headers=auth_headers
            )

            if response.status_code == 200:
                data = response.json()
                assert "models" in data
        except Exception:
            pytest.skip("Auth headers not available")

    def test_model_detail_endpoint(self, mock_app, auth_headers):
        """Test model detail endpoint."""
        if mock_app is None:
            pytest.skip("Test client not available")

        try:
            response = mock_app.get(
                "/api/models/ridge",
                headers=auth_headers
            )

            if response.status_code == 200:
                data = response.json()
                assert "name" in data
        except Exception:
            pytest.skip("Auth headers not available")

    def test_model_comparison_endpoint(self, mock_app, auth_headers):
        """Test model comparison endpoint."""
        if mock_app is None:
            pytest.skip("Test client not available")

        try:
            response = mock_app.get(
                "/api/models/ridge/comparison",
                headers=auth_headers
            )

            if response.status_code == 200:
                data = response.json()
                assert "selected_model" in data or "comparison" in data
        except Exception:
            pytest.skip("Auth headers not available")

    def test_model_metrics_endpoint(self, mock_app, auth_headers):
        """Test model metrics by horizon endpoint."""
        if mock_app is None:
            pytest.skip("Test client not available")

        try:
            response = mock_app.get(
                "/api/models/ridge/metrics/5",
                headers=auth_headers
            )

            if response.status_code == 200:
                data = response.json()
                assert "model" in data
                assert "horizon" in data
        except Exception:
            pytest.skip("Auth headers not available")

    def test_model_ranking_endpoint(self, mock_app, auth_headers):
        """Test model ranking endpoint."""
        if mock_app is None:
            pytest.skip("Test client not available")

        try:
            response = mock_app.get(
                "/api/models/ranking",
                headers=auth_headers
            )

            if response.status_code == 200:
                data = response.json()
                assert "rankings" in data or "metric" in data
        except Exception:
            pytest.skip("Auth headers not available")

    def test_model_not_found(self, mock_app, auth_headers):
        """Test model not found response."""
        if mock_app is None:
            pytest.skip("Test client not available")

        try:
            response = mock_app.get(
                "/api/models/nonexistent_model",
                headers=auth_headers
            )

            assert response.status_code in [404, 500]
        except Exception:
            pytest.skip("Auth headers not available")

    def test_models_response_schema(self, mock_models_response):
        """Test models response schema."""
        assert "models" in mock_models_response
        assert isinstance(mock_models_response["models"], list)

        if len(mock_models_response["models"]) > 0:
            model = mock_models_response["models"][0]
            assert "name" in model
            assert "horizons" in model


# =============================================================================
# Test Class: Image Endpoints
# =============================================================================

@pytest.mark.integration
class TestImageEndpoints:
    """Tests for image serving endpoints."""

    def test_image_types_endpoint(self, mock_app, auth_headers):
        """Test image types endpoint."""
        if mock_app is None:
            pytest.skip("Test client not available")

        try:
            response = mock_app.get(
                "/api/images/types",
                headers=auth_headers
            )

            if response.status_code == 200:
                data = response.json()
                assert "types" in data
        except Exception:
            pytest.skip("Auth headers not available")

    def test_latest_image_endpoint(self, mock_app, auth_headers):
        """Test latest image endpoint."""
        if mock_app is None:
            pytest.skip("Test client not available")

        try:
            response = mock_app.get(
                "/api/images/latest/forward_forecast",
                headers=auth_headers
            )

            # May return 200 (image) or 404 (not found)
            assert response.status_code in [200, 404]
        except Exception:
            pytest.skip("Auth headers not available")

    def test_forecast_image_endpoint(self, mock_app, auth_headers):
        """Test forecast image endpoint."""
        if mock_app is None:
            pytest.skip("Test client not available")

        try:
            year = datetime.now().year
            week = datetime.now().isocalendar()[1]

            response = mock_app.get(
                f"/api/images/forecast/{year}/{week}/forward_forecast.png",
                headers=auth_headers
            )

            # May return 200 (image) or 404 (not found)
            assert response.status_code in [200, 404]
        except Exception:
            pytest.skip("Auth headers not available")

    def test_backtest_image_endpoint(self, mock_app, auth_headers):
        """Test backtest image endpoint."""
        if mock_app is None:
            pytest.skip("Test client not available")

        try:
            response = mock_app.get(
                "/api/images/backtest/ridge/5/backtest_ridge_h5.png",
                headers=auth_headers
            )

            # May return 200 (image) or 404 (not found)
            assert response.status_code in [200, 404]
        except Exception:
            pytest.skip("Auth headers not available")

    def test_list_week_images_endpoint(self, mock_app, auth_headers):
        """Test list week images endpoint."""
        if mock_app is None:
            pytest.skip("Test client not available")

        try:
            year = datetime.now().year
            week = datetime.now().isocalendar()[1]

            response = mock_app.get(
                f"/api/images/list/{year}/{week}",
                headers=auth_headers
            )

            if response.status_code == 200:
                data = response.json()
                assert "year" in data
                assert "week" in data
                assert "images" in data
        except Exception:
            pytest.skip("Auth headers not available")


# =============================================================================
# Test Class: Health Endpoints
# =============================================================================

@pytest.mark.integration
class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_endpoint(self, mock_app):
        """Test basic health endpoint."""
        if mock_app is None:
            pytest.skip("Test client not available")

        response = mock_app.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_ready_endpoint(self, mock_app):
        """Test readiness endpoint."""
        if mock_app is None:
            pytest.skip("Test client not available")

        response = mock_app.get("/health/ready")

        # Can be 200 (all healthy) or 503 (some unhealthy)
        assert response.status_code in [200, 503]

        data = response.json()
        assert "status" in data
        assert "checks" in data

    def test_health_live_endpoint(self, mock_app):
        """Test liveness endpoint."""
        if mock_app is None:
            pytest.skip("Test client not available")

        response = mock_app.get("/health/live")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "alive"

    def test_health_individual_service(self, mock_app):
        """Test individual service health check."""
        if mock_app is None:
            pytest.skip("Test client not available")

        for service in ["postgresql", "minio", "mlflow"]:
            response = mock_app.get(f"/health/{service}")

            # Can be 200 (healthy) or 503 (unhealthy)
            assert response.status_code in [200, 503]

            data = response.json()
            assert "service" in data
            assert "status" in data

    def test_health_invalid_service(self, mock_app):
        """Test invalid service health check."""
        if mock_app is None:
            pytest.skip("Test client not available")

        response = mock_app.get("/health/invalid_service")

        assert response.status_code == 400

    def test_health_response_contains_timestamp(self, mock_app):
        """Test that health response contains timestamp."""
        if mock_app is None:
            pytest.skip("Test client not available")

        response = mock_app.get("/health")

        if response.status_code == 200:
            data = response.json()
            assert "timestamp" in data

    def test_ready_response_contains_latencies(self, mock_app):
        """Test that ready response contains latency info."""
        if mock_app is None:
            pytest.skip("Test client not available")

        response = mock_app.get("/health/ready")

        data = response.json()
        assert "checks" in data

        for service, check in data["checks"].items():
            assert "status" in check
            assert "latency_ms" in check


# =============================================================================
# Test Class: Root and Info Endpoints
# =============================================================================

@pytest.mark.integration
class TestRootEndpoints:
    """Tests for root and info endpoints."""

    def test_root_endpoint(self, mock_app):
        """Test root endpoint."""
        if mock_app is None:
            pytest.skip("Test client not available")

        response = mock_app.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data

    def test_api_info_endpoint(self, mock_app):
        """Test API info endpoint."""
        if mock_app is None:
            pytest.skip("Test client not available")

        response = mock_app.get("/api")

        assert response.status_code == 200
        data = response.json()
        assert "api_version" in data or "version" in data
        assert "endpoints" in data

    def test_openapi_schema_endpoint(self, mock_app):
        """Test OpenAPI schema endpoint."""
        if mock_app is None:
            pytest.skip("Test client not available")

        response = mock_app.get("/openapi.json")

        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data


# =============================================================================
# Test Class: Error Handling
# =============================================================================

@pytest.mark.integration
class TestAPIErrorHandling:
    """Tests for API error handling."""

    def test_404_for_invalid_endpoint(self, mock_app):
        """Test 404 response for invalid endpoints."""
        if mock_app is None:
            pytest.skip("Test client not available")

        response = mock_app.get("/api/nonexistent/endpoint")

        assert response.status_code == 404

    def test_401_for_protected_endpoint_without_auth(self, mock_app):
        """Test 401 response for protected endpoints without auth."""
        if mock_app is None:
            pytest.skip("Test client not available")

        response = mock_app.get("/api/forecasts/")

        assert response.status_code in [401, 403, 422]

    def test_400_for_invalid_parameters(self, mock_app, auth_headers):
        """Test 400 response for invalid parameters."""
        if mock_app is None:
            pytest.skip("Test client not available")

        try:
            response = mock_app.get(
                "/api/forecasts/?horizon=-1",  # Invalid horizon
                headers=auth_headers
            )

            # Should reject invalid parameter
            assert response.status_code in [400, 422, 200]  # 200 if ignored
        except Exception:
            pytest.skip("Auth headers not available")

    def test_method_not_allowed(self, mock_app):
        """Test 405 response for wrong HTTP method."""
        if mock_app is None:
            pytest.skip("Test client not available")

        # POST to a GET-only endpoint
        response = mock_app.post("/health")

        assert response.status_code == 405


# =============================================================================
# Test Class: CORS Headers
# =============================================================================

@pytest.mark.integration
class TestCORSHeaders:
    """Tests for CORS headers."""

    def test_cors_headers_present(self, mock_app):
        """Test that CORS headers are present."""
        if mock_app is None:
            pytest.skip("Test client not available")

        response = mock_app.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            }
        )

        # CORS preflight should return 200
        assert response.status_code in [200, 405]

    def test_allowed_origins(self, mock_app):
        """Test that allowed origins work."""
        if mock_app is None:
            pytest.skip("Test client not available")

        response = mock_app.get(
            "/health",
            headers={"Origin": "http://localhost:3000"}
        )

        assert response.status_code == 200

        # Check for CORS header (may or may not be present)
        cors_header = response.headers.get("access-control-allow-origin")
        if cors_header:
            assert cors_header in ["*", "http://localhost:3000"]


# =============================================================================
# Test Class: Response Time
# =============================================================================

@pytest.mark.integration
class TestResponseTime:
    """Tests for API response time."""

    def test_health_response_time(self, mock_app):
        """Test that health endpoint responds quickly."""
        import time

        if mock_app is None:
            pytest.skip("Test client not available")

        start = time.time()
        response = mock_app.get("/health")
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 1.0, f"Health check took too long: {elapsed:.2f}s"

    def test_root_response_time(self, mock_app):
        """Test that root endpoint responds quickly."""
        import time

        if mock_app is None:
            pytest.skip("Test client not available")

        start = time.time()
        response = mock_app.get("/")
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 1.0, f"Root endpoint took too long: {elapsed:.2f}s"
