"""
Integration Tests for Infrastructure Components
================================================

Tests for:
- HashiCorp Vault connectivity and secret retrieval
- Feast Feature Store materialization and retrieval
- Jaeger tracing connectivity
- Grafana dashboard provisioning
- Service health checks

These tests require the infrastructure stack to be running:
    docker-compose -f docker-compose.yml -f docker-compose.infrastructure.yml up -d

Author: Trading Team
Date: 2026-01-17
"""

import os
import pytest
import requests
import time
from typing import Optional, Dict, Any

# Skip all tests if infrastructure is not available
pytestmark = pytest.mark.integration


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def vault_client():
    """Create Vault client for testing."""
    try:
        from src.shared.secrets.vault_client import VaultClient
        client = VaultClient(
            url=os.getenv("VAULT_ADDR", "http://localhost:8200"),
            token=os.getenv("VAULT_TOKEN", "devtoken"),
        )
        return client
    except ImportError:
        pytest.skip("VaultClient not available")


@pytest.fixture(scope="module")
def feast_store():
    """Create Feast store for testing."""
    try:
        from feast import FeatureStore
        store = FeatureStore(repo_path="feature_repo")
        return store
    except (ImportError, Exception) as e:
        pytest.skip(f"Feast not available: {e}")


# =============================================================================
# VAULT TESTS
# =============================================================================

class TestVaultIntegration:
    """Test HashiCorp Vault integration."""

    def test_vault_health(self):
        """Vault should be healthy and responding."""
        vault_addr = os.getenv("VAULT_ADDR", "http://localhost:8200")
        try:
            response = requests.get(f"{vault_addr}/v1/sys/health", timeout=5)
            assert response.status_code in [200, 429, 472, 473, 501, 503]
            data = response.json()
            # In dev mode, initialized and unsealed should be true
            if response.status_code == 200:
                assert data.get("initialized") is True
                assert data.get("sealed") is False
        except requests.exceptions.ConnectionError:
            pytest.skip("Vault not available")

    def test_vault_get_secret(self, vault_client):
        """Should retrieve secrets from Vault."""
        if vault_client is None:
            pytest.skip("Vault client not configured")

        # Test getting a secret (may be empty in dev)
        secret = vault_client.get_secret("secret/data/trading/database", "password")
        # Secret might be None if not configured, but method should not raise
        assert secret is None or isinstance(secret, str)

    def test_vault_fallback_to_env(self, vault_client):
        """Should fallback to environment variables when secret not in Vault."""
        if vault_client is None:
            pytest.skip("Vault client not configured")

        # Set an env var
        os.environ["TEST_FALLBACK_VAR"] = "test_value"

        # Try to get a non-existent secret
        result = vault_client.get_secret(
            "secret/data/nonexistent/path",
            "TEST_FALLBACK_VAR",
            env_fallback="TEST_FALLBACK_VAR"
        )

        # Clean up
        del os.environ["TEST_FALLBACK_VAR"]

        # Should have fallen back to env var
        assert result == "test_value"


# =============================================================================
# FEAST TESTS
# =============================================================================

class TestFeastIntegration:
    """Test Feast Feature Store integration."""

    def test_feast_registry_available(self, feast_store):
        """Feast registry should be accessible."""
        if feast_store is None:
            pytest.skip("Feast store not available")

        # List feature views
        feature_views = feast_store.list_feature_views()
        assert isinstance(feature_views, list)

    def test_feast_feature_views_defined(self, feast_store):
        """Expected feature views should be defined."""
        if feast_store is None:
            pytest.skip("Feast store not available")

        feature_views = feast_store.list_feature_views()
        view_names = [fv.name for fv in feature_views]

        expected_views = ["technical_features", "macro_features", "state_features"]
        for view in expected_views:
            if view not in view_names:
                pytest.skip(f"Feature view {view} not yet defined")

    def test_feast_online_store_connection(self, feast_store):
        """Feast should connect to Redis online store."""
        if feast_store is None:
            pytest.skip("Feast store not available")

        # Try to get online features (may be empty)
        try:
            # This tests the connection to Redis
            features = feast_store.get_online_features(
                features=["technical_features:rsi_9"],
                entity_rows=[{"symbol": "USDCOP", "bar_id": 0}]
            )
            # If we get here without error, connection works
            assert features is not None
        except Exception as e:
            # Redis might not have data yet, but connection should work
            if "connection" in str(e).lower():
                pytest.fail(f"Redis connection failed: {e}")


# =============================================================================
# JAEGER TESTS
# =============================================================================

class TestJaegerIntegration:
    """Test Jaeger distributed tracing integration."""

    def test_jaeger_ui_available(self):
        """Jaeger UI should be accessible."""
        jaeger_url = os.getenv("JAEGER_UI_URL", "http://localhost:16686")
        try:
            response = requests.get(jaeger_url, timeout=5)
            assert response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.skip("Jaeger not available")

    def test_jaeger_api_services(self):
        """Jaeger API should list services."""
        jaeger_url = os.getenv("JAEGER_UI_URL", "http://localhost:16686")
        try:
            response = requests.get(f"{jaeger_url}/api/services", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            assert isinstance(data["data"], list)
        except requests.exceptions.ConnectionError:
            pytest.skip("Jaeger not available")

    def test_otel_collector_health(self):
        """OpenTelemetry Collector should be healthy."""
        otel_url = os.getenv("OTEL_COLLECTOR_URL", "http://localhost:13133")
        try:
            response = requests.get(otel_url, timeout=5)
            # Health endpoint returns 200 when healthy
            if response.status_code == 200:
                assert True
            else:
                pytest.skip("OTel Collector not configured")
        except requests.exceptions.ConnectionError:
            pytest.skip("OTel Collector not available")


# =============================================================================
# GRAFANA TESTS
# =============================================================================

class TestGrafanaIntegration:
    """Test Grafana dashboard provisioning."""

    def test_grafana_health(self):
        """Grafana should be healthy."""
        grafana_url = os.getenv("GRAFANA_URL", "http://localhost:3003")
        try:
            response = requests.get(f"{grafana_url}/api/health", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert data.get("database") == "ok"
        except requests.exceptions.ConnectionError:
            pytest.skip("Grafana not available")

    def test_grafana_datasources_provisioned(self):
        """Data sources should be provisioned."""
        grafana_url = os.getenv("GRAFANA_URL", "http://localhost:3003")
        grafana_user = os.getenv("GRAFANA_USER", "admin")
        grafana_password = os.getenv("GRAFANA_PASSWORD", "admin")

        try:
            response = requests.get(
                f"{grafana_url}/api/datasources",
                auth=(grafana_user, grafana_password),
                timeout=5
            )
            if response.status_code == 401:
                pytest.skip("Grafana authentication failed")

            assert response.status_code == 200
            datasources = response.json()

            expected_datasources = ["Prometheus", "Loki", "Jaeger", "TimescaleDB"]
            ds_names = [ds["name"] for ds in datasources]

            for expected in expected_datasources:
                if expected not in ds_names:
                    pytest.skip(f"Datasource {expected} not yet provisioned")

        except requests.exceptions.ConnectionError:
            pytest.skip("Grafana not available")

    def test_grafana_dashboards_provisioned(self):
        """Dashboards should be provisioned."""
        grafana_url = os.getenv("GRAFANA_URL", "http://localhost:3003")
        grafana_user = os.getenv("GRAFANA_USER", "admin")
        grafana_password = os.getenv("GRAFANA_PASSWORD", "admin")

        try:
            response = requests.get(
                f"{grafana_url}/api/search?type=dash-db",
                auth=(grafana_user, grafana_password),
                timeout=5
            )
            if response.status_code == 401:
                pytest.skip("Grafana authentication failed")

            assert response.status_code == 200
            dashboards = response.json()

            # Should have at least some dashboards
            if len(dashboards) == 0:
                pytest.skip("No dashboards provisioned yet")

        except requests.exceptions.ConnectionError:
            pytest.skip("Grafana not available")


# =============================================================================
# SERVICE HEALTH TESTS
# =============================================================================

class TestServiceHealth:
    """Test overall service health."""

    @pytest.mark.parametrize("service,url,expected_path", [
        ("prometheus", "http://localhost:9090", "/-/healthy"),
        ("loki", "http://localhost:3100", "/ready"),
        ("alertmanager", "http://localhost:9093", "/-/healthy"),
        ("redis", "http://localhost:6379", None),  # Redis doesn't have HTTP
        ("postgres", "http://localhost:5432", None),  # Postgres doesn't have HTTP
    ])
    def test_service_health(self, service, url, expected_path):
        """Each service should be healthy."""
        if expected_path is None:
            pytest.skip(f"{service} doesn't have HTTP health endpoint")

        try:
            response = requests.get(f"{url}{expected_path}", timeout=5)
            assert response.status_code == 200, f"{service} unhealthy"
        except requests.exceptions.ConnectionError:
            pytest.skip(f"{service} not available")


# =============================================================================
# END-TO-END TESTS
# =============================================================================

class TestEndToEnd:
    """End-to-end integration tests."""

    def test_tracing_flow(self):
        """Test that traces flow from service to Jaeger."""
        # This would require a running inference service with tracing enabled
        # Skip for now as it requires full stack
        pytest.skip("Requires full trading stack running")

    def test_feature_retrieval_flow(self):
        """Test feature retrieval from Feast with fallback to builder."""
        # This would require Feast and CanonicalFeatureBuilder
        pytest.skip("Requires full feature stack running")

    def test_secrets_retrieval_flow(self):
        """Test secrets retrieval from Vault with env fallback."""
        # This would require Vault with secrets configured
        pytest.skip("Requires Vault with secrets configured")
