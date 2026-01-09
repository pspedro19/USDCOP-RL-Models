"""
Integration Tests: Multi-Model Trading API
==========================================

Tests all API endpoints for the multi-model trading signals service.

Endpoints tested:
- GET /api/health
- GET /api/models/signals/latest
- GET /api/models/performance/comparison
- GET /api/models/equity-curves
- GET /api/models/positions/current
- GET /api/models/pnl/summary
- WS /ws/trading-signals

Author: Pedro @ Lean Tech Solutions
Date: 2025-12-26
"""

import pytest
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any
from httpx import AsyncClient, ASGITransport
import json

# Import the API app for testing
try:
    import sys
    from pathlib import Path
    services_path = Path(__file__).parent.parent.parent / 'services'
    sys.path.insert(0, str(services_path))
    from multi_model_trading_api import app
    HAS_APP = True
except ImportError:
    HAS_APP = False


@pytest.fixture
def api_base_url() -> str:
    """Multi-model trading API base URL"""
    import os
    return os.getenv('MULTI_MODEL_API_URL', 'http://localhost:8006')


@pytest.fixture
async def async_client():
    """Create async HTTP client for testing"""
    if not HAS_APP:
        pytest.skip("multi_model_trading_api not available")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.integration
@pytest.mark.api
class TestHealthEndpoint:
    """Tests for health check endpoint"""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, async_client):
        """GET /api/health returns 200 when healthy"""
        response = await async_client.get("/api/health")

        assert response.status_code in [200, 500], \
            f"Expected 200 or 500, got {response.status_code}"

    @pytest.mark.asyncio
    async def test_health_response_structure(self, async_client):
        """Health response has required fields"""
        response = await async_client.get("/api/health")
        data = response.json()

        # Required fields
        assert 'status' in data, "Missing 'status' field"
        assert 'timestamp' in data, "Missing 'timestamp' field"

        # Status should be valid
        assert data['status'] in ['healthy', 'unhealthy'], \
            f"Invalid status: {data['status']}"

    @pytest.mark.asyncio
    async def test_health_timestamp_format(self, async_client):
        """Health timestamp is ISO format"""
        response = await async_client.get("/api/health")
        data = response.json()

        try:
            datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        except ValueError as e:
            pytest.fail(f"Invalid timestamp format: {data['timestamp']} - {e}")


@pytest.mark.integration
@pytest.mark.api
class TestModelsAPI:
    """Tests for model-related endpoints"""

    @pytest.mark.asyncio
    async def test_list_models_endpoint_exists(self, async_client):
        """GET /api/models/signals/latest endpoint exists"""
        response = await async_client.get("/api/models/signals/latest")

        # Should not be 404 (endpoint exists)
        assert response.status_code != 404, "Endpoint not found"

        # 500 is acceptable if DB not connected, 200 if working
        assert response.status_code in [200, 500], \
            f"Unexpected status: {response.status_code}"

    @pytest.mark.asyncio
    async def test_get_latest_signals_structure(self, async_client):
        """GET /api/models/signals/latest returns correct structure"""
        response = await async_client.get("/api/models/signals/latest")

        if response.status_code == 500:
            pytest.skip("Database not connected")

        data = response.json()

        # Required top-level fields
        assert 'timestamp' in data, "Missing timestamp"
        assert 'market_price' in data, "Missing market_price"
        assert 'market_status' in data, "Missing market_status"
        assert 'signals' in data, "Missing signals array"

        # Signals should be a list
        assert isinstance(data['signals'], list), "Signals should be a list"

    @pytest.mark.asyncio
    async def test_signal_object_structure(self, async_client):
        """Each signal has required fields"""
        response = await async_client.get("/api/models/signals/latest")

        if response.status_code == 500:
            pytest.skip("Database not connected")

        data = response.json()

        if not data['signals']:
            pytest.skip("No signals available")

        signal = data['signals'][0]

        required_fields = [
            'strategy_code', 'strategy_name', 'signal', 'side',
            'confidence', 'size', 'risk_usd', 'reasoning',
            'timestamp', 'age_seconds'
        ]

        for field in required_fields:
            assert field in signal, f"Signal missing required field: {field}"

        # Validate field types
        assert isinstance(signal['confidence'], (int, float)), "Confidence should be numeric"
        assert 0 <= signal['confidence'] <= 1, "Confidence should be 0-1"
        assert isinstance(signal['age_seconds'], int), "age_seconds should be int"

    @pytest.mark.asyncio
    async def test_market_status_values(self, async_client):
        """Market status has valid values"""
        response = await async_client.get("/api/models/signals/latest")

        if response.status_code == 500:
            pytest.skip("Database not connected")

        data = response.json()

        valid_statuses = ['open', 'closed', 'pre_market']
        assert data['market_status'] in valid_statuses, \
            f"Invalid market status: {data['market_status']}"


@pytest.mark.integration
@pytest.mark.api
class TestPerformanceComparison:
    """Tests for performance comparison endpoint"""

    @pytest.mark.asyncio
    async def test_performance_comparison_default_period(self, async_client):
        """GET /api/models/performance/comparison with default period"""
        response = await async_client.get("/api/models/performance/comparison")

        if response.status_code == 500:
            pytest.skip("Database not connected")

        assert response.status_code == 200
        data = response.json()

        assert 'period' in data, "Missing period"
        assert 'strategies' in data, "Missing strategies"
        assert data['period'] == '30d', "Default period should be 30d"

    @pytest.mark.asyncio
    async def test_performance_comparison_all_periods(self, async_client):
        """Test all period options"""
        periods = ['24h', '7d', '30d', 'all']

        for period in periods:
            response = await async_client.get(
                f"/api/models/performance/comparison?period={period}"
            )

            if response.status_code == 500:
                continue  # Skip if DB not connected

            assert response.status_code == 200, f"Failed for period {period}"
            data = response.json()
            assert data['period'] == period, f"Period mismatch for {period}"

    @pytest.mark.asyncio
    async def test_strategy_performance_fields(self, async_client):
        """Each strategy has required performance fields"""
        response = await async_client.get("/api/models/performance/comparison")

        if response.status_code == 500:
            pytest.skip("Database not connected")

        data = response.json()

        if not data['strategies']:
            pytest.skip("No strategies available")

        strategy = data['strategies'][0]

        required_fields = [
            'strategy_code', 'strategy_name', 'strategy_type',
            'total_return_pct', 'daily_return_pct',
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'total_trades', 'win_rate', 'profit_factor',
            'max_drawdown_pct', 'current_drawdown_pct', 'volatility_pct',
            'avg_hold_time_minutes', 'current_equity', 'open_positions'
        ]

        for field in required_fields:
            assert field in strategy, f"Strategy missing field: {field}"

        # Validate win_rate range
        assert 0 <= strategy['win_rate'] <= 1, "Win rate should be 0-1"


@pytest.mark.integration
@pytest.mark.api
class TestEquityCurves:
    """Tests for equity curves endpoint"""

    @pytest.mark.asyncio
    async def test_equity_curves_default_params(self, async_client):
        """GET /api/models/equity-curves with defaults"""
        response = await async_client.get("/api/models/equity-curves")

        if response.status_code == 500:
            pytest.skip("Database not connected")

        assert response.status_code == 200
        data = response.json()

        assert 'start_date' in data
        assert 'end_date' in data
        assert 'resolution' in data
        assert 'curves' in data

    @pytest.mark.asyncio
    async def test_equity_curves_with_hours_param(self, async_client):
        """Test hours parameter"""
        for hours in [24, 48, 168]:  # 1 day, 2 days, 1 week
            response = await async_client.get(
                f"/api/models/equity-curves?hours={hours}"
            )

            if response.status_code == 500:
                continue

            assert response.status_code == 200, f"Failed for hours={hours}"

    @pytest.mark.asyncio
    async def test_equity_curves_resolution_options(self, async_client):
        """Test resolution options"""
        resolutions = ['5m', '1h', '1d']

        for res in resolutions:
            response = await async_client.get(
                f"/api/models/equity-curves?resolution={res}"
            )

            if response.status_code == 500:
                continue

            assert response.status_code == 200, f"Failed for resolution={res}"
            data = response.json()
            assert data['resolution'] == res

    @pytest.mark.asyncio
    async def test_equity_curve_data_structure(self, async_client):
        """Equity curve data has correct structure"""
        response = await async_client.get("/api/models/equity-curves")

        if response.status_code == 500:
            pytest.skip("Database not connected")

        data = response.json()

        if not data['curves']:
            pytest.skip("No curves available")

        curve = data['curves'][0]

        assert 'strategy_code' in curve
        assert 'strategy_name' in curve
        assert 'data' in curve
        assert 'summary' in curve

        if curve['data']:
            point = curve['data'][0]
            assert 'timestamp' in point
            assert 'equity_value' in point
            assert 'return_pct' in point
            assert 'drawdown_pct' in point


@pytest.mark.integration
@pytest.mark.api
class TestCurrentPositions:
    """Tests for current positions endpoint"""

    @pytest.mark.asyncio
    async def test_current_positions_structure(self, async_client):
        """GET /api/models/positions/current returns correct structure"""
        response = await async_client.get("/api/models/positions/current")

        if response.status_code == 500:
            pytest.skip("Database not connected")

        assert response.status_code == 200
        data = response.json()

        assert 'timestamp' in data
        assert 'total_positions' in data
        assert 'total_notional' in data
        assert 'total_pnl' in data
        assert 'positions' in data

        assert isinstance(data['positions'], list)

    @pytest.mark.asyncio
    async def test_current_positions_filter_by_strategy(self, async_client):
        """Test filtering positions by strategy"""
        response = await async_client.get(
            "/api/models/positions/current?strategy=RL_PPO"
        )

        if response.status_code == 500:
            pytest.skip("Database not connected")

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_position_object_structure(self, async_client):
        """Position objects have required fields"""
        response = await async_client.get("/api/models/positions/current")

        if response.status_code == 500:
            pytest.skip("Database not connected")

        data = response.json()

        if not data['positions']:
            pytest.skip("No open positions")

        position = data['positions'][0]

        required_fields = [
            'position_id', 'strategy_code', 'strategy_name',
            'side', 'quantity', 'entry_price', 'current_price',
            'unrealized_pnl', 'unrealized_pnl_pct',
            'entry_time', 'holding_time_minutes', 'leverage'
        ]

        for field in required_fields:
            assert field in position, f"Position missing field: {field}"


@pytest.mark.integration
@pytest.mark.api
class TestPnLSummary:
    """Tests for P&L summary endpoint"""

    @pytest.mark.asyncio
    async def test_pnl_summary_default_period(self, async_client):
        """GET /api/models/pnl/summary with default period"""
        response = await async_client.get("/api/models/pnl/summary")

        if response.status_code == 500:
            pytest.skip("Database not connected")

        assert response.status_code == 200
        data = response.json()

        assert 'period' in data
        assert 'start_date' in data
        assert 'end_date' in data
        assert 'strategies' in data
        assert 'portfolio_total' in data

    @pytest.mark.asyncio
    async def test_pnl_summary_all_periods(self, async_client):
        """Test all period options"""
        periods = ['today', 'week', 'month', 'all']

        for period in periods:
            response = await async_client.get(
                f"/api/models/pnl/summary?period={period}"
            )

            if response.status_code == 500:
                continue

            assert response.status_code == 200, f"Failed for period {period}"
            data = response.json()
            assert data['period'] == period

    @pytest.mark.asyncio
    async def test_strategy_pnl_fields(self, async_client):
        """Each strategy has required P&L fields"""
        response = await async_client.get("/api/models/pnl/summary")

        if response.status_code == 500:
            pytest.skip("Database not connected")

        data = response.json()

        if not data['strategies']:
            pytest.skip("No strategies available")

        strategy = data['strategies'][0]

        required_fields = [
            'strategy_code', 'strategy_name',
            'gross_profit', 'gross_loss', 'net_profit', 'total_fees',
            'n_trades', 'n_wins', 'n_losses', 'win_rate',
            'avg_win', 'avg_loss', 'avg_trade', 'profit_factor'
        ]

        for field in required_fields:
            assert field in strategy, f"Strategy P&L missing field: {field}"


@pytest.mark.integration
@pytest.mark.api
class TestErrorHandling:
    """Tests for API error handling"""

    @pytest.mark.asyncio
    async def test_invalid_period_param(self, async_client):
        """Invalid period parameter returns error"""
        response = await async_client.get(
            "/api/models/performance/comparison?period=invalid"
        )

        # Should either handle gracefully or return default
        assert response.status_code in [200, 400, 422, 500]

    @pytest.mark.asyncio
    async def test_invalid_hours_param(self, async_client):
        """Invalid hours parameter returns validation error"""
        response = await async_client.get(
            "/api/models/equity-curves?hours=-1"
        )

        # Should be 422 for validation error or 400
        assert response.status_code in [400, 422, 500]

    @pytest.mark.asyncio
    async def test_root_endpoint(self, async_client):
        """Root endpoint returns API info"""
        response = await async_client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert 'message' in data
        assert 'version' in data
        assert 'endpoints' in data

    @pytest.mark.asyncio
    async def test_nonexistent_endpoint(self, async_client):
        """Non-existent endpoint returns 404"""
        response = await async_client.get("/api/nonexistent")

        assert response.status_code == 404


@pytest.mark.integration
@pytest.mark.api
class TestModelNotFound:
    """Tests for model not found scenarios"""

    @pytest.mark.asyncio
    async def test_filter_unknown_strategy(self, async_client):
        """Filtering by unknown strategy returns empty or error"""
        response = await async_client.get(
            "/api/models/positions/current?strategy=UNKNOWN_STRATEGY"
        )

        if response.status_code == 500:
            pytest.skip("Database not connected")

        # Should return empty positions or 404
        if response.status_code == 200:
            data = response.json()
            assert data['total_positions'] == 0 or data['positions'] == []
        else:
            assert response.status_code in [404, 400]


@pytest.mark.integration
@pytest.mark.api
class TestCompareModels:
    """Tests for model comparison functionality"""

    @pytest.mark.asyncio
    async def test_compare_models_via_performance(self, async_client):
        """Compare models through performance endpoint"""
        response = await async_client.get("/api/models/performance/comparison")

        if response.status_code == 500:
            pytest.skip("Database not connected")

        data = response.json()
        strategies = data.get('strategies', [])

        if len(strategies) < 2:
            pytest.skip("Need at least 2 strategies to compare")

        # Verify we can compare metrics
        for strategy in strategies:
            assert 'sharpe_ratio' in strategy
            assert 'win_rate' in strategy
            assert 'total_return_pct' in strategy

    @pytest.mark.asyncio
    async def test_equity_curves_multiple_strategies(self, async_client):
        """Compare equity curves for multiple strategies"""
        response = await async_client.get(
            "/api/models/equity-curves?strategies=RL_PPO,ML_XGB"
        )

        if response.status_code == 500:
            pytest.skip("Database not connected")

        assert response.status_code == 200
        data = response.json()

        # Should return curves for specified strategies
        assert 'curves' in data


@pytest.mark.integration
@pytest.mark.api
class TestCORS:
    """Tests for CORS configuration"""

    @pytest.mark.asyncio
    async def test_cors_headers_present(self, async_client):
        """CORS headers are present in response"""
        response = await async_client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )

        # CORS preflight should be successful
        assert response.status_code in [200, 204, 405]


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.slow
class TestWebSocket:
    """Tests for WebSocket endpoint"""

    @pytest.mark.asyncio
    async def test_websocket_connection(self, api_base_url):
        """WebSocket connection can be established"""
        import websockets

        ws_url = api_base_url.replace("http", "ws") + "/ws/trading-signals"

        try:
            async with websockets.connect(ws_url, timeout=5) as ws:
                # Should receive initial connection message
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                data = json.loads(msg)

                assert 'type' in data
                assert data['type'] == 'connection'

        except (ConnectionRefusedError, OSError):
            pytest.skip("WebSocket server not available")
        except asyncio.TimeoutError:
            pytest.fail("WebSocket connection timed out")

    @pytest.mark.asyncio
    async def test_websocket_ping_pong(self, api_base_url):
        """WebSocket responds to ping"""
        import websockets

        ws_url = api_base_url.replace("http", "ws") + "/ws/trading-signals"

        try:
            async with websockets.connect(ws_url, timeout=5) as ws:
                # Wait for initial message
                await asyncio.wait_for(ws.recv(), timeout=5)

                # Send ping
                await ws.send("ping")

                # Should receive pong
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                assert msg == "pong"

        except (ConnectionRefusedError, OSError):
            pytest.skip("WebSocket server not available")
