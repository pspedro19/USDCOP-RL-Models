"""
Unit tests for TradingFlagsRedis.

Tests the Redis-backed kill switch implementation (P2 Remediation).

These tests use fakeredis to mock Redis without requiring a real connection.

Author: Trading Team
Version: 1.0.0
Created: 2026-01-18
"""

import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import fakeredis for mocking
try:
    import fakeredis
    FAKEREDIS_AVAILABLE = True
except ImportError:
    FAKEREDIS_AVAILABLE = False
    fakeredis = None

# Import trading flags module directly to avoid circular imports
import importlib.util
spec = importlib.util.spec_from_file_location(
    "trading_flags",
    PROJECT_ROOT / "src" / "config" / "trading_flags.py"
)
trading_flags_module = importlib.util.module_from_spec(spec)

# Mock redis import if fakeredis is available
if FAKEREDIS_AVAILABLE:
    sys.modules['redis'] = fakeredis
    spec.loader.exec_module(trading_flags_module)
    TradingFlagsRedis = trading_flags_module.TradingFlagsRedis
else:
    TradingFlagsRedis = None


@pytest.mark.skipif(not FAKEREDIS_AVAILABLE, reason="fakeredis not installed")
class TestTradingFlagsRedisBasics:
    """Basic tests for TradingFlagsRedis."""

    @pytest.fixture
    def redis_client(self):
        """Create a fake Redis client."""
        return fakeredis.FakeRedis(decode_responses=True)

    @pytest.fixture
    def flags(self, redis_client):
        """Create TradingFlagsRedis with fake Redis."""
        return TradingFlagsRedis(redis_client=redis_client, enable_pubsub=False)

    def test_default_kill_switch_is_false(self, flags):
        """Kill switch should be False by default."""
        assert flags.kill_switch == False  # noqa: E712

    def test_default_inference_enabled_is_true(self, flags):
        """Inference should be enabled by default."""
        assert flags.inference_enabled == True  # noqa: E712

    def test_default_paper_trading_is_true(self, flags):
        """Paper trading should be enabled by default."""
        assert flags.paper_trading == True  # noqa: E712

    def test_default_maintenance_mode_is_false(self, flags):
        """Maintenance mode should be False by default."""
        assert flags.maintenance_mode == False  # noqa: E712


@pytest.mark.skipif(not FAKEREDIS_AVAILABLE, reason="fakeredis not installed")
class TestKillSwitch:
    """Tests for kill switch functionality."""

    @pytest.fixture
    def redis_client(self):
        return fakeredis.FakeRedis(decode_responses=True)

    @pytest.fixture
    def flags(self, redis_client):
        return TradingFlagsRedis(redis_client=redis_client, enable_pubsub=False)

    def test_activate_kill_switch(self, flags):
        """Should be able to activate kill switch."""
        flags.set_kill_switch(True, reason="Test activation")
        assert flags.kill_switch == True  # noqa: E712

    def test_deactivate_kill_switch(self, flags):
        """Should be able to deactivate kill switch."""
        flags.set_kill_switch(True, reason="Activate first")
        flags.set_kill_switch(False, reason="Test deactivation")
        assert flags.kill_switch == False  # noqa: E712

    def test_kill_switch_blocks_trading(self, flags):
        """Kill switch should block trading."""
        flags.set_kill_switch(True, reason="Emergency")
        allowed, reason = flags.is_trading_allowed()
        assert allowed == False  # noqa: E712
        assert "kill switch" in reason.lower()

    def test_kill_switch_blocks_inference(self, flags):
        """Kill switch should block inference."""
        flags.set_kill_switch(True, reason="Emergency")
        allowed, reason = flags.is_inference_allowed()
        assert allowed == False  # noqa: E712
        assert "kill switch" in reason.lower()

    def test_kill_switch_state_persists(self, redis_client):
        """Kill switch state should persist across instances."""
        # Create first instance and set kill switch
        flags1 = TradingFlagsRedis(redis_client=redis_client, enable_pubsub=False)
        flags1.set_kill_switch(True, reason="Persist test")

        # Create second instance with same Redis
        flags2 = TradingFlagsRedis(redis_client=redis_client, enable_pubsub=False)

        # State should be visible in second instance
        assert flags2.kill_switch == True  # noqa: E712


@pytest.mark.skipif(not FAKEREDIS_AVAILABLE, reason="fakeredis not installed")
class TestFlagState:
    """Tests for flag state storage."""

    @pytest.fixture
    def redis_client(self):
        return fakeredis.FakeRedis(decode_responses=True)

    @pytest.fixture
    def flags(self, redis_client):
        return TradingFlagsRedis(redis_client=redis_client, enable_pubsub=False)

    def test_get_all_flags_returns_all_states(self, flags):
        """Should return all flag states."""
        all_flags = flags.get_all_flags()

        assert "kill_switch" in all_flags
        assert "maintenance_mode" in all_flags
        assert "paper_trading" in all_flags
        assert "inference_enabled" in all_flags

    def test_flag_state_includes_reason(self, flags):
        """Flag state should include reason."""
        flags.set_kill_switch(True, reason="Test reason")
        all_flags = flags.get_all_flags()

        assert all_flags["kill_switch"]["reason"] == "Test reason"

    def test_flag_state_includes_updated_by(self, flags):
        """Flag state should include who updated it."""
        flags.set_kill_switch(True, reason="Test", updated_by="test_user")
        all_flags = flags.get_all_flags()

        assert all_flags["kill_switch"]["updated_by"] == "test_user"


@pytest.mark.skipif(not FAKEREDIS_AVAILABLE, reason="fakeredis not installed")
class TestHealthCheck:
    """Tests for health check functionality."""

    @pytest.fixture
    def redis_client(self):
        return fakeredis.FakeRedis(decode_responses=True)

    @pytest.fixture
    def flags(self, redis_client):
        return TradingFlagsRedis(redis_client=redis_client, enable_pubsub=False)

    def test_health_check_returns_true_when_healthy(self, flags):
        """Health check should return True when Redis is available."""
        assert flags.health_check() == True  # noqa: E712


@pytest.mark.skipif(not FAKEREDIS_AVAILABLE, reason="fakeredis not installed")
class TestErrorHandling:
    """Tests for error handling."""

    def test_kill_switch_fails_safe_on_error(self):
        """Kill switch should return True (safe) on Redis error."""
        # Create a mock Redis that raises errors
        mock_redis = Mock()
        mock_redis.exists.return_value = True  # Skip default initialization
        mock_redis.get.side_effect = Exception("Connection error")

        # Patch the initialization to avoid connection attempts
        with patch.object(TradingFlagsRedis, '_ensure_defaults', return_value=None):
            flags = TradingFlagsRedis.__new__(TradingFlagsRedis)
            flags._redis = mock_redis
            flags._enable_pubsub = False

            # Should fail safe - assume kill switch is ON
            assert flags.kill_switch == True  # noqa: E712


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
