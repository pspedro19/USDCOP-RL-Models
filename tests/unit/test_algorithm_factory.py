"""
Unit tests for Algorithm Factory (Phase 1).

Tests the factory pattern, adapter creation, algorithm resolution,
and kwargs extraction from SSOT config.
"""

import pytest
from unittest.mock import MagicMock, patch


# ============================================================================
# Factory tests
# ============================================================================


class TestCreateAlgorithm:
    """Test create_algorithm() factory dispatch."""

    def test_create_ppo_adapter(self):
        from src.training.algorithm_factory import create_algorithm, PPOAdapter

        adapter = create_algorithm("ppo")
        assert isinstance(adapter, PPOAdapter)
        assert adapter.name() == "ppo"
        assert adapter.is_recurrent() is False

    def test_create_recurrent_ppo_adapter(self):
        from src.training.algorithm_factory import create_algorithm, RecurrentPPOAdapter

        adapter = create_algorithm("recurrent_ppo", lstm_hidden_size=64, n_lstm_layers=2)
        assert isinstance(adapter, RecurrentPPOAdapter)
        assert adapter.name() == "recurrent_ppo"
        assert adapter.is_recurrent() is True
        assert adapter._lstm_hidden_size == 64
        assert adapter._n_lstm_layers == 2

    def test_create_sac_adapter(self):
        from src.training.algorithm_factory import create_algorithm, SACAdapter

        adapter = create_algorithm("sac")
        assert isinstance(adapter, SACAdapter)
        assert adapter.name() == "sac"
        assert adapter.is_recurrent() is False

    def test_create_unknown_raises(self):
        from src.training.algorithm_factory import create_algorithm

        with pytest.raises(ValueError, match="Unknown algorithm 'unknown'"):
            create_algorithm("unknown")

    def test_extra_kwargs_ignored_gracefully(self):
        from src.training.algorithm_factory import create_algorithm

        # PPOAdapter.__init__ has no custom params, so extra kwargs should be ignored
        adapter = create_algorithm("ppo", lstm_hidden_size=128)
        assert adapter.name() == "ppo"


class TestRegisterAlgorithm:
    """Test register_algorithm() extensibility."""

    def test_register_custom_adapter(self):
        from src.training.algorithm_factory import (
            register_algorithm,
            create_algorithm,
            AlgorithmAdapter,
            _ALGORITHM_REGISTRY,
        )

        class CustomAdapter(AlgorithmAdapter):
            def create(self, env, **kwargs):
                return MagicMock()

            def load(self, path, device="cpu"):
                return MagicMock()

            def is_recurrent(self):
                return False

            def name(self):
                return "custom"

        register_algorithm("custom_test", CustomAdapter)
        try:
            adapter = create_algorithm("custom_test")
            assert adapter.name() == "custom"
        finally:
            # Clean up
            _ALGORITHM_REGISTRY.pop("custom_test", None)


# ============================================================================
# Algorithm name resolution
# ============================================================================


class TestResolveAlgorithmName:
    """Test resolve_algorithm_name() with SSOT config."""

    def test_explicit_algorithm_field(self):
        from src.training.algorithm_factory import resolve_algorithm_name

        config = MagicMock()
        config._raw = {"training": {"algorithm": "sac"}}
        config.lstm.enabled = False

        assert resolve_algorithm_name(config) == "sac"

    def test_lstm_enabled_implies_recurrent_ppo(self):
        from src.training.algorithm_factory import resolve_algorithm_name

        config = MagicMock()
        config._raw = {"training": {}}
        config.lstm.enabled = True

        assert resolve_algorithm_name(config) == "recurrent_ppo"

    def test_default_is_ppo(self):
        from src.training.algorithm_factory import resolve_algorithm_name

        config = MagicMock()
        config._raw = {"training": {}}
        config.lstm.enabled = False

        assert resolve_algorithm_name(config) == "ppo"

    def test_explicit_overrides_lstm(self):
        from src.training.algorithm_factory import resolve_algorithm_name

        config = MagicMock()
        config._raw = {"training": {"algorithm": "ppo"}}
        config.lstm.enabled = True  # Explicit "ppo" overrides

        assert resolve_algorithm_name(config) == "ppo"


# ============================================================================
# Kwargs extraction
# ============================================================================


class TestGetAlgorithmKwargs:
    """Test get_algorithm_kwargs() from config."""

    def test_ppo_kwargs(self):
        from src.training.algorithm_factory import get_algorithm_kwargs

        config = MagicMock()
        config._raw = {
            "training": {
                "network": {"policy_layers": [128, 128]},
            }
        }
        config.lstm.enabled = False
        config.ppo.learning_rate = 3e-4
        config.ppo.n_steps = 2048
        config.ppo.batch_size = 64
        config.ppo.n_epochs = 10
        config.ppo.gamma = 0.99
        config.ppo.gae_lambda = 0.95
        config.ppo.clip_range = 0.2
        config.ppo.ent_coef = 0.01
        config.ppo.vf_coef = 0.5
        config.ppo.max_grad_norm = 0.5

        kwargs = get_algorithm_kwargs(config)

        assert kwargs["learning_rate"] == 3e-4
        assert kwargs["gamma"] == 0.99
        assert kwargs["policy_kwargs"]["net_arch"] == [128, 128]

    def test_sac_kwargs(self):
        from src.training.algorithm_factory import get_algorithm_kwargs

        config = MagicMock()
        config._raw = {
            "training": {
                "algorithm": "sac",
                "sac": {
                    "learning_rate": 1e-4,
                    "buffer_size": 100_000,
                    "batch_size": 256,
                    "tau": 0.005,
                    "gamma": 0.99,
                    "ent_coef": "auto",
                    "target_entropy": "auto",
                    "train_freq": 4,
                    "gradient_steps": 4,
                },
                "network": {"policy_layers": [256, 256]},
            }
        }
        config.lstm.enabled = False

        kwargs = get_algorithm_kwargs(config)

        assert kwargs["learning_rate"] == 1e-4
        assert kwargs["buffer_size"] == 100_000
        assert kwargs["ent_coef"] == "auto"
        assert "target_entropy" not in kwargs  # "auto" is filtered out
        assert kwargs["policy_kwargs"]["net_arch"] == [256, 256]
