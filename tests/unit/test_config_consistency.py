"""
Test Config Consistency
=======================

Ensures all services use the same thresholds and costs.
This test prevents bugs where training uses different thresholds than inference.

Run with: pytest tests/unit/test_config_consistency.py -v
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestConfigConsistency:
    """Verify all services use consistent configuration."""

    # Expected values
    EXPECTED_THRESHOLD_LONG = 0.33
    EXPECTED_THRESHOLD_SHORT = -0.33
    EXPECTED_TRANSACTION_COST_BPS = 75.0
    EXPECTED_SLIPPAGE_BPS = 15.0
    EXPECTED_GAMMA = 0.90
    EXPECTED_ENT_COEF = 0.05

    def test_trading_env_thresholds(self):
        """Verify trading_env.py uses correct thresholds."""
        from src.training.environments.trading_env import TradingEnvConfig

        config = TradingEnvConfig()
        assert config.threshold_long == self.EXPECTED_THRESHOLD_LONG, \
            f"trading_env threshold_long: {config.threshold_long} != {self.EXPECTED_THRESHOLD_LONG}"
        assert config.threshold_short == self.EXPECTED_THRESHOLD_SHORT, \
            f"trading_env threshold_short: {config.threshold_short} != {self.EXPECTED_THRESHOLD_SHORT}"

    def test_trading_env_costs(self):
        """Verify trading_env.py uses correct transaction costs."""
        from src.training.environments.trading_env import TradingEnvConfig

        config = TradingEnvConfig()
        assert config.transaction_cost_bps == self.EXPECTED_TRANSACTION_COST_BPS, \
            f"trading_env transaction_cost: {config.transaction_cost_bps} != {self.EXPECTED_TRANSACTION_COST_BPS}"
        assert config.slippage_bps == self.EXPECTED_SLIPPAGE_BPS, \
            f"trading_env slippage: {config.slippage_bps} != {self.EXPECTED_SLIPPAGE_BPS}"

    def test_reward_calculator_loss_penalty(self):
        """Verify reward_calculator uses correct loss penalty."""
        from src.training.reward_calculator import RewardConfig

        config = RewardConfig()
        assert config.loss_penalty_multiplier == 2.0, \
            f"reward loss_penalty: {config.loss_penalty_multiplier} != 2.0"

    def test_reward_calculator_hold_bonus_disabled(self):
        """Verify hold bonus is disabled."""
        from src.training.reward_calculator import RewardConfig

        config = RewardConfig()
        assert config.hold_bonus_per_bar == 0.0, \
            f"reward hold_bonus should be 0.0, got {config.hold_bonus_per_bar}"

    def test_no_hardcoded_095_thresholds(self):
        """Verify no files still have 0.95/-0.95 hardcoded thresholds."""
        import re

        files_to_check = [
            PROJECT_ROOT / "services" / "inference_api" / "core" / "inference_engine.py",
            PROJECT_ROOT / "services" / "inference_api" / "core" / "trade_simulator.py",
            PROJECT_ROOT / "airflow" / "dags" / "services" / "backtest_factory.py",
        ]

        for filepath in files_to_check:
            if not filepath.exists():
                continue

            content = filepath.read_text()

            # Check for dangerous 0.95 patterns that aren't in comments
            lines = content.split("\n")
            for i, line in enumerate(lines, 1):
                # Skip comments
                if line.strip().startswith("#"):
                    continue

                # Check for 0.95 threshold (legacy bug)
                if re.search(r'["\']?long["\']?\s*[:=]\s*0\.95', line):
                    pytest.fail(
                        f"Found hardcoded 0.95 long threshold in {filepath.name}:{i}\n"
                        f"Line: {line.strip()}"
                    )
                if re.search(r'["\']?short["\']?\s*[:=]\s*-0\.95', line):
                    pytest.fail(
                        f"Found hardcoded -0.95 short threshold in {filepath.name}:{i}\n"
                        f"Line: {line.strip()}"
                    )


class TestCircuitBreaker:
    """Test circuit breaker configuration."""

    def test_circuit_breaker_config_exists(self):
        """Verify circuit breaker is configured in trading_env."""
        from src.training.environments.trading_env import TradingEnvConfig

        config = TradingEnvConfig()
        assert hasattr(config, 'max_consecutive_losses'), \
            "TradingEnvConfig missing max_consecutive_losses"
        assert hasattr(config, 'cooldown_bars_after_losses'), \
            "TradingEnvConfig missing cooldown_bars_after_losses"

    def test_circuit_breaker_values(self):
        """Verify circuit breaker has sensible defaults."""
        from src.training.environments.trading_env import TradingEnvConfig

        config = TradingEnvConfig()
        assert config.max_consecutive_losses == 5, \
            f"max_consecutive_losses: {config.max_consecutive_losses} != 5"
        assert config.cooldown_bars_after_losses == 12, \
            f"cooldown_bars: {config.cooldown_bars_after_losses} != 12"


class TestVolatilityFilter:
    """Test volatility filter configuration."""

    def test_volatility_filter_config_exists(self):
        """Verify volatility filter is configured."""
        from src.training.environments.trading_env import TradingEnvConfig

        config = TradingEnvConfig()
        assert hasattr(config, 'enable_volatility_filter'), \
            "TradingEnvConfig missing enable_volatility_filter"
        assert hasattr(config, 'max_atr_multiplier'), \
            "TradingEnvConfig missing max_atr_multiplier"

    def test_volatility_filter_enabled(self):
        """Verify volatility filter is enabled by default."""
        from src.training.environments.trading_env import TradingEnvConfig

        config = TradingEnvConfig()
        assert config.enable_volatility_filter is True
        assert config.max_atr_multiplier == 2.0


class TestRewardIntegration:
    """Test reward calculator is integrated with TradingEnvironment."""

    def test_reward_adapter_is_default(self):
        """Verify RewardStrategyAdapter is used by default."""
        from src.training.environments.trading_env import (
            TradingEnvironment, TradingEnvConfig, RewardStrategyAdapter
        )
        import pandas as pd
        import numpy as np

        # Create minimal test DataFrame
        n_bars = 1500
        df = pd.DataFrame({
            "log_ret_5m": np.random.randn(n_bars) * 0.001,
            "log_ret_1h": np.random.randn(n_bars) * 0.002,
            "log_ret_4h": np.random.randn(n_bars) * 0.003,
            "rsi_9": np.random.rand(n_bars) * 100,
            "atr_pct": np.random.rand(n_bars) * 0.02,
            "adx_14": np.random.rand(n_bars) * 100,
            "dxy_z": np.random.randn(n_bars),
            "dxy_change_1d": np.random.randn(n_bars) * 0.01,
            "vix_z": np.random.randn(n_bars),
            "embi_z": np.random.randn(n_bars),
            "brent_change_1d": np.random.randn(n_bars) * 0.02,
            "rate_spread": np.random.randn(n_bars),
            "usdmxn_change_1d": np.random.randn(n_bars) * 0.01,
        })

        # Create environment with defaults
        config = TradingEnvConfig()
        norm_stats = {f: {"mean": 0, "std": 1} for f in config.core_features}

        env = TradingEnvironment(df=df, norm_stats=norm_stats, config=config)

        # Verify RewardStrategyAdapter is being used
        assert isinstance(env.reward_strategy, RewardStrategyAdapter), \
            f"Expected RewardStrategyAdapter, got {type(env.reward_strategy).__name__}"

    def test_reward_adapter_has_intratrade_penalty(self):
        """Verify reward adapter uses intratrade DD penalty."""
        from src.training.reward_calculator import RewardConfig

        config = RewardConfig()
        assert config.intratrade_dd_penalty == 0.5, \
            f"intratrade_dd_penalty should be 0.5, got {config.intratrade_dd_penalty}"
        assert config.max_intratrade_dd == 0.02, \
            f"max_intratrade_dd should be 0.02, got {config.max_intratrade_dd}"

    def test_reward_adapter_has_time_decay(self):
        """Verify reward adapter uses time decay."""
        from src.training.reward_calculator import RewardConfig

        config = RewardConfig()
        assert config.time_decay_start_bars == 24, \
            f"time_decay_start_bars should be 24, got {config.time_decay_start_bars}"
        assert config.time_decay_per_bar == 0.0001, \
            f"time_decay_per_bar should be 0.0001, got {config.time_decay_per_bar}"


class TestInferenceFilesDirectly:
    """Test inference files directly by reading source code (no import required)."""

    EXPECTED_THRESHOLD_LONG = 0.33
    EXPECTED_THRESHOLD_SHORT = -0.33

    def test_inference_engine_thresholds_in_source(self):
        """Verify inference_engine.py has correct thresholds in source."""
        inference_path = PROJECT_ROOT / "services" / "inference_api" / "core" / "inference_engine.py"

        if not inference_path.exists():
            pytest.skip(f"inference_engine.py not found at {inference_path}")

        content = inference_path.read_text()

        # Check thresholds are 0.33/-0.33 (not 0.95/-0.95)
        assert '{"long": 0.33, "short": -0.33}' in content, \
            "Thresholds 0.33/-0.33 not found in inference_engine.py"

    def test_trade_simulator_thresholds_in_source(self):
        """Verify trade_simulator.py has correct thresholds in source."""
        simulator_path = PROJECT_ROOT / "services" / "inference_api" / "core" / "trade_simulator.py"

        if not simulator_path.exists():
            pytest.skip(f"trade_simulator.py not found at {simulator_path}")

        content = simulator_path.read_text()

        # Check thresholds are 0.33/-0.33
        assert '"long_entry": 0.33, "short_entry": -0.33' in content, \
            "Thresholds 0.33/-0.33 not found in trade_simulator.py"

    def test_backtest_factory_defaults_in_source(self):
        """Verify backtest_factory.py has correct defaults."""
        factory_path = PROJECT_ROOT / "airflow" / "dags" / "services" / "backtest_factory.py"

        if not factory_path.exists():
            pytest.skip(f"backtest_factory.py not found at {factory_path}")

        content = factory_path.read_text()

        # Check BacktestConfigBuilder defaults
        assert "_long_entry: float = 0.33" in content, \
            "BacktestConfigBuilder long_entry default not 0.33"
        assert "_short_entry: float = -0.33" in content, \
            "BacktestConfigBuilder short_entry default not -0.33"
        assert "_transaction_cost_bps: float = 75.0" in content, \
            "BacktestConfigBuilder transaction_cost_bps default not 75.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
