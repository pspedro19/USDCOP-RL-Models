"""
SSOT (Single Source of Truth) Consistency Tests
================================================
Validates that all configuration values across the codebase are consistent
with the canonical config.yaml.

This test suite prevents configuration drift by:
1. Verifying TradingConfig loads correctly from YAML
2. Detecting hardcoded values that conflict with SSOT
3. Validating all critical modules use consistent parameters

Run with: pytest tests/unit/test_ssot_consistency.py -v

Author: Trading Team
Date: 2026-01-13
"""

import re
import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestTradingConfigLoader:
    """Test the TradingConfig SSOT loader."""

    def test_load_current_config(self):
        """Verify current config loads successfully."""
        from src.config.trading_config import load_trading_config, reset_trading_config

        reset_trading_config()  # Clean state
        config = load_trading_config("current")

        assert config.version == "current"
        assert config.ppo.gamma == 0.90
        assert config.ppo.ent_coef == 0.05
        assert config.thresholds.long == 0.33
        assert config.thresholds.short == -0.33
        assert config.costs.transaction_cost_bps == 75.0
        assert config.costs.slippage_bps == 15.0

    def test_config_immutability(self):
        """Verify config is frozen (immutable)."""
        from src.config.trading_config import load_trading_config, reset_trading_config

        reset_trading_config()
        config = load_trading_config("current")

        with pytest.raises(Exception):  # FrozenInstanceError
            config.ppo.gamma = 0.99  # Should fail

    def test_singleton_behavior(self):
        """Verify singleton returns same instance."""
        from src.config.trading_config import (
            load_trading_config,
            get_trading_config,
            reset_trading_config,
        )

        reset_trading_config()
        config1 = load_trading_config("current")
        config2 = get_trading_config()

        assert config1 is config2

    def test_version_mismatch_error(self):
        """Verify loading different version after init raises error."""
        from src.config.trading_config import (
            load_trading_config,
            reset_trading_config,
            ConfigVersionMismatchError,
        )

        reset_trading_config()
        load_trading_config("current")

        with pytest.raises(ConfigVersionMismatchError):
            load_trading_config("legacy")  # Should fail - already loaded current

    def test_helper_functions(self):
        """Verify backward compatibility helper functions work."""
        from src.config.trading_config import (
            load_trading_config,
            reset_trading_config,
            get_ppo_hyperparameters,
            get_env_config,
            get_reward_config,
        )

        reset_trading_config()
        load_trading_config("current")

        ppo_params = get_ppo_hyperparameters()
        assert ppo_params["gamma"] == 0.90
        assert ppo_params["ent_coef"] == 0.05

        env_config = get_env_config()
        assert env_config["threshold_long"] == 0.33
        assert env_config["threshold_short"] == -0.33
        assert env_config["transaction_cost_bps"] == 75.0

        reward_config = get_reward_config()
        assert reward_config["loss_penalty_multiplier"] == 2.0
        assert reward_config["hold_bonus_per_bar"] == 0.0


class TestNoHardcodedConflicts:
    """Detect hardcoded values that conflict with SSOT."""

    # Forbidden patterns - these values should come from config, not hardcoded
    FORBIDDEN_PATTERNS = [
        # Old gamma/ent_coef values
        (r"gamma\s*[=:]\s*0\.99\b", "gamma 0.99 (should be 0.90)"),
        (r"ent_coef\s*[=:]\s*0\.01\b", "ent_coef 0.01 (should be 0.05)"),
        # Old threshold values (excluding threshold_exit which is intentionally lower)
        (r"threshold(?!_exit).*0\.95\b", "threshold 0.95 (should be 0.33)"),
        (r"threshold(?!_exit).*-0\.95\b", "threshold -0.95 (should be -0.33)"),
        (r"threshold_(?:long|short).*0\.10\b", "threshold_long/short 0.10 (should be 0.33)"),
        (r"threshold_(?:long|short).*-0\.10\b", "threshold_long/short -0.10 (should be -0.33)"),
        # Old cost values
        (r"transaction_cost_bps\s*[=:]\s*25\.0", "cost 25.0 (should be 75.0)"),
        (r"transaction_cost_bps\s*[=:]\s*5\.0", "cost 5.0 (should be 75.0)"),
        (r"slippage_bps\s*[=:]\s*2\.0\b", "slippage 2.0 (should be 15.0)"),
    ]

    # Directories to check
    CHECK_DIRS = [
        PROJECT_ROOT / "src" / "training",
        PROJECT_ROOT / "src" / "config",
        PROJECT_ROOT / "services" / "inference_api",
        PROJECT_ROOT / "airflow" / "dags",
    ]

    # Files to skip (legacy, tests, etc.)
    SKIP_PATTERNS = [
        r".*test.*\.py$",
        r".*__pycache__.*",
        r".*\.pyc$",
        r".*legacy_config\.yaml$",  # Legacy config is OK
        r".*archive.*",
    ]

    def _should_skip(self, filepath: Path) -> bool:
        """Check if file should be skipped."""
        filepath_str = str(filepath)
        for pattern in self.SKIP_PATTERNS:
            if re.search(pattern, filepath_str, re.IGNORECASE):
                return True
        return False

    def _check_file(self, filepath: Path) -> list:
        """Check file for forbidden patterns."""
        violations = []

        try:
            content = filepath.read_text(encoding="utf-8")
        except Exception:
            return violations

        lines = content.split("\n")
        for line_num, line in enumerate(lines, 1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("//"):
                continue
            # Skip strings that look like documentation
            if "from" in line.lower() and "->" in line:
                continue
            # Skip lines with migration notes
            if "MIGRATION:" in line or "migration:" in line:
                continue

            for pattern, description in self.FORBIDDEN_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    violations.append(
                        f"{filepath.relative_to(PROJECT_ROOT)}:{line_num} - {description}\n"
                        f"  Line: {line.strip()[:100]}"
                    )

        return violations

    def test_no_old_gamma_values(self):
        """Check no files have old gamma=0.99 hardcoded."""
        violations = []

        for check_dir in self.CHECK_DIRS:
            if not check_dir.exists():
                continue

            for py_file in check_dir.rglob("*.py"):
                if self._should_skip(py_file):
                    continue

                file_violations = self._check_file(py_file)
                # Filter only gamma violations
                gamma_violations = [v for v in file_violations if "gamma" in v]
                violations.extend(gamma_violations)

        assert not violations, (
            f"Found old gamma values (should be 0.90):\n" + "\n".join(violations)
        )

    def test_no_old_entcoef_values(self):
        """Check no files have old ent_coef=0.01 hardcoded."""
        violations = []

        for check_dir in self.CHECK_DIRS:
            if not check_dir.exists():
                continue

            for py_file in check_dir.rglob("*.py"):
                if self._should_skip(py_file):
                    continue

                file_violations = self._check_file(py_file)
                # Filter only ent_coef violations
                entcoef_violations = [v for v in file_violations if "ent_coef" in v]
                violations.extend(entcoef_violations)

        assert not violations, (
            f"Found old ent_coef values (should be 0.05):\n" + "\n".join(violations)
        )

    def test_no_old_threshold_values(self):
        """Check no files have old thresholds hardcoded."""
        violations = []

        for check_dir in self.CHECK_DIRS:
            if not check_dir.exists():
                continue

            for py_file in check_dir.rglob("*.py"):
                if self._should_skip(py_file):
                    continue

                file_violations = self._check_file(py_file)
                # Filter only threshold violations
                threshold_violations = [v for v in file_violations if "threshold" in v.lower()]
                violations.extend(threshold_violations)

        assert not violations, (
            f"Found old threshold values (should be ±0.33):\n" + "\n".join(violations)
        )


class TestCriticalFilesConsistency:
    """Verify critical files have correct SSOT values."""

    EXPECTED_GAMMA = 0.90
    EXPECTED_ENT_COEF = 0.05
    EXPECTED_THRESHOLD_LONG = 0.33
    EXPECTED_THRESHOLD_SHORT = -0.33
    EXPECTED_TRANSACTION_COST_BPS = 75.0
    EXPECTED_SLIPPAGE_BPS = 15.0

    def test_trading_env_config_defaults(self):
        """Verify TradingEnvConfig has correct defaults."""
        from src.training.environments.trading_env import TradingEnvConfig

        config = TradingEnvConfig()
        assert config.threshold_long == self.EXPECTED_THRESHOLD_LONG
        assert config.threshold_short == self.EXPECTED_THRESHOLD_SHORT
        assert config.transaction_cost_bps == self.EXPECTED_TRANSACTION_COST_BPS
        assert config.slippage_bps == self.EXPECTED_SLIPPAGE_BPS

    def test_reward_config_defaults(self):
        """Verify RewardConfig has correct defaults."""
        from src.training.reward_calculator import RewardConfig

        config = RewardConfig()
        assert config.loss_penalty_multiplier == 2.0
        assert config.hold_bonus_per_bar == 0.0  # Disabled

    def test_ppo_config_defaults(self):
        """Verify PPOConfig in ppo_trainer.py has correct defaults."""
        from src.training.trainers.ppo_trainer import PPOConfig

        config = PPOConfig()
        assert config.gamma == self.EXPECTED_GAMMA, \
            f"PPOConfig gamma: {config.gamma} != {self.EXPECTED_GAMMA}"
        assert config.ent_coef == self.EXPECTED_ENT_COEF, \
            f"PPOConfig ent_coef: {config.ent_coef} != {self.EXPECTED_ENT_COEF}"

    def test_inference_api_config(self):
        """Verify inference API config has correct defaults."""
        inference_config_path = PROJECT_ROOT / "services" / "inference_api" / "config.py"

        if not inference_config_path.exists():
            pytest.skip("inference_api/config.py not found")

        content = inference_config_path.read_text()

        # Check thresholds
        assert 'threshold_long: float = 0.33' in content, \
            "inference_api config threshold_long should be 0.33"
        assert 'threshold_short: float = -0.33' in content, \
            "inference_api config threshold_short should be -0.33"
        assert 'transaction_cost_bps: float = 75.0' in content, \
            "inference_api config transaction_cost_bps should be 75.0"

    def test_l3_dag_config(self):
        """Verify L3 training DAG has correct config."""
        dag_path = PROJECT_ROOT / "airflow" / "dags" / "l3_model_training.py"

        if not dag_path.exists():
            pytest.skip("l3_model_training.py not found")

        content = dag_path.read_text(encoding="utf-8")

        # Check gamma and ent_coef
        assert '"gamma": 0.90' in content or "'gamma': 0.90" in content, \
            "L3 DAG gamma should be 0.90"
        assert '"ent_coef": 0.05' in content or "'ent_coef': 0.05" in content, \
            "L3 DAG ent_coef should be 0.05"


class TestConfigValidation:
    """Test configuration validation catches invalid values."""

    def test_gamma_validation(self):
        """Verify gamma must be in (0, 1)."""
        from src.config.trading_config import PPOHyperparameters, ConfigValidationError

        with pytest.raises(ConfigValidationError):
            PPOHyperparameters(
                learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
                gamma=1.5,  # Invalid
                gae_lambda=0.95, clip_range=0.2, ent_coef=0.05,
                total_timesteps=500000
            )

    def test_threshold_validation(self):
        """Verify thresholds must have correct signs."""
        from src.config.trading_config import ThresholdConfig, ConfigValidationError

        with pytest.raises(ConfigValidationError):
            ThresholdConfig(long=-0.33, short=0.33)  # Signs reversed

    def test_cost_warning_for_low_values(self, caplog):
        """Verify low transaction costs trigger warning (logged)."""
        import logging
        from src.config.trading_config import CostConfig

        with caplog.at_level(logging.WARNING):
            config = CostConfig(
                transaction_cost_bps=25.0,  # Too low for USDCOP - will log warning
                slippage_bps=2.0
            )

        # Config should still be created successfully
        assert config.transaction_cost_bps == 25.0

        # Verify warning was logged
        assert any("very low for USDCOP" in record.message for record in caplog.records)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
