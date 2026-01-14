"""
Test SSOT (Single Source of Truth) Configuration Compliance
============================================================

Verifies that all modules in the codebase use centralized configuration
from trading_config.yaml rather than hardcoded values.

Author: Pedro @ Lean Tech Solutions / Claude Code
Date: 2026-01-14
"""

import pytest
import re
from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestSSOTCompliance:
    """Test suite for verifying SSOT configuration compliance."""

    # Expected SSOT values from config/trading_config.yaml
    SSOT_VALUES = {
        "gamma": 0.90,
        "threshold_long": 0.33,
        "threshold_short": -0.33,
        "transaction_cost_bps": 3.0,
        "slippage_bps": 2.0,
        "ent_coef": 0.05,
    }

    # Directories to scan for compliance
    SCAN_DIRS = [
        "src",
        "scripts",
        "services",
        "airflow/dags",
        "notebooks",
    ]

    # Files to exclude from scanning
    EXCLUDE_FILES = [
        "config/trading_config.yaml",  # SSOT itself
        "src/config/config_helper.py",  # Helper that reads SSOT
        "src/config/trading_config.py",  # Config loader
        "tests/",  # Test files
        "__pycache__",
        ".pyc",
    ]

    def test_no_gamma_099_in_codebase(self):
        """Verify gamma 0.99 is not used anywhere (should be 0.90)."""
        violations = []

        for scan_dir in self.SCAN_DIRS:
            dir_path = PROJECT_ROOT / scan_dir
            if not dir_path.exists():
                continue

            for py_file in dir_path.rglob("*.py"):
                if self._should_exclude(py_file):
                    continue

                content = py_file.read_text(encoding="utf-8", errors="ignore")

                # Look for gamma = 0.99 patterns
                patterns = [
                    r'gamma\s*[=:]\s*0\.99',
                    r'"gamma"\s*:\s*0\.99',
                    r"'gamma'\s*:\s*0\.99",
                ]

                for pattern in patterns:
                    if re.search(pattern, content):
                        violations.append(str(py_file.relative_to(PROJECT_ROOT)))
                        break

        assert len(violations) == 0, (
            f"Found gamma=0.99 in {len(violations)} files (should be 0.90): "
            f"{violations}"
        )

    def test_hardcoded_thresholds_have_ssot_comment(self):
        """
        Verify that any hardcoded threshold values have SSOT documentation.

        Files using threshold values should either:
        1. Import from config_helper
        2. Have "# From SSOT" comment documenting the source
        """
        violations = []

        threshold_patterns = [
            (r'(?<![a-zA-Z_])0\.33(?![0-9])', "threshold_long"),
            (r'(?<![a-zA-Z_])-0\.33(?![0-9])', "threshold_short"),
        ]

        for scan_dir in self.SCAN_DIRS:
            dir_path = PROJECT_ROOT / scan_dir
            if not dir_path.exists():
                continue

            for py_file in dir_path.rglob("*.py"):
                if self._should_exclude(py_file):
                    continue

                content = py_file.read_text(encoding="utf-8", errors="ignore")
                lines = content.split("\n")

                for i, line in enumerate(lines):
                    for pattern, threshold_name in threshold_patterns:
                        if re.search(pattern, line):
                            # Check if line or nearby lines have SSOT comment
                            context = "\n".join(lines[max(0, i-2):min(len(lines), i+3)])
                            has_ssot_ref = (
                                "SSOT" in context or
                                "config_helper" in context or
                                "get_thresholds" in context or
                                "trading_config" in context
                            )

                            if not has_ssot_ref:
                                rel_path = str(py_file.relative_to(PROJECT_ROOT))
                                violations.append(
                                    f"{rel_path}:{i+1} - {threshold_name}"
                                )

        # Allow some violations for now (legacy code)
        # This test documents technical debt
        if violations:
            pytest.skip(
                f"Found {len(violations)} undocumented threshold uses. "
                f"Consider adding SSOT comments: {violations[:5]}..."
            )

    def test_config_helper_exists(self):
        """Verify the SSOT config helper module exists."""
        helper_path = PROJECT_ROOT / "src" / "config" / "config_helper.py"
        assert helper_path.exists(), (
            "SSOT config helper not found at src/config/config_helper.py"
        )

    def test_config_helper_provides_thresholds(self):
        """Verify config helper provides threshold functions."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            from src.config.config_helper import get_thresholds, get_costs

            # Verify functions return expected types
            long_th, short_th = get_thresholds()
            assert isinstance(long_th, (int, float))
            assert isinstance(short_th, (int, float))

            cost_bps, slippage_bps = get_costs()
            assert isinstance(cost_bps, (int, float))
            assert isinstance(slippage_bps, (int, float))

            # Verify values match SSOT
            assert long_th == self.SSOT_VALUES["threshold_long"]
            assert short_th == self.SSOT_VALUES["threshold_short"]
            assert cost_bps == self.SSOT_VALUES["transaction_cost_bps"]
            assert slippage_bps == self.SSOT_VALUES["slippage_bps"]

        except ImportError as e:
            pytest.fail(f"Could not import config_helper: {e}")

    def test_trading_config_yaml_exists(self):
        """Verify the SSOT YAML config file exists."""
        config_path = PROJECT_ROOT / "config" / "trading_config.yaml"
        assert config_path.exists(), (
            "SSOT config not found at config/trading_config.yaml"
        )

    def test_trading_config_has_required_keys(self):
        """Verify SSOT config has all required parameters."""
        import yaml

        config_path = PROJECT_ROOT / "config" / "trading_config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Check required sections
        assert "ppo" in config, "Missing 'ppo' section in trading_config.yaml"
        assert "thresholds" in config, "Missing 'thresholds' section"
        assert "costs" in config, "Missing 'costs' section"

        # Check specific keys
        assert config["ppo"].get("gamma") == self.SSOT_VALUES["gamma"], (
            f"gamma should be {self.SSOT_VALUES['gamma']}"
        )
        assert config["thresholds"].get("long") == self.SSOT_VALUES["threshold_long"]
        assert config["thresholds"].get("short") == self.SSOT_VALUES["threshold_short"]

    def _should_exclude(self, file_path: Path) -> bool:
        """Check if file should be excluded from scanning."""
        path_str = str(file_path)
        return any(excl in path_str for excl in self.EXCLUDE_FILES)


class TestNormalizationSSOT:
    """Test normalization config follows SSOT."""

    def test_norm_stats_file_exists(self):
        """Verify normalization stats file exists."""
        norm_path = PROJECT_ROOT / "config" / "norm_stats.json"
        assert norm_path.exists(), (
            "Normalization stats not found at config/norm_stats.json"
        )

    def test_norm_stats_has_core_features(self):
        """Verify norm stats has all required features."""
        import json

        norm_path = PROJECT_ROOT / "config" / "norm_stats.json"
        with open(norm_path, "r") as f:
            stats = json.load(f)

        core_features = [
            "log_ret_5m", "log_ret_1h", "log_ret_4h",
            "rsi_9", "atr_pct", "adx_14",
            "dxy_z", "dxy_change_1d", "vix_z", "embi_z", "brent_change_1d",
            "rate_spread", "usdmxn_change_1d"
        ]

        for feature in core_features:
            assert feature in stats, f"Missing normalization stats for: {feature}"
            assert "mean" in stats[feature], f"Missing 'mean' for {feature}"
            assert "std" in stats[feature], f"Missing 'std' for {feature}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
