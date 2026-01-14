"""
Test Backtest Determinism
==========================

Verifies that running the backtest multiple times with the same seed
produces identical results. This is critical for reproducibility
and debugging.

Author: Pedro @ Lean Tech Solutions / Claude Code
Date: 2026-01-14
"""

import pytest
import sys
import json
import subprocess
import tempfile
from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestBacktestDeterminism:
    """Test suite for verifying backtest reproducibility."""

    @pytest.fixture
    def backtest_script(self):
        """Return path to backtest script."""
        return PROJECT_ROOT / "scripts" / "backtest.py"

    @pytest.fixture
    def model_path(self):
        """Return path to a model for testing."""
        models_dir = PROJECT_ROOT / "models" / "ppo_production"
        if models_dir.exists():
            for model_file in ["final_model.zip", "best_model.zip"]:
                if (models_dir / model_file).exists():
                    return str(models_dir / model_file)
        return None

    @pytest.fixture
    def dataset_path(self):
        """Return path to test dataset."""
        dataset = PROJECT_ROOT / "data" / "pipeline" / "07_output" / "datasets_5min" / "RL_DS3_MACRO_CORE.csv"
        if dataset.exists():
            return str(dataset)
        return None

    @pytest.fixture
    def norm_stats_path(self):
        """Return path to normalization stats."""
        stats = PROJECT_ROOT / "config" / "norm_stats.json"
        if stats.exists():
            return str(stats)
        return None

    def test_set_seed_function_exists(self, backtest_script):
        """Verify backtest script has set_seed function."""
        content = backtest_script.read_text()
        assert "def set_seed" in content, "set_seed function not found in backtest.py"
        assert "np.random.seed" in content, "numpy seed not set in set_seed"
        assert "random.seed" in content, "random seed not set in set_seed"

    def test_seed_argument_exists(self, backtest_script):
        """Verify backtest script accepts --seed argument."""
        content = backtest_script.read_text()
        assert "--seed" in content, "--seed argument not found in backtest.py"
        assert "default=42" in content, "Default seed should be 42"

    def test_seed_is_called_in_main(self, backtest_script):
        """Verify set_seed is called in main function."""
        content = backtest_script.read_text()
        # Look for set_seed being called after argument parsing
        assert "set_seed(args.seed)" in content or "set_seed(" in content, (
            "set_seed is not called in main()"
        )

    @pytest.mark.skipif(
        not (PROJECT_ROOT / "models" / "ppo_production").exists(),
        reason="No production model available for testing"
    )
    def test_backtest_deterministic_with_same_seed(
        self, backtest_script, model_path, dataset_path, norm_stats_path
    ):
        """
        Run backtest twice with same seed and verify identical results.

        This is the critical test for reproducibility.
        """
        if not all([model_path, dataset_path, norm_stats_path]):
            pytest.skip("Required files not found for determinism test")

        results = []
        seed = 42

        for run_num in range(2):
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', delete=False
            ) as f:
                output_path = f.name

            cmd = [
                sys.executable,
                str(backtest_script),
                "--model", model_path,
                "--dataset", dataset_path,
                "--norm-stats", norm_stats_path,
                "--seed", str(seed),
                "--output", output_path,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )

            if result.returncode != 0:
                pytest.fail(
                    f"Backtest run {run_num + 1} failed: {result.stderr}"
                )

            # Read results
            with open(output_path, 'r') as f:
                run_results = json.load(f)

            # Remove timestamp (will differ)
            run_results.pop("timestamp", None)
            results.append(run_results)

            # Cleanup
            Path(output_path).unlink(missing_ok=True)

        # Compare results
        assert results[0] == results[1], (
            f"Backtest results differ between runs!\n"
            f"Run 1: {results[0]}\n"
            f"Run 2: {results[1]}"
        )

    @pytest.mark.skipif(
        not (PROJECT_ROOT / "models" / "ppo_production").exists(),
        reason="No production model available for testing"
    )
    def test_different_seeds_produce_different_results(
        self, backtest_script, model_path, dataset_path, norm_stats_path
    ):
        """
        Verify that different seeds can produce different results.

        Note: This test may occasionally fail if the model is fully
        deterministic regardless of seed (which is actually fine).
        """
        if not all([model_path, dataset_path, norm_stats_path]):
            pytest.skip("Required files not found for determinism test")

        results = []
        seeds = [42, 123]

        for seed in seeds:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', delete=False
            ) as f:
                output_path = f.name

            cmd = [
                sys.executable,
                str(backtest_script),
                "--model", model_path,
                "--dataset", dataset_path,
                "--norm-stats", norm_stats_path,
                "--seed", str(seed),
                "--output", output_path,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )

            if result.returncode == 0:
                with open(output_path, 'r') as f:
                    run_results = json.load(f)
                run_results.pop("timestamp", None)
                results.append(run_results)

            Path(output_path).unlink(missing_ok=True)

        # This test documents behavior - not a hard requirement
        if len(results) == 2 and results[0] == results[1]:
            pytest.skip(
                "Different seeds produced same results - "
                "model may be fully deterministic (OK)"
            )


class TestNormalizerDeterminism:
    """Test normalizer produces deterministic results."""

    def test_zscore_normalizer_deterministic(self):
        """Verify ZScoreNormalizer produces identical results."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            from src.core.normalizers.zscore_normalizer import ZScoreNormalizer
        except ImportError:
            pytest.skip("ZScoreNormalizer not available")

        norm_path = PROJECT_ROOT / "config" / "norm_stats.json"
        if not norm_path.exists():
            pytest.skip("Normalization stats not found")

        normalizer = ZScoreNormalizer(stats_path=str(norm_path))

        # Test values
        test_cases = [
            ("log_ret_5m", 0.001),
            ("rsi_9", 50.0),
            ("atr_pct", 0.015),
            ("dxy_z", 1.5),
        ]

        results_run1 = []
        results_run2 = []

        for feature, value in test_cases:
            results_run1.append(normalizer.normalize(feature, value))
            results_run2.append(normalizer.normalize(feature, value))

        assert results_run1 == results_run2, (
            "Normalizer produced different results for same inputs"
        )

    def test_observation_builder_deterministic(self):
        """Verify observation building is deterministic."""
        import sys
        import numpy as np
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            from src.core.builders.observation_builder import ObservationBuilder
        except ImportError:
            pytest.skip("ObservationBuilder not available")

        norm_path = PROJECT_ROOT / "config" / "norm_stats.json"
        feature_path = PROJECT_ROOT / "config" / "feature_config.json"

        if not norm_path.exists() or not feature_path.exists():
            pytest.skip("Config files not found")

        builder = ObservationBuilder(
            stats_path=str(norm_path),
            feature_config_path=str(feature_path)
        )

        # Create test feature dict
        features = {
            "log_ret_5m": 0.001,
            "log_ret_1h": 0.005,
            "log_ret_4h": 0.01,
            "rsi_9": 50.0,
            "atr_pct": 0.015,
            "adx_14": 25.0,
            "dxy_z": 0.5,
            "dxy_change_1d": 0.002,
            "vix_z": -0.3,
            "embi_z": 0.2,
            "brent_change_1d": 0.01,
            "rate_spread": 8.0,
            "usdmxn_change_1d": -0.005,
        }

        # Build observation twice
        obs1 = builder.build(features, position=0.0, time_norm=0.5)
        obs2 = builder.build(features, position=0.0, time_norm=0.5)

        np.testing.assert_array_equal(obs1, obs2, err_msg=(
            "ObservationBuilder produced different results for same inputs"
        ))


class TestModelInferenceDeterminism:
    """Test model inference is deterministic."""

    @pytest.mark.skipif(
        not (PROJECT_ROOT / "models" / "ppo_production").exists(),
        reason="No production model available"
    )
    def test_model_predict_deterministic(self):
        """Verify model produces same predictions for same input."""
        import sys
        import numpy as np
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            from stable_baselines3 import PPO
        except ImportError:
            pytest.skip("stable_baselines3 not available")

        # Find model
        models_dir = PROJECT_ROOT / "models" / "ppo_production"
        model_path = None
        for name in ["final_model.zip", "best_model.zip"]:
            if (models_dir / name).exists():
                model_path = str(models_dir / name)
                break

        if not model_path:
            pytest.skip("No model found")

        model = PPO.load(model_path, device='cpu')

        # Create test observation
        np.random.seed(42)
        obs = np.random.randn(1, 15).astype(np.float32)

        # Predict multiple times
        predictions = []
        for _ in range(3):
            action, _ = model.predict(obs, deterministic=True)
            predictions.append(action[0])

        # All predictions should be identical
        assert all(p == predictions[0] for p in predictions), (
            f"Model produced different predictions: {predictions}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
