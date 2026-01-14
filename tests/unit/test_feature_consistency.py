"""
Feature Consistency Tests
=========================
Tests to ensure feature store consistency between training and inference.

These tests validate:
1. ModelContract registration and validation
2. BuilderFactory explicit registration (no string matching)
3. Fail-fast behavior on missing norm_stats
4. Hash verification for model integrity
5. Consistency between training and inference norm_stats

CRITICAL: These tests are designed to catch deployment errors
that would cause wrong predictions.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "services" / "inference_api"))


class TestModelContract:
    """Tests for ModelContract and ModelRegistry"""

    def test_model_contract_immutable(self):
        """ModelContract should be immutable (frozen dataclass)"""
        from services.inference_api.contracts.model_contract import (
            ModelContract,
            BuilderType,
        )

        contract = ModelContract(
            model_id="test_model",
            version="1.0.0",
            builder_type=BuilderType.CURRENT_15DIM,
            observation_dim=15,
            norm_stats_path="config/test.json",
            model_path="models/test.zip",
        )

        # Should not be able to modify
        with pytest.raises(Exception):  # FrozenInstanceError
            contract.model_id = "modified"

    def test_model_contract_dimension_validation(self):
        """ModelContract should validate builder_type matches observation_dim"""
        from services.inference_api.contracts.model_contract import (
            ModelContract,
            BuilderType,
        )

        # Valid: current builder with 15 dims
        contract = ModelContract(
            model_id="valid_current",
            version="1.0.0",
            builder_type=BuilderType.CURRENT_15DIM,
            observation_dim=15,
            norm_stats_path="config/test.json",
            model_path="models/test.zip",
        )
        assert contract.observation_dim == 15

        # Invalid: current builder with 32 dims should raise
        with pytest.raises(ValueError, match="Observation dim mismatch"):
            ModelContract(
                model_id="invalid_current",
                version="1.0.0",
                builder_type=BuilderType.CURRENT_15DIM,
                observation_dim=32,  # Wrong!
                norm_stats_path="config/test.json",
                model_path="models/test.zip",
            )

    def test_model_registry_explicit_registration(self):
        """ModelRegistry should require explicit registration"""
        from services.inference_api.contracts.model_contract import (
            ModelRegistry,
            BuilderNotRegisteredError,
        )

        # Registered models should work
        contract = ModelRegistry.get("ppo_primary")
        assert contract.model_id == "ppo_primary"
        assert contract.observation_dim == 15

        # Unregistered models should fail explicitly
        with pytest.raises(BuilderNotRegisteredError, match="not registered"):
            ModelRegistry.get("nonexistent_model")

    def test_builder_type_enum_explicit(self):
        """BuilderType should be an explicit enum, not string matching"""
        from services.inference_api.contracts.model_contract import BuilderType

        # Should be explicit enum values
        assert BuilderType.CURRENT_15DIM.value == "current_15dim"

        # Should not be able to create arbitrary values
        with pytest.raises(ValueError):
            BuilderType("arbitrary_string")


class TestBuilderFactory:
    """Tests for BuilderFactory explicit registration"""

    def test_factory_uses_registry_not_string_matching(self):
        """BuilderFactory should use ModelRegistry, NOT string matching"""
        from services.inference_api.core.builder_factory import BuilderFactory
        from services.inference_api.contracts.model_contract import BuilderType

        # Clear cache
        BuilderFactory.clear_cache()

        # Get by model_id - should use registry
        # This test would fail if string matching like "v1" in model_id was used
        builder = BuilderFactory.get_builder("ppo_primary")
        assert builder.OBSERVATION_DIM == 15

    def test_factory_get_by_type_explicit(self):
        """BuilderFactory.get_builder_by_type should work with explicit enum"""
        from services.inference_api.core.builder_factory import BuilderFactory
        from services.inference_api.contracts.model_contract import BuilderType

        BuilderFactory.clear_cache()

        builder_current = BuilderFactory.get_builder_by_type(BuilderType.CURRENT_15DIM)
        assert builder_current.OBSERVATION_DIM == 15

    def test_factory_rejects_unregistered_builder_type(self):
        """BuilderFactory should reject unregistered builder types"""
        from services.inference_api.core.builder_factory import BuilderFactory
        from services.inference_api.contracts.model_contract import (
            BuilderNotRegisteredError,
        )

        # Create a mock BuilderType that's not registered
        with pytest.raises(BuilderNotRegisteredError):
            BuilderFactory.get_builder("definitely_not_a_real_model")


class TestObservationBuilderFailFast:
    """Tests for fail-fast behavior on missing norm_stats"""

    def test_observation_builder_fails_without_norm_stats(self):
        """ObservationBuilder should FAIL if norm_stats not found"""
        from services.inference_api.core.observation_builder import (
            ObservationBuilder,
            NormStatsNotFoundError,
        )

        # Try to create builder with non-existent path
        fake_path = Path("/nonexistent/path/norm_stats.json")

        with pytest.raises(NormStatsNotFoundError, match="CRITICAL"):
            ObservationBuilder(norm_stats_path=fake_path)

    def test_observation_builder_no_default_stats(self):
        """ObservationBuilder should NOT have hardcoded default stats"""
        from services.inference_api.core.observation_builder import ObservationBuilder
        import inspect

        # Check that _default_norm_stats method doesn't exist
        assert not hasattr(ObservationBuilder, '_default_norm_stats'), \
            "ObservationBuilder should NOT have _default_norm_stats method"

        # Check that the class doesn't have any hardcoded default values
        source = inspect.getsource(ObservationBuilder)
        dangerous_patterns = [
            '"dxy_z": {"mean": 100.0',  # Wrong default
            '"vix_z": {"mean": 20.0',   # Wrong default
            '"embi_z": {"mean": 300.0',  # Wrong default
        ]

        for pattern in dangerous_patterns:
            assert pattern not in source, \
                f"ObservationBuilder contains dangerous hardcoded default: {pattern}"

class TestNormStatsConsistency:
    """Tests for norm_stats consistency between training and inference"""

    def test_norm_stats_file_exists(self):
        """norm_stats file should exist"""
        norm_stats_path = project_root / "config" / "norm_stats.json"
        assert norm_stats_path.exists(), \
            f"CRITICAL: Norm stats file not found at {norm_stats_path}"

    def test_norm_stats_has_required_features(self):
        """Norm stats should have all required features"""
        norm_stats_path = project_root / "config" / "norm_stats.json"

        if not norm_stats_path.exists():
            pytest.skip("Norm stats file not found")

        with open(norm_stats_path) as f:
            stats = json.load(f)

        required_features = [
            "log_ret_5m", "log_ret_1h", "log_ret_4h",
            "rsi_9", "atr_pct", "adx_14",
            "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
            "brent_change_1d", "rate_spread", "usdmxn_change_1d",
        ]

        for feature in required_features:
            assert feature in stats, f"Missing required feature: {feature}"
            assert "mean" in stats[feature], f"Missing 'mean' for {feature}"
            assert "std" in stats[feature], f"Missing 'std' for {feature}"

    def test_norm_stats_not_hardcoded_defaults(self):
        """Norm stats should NOT be the wrong hardcoded defaults"""
        norm_stats_path = project_root / "config" / "norm_stats.json"

        if not norm_stats_path.exists():
            pytest.skip("Norm stats file not found")

        with open(norm_stats_path) as f:
            stats = json.load(f)

        # These are WRONG defaults that were hardcoded before
        wrong_defaults = {
            "dxy_z": {"mean": 100.0, "std": 5.0},     # Should be ~0, ~1
            "vix_z": {"mean": 20.0, "std": 8.0},      # Should be ~0, ~0.9
            "embi_z": {"mean": 300.0, "std": 60.0},   # Should be ~0, ~1
        }

        for feature, wrong_values in wrong_defaults.items():
            if feature in stats:
                actual_mean = stats[feature].get("mean", 0)
                actual_std = stats[feature].get("std", 1)

                # Check that values are NOT close to wrong defaults
                # Correct values should be z-scored (mean~0, std~1)
                assert abs(actual_mean - wrong_values["mean"]) > 1.0, \
                    f"{feature} mean looks like wrong hardcoded default"

    def test_norm_stats_are_z_scored(self):
        """Macro z-score features should have mean~0, std~1"""
        norm_stats_path = project_root / "config" / "norm_stats.json"

        if not norm_stats_path.exists():
            pytest.skip("Norm stats file not found")

        with open(norm_stats_path) as f:
            stats = json.load(f)

        z_scored_features = ["dxy_z", "vix_z", "embi_z"]

        for feature in z_scored_features:
            if feature in stats:
                mean = stats[feature].get("mean", 999)
                std = stats[feature].get("std", 999)

                # Z-scored features should have mean close to 0 and std close to 1
                assert abs(mean) < 1.0, \
                    f"{feature} mean={mean} is not z-scored (expected ~0)"
                assert 0.1 < std < 2.0, \
                    f"{feature} std={std} is not z-scored (expected ~1)"


class TestConsistencyValidator:
    """Tests for ConsistencyValidatorService"""

    def test_validator_catches_missing_norm_stats(self):
        """Validator should catch missing norm_stats"""
        from services.inference_api.services.consistency_validator import (
            ConsistencyValidatorService,
            ValidationStatus,
        )
        from services.inference_api.contracts.model_contract import (
            ModelContract,
            BuilderType,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            validator = ConsistencyValidatorService(Path(tmpdir))

            # Create a contract pointing to non-existent norm_stats
            contract = ModelContract(
                model_id="test_missing",
                version="1.0.0",
                builder_type=BuilderType.CURRENT_15DIM,
                observation_dim=15,
                norm_stats_path="nonexistent.json",
                model_path="models/test.zip",
            )

            # Validator should detect this
            result = validator._check_norm_stats_exists(contract)
            assert result.status == ValidationStatus.FAILED
            assert "NOT FOUND" in result.message

    def test_validator_full_report(self):
        """Validator should produce complete report for registered model"""
        from services.inference_api.services.consistency_validator import (
            validate_model_consistency,
            ValidationStatus,
        )

        # Validate the production model
        report = validate_model_consistency("ppo_primary", project_root)

        assert report.model_id == "ppo_primary"
        assert len(report.checks) > 0

        # Should have checked registration
        registration_checks = [c for c in report.checks if c.check_name == "model_registration"]
        assert len(registration_checks) == 1

        # Print report for debugging
        print("\nConsistency Report for ppo_primary:")
        for check in report.checks:
            print(f"  {check.check_name}: {check.status.value} - {check.message}")


class TestHashVerification:
    """Tests for hash verification functionality"""

    def test_compute_json_hash_deterministic(self):
        """JSON hash should be deterministic"""
        from services.inference_api.contracts.model_contract import compute_json_hash

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"a": 1, "b": 2}, f)
            f.flush()

            hash1 = compute_json_hash(Path(f.name))
            hash2 = compute_json_hash(Path(f.name))

            assert hash1 == hash2, "Hash should be deterministic"
            assert len(hash1) == 64, "Should be SHA256 hex digest"

    def test_compute_json_hash_normalized(self):
        """JSON hash should normalize key order"""
        from services.inference_api.contracts.model_contract import compute_json_hash

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f1:
            json.dump({"a": 1, "b": 2}, f1)
            f1.flush()

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f2:
                json.dump({"b": 2, "a": 1}, f2)  # Different order
                f2.flush()

                hash1 = compute_json_hash(Path(f1.name))
                hash2 = compute_json_hash(Path(f2.name))

                assert hash1 == hash2, "Hash should be same regardless of key order"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
