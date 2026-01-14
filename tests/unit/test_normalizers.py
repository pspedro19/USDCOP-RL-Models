"""
Tests for Normalizer Factory and Strategy Pattern implementations.

These tests verify:
- ZScoreNormalizer correctness and edge cases
- MinMaxNormalizer correctness and edge cases
- ClipNormalizer correctness and edge cases
- NormalizerFactory creation from config
- Batch normalization vectorization correctness

Contrato: CTR-006
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from features.normalizers import (
    ZScoreNormalizer,
    MinMaxNormalizer,
    ClipNormalizer,
    NoOpNormalizer,
    NormalizerFactory,
)


class TestZScoreNormalizer:
    """Tests for Z-Score normalization strategy."""

    def test_basic_normalization(self):
        """Standard z-score: (value - mean) / std."""
        normalizer = ZScoreNormalizer(mean=100.0, std=10.0)

        # Value at mean should be 0
        assert normalizer.normalize(100.0) == 0.0

        # One std above mean should be 1
        assert normalizer.normalize(110.0) == 1.0

        # One std below mean should be -1
        assert normalizer.normalize(90.0) == -1.0

    def test_clipping(self):
        """Values should be clipped to clip range."""
        normalizer = ZScoreNormalizer(mean=0.0, std=1.0, clip=(-2.0, 2.0))

        # 5 std above mean should clip to 2
        assert normalizer.normalize(5.0) == 2.0

        # 5 std below mean should clip to -2
        assert normalizer.normalize(-5.0) == -2.0

    def test_denormalization(self):
        """Denormalization should reverse normalization."""
        normalizer = ZScoreNormalizer(mean=50.0, std=10.0)

        original = 65.0
        normalized = normalizer.normalize(original)
        denormalized = normalizer.denormalize(normalized)

        assert abs(denormalized - original) < 1e-10

    def test_nan_handling(self):
        """NaN values should return 0.0."""
        normalizer = ZScoreNormalizer(mean=0.0, std=1.0)
        assert normalizer.normalize(float('nan')) == 0.0

    def test_inf_handling(self):
        """Inf values should return 0.0."""
        normalizer = ZScoreNormalizer(mean=0.0, std=1.0)
        assert normalizer.normalize(float('inf')) == 0.0
        assert normalizer.normalize(float('-inf')) == 0.0

    def test_batch_normalization(self):
        """Batch normalization should match scalar normalization."""
        normalizer = ZScoreNormalizer(mean=100.0, std=10.0, clip=(-5.0, 5.0))

        values = np.array([90.0, 100.0, 110.0, 150.0])
        batch_result = normalizer.normalize_batch(values)

        # Compare with scalar
        for i, val in enumerate(values):
            scalar_result = normalizer.normalize(val)
            assert abs(batch_result[i] - scalar_result) < 1e-10

    def test_zero_std_protection(self):
        """Zero std should default to 1.0 to prevent division by zero."""
        normalizer = ZScoreNormalizer(mean=0.0, std=0.0)
        assert normalizer.std == 1.0
        # Should not raise
        result = normalizer.normalize(5.0)
        assert result == 5.0


class TestMinMaxNormalizer:
    """Tests for Min-Max normalization strategy."""

    def test_basic_normalization(self):
        """Min-Max: (value - min) / (max - min)."""
        normalizer = MinMaxNormalizer(min_val=0.0, max_val=100.0)

        assert normalizer.normalize(0.0) == 0.0
        assert normalizer.normalize(50.0) == 0.5
        assert normalizer.normalize(100.0) == 1.0

    def test_custom_output_range(self):
        """Output range can be customized."""
        normalizer = MinMaxNormalizer(
            min_val=0.0, max_val=100.0,
            output_range=(-1.0, 1.0)
        )

        assert normalizer.normalize(0.0) == -1.0
        assert normalizer.normalize(50.0) == 0.0
        assert normalizer.normalize(100.0) == 1.0

    def test_clipping_to_output_range(self):
        """Values outside input range should clip to output range."""
        normalizer = MinMaxNormalizer(min_val=0.0, max_val=100.0)

        assert normalizer.normalize(-50.0) == 0.0
        assert normalizer.normalize(150.0) == 1.0

    def test_denormalization(self):
        """Denormalization should reverse normalization."""
        normalizer = MinMaxNormalizer(min_val=10.0, max_val=50.0)

        original = 30.0
        normalized = normalizer.normalize(original)
        denormalized = normalizer.denormalize(normalized)

        assert abs(denormalized - original) < 1e-10

    def test_batch_normalization(self):
        """Batch normalization should match scalar normalization."""
        normalizer = MinMaxNormalizer(min_val=0.0, max_val=100.0)

        values = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        batch_result = normalizer.normalize_batch(values)

        for i, val in enumerate(values):
            assert abs(batch_result[i] - normalizer.normalize(val)) < 1e-10


class TestClipNormalizer:
    """Tests for Clip-only normalization strategy."""

    def test_basic_clipping(self):
        """Values should be clipped without scaling."""
        normalizer = ClipNormalizer(clip=(-1.0, 1.0))

        assert normalizer.normalize(0.5) == 0.5
        assert normalizer.normalize(-0.5) == -0.5
        assert normalizer.normalize(2.0) == 1.0
        assert normalizer.normalize(-2.0) == -1.0

    def test_denormalization_is_identity(self):
        """Denormalization should be identity for ClipNormalizer."""
        normalizer = ClipNormalizer(clip=(-1.0, 1.0))

        assert normalizer.denormalize(0.5) == 0.5
        assert normalizer.denormalize(1.0) == 1.0

    def test_nan_handling(self):
        """NaN values should return 0.0."""
        normalizer = ClipNormalizer(clip=(-1.0, 1.0))
        assert normalizer.normalize(float('nan')) == 0.0


class TestNoOpNormalizer:
    """Tests for No-Op normalization strategy."""

    def test_passthrough(self):
        """Values should pass through unchanged."""
        normalizer = NoOpNormalizer()

        assert normalizer.normalize(5.0) == 5.0
        assert normalizer.normalize(-3.14) == -3.14

    def test_nan_handling(self):
        """NaN values should return 0.0."""
        normalizer = NoOpNormalizer()
        assert normalizer.normalize(float('nan')) == 0.0


class TestNormalizerFactory:
    """Tests for NormalizerFactory."""

    def test_create_zscore(self):
        """Factory should create ZScoreNormalizer."""
        normalizer = NormalizerFactory.create(
            "zscore",
            mean=10.0,
            std=2.0,
            clip=(-3.0, 3.0)
        )

        assert isinstance(normalizer, ZScoreNormalizer)
        assert normalizer.mean == 10.0
        assert normalizer.std == 2.0
        assert normalizer.clip == (-3.0, 3.0)

    def test_create_minmax(self):
        """Factory should create MinMaxNormalizer."""
        normalizer = NormalizerFactory.create(
            "minmax",
            min_val=0.0,
            max_val=100.0
        )

        assert isinstance(normalizer, MinMaxNormalizer)
        assert normalizer.min_val == 0.0
        assert normalizer.max_val == 100.0

    def test_create_clip(self):
        """Factory should create ClipNormalizer."""
        normalizer = NormalizerFactory.create(
            "clip",
            clip=(-5.0, 5.0)
        )

        assert isinstance(normalizer, ClipNormalizer)
        assert normalizer.clip == (-5.0, 5.0)

    def test_create_noop(self):
        """Factory should create NoOpNormalizer."""
        normalizer = NormalizerFactory.create("none")
        assert isinstance(normalizer, NoOpNormalizer)

    def test_from_config(self):
        """Factory should create normalizer from config dict."""
        config = {
            "method": "zscore",
            "mean": 49.27,
            "std": 23.07,
            "clip": [-3.0, 3.0]
        }

        normalizer = NormalizerFactory.from_config(config)

        assert isinstance(normalizer, ZScoreNormalizer)
        assert normalizer.mean == 49.27
        assert normalizer.std == 23.07

    def test_unknown_method_raises(self):
        """Unknown method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown normalization method"):
            NormalizerFactory.create("unknown_method")

    def test_list_methods(self):
        """list_methods should return available methods."""
        methods = NormalizerFactory.list_methods()

        assert "zscore" in methods
        assert "minmax" in methods
        assert "clip" in methods
        assert "none" in methods

    def test_register_custom_normalizer(self):
        """Should be able to register custom normalizer."""
        class CustomNormalizer:
            def normalize(self, value: float) -> float:
                return value * 2

            def denormalize(self, value: float) -> float:
                return value / 2

            def normalize_batch(self, values: np.ndarray) -> np.ndarray:
                return values * 2

        NormalizerFactory.register("custom", CustomNormalizer)

        normalizer = NormalizerFactory.create("custom")
        assert normalizer.normalize(5.0) == 10.0


class TestNormalizationConsistency:
    """Tests for consistency with feature_registry.yaml definitions."""

    @pytest.fixture
    def registry_config(self):
        """Load feature registry YAML for testing."""
        import yaml
        config_path = project_root / "config" / "feature_registry.yaml"

        if not config_path.exists():
            pytest.skip("feature_registry.yaml not found")

        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def test_all_features_have_normalization_config(self, registry_config):
        """All features in registry should have normalization config."""
        for feature in registry_config.get("features", []):
            assert "normalization" in feature, \
                f"Feature {feature['name']} missing normalization config"

            norm_config = feature["normalization"]
            assert "method" in norm_config, \
                f"Feature {feature['name']} missing normalization method"

    def test_factory_can_create_all_registry_normalizers(self, registry_config):
        """Factory should be able to create normalizers for all features."""
        for feature in registry_config.get("features", []):
            norm_config = feature["normalization"]

            # Should not raise
            normalizer = NormalizerFactory.from_config(norm_config)
            assert normalizer is not None, \
                f"Failed to create normalizer for {feature['name']}"
