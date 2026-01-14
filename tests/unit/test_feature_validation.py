"""
Tests para Feature Validation.
CLAUDE-T14 | Plan Item: P1-5

Valida:
- validate_feature_order_at_startup
- FeatureOrderMismatchError
- FeatureCountMismatchError
- validate_observation_shape
- validate_no_nan_inf
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from features.validation import (
    validate_feature_order_at_startup,
    validate_observation_shape,
    validate_no_nan_inf,
    FeatureOrderMismatchError,
    FeatureCountMismatchError,
    create_startup_validator,
)
from features.builder import FeatureBuilder


class TestFeatureOrderValidation:
    """Tests para validacion de feature order."""

    @pytest.fixture
    def builder(self):
        return FeatureBuilder(version="current")

    def test_validation_passes_with_no_model_metadata(self, builder, tmp_path):
        """Validacion DEBE pasar si modelo no tiene metadata (con warning)."""
        # Crear archivo ONNX vacio (mock)
        model_path = tmp_path / "test_model.onnx"
        model_path.touch()

        with patch("features.validation._extract_feature_order_from_onnx") as mock_extract:
            mock_extract.return_value = None
            result = validate_feature_order_at_startup(builder, model_path, strict=False)
            assert result is True

    def test_validation_raises_on_missing_model_strict(self, builder, tmp_path):
        """Validacion DEBE lanzar error si modelo no existe en modo strict."""
        model_path = tmp_path / "nonexistent.onnx"

        with pytest.raises(FileNotFoundError):
            validate_feature_order_at_startup(builder, model_path, strict=True)

    def test_validation_returns_false_on_missing_model_non_strict(self, builder, tmp_path):
        """Validacion DEBE retornar False si modelo no existe en modo non-strict."""
        model_path = tmp_path / "nonexistent.onnx"
        result = validate_feature_order_at_startup(builder, model_path, strict=False)
        assert result is False

    def test_validation_raises_on_order_mismatch(self, builder, tmp_path):
        """Validacion DEBE lanzar FeatureOrderMismatchError si orden difiere."""
        model_path = tmp_path / "test_model.onnx"
        model_path.touch()

        # Simular orden diferente
        wrong_order = list(builder.get_feature_names())
        wrong_order[0], wrong_order[1] = wrong_order[1], wrong_order[0]  # Swap

        with patch("features.validation._extract_feature_order_from_onnx") as mock_extract:
            mock_extract.return_value = wrong_order
            with pytest.raises(FeatureOrderMismatchError):
                validate_feature_order_at_startup(builder, model_path, strict=True)

    def test_validation_raises_on_count_mismatch(self, builder, tmp_path):
        """Validacion DEBE lanzar FeatureCountMismatchError si cantidad difiere."""
        model_path = tmp_path / "test_model.onnx"
        model_path.touch()

        # Simular menos features
        wrong_order = list(builder.get_feature_names())[:-2]

        with patch("features.validation._extract_feature_order_from_onnx") as mock_extract:
            mock_extract.return_value = wrong_order
            with pytest.raises(FeatureCountMismatchError):
                validate_feature_order_at_startup(builder, model_path, strict=True)

    def test_validation_passes_with_matching_order(self, builder, tmp_path):
        """Validacion DEBE pasar si orden coincide exactamente."""
        model_path = tmp_path / "test_model.onnx"
        model_path.touch()

        correct_order = list(builder.get_feature_names())

        with patch("features.validation._extract_feature_order_from_onnx") as mock_extract:
            mock_extract.return_value = correct_order
            result = validate_feature_order_at_startup(builder, model_path, strict=True)
            assert result is True


class TestObservationShapeValidation:
    """Tests para validacion de observation shape."""

    def test_valid_shape_passes(self):
        """Shape correcto DEBE pasar validacion."""
        obs = np.zeros(15, dtype=np.float32)
        result = validate_observation_shape(obs, expected_dim=15)
        assert result is True

    def test_invalid_shape_raises_in_strict_mode(self):
        """Shape incorrecto DEBE lanzar ValueError en modo strict."""
        obs = np.zeros(10, dtype=np.float32)
        with pytest.raises(ValueError, match="Shape incorrecto"):
            validate_observation_shape(obs, expected_dim=15, strict=True)

    def test_invalid_shape_returns_false_in_non_strict_mode(self):
        """Shape incorrecto DEBE retornar False en modo non-strict."""
        obs = np.zeros(10, dtype=np.float32)
        result = validate_observation_shape(obs, expected_dim=15, strict=False)
        assert result is False

    def test_non_array_raises_typeerror(self):
        """Input no-array DEBE lanzar TypeError."""
        with pytest.raises(TypeError):
            validate_observation_shape([1, 2, 3], expected_dim=3, strict=True)


class TestNanInfValidation:
    """Tests para validacion de NaN/Inf."""

    def test_valid_array_passes(self):
        """Array sin NaN/Inf DEBE pasar validacion."""
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = validate_no_nan_inf(obs)
        assert result is True

    def test_nan_raises_in_strict_mode(self):
        """NaN DEBE lanzar ValueError en modo strict."""
        obs = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        with pytest.raises(ValueError, match="NaN"):
            validate_no_nan_inf(obs, strict=True)

    def test_inf_raises_in_strict_mode(self):
        """Inf DEBE lanzar ValueError en modo strict."""
        obs = np.array([1.0, np.inf, 3.0], dtype=np.float32)
        with pytest.raises(ValueError, match="Inf"):
            validate_no_nan_inf(obs, strict=True)

    def test_nan_returns_false_in_non_strict_mode(self):
        """NaN DEBE retornar False en modo non-strict."""
        obs = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        result = validate_no_nan_inf(obs, strict=False)
        assert result is False


class TestStartupValidator:
    """Tests para create_startup_validator."""

    def test_startup_validator_created(self):
        """Validator DEBE ser creado correctamente."""
        builder = FeatureBuilder(version="current")
        model_path = Path("test.onnx")

        validator = create_startup_validator(builder, model_path)
        assert callable(validator)

    def test_startup_validator_raises_on_missing_model(self, tmp_path):
        """Validator DEBE lanzar error si modelo no existe."""
        builder = FeatureBuilder(version="current")
        model_path = tmp_path / "nonexistent.onnx"

        validator = create_startup_validator(builder, model_path)

        with pytest.raises(FileNotFoundError):
            validator()
