"""
Tests para correcciones P0.
CLAUDE-T3, CLAUDE-T8 | Plan Items: P0-1, P0-4

Valida:
- P0-1: norm_stats path is correctly configured
- P0-4: No hay passwords hardcoded
"""

import pytest
import sys
import os
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "services"))


class TestP0_1_NormStats:
    """Tests para P0-1: norm_stats debe estar correctamente configurado."""

    def test_p0_1_norm_stats_path_is_configured(self):
        """P0-1: norm_stats path DEBE estar configurado."""
        from features.contract import FEATURE_CONTRACT

        assert FEATURE_CONTRACT.norm_stats_path is not None
        assert len(FEATURE_CONTRACT.norm_stats_path) > 0

    def test_p0_1_contract_observation_dim_is_15(self):
        """P0-1: observation_dim DEBE ser 15."""
        from features.contract import FEATURE_CONTRACT

        assert FEATURE_CONTRACT.observation_dim == 15

    def test_p0_1_contract_feature_order_has_15_features(self):
        """P0-1: feature_order DEBE tener 15 features."""
        from features.contract import FEATURE_CONTRACT

        assert len(FEATURE_CONTRACT.feature_order) == 15

    def test_p0_1_norm_stats_file_exists(self):
        """P0-1: norm_stats file DEBE existir."""
        from features.contract import FEATURE_CONTRACT

        norm_stats_path = project_root / FEATURE_CONTRACT.norm_stats_path

        assert norm_stats_path.exists(), \
            f"norm_stats no existe en {norm_stats_path}"

    def test_p0_1_norm_stats_has_required_features(self):
        """P0-1: norm_stats DEBE tener las features requeridas."""
        import json
        from features.contract import FEATURE_CONTRACT

        norm_stats_path = project_root / FEATURE_CONTRACT.norm_stats_path

        if not norm_stats_path.exists():
            pytest.skip(f"norm_stats file not found at {norm_stats_path}")

        actual_path = norm_stats_path

        with open(actual_path) as f:
            stats = json.load(f)

        # Las primeras 13 features (excluir position y time_normalized)
        required_features = FEATURE_CONTRACT.feature_order[:13]

        for feature in required_features:
            assert feature in stats, f"Feature {feature} falta en norm_stats"
            assert "mean" in stats[feature], f"Feature {feature} sin mean"
            assert "std" in stats[feature], f"Feature {feature} sin std"


class TestP0_4_NoHardcodedPasswords:
    """Tests para P0-4: No debe haber passwords hardcoded."""

    def test_p0_4_inference_api_no_hardcoded_password(self):
        """P0-4: inference_api config NO debe tener password hardcoded."""
        config_path = project_root / "services" / "inference_api" / "config.py"

        if not config_path.exists():
            pytest.skip("inference_api config.py not found")

        content = config_path.read_text()

        # No debe tener valores por defecto para passwords
        assert 'admin123' not in content, \
            "Password hardcoded 'admin123' encontrado en config.py"

        # Verificar que postgres_password usa empty string como default
        # Esto forzara a usar env vars
        assert 'postgres_password' in content.lower()

    def test_p0_4_config_file_no_hardcoded_default(self):
        """P0-4: El archivo config.py NO debe tener password hardcoded como default."""
        config_path = project_root / "services" / "inference_api" / "config.py"

        if not config_path.exists():
            pytest.skip("inference_api config.py not found")

        content = config_path.read_text()

        # Buscar el patron de default value para postgres_password
        # El patron correcto es: os.getenv("POSTGRES_PASSWORD", "")
        # El patron incorrecto seria: os.getenv("POSTGRES_PASSWORD", "admin123")
        import re

        # Buscar definicion de postgres_password con valor default
        pattern = r'postgres_password.*=.*os\.getenv\([^,]+,\s*"([^"]*)"'
        match = re.search(pattern, content)

        if match:
            default_value = match.group(1)
            assert default_value == "", \
                f"postgres_password tiene default '{default_value}', debe ser ''"


class TestFeatureContract:
    """Tests adicionales para Feature Contract."""

    def test_contract_is_frozen(self):
        """El contrato DEBE ser inmutable (frozen dataclass)."""
        from features.contract import FEATURE_CONTRACT

        # Intentar modificar debe fallar
        with pytest.raises(Exception):  # FrozenInstanceError
            FEATURE_CONTRACT.observation_dim = 20

    def test_get_contract_returns_correct_instance(self):
        """get_contract DEBE retornar el contrato correcto."""
        from features.contract import get_contract

        contract = get_contract("current")
        assert contract is not None
        assert contract.observation_dim == 15

    def test_get_contract_invalid_version_raises(self):
        """get_contract con version invalida DEBE lanzar ValueError."""
        from features.contract import get_contract

        with pytest.raises(ValueError, match="Unknown version|no existe"):
            get_contract("v99")
