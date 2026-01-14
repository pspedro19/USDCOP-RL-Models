"""
Tests para Features Snapshot Schema.
CLAUDE-T4 | Plan Item: P0-8 | Contrato: CTR-004

Valida:
- Schema de FeaturesSnapshot
- Serializacion para BD
- Validacion de feature count
"""

import pytest
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src" / "models"))

from trade_snapshot import (
    FeaturesSnapshot,
    ModelMetadata,
    compute_model_hash
)


class TestFeaturesSnapshot:
    """Tests para FeaturesSnapshot schema."""

    @pytest.fixture
    def valid_snapshot_data(self):
        """Datos validos para un snapshot."""
        return {
            "version": "current",
            "timestamp": datetime.now(),
            "bar_idx": 288,
            "raw_features": {
                "log_ret_5m": -0.0012,
                "log_ret_1h": 0.0035,
                "log_ret_4h": 0.0078,
                "rsi_9": 45.2,
                "atr_pct": 0.08,
                "adx_14": 28.5,
                "dxy_z": 0.5,
                "dxy_change_1d": 0.002,
                "vix_z": -0.3,
                "embi_z": 0.1,
                "brent_change_1d": -0.01,
                "rate_spread": 5.5,
                "usdmxn_change_1d": 0.003
            },
            "normalized_features": {
                "log_ret_5m": -0.5,
                "log_ret_1h": 0.8,
                "log_ret_4h": 1.2,
                "rsi_9": -0.3,
                "atr_pct": 0.4,
                "adx_14": -0.2,
                "dxy_z": 0.5,
                "dxy_change_1d": 0.2,
                "vix_z": -0.3,
                "embi_z": 0.1,
                "brent_change_1d": -0.2,
                "rate_spread": 0.6,
                "usdmxn_change_1d": 0.15,
                "position": 0.0,
                "time_normalized": 0.5
            }
        }

    def test_snapshot_parses_correctly(self, valid_snapshot_data):
        """FeaturesSnapshot DEBE parsear datos validos."""
        snapshot = FeaturesSnapshot(**valid_snapshot_data)

        assert snapshot.version == "current"
        assert snapshot.bar_idx == 288
        assert len(snapshot.raw_features) == 13
        assert len(snapshot.normalized_features) == 15

    def test_snapshot_validates_feature_count(self, valid_snapshot_data):
        """validate_feature_count DEBE pasar para datos validos."""
        snapshot = FeaturesSnapshot(**valid_snapshot_data)
        assert snapshot.validate_feature_count() is True

    def test_snapshot_rejects_wrong_raw_count(self, valid_snapshot_data):
        """validate_feature_count DEBE fallar si raw_features != 13."""
        valid_snapshot_data["raw_features"]["extra"] = 0.0  # 14 features
        snapshot = FeaturesSnapshot(**valid_snapshot_data)

        with pytest.raises(ValueError, match="13 features"):
            snapshot.validate_feature_count()

    def test_snapshot_rejects_wrong_norm_count(self, valid_snapshot_data):
        """validate_feature_count DEBE fallar si normalized_features != 15."""
        valid_snapshot_data["normalized_features"]["extra"] = 0.0  # 16 features
        snapshot = FeaturesSnapshot(**valid_snapshot_data)

        with pytest.raises(ValueError, match="15 features"):
            snapshot.validate_feature_count()

    def test_to_db_json_is_serializable(self, valid_snapshot_data):
        """to_db_json DEBE retornar JSON serializable."""
        snapshot = FeaturesSnapshot(**valid_snapshot_data)
        db_json = snapshot.to_db_json()

        # Debe ser serializable
        json_str = json.dumps(db_json)
        assert isinstance(json_str, str)

        # Round-trip
        recovered = json.loads(json_str)
        assert recovered["version"] == "current"
        assert len(recovered["raw_features"]) == 13

    def test_timestamp_iso_format(self, valid_snapshot_data):
        """timestamp DEBE serializar como ISO format."""
        snapshot = FeaturesSnapshot(**valid_snapshot_data)
        db_json = snapshot.to_db_json()

        # Debe ser string ISO
        assert isinstance(db_json["timestamp"], str)
        # Debe parsear sin error
        datetime.fromisoformat(db_json["timestamp"].replace('Z', '+00:00'))


class TestModelMetadata:
    """Tests para ModelMetadata schema."""

    def test_model_metadata_parses_correctly(self):
        """ModelMetadata DEBE parsear datos validos."""
        data = {
            "confidence": 0.85,
            "action_probs": [0.1, 0.85, 0.05],
            "critic_value": 0.023,
            "entropy": 0.32
        }
        metadata = ModelMetadata(**data)

        assert metadata.confidence == 0.85
        assert len(metadata.action_probs) == 3

    def test_confidence_must_be_0_to_1(self):
        """confidence DEBE estar en [0, 1]."""
        data = {
            "confidence": 1.5,  # Invalido
            "action_probs": [0.33, 0.33, 0.34],
            "critic_value": 0.0,
            "entropy": 0.5
        }

        with pytest.raises(Exception):  # ValidationError
            ModelMetadata(**data)

    def test_action_probs_must_have_3_elements(self):
        """action_probs DEBE tener exactamente 3 elementos."""
        data = {
            "confidence": 0.5,
            "action_probs": [0.5, 0.5],  # Solo 2
            "critic_value": 0.0,
            "entropy": 0.5
        }

        with pytest.raises(Exception):  # ValidationError
            ModelMetadata(**data)


class TestModelHash:
    """Tests para compute_model_hash."""

    def test_hash_is_deterministic(self):
        """compute_model_hash DEBE ser deterministico."""
        data = b"test model bytes"

        hash1 = compute_model_hash(data)
        hash2 = compute_model_hash(data)

        assert hash1 == hash2

    def test_hash_is_64_chars(self):
        """Hash SHA256 DEBE ser 64 caracteres."""
        data = b"test model bytes"
        hash_str = compute_model_hash(data)

        assert len(hash_str) == 64

    def test_different_data_different_hash(self):
        """Datos diferentes DEBEN producir hashes diferentes."""
        hash1 = compute_model_hash(b"model version 1")
        hash2 = compute_model_hash(b"model version 2")

        assert hash1 != hash2


class TestMigrationFileExists:
    """Tests que verifican existencia de archivos de migracion."""

    def test_migration_file_exists(self):
        """Archivo de migracion DEBE existir."""
        migration_path = project_root / "database" / "migrations" / "003_add_model_hash_and_constraints.sql"
        assert migration_path.exists(), f"Migracion no existe: {migration_path}"

    def test_init_script_exists(self):
        """Init script de trades metadata DEBE existir."""
        script_path = project_root / "init-scripts" / "12-trades-metadata.sql"
        assert script_path.exists(), f"Init script no existe: {script_path}"
