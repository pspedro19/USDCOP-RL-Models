"""
Tests para Model Metadata.
CLAUDE-T13 | Plan Item: P1-2
Contrato: CTR-009

Valida:
- PredictionMetadata dataclass
- capture_prediction_metadata function
- Serialization/deserialization
- Validation
- Entropy and confidence computation
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from models.model_metadata import (
    PredictionMetadata,
    capture_prediction_metadata,
    compute_entropy,
    compute_confidence,
)


class TestPredictionMetadataCreation:
    """Tests para creacion de PredictionMetadata."""

    def test_create_with_required_fields(self):
        """PredictionMetadata DEBE ser creatable con campos minimos."""
        metadata = PredictionMetadata(
            model_id="ppo_primary_abc12345",
            model_version="current",
            model_hash="a" * 64,
            norm_stats_hash="b" * 64,
        )

        assert metadata.model_id == "ppo_primary_abc12345"
        assert metadata.model_version == "current"
        assert len(metadata.model_hash) == 64
        assert len(metadata.norm_stats_hash) == 64

    def test_default_values_are_correct(self):
        """Valores por defecto DEBEN ser correctos."""
        metadata = PredictionMetadata(
            model_id="test",
            model_version="current",
            model_hash="a" * 64,
            norm_stats_hash="b" * 64,
        )

        assert metadata.action == 0
        assert metadata.bid_ask_spread == 0.0
        assert metadata.market_volatility == 0.0
        assert metadata.confidence == 0.0
        assert metadata.entropy == 0.0
        assert metadata.observation == []
        assert metadata.raw_features == {}

    def test_action_probabilities_default(self):
        """action_probabilities default DEBE ser ~uniforme."""
        metadata = PredictionMetadata(
            model_id="test",
            model_version="current",
            model_hash="a" * 64,
            norm_stats_hash="b" * 64,
        )

        assert len(metadata.action_probabilities) == 3
        assert sum(metadata.action_probabilities) == pytest.approx(1.0, rel=0.01)


class TestPredictionMetadataSerialization:
    """Tests para serializacion."""

    @pytest.fixture
    def sample_metadata(self):
        """Metadata de ejemplo."""
        return PredictionMetadata(
            model_id="ppo_primary_abc12345",
            model_version="current",
            model_hash="a" * 64,
            norm_stats_hash="b" * 64,
            observation=[0.5] * 15,
            raw_features={"rsi_9": 55.0, "atr_pct": 0.01},
            bid_ask_spread=0.0012,
            market_volatility=0.05,
            action=1,
            action_probabilities=[0.1, 0.85, 0.05],
            confidence=0.85,
            entropy=0.5,
            bar_idx=100,
        )

    def test_to_dict_returns_dict(self, sample_metadata):
        """to_dict DEBE retornar un diccionario."""
        result = sample_metadata.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_has_all_fields(self, sample_metadata):
        """to_dict DEBE incluir todos los campos."""
        result = sample_metadata.to_dict()

        assert "model_id" in result
        assert "model_version" in result
        assert "model_hash" in result
        assert "norm_stats_hash" in result
        assert "observation" in result
        assert "raw_features" in result
        assert "bid_ask_spread" in result
        assert "action" in result
        assert "action_probabilities" in result
        assert "confidence" in result
        assert "entropy" in result
        assert "timestamp" in result

    def test_timestamp_serialized_as_iso(self, sample_metadata):
        """timestamp DEBE ser serializado como ISO string."""
        result = sample_metadata.to_dict()
        assert isinstance(result["timestamp"], str)
        # Should be parseable
        datetime.fromisoformat(result["timestamp"])

    def test_from_dict_roundtrip(self, sample_metadata):
        """from_dict DEBE reconstruir metadata correctamente."""
        d = sample_metadata.to_dict()
        restored = PredictionMetadata.from_dict(d)

        assert restored.model_id == sample_metadata.model_id
        assert restored.model_version == sample_metadata.model_version
        assert restored.action == sample_metadata.action
        assert restored.confidence == sample_metadata.confidence
        assert restored.bid_ask_spread == sample_metadata.bid_ask_spread

    def test_from_dict_handles_z_suffix(self):
        """from_dict DEBE manejar timestamps con Z."""
        d = {
            "model_id": "test",
            "model_version": "current",
            "model_hash": "a" * 64,
            "norm_stats_hash": "b" * 64,
            "observation": [],
            "raw_features": {},
            "bid_ask_spread": 0.0,
            "market_volatility": 0.0,
            "timestamp": "2024-01-15T10:30:00Z",
            "action": 0,
            "action_probabilities": [0.33, 0.34, 0.33],
            "value_estimate": 0.0,
            "entropy": 0.0,
            "confidence": 0.0,
            "bar_idx": None,
            "feature_contract_version": "current",
            "config_hash": None,
        }

        metadata = PredictionMetadata.from_dict(d)
        assert metadata.model_id == "test"
        assert isinstance(metadata.timestamp, datetime)


class TestPredictionMetadataValidation:
    """Tests para validacion."""

    def test_valid_metadata_passes(self):
        """Metadata valida DEBE pasar validacion."""
        metadata = PredictionMetadata(
            model_id="ppo_primary_abc12345",
            model_version="current",
            model_hash="a" * 64,
            norm_stats_hash="b" * 64,
            observation=[0.5] * 15,
            action_probabilities=[0.1, 0.85, 0.05],
            confidence=0.85,
            action=1,
        )

        assert metadata.validate() is True

    def test_missing_model_id_raises(self):
        """model_id vacio DEBE lanzar ValueError."""
        metadata = PredictionMetadata(
            model_id="",
            model_version="current",
            model_hash="a" * 64,
            norm_stats_hash="b" * 64,
            observation=[0.5] * 15,
        )

        with pytest.raises(ValueError, match="model_id"):
            metadata.validate()

    def test_missing_model_hash_raises(self):
        """model_hash vacio DEBE lanzar ValueError."""
        metadata = PredictionMetadata(
            model_id="test",
            model_version="current",
            model_hash="",
            norm_stats_hash="b" * 64,
            observation=[0.5] * 15,
        )

        with pytest.raises(ValueError, match="model_hash"):
            metadata.validate()

    def test_wrong_observation_count_raises(self):
        """observation con != 15 elementos DEBE lanzar ValueError."""
        metadata = PredictionMetadata(
            model_id="test",
            model_version="current",
            model_hash="a" * 64,
            norm_stats_hash="b" * 64,
            observation=[0.5] * 10,  # Wrong count
        )

        with pytest.raises(ValueError, match="observation"):
            metadata.validate()

    def test_wrong_action_probs_count_raises(self):
        """action_probabilities con != 3 elementos DEBE lanzar ValueError."""
        metadata = PredictionMetadata(
            model_id="test",
            model_version="current",
            model_hash="a" * 64,
            norm_stats_hash="b" * 64,
            observation=[0.5] * 15,
            action_probabilities=[0.5, 0.5],  # Only 2
        )

        with pytest.raises(ValueError, match="action_probabilities"):
            metadata.validate()

    def test_invalid_confidence_raises(self):
        """confidence fuera de [0,1] DEBE lanzar ValueError."""
        metadata = PredictionMetadata(
            model_id="test",
            model_version="current",
            model_hash="a" * 64,
            norm_stats_hash="b" * 64,
            observation=[0.5] * 15,
            confidence=1.5,  # Invalid
        )

        with pytest.raises(ValueError, match="confidence"):
            metadata.validate()

    def test_invalid_action_raises(self):
        """action fuera de [0,2] DEBE lanzar ValueError."""
        metadata = PredictionMetadata(
            model_id="test",
            model_version="current",
            model_hash="a" * 64,
            norm_stats_hash="b" * 64,
            observation=[0.5] * 15,
            action=5,  # Invalid
        )

        with pytest.raises(ValueError, match="action"):
            metadata.validate()


class TestCapturePredictonMetadata:
    """Tests para capture_prediction_metadata."""

    def test_captures_all_fields(self):
        """capture_prediction_metadata DEBE capturar todos los campos."""
        obs = np.zeros(15, dtype=np.float32)
        raw = {"rsi_9": 55.0, "atr_pct": 0.01}

        metadata = capture_prediction_metadata(
            model_id="ppo_primary_abc12345",
            model_version="current",
            model_hash="a" * 64,
            norm_stats_hash="b" * 64,
            observation=obs,
            raw_features=raw,
            action=1,
            action_probabilities=[0.1, 0.85, 0.05],
            value_estimate=0.5,
            bid_ask_spread=0.0012,
            market_volatility=0.05,
            bar_idx=100,
        )

        assert metadata.model_id == "ppo_primary_abc12345"
        assert metadata.action == 1
        assert metadata.bid_ask_spread == 0.0012
        assert metadata.bar_idx == 100
        assert len(metadata.observation) == 15

    def test_computes_entropy_from_probs(self):
        """capture_prediction_metadata DEBE computar entropy."""
        obs = np.zeros(15, dtype=np.float32)

        # Uniform probs = high entropy
        metadata = capture_prediction_metadata(
            model_id="test",
            model_version="current",
            model_hash="a" * 64,
            norm_stats_hash="b" * 64,
            observation=obs,
            raw_features={},
            action=0,
            action_probabilities=[0.33, 0.34, 0.33],
        )

        assert metadata.entropy > 0.9  # Near max entropy for 3 classes

    def test_computes_confidence_from_probs(self):
        """capture_prediction_metadata DEBE computar confidence como max prob."""
        obs = np.zeros(15, dtype=np.float32)

        metadata = capture_prediction_metadata(
            model_id="test",
            model_version="current",
            model_hash="a" * 64,
            norm_stats_hash="b" * 64,
            observation=obs,
            raw_features={},
            action=1,
            action_probabilities=[0.1, 0.85, 0.05],
        )

        assert metadata.confidence == 0.85

    def test_converts_numpy_observation_to_list(self):
        """observation numpy DEBE ser convertido a lista."""
        obs = np.array([1.0, 2.0, 3.0] * 5, dtype=np.float32)

        metadata = capture_prediction_metadata(
            model_id="test",
            model_version="current",
            model_hash="a" * 64,
            norm_stats_hash="b" * 64,
            observation=obs,
            raw_features={},
            action=0,
            action_probabilities=[0.33, 0.34, 0.33],
        )

        assert isinstance(metadata.observation, list)
        assert len(metadata.observation) == 15

    def test_sets_timestamp_automatically(self):
        """timestamp DEBE ser seteado automaticamente."""
        obs = np.zeros(15, dtype=np.float32)

        before = datetime.utcnow()
        metadata = capture_prediction_metadata(
            model_id="test",
            model_version="current",
            model_hash="a" * 64,
            norm_stats_hash="b" * 64,
            observation=obs,
            raw_features={},
            action=0,
            action_probabilities=[0.33, 0.34, 0.33],
        )
        after = datetime.utcnow()

        assert before <= metadata.timestamp <= after


class TestEntropyComputation:
    """Tests para compute_entropy."""

    def test_uniform_distribution_max_entropy(self):
        """Distribucion uniforme DEBE tener max entropy."""
        probs = [1/3, 1/3, 1/3]
        entropy = compute_entropy(probs)

        # Max entropy for 3 classes = ln(3) ≈ 1.0986
        assert entropy == pytest.approx(np.log(3), rel=0.01)

    def test_deterministic_distribution_zero_entropy(self):
        """Distribucion deterministica DEBE tener ~0 entropy."""
        probs = [1.0, 0.0, 0.0]
        entropy = compute_entropy(probs)

        assert entropy == pytest.approx(0.0, abs=0.01)

    def test_handles_zero_probabilities(self):
        """compute_entropy DEBE manejar probabilidades cero."""
        probs = [0.5, 0.5, 0.0]
        entropy = compute_entropy(probs)

        # Should not raise, should return valid value
        assert entropy >= 0
        assert not np.isnan(entropy)


class TestConfidenceComputation:
    """Tests para compute_confidence."""

    def test_returns_max_probability(self):
        """compute_confidence DEBE retornar probabilidad maxima."""
        probs = [0.1, 0.85, 0.05]
        confidence = compute_confidence(probs)

        assert confidence == 0.85

    def test_uniform_distribution_third(self):
        """Distribucion uniforme DEBE dar confidence ~0.33."""
        probs = [0.33, 0.34, 0.33]
        confidence = compute_confidence(probs)

        assert confidence == 0.34

    def test_deterministic_one(self):
        """Distribucion deterministica DEBE dar confidence 1.0."""
        probs = [1.0, 0.0, 0.0]
        confidence = compute_confidence(probs)

        assert confidence == 1.0


class TestBidAskSpreadCapture:
    """Tests especificos para bid_ask_spread (P1-2 requirement)."""

    def test_bid_ask_spread_stored_correctly(self):
        """bid_ask_spread DEBE ser almacenado correctamente."""
        obs = np.zeros(15, dtype=np.float32)

        metadata = capture_prediction_metadata(
            model_id="test",
            model_version="current",
            model_hash="a" * 64,
            norm_stats_hash="b" * 64,
            observation=obs,
            raw_features={},
            action=0,
            action_probabilities=[0.33, 0.34, 0.33],
            bid_ask_spread=0.00125,
        )

        assert metadata.bid_ask_spread == 0.00125

    def test_bid_ask_spread_in_serialized_dict(self):
        """bid_ask_spread DEBE aparecer en dict serializado."""
        obs = np.zeros(15, dtype=np.float32)

        metadata = capture_prediction_metadata(
            model_id="test",
            model_version="current",
            model_hash="a" * 64,
            norm_stats_hash="b" * 64,
            observation=obs,
            raw_features={},
            action=0,
            action_probabilities=[0.33, 0.34, 0.33],
            bid_ask_spread=0.0015,
        )

        d = metadata.to_dict()
        assert d["bid_ask_spread"] == 0.0015

    def test_bid_ask_spread_roundtrip(self):
        """bid_ask_spread DEBE sobrevivir roundtrip serialization."""
        obs = np.zeros(15, dtype=np.float32)

        original = capture_prediction_metadata(
            model_id="test",
            model_version="current",
            model_hash="a" * 64,
            norm_stats_hash="b" * 64,
            observation=obs,
            raw_features={},
            action=0,
            action_probabilities=[0.33, 0.34, 0.33],
            bid_ask_spread=0.00175,
        )

        d = original.to_dict()
        restored = PredictionMetadata.from_dict(d)

        assert restored.bid_ask_spread == 0.00175
