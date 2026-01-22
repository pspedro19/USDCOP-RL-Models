"""
Tests para Feature Contract SSOT.
"""
import pytest
import numpy as np
import subprocess
from pathlib import Path

from src.core.contracts.feature_contract import (
    FEATURE_ORDER,
    OBSERVATION_DIM,
    FEATURE_CONTRACT,
    FEATURE_SPECS,
    FEATURE_ORDER_HASH,
    FeatureType,
    validate_feature_vector,
    get_feature_index,
    features_dict_to_array,
    FeatureContractError,
)


REPO_ROOT = Path(__file__).parent.parent.parent


class TestFeatureOrderSSoT:
    """Tests para FEATURE_ORDER como SSOT."""

    def test_feature_order_has_15_elements(self):
        assert len(FEATURE_ORDER) == 15
        assert len(FEATURE_ORDER) == OBSERVATION_DIM

    def test_feature_order_is_immutable(self):
        assert isinstance(FEATURE_ORDER, tuple)

    def test_all_expected_features_present(self):
        expected = {
            "log_ret_5m", "log_ret_1h", "log_ret_4h",
            "rsi_9", "atr_pct", "adx_14",
            "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
            "brent_change_1d", "rate_spread", "usdmxn_change_1d",
            "position", "time_normalized"
        }
        assert set(FEATURE_ORDER) == expected

    def test_no_session_progress_in_features(self):
        assert "session_progress" not in FEATURE_ORDER
        assert "time_normalized" in FEATURE_ORDER

    def test_time_normalized_is_last(self):
        assert FEATURE_ORDER[14] == "time_normalized"
        assert get_feature_index("time_normalized") == 14

    def test_position_is_second_to_last(self):
        assert FEATURE_ORDER[13] == "position"
        assert get_feature_index("position") == 13

    def test_feature_order_hash_is_deterministic(self):
        import hashlib
        expected_hash = hashlib.sha256(",".join(FEATURE_ORDER).encode("utf-8")).hexdigest()[:16]
        assert FEATURE_ORDER_HASH == expected_hash


class TestFeatureSpecs:
    """Tests para especificaciones de features."""

    def test_all_features_have_specs(self):
        for name in FEATURE_ORDER:
            assert name in FEATURE_SPECS

    def test_spec_indices_match_order(self):
        for i, name in enumerate(FEATURE_ORDER):
            assert FEATURE_SPECS[name].index == i

    def test_technical_features_count(self):
        technical = FEATURE_CONTRACT.get_features_by_type(FeatureType.TECHNICAL)
        assert len(technical) == 6

    def test_macro_features_count(self):
        macro = FEATURE_CONTRACT.get_features_by_type(FeatureType.MACRO)
        assert len(macro) == 7

    def test_normalizable_features(self):
        normalizable = FEATURE_CONTRACT.get_normalizable_features()
        assert "position" not in normalizable
        assert "time_normalized" not in normalizable
        assert len(normalizable) == 13


class TestFeatureValidation:
    """Tests para validación de features."""

    @pytest.fixture
    def valid_observation(self):
        return np.zeros(15, dtype=np.float32)

    def test_validate_valid_observation(self, valid_observation):
        assert validate_feature_vector(valid_observation)

    def test_validate_wrong_shape_14(self):
        obs = np.zeros(14, dtype=np.float32)
        with pytest.raises(FeatureContractError, match="Invalid shape"):
            validate_feature_vector(obs)

    def test_validate_wrong_shape_16(self):
        obs = np.zeros(16, dtype=np.float32)
        with pytest.raises(FeatureContractError, match="Invalid shape"):
            validate_feature_vector(obs)

    def test_validate_detects_nan(self):
        obs = np.zeros(15, dtype=np.float32)
        obs[5] = np.nan
        with pytest.raises(FeatureContractError, match="NaN"):
            validate_feature_vector(obs)

    def test_validate_detects_inf(self):
        obs = np.zeros(15, dtype=np.float32)
        obs[3] = np.inf
        with pytest.raises(FeatureContractError, match="Inf"):
            validate_feature_vector(obs)

    def test_validate_detects_out_of_range(self):
        obs = np.zeros(15, dtype=np.float32)
        obs[0] = 10.0
        with pytest.raises(FeatureContractError, match="outside range"):
            validate_feature_vector(obs, strict=True)

    def test_validate_position_range(self):
        obs = np.zeros(15, dtype=np.float32)
        obs[13] = 2.0
        with pytest.raises(FeatureContractError, match="position"):
            validate_feature_vector(obs, strict=True)

    def test_validate_time_normalized_range(self):
        obs = np.zeros(15, dtype=np.float32)
        obs[14] = 1.5
        with pytest.raises(FeatureContractError, match="time_normalized"):
            validate_feature_vector(obs, strict=True)


class TestFeatureDict:
    """Tests para conversión dict → array."""

    def test_dict_to_array_valid(self):
        features = {name: 0.0 for name in FEATURE_ORDER}
        result = features_dict_to_array(features)
        assert result.shape == (15,)
        assert result.dtype == np.float32

    def test_dict_to_array_preserves_order(self):
        features = {name: float(i) for i, name in enumerate(FEATURE_ORDER)}
        result = features_dict_to_array(features)
        for i, name in enumerate(FEATURE_ORDER):
            assert result[i] == float(i)

    def test_dict_missing_feature_fails(self):
        features = {name: 0.0 for name in FEATURE_ORDER}
        del features["position"]
        with pytest.raises(FeatureContractError, match="Missing"):
            features_dict_to_array(features)

    def test_dict_extra_feature_fails(self):
        features = {name: 0.0 for name in FEATURE_ORDER}
        features["extra_feature"] = 0.0
        with pytest.raises(FeatureContractError, match="Extra"):
            features_dict_to_array(features)
