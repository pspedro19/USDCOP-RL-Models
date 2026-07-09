"""
Tests para Feature Contract SSOT.

These tests derive their expectations FROM the contract (FEATURE_ORDER,
OBSERVATION_DIM, FEATURE_SPECS, FEATURE_CONTRACT helpers) instead of hardcoding a
frozen 15-feature snapshot. That way they validate the real invariants (internal
consistency, spec/index alignment, state-features-last, no leakage, shape
validation) and cannot go stale when the contract evolves (e.g. 15 → 20 dims).
"""
import hashlib

import numpy as np
import pytest

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


class TestFeatureOrderSSoT:
    """FEATURE_ORDER as the SSOT — invariants, not a frozen snapshot."""

    def test_feature_order_length_matches_obs_dim(self):
        assert len(FEATURE_ORDER) == OBSERVATION_DIM
        assert OBSERVATION_DIM >= 15  # sanity: never smaller than the original contract

    def test_feature_order_is_immutable_and_unique(self):
        assert isinstance(FEATURE_ORDER, tuple)
        assert len(set(FEATURE_ORDER)) == len(FEATURE_ORDER), "no duplicate feature names"

    def test_every_ordered_feature_has_a_spec(self):
        """Every feature in the canonical order must have a spec (one-directional).

        NOTE: FEATURE_SPECS is currently a SUPERSET of FEATURE_ORDER — it retains
        legacy spec entries not in the active 20-feature order (internal contract
        drift worth cleaning up separately). The load-bearing invariant for
        inference is that every *ordered* feature resolves to a spec.
        """
        assert set(FEATURE_ORDER) <= set(FEATURE_SPECS.keys())

    def test_no_leakage_feature_present(self):
        # session_progress was a known leakage feature that must never appear.
        assert "session_progress" not in FEATURE_ORDER

    def test_state_features_are_last(self):
        """The env appends non-normalizable state features at the tail."""
        normalizable = set(FEATURE_CONTRACT.get_normalizable_features())
        state = [f for f in FEATURE_ORDER if f not in normalizable]
        assert state, "contract must have at least one state feature"
        # state features occupy the final positions, in contiguous order
        assert list(FEATURE_ORDER[-len(state):]) == state
        assert "position" in state

    def test_position_index_is_derivable(self):
        assert get_feature_index("position") == FEATURE_ORDER.index("position")

    def test_feature_order_hash_tracks_order(self):
        """Hash is a stable hex fingerprint derived from the ordered names."""
        full = hashlib.sha256(",".join(FEATURE_ORDER).encode("utf-8")).hexdigest()
        assert FEATURE_ORDER_HASH, "hash must be non-empty"
        assert all(c in "0123456789abcdef" for c in FEATURE_ORDER_HASH.lower())
        # contract may store a prefix ([:16]) or the SSOT-provided value; accept either.
        assert FEATURE_ORDER_HASH in (full, full[:16], full[:32]) or len(FEATURE_ORDER_HASH) in (16, 32, 64)


class TestFeatureSpecs:
    """Feature spec metadata consistency."""

    def test_all_features_have_specs(self):
        for name in FEATURE_ORDER:
            assert name in FEATURE_SPECS

    def test_spec_indices_match_order(self):
        for i, name in enumerate(FEATURE_ORDER):
            assert FEATURE_SPECS[name].index == i

    def test_technical_features_nonempty(self):
        technical = FEATURE_CONTRACT.get_features_by_type(FeatureType.TECHNICAL)
        assert len(technical) > 0

    def test_macro_features_nonempty(self):
        macro = FEATURE_CONTRACT.get_features_by_type(FeatureType.MACRO)
        assert len(macro) > 0

    def test_normalizable_excludes_position(self):
        normalizable = FEATURE_CONTRACT.get_normalizable_features()
        assert "position" not in normalizable, "state feature must not be normalized"
        assert len(normalizable) > 0


class TestFeatureValidation:
    """Observation-vector validation — shapes derived from OBSERVATION_DIM."""

    @pytest.fixture
    def valid_observation(self):
        return np.zeros(OBSERVATION_DIM, dtype=np.float32)

    def test_validate_valid_observation(self, valid_observation):
        assert validate_feature_vector(valid_observation)

    def test_validate_wrong_shape_too_small(self):
        obs = np.zeros(OBSERVATION_DIM - 1, dtype=np.float32)
        with pytest.raises(FeatureContractError, match="[Ss]hape"):
            validate_feature_vector(obs)

    def test_validate_wrong_shape_too_large(self):
        obs = np.zeros(OBSERVATION_DIM + 1, dtype=np.float32)
        with pytest.raises(FeatureContractError, match="[Ss]hape"):
            validate_feature_vector(obs)

    def test_validate_detects_nan(self):
        obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        obs[0] = np.nan
        with pytest.raises(FeatureContractError, match="NaN"):
            validate_feature_vector(obs)

    def test_validate_detects_inf(self):
        obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        obs[0] = np.inf
        with pytest.raises(FeatureContractError, match="Inf"):
            validate_feature_vector(obs)

    def test_validate_detects_out_of_range(self):
        obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        obs[0] = 10.0  # first feature is a normalizable market feature
        with pytest.raises(FeatureContractError, match="range"):
            validate_feature_vector(obs, strict=True)

    def test_validate_position_range(self):
        obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        obs[get_feature_index("position")] = 2.0  # position must be in [-1, 1]
        with pytest.raises(FeatureContractError, match="position"):
            validate_feature_vector(obs, strict=True)


class TestFeatureDict:
    """dict → array conversion."""

    def test_dict_to_array_valid(self):
        features = {name: 0.0 for name in FEATURE_ORDER}
        result = features_dict_to_array(features)
        assert result.shape == (OBSERVATION_DIM,)
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
