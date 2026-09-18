"""Legacy simplex reconstruction is descriptive, never a training repair."""

import numpy as np
import pytest

from src.research.retrospective_regime import recover_regime_categories


def test_fifth_state_is_recoverable_and_zero_row_is_not_state_zero():
    stored = np.array([[0, 0, 0, 0], [0.1, 0.2, 0.1, 0.2], [0.7, 0.1, 0.1, 0.1]])
    original = stored.copy()
    ids, evidence = recover_regime_categories(stored, declared_k=5)
    assert ids.tolist() == [4, 4, 0]
    assert evidence["changed_argmax_sessions"] == 2
    assert evidence["session_counts"] == [1, 0, 0, 0, 2]
    assert evidence["full_probabilities"][1] == pytest.approx([0.1, 0.2, 0.1, 0.2, 0.4])
    assert evidence["method"] == "RESIDUAL_FIFTH_COORDINATE_DESCRIPTIVE_ONLY"
    np.testing.assert_array_equal(stored, original)
    assert evidence["training_inputs_modified"] is False
    assert evidence["independent_hmm_forward_recomputed"] is False


def test_float32_rounding_is_disclosed_not_renormalized():
    stored = np.array([[0.7, 0.1, 0.1, 0.1]], dtype=np.float32)
    ids, evidence = recover_regime_categories(stored, declared_k=5)
    assert ids.tolist() == [0]
    assert evidence["full_probabilities"][0][:4] == stored.astype(float).tolist()[0]
    assert evidence["max_sum_excess"] < 1e-6


def test_full_four_state_contract_is_not_reinterpreted_as_five():
    ids, evidence = recover_regime_categories([[0.1, 0.2, 0.3, 0.4]], declared_k=4)
    assert ids.tolist() == [3]
    assert evidence["method"] == "FULL_STORED_POSTERIOR"
    assert len(evidence["full_probabilities"][0]) == 4


def test_smaller_model_requires_zero_padding():
    ids, _ = recover_regime_categories([[0.3, 0.7, 0, 0]], declared_k=2)
    assert ids.tolist() == [1]
    with pytest.raises(ValueError, match="padding"):
        recover_regime_categories([[0.3, 0.6, 0.1, 0]], declared_k=2)


@pytest.mark.parametrize("k", [None, True, 1, 6, 5.0, "5"])
def test_missing_or_unsupported_metadata_cannot_silently_guess_k(k):
    with pytest.raises(ValueError, match="declared_k"):
        recover_regime_categories([[0, 0, 0, 0]], declared_k=k)


@pytest.mark.parametrize(
    "stored",
    [
        [],
        [0, 0, 0, 0],
        [[0, 0, 0]],
        [[float("nan"), 0, 0, 0]],
        [[float("inf"), 0, 0, 0]],
        [[-0.01, 0, 0, 0]],
        [[1.01, 0, 0, 0]],
        [[0.6, 0.6, 0, 0]],
    ],
)
def test_invalid_posteriors_fail_closed(stored):
    with pytest.raises(ValueError):
        recover_regime_categories(stored, declared_k=5)


def test_missing_mass_in_declared_full_model_is_not_silently_filled():
    with pytest.raises(ValueError, match="sum"):
        recover_regime_categories([[0, 0, 0, 0]], declared_k=4)


def test_tie_convention_is_explicit():
    ids, evidence = recover_regime_categories([[0.5, 0.5, 0, 0]], declared_k=5)
    assert ids.tolist() == [0]
    assert evidence["near_tie_sessions"] == 1
    assert evidence["tie_break"] == "lowest_state_id_numpy_argmax"
