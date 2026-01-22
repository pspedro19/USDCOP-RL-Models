"""
Tests para Action Contract SSOT.
"""
import pytest
import subprocess
from pathlib import Path

from src.core.contracts.action_contract import (
    Action,
    InvalidActionError,
    validate_model_output,
    MODEL_OUTPUT_CONTRACT,
    ACTION_COUNT,
    VALID_ACTIONS,
)


REPO_ROOT = Path(__file__).parent.parent.parent


class TestActionEnum:
    """Tests para Action enum."""

    def test_action_values_are_correct(self):
        assert Action.SELL.value == 0
        assert Action.HOLD.value == 1
        assert Action.BUY.value == 2

    def test_action_count_is_three(self):
        assert len(Action) == 3
        assert ACTION_COUNT == 3

    def test_from_int_valid_values(self):
        assert Action.from_int(0) == Action.SELL
        assert Action.from_int(1) == Action.HOLD
        assert Action.from_int(2) == Action.BUY

    def test_from_int_invalid_negative(self):
        with pytest.raises(InvalidActionError, match="Invalid action value: -1"):
            Action.from_int(-1)

    def test_from_int_invalid_too_large(self):
        with pytest.raises(InvalidActionError, match="Invalid action value: 3"):
            Action.from_int(3)

    def test_from_string_sell_aliases(self):
        for alias in ["sell", "SELL", "short", "SHORT", "s", "-1"]:
            assert Action.from_string(alias) == Action.SELL

    def test_from_string_hold_aliases(self):
        for alias in ["hold", "HOLD", "flat", "FLAT", "h", "0"]:
            assert Action.from_string(alias) == Action.HOLD

    def test_from_string_buy_aliases(self):
        for alias in ["buy", "BUY", "long", "LONG", "b", "1"]:
            assert Action.from_string(alias) == Action.BUY

    def test_from_string_invalid(self):
        with pytest.raises(InvalidActionError):
            Action.from_string("invalid")

    def test_from_model_output(self):
        assert Action.from_model_output([0.8, 0.1, 0.1]) == Action.SELL
        assert Action.from_model_output([0.1, 0.8, 0.1]) == Action.HOLD
        assert Action.from_model_output([0.1, 0.1, 0.8]) == Action.BUY

    def test_from_model_output_invalid_length(self):
        with pytest.raises(InvalidActionError, match="must have 3 values"):
            Action.from_model_output([0.5, 0.5])

    def test_to_signal(self):
        assert Action.SELL.to_signal() == "SHORT"
        assert Action.HOLD.to_signal() == "FLAT"
        assert Action.BUY.to_signal() == "LONG"

    def test_to_position(self):
        assert Action.SELL.to_position() == -1
        assert Action.HOLD.to_position() == 0
        assert Action.BUY.to_position() == 1

    def test_is_entry(self):
        assert Action.BUY.is_entry is True
        assert Action.SELL.is_entry is True
        assert Action.HOLD.is_entry is False

    def test_direction(self):
        assert Action.SELL.direction == -1
        assert Action.HOLD.direction == 0
        assert Action.BUY.direction == 1


class TestModelOutputContract:
    """Tests para ModelOutputContract."""

    def test_validate_valid_output(self):
        is_valid, errors = MODEL_OUTPUT_CONTRACT.validate_output(action=0, confidence=0.85)
        assert is_valid
        assert len(errors) == 0

    def test_validate_invalid_action(self):
        is_valid, errors = MODEL_OUTPUT_CONTRACT.validate_output(action=5, confidence=0.5)
        assert not is_valid
        assert any("Invalid action" in e for e in errors)

    def test_validate_confidence_negative(self):
        is_valid, errors = MODEL_OUTPUT_CONTRACT.validate_output(action=1, confidence=-0.5)
        assert not is_valid
        assert any("out of range" in e for e in errors)

    def test_validate_confidence_above_one(self):
        is_valid, errors = MODEL_OUTPUT_CONTRACT.validate_output(action=1, confidence=1.5)
        assert not is_valid

    def test_validate_with_valid_probs(self):
        is_valid, errors = MODEL_OUTPUT_CONTRACT.validate_output(
            action=1, confidence=0.7, action_probs=[0.1, 0.7, 0.2]
        )
        assert is_valid

    def test_validate_probs_wrong_length(self):
        is_valid, errors = MODEL_OUTPUT_CONTRACT.validate_output(
            action=1, confidence=0.5, action_probs=[0.5, 0.5]
        )
        assert not is_valid

    def test_validate_probs_dont_sum_to_one(self):
        is_valid, errors = MODEL_OUTPUT_CONTRACT.validate_output(
            action=1, confidence=0.5, action_probs=[0.5, 0.5, 0.5]
        )
        assert not is_valid


class TestValidateModelOutput:
    """Tests para función validate_model_output."""

    def test_valid_returns_true(self):
        assert validate_model_output(action=0, confidence=0.9) is True
        assert validate_model_output(action=1, confidence=0.5) is True
        assert validate_model_output(action=2, confidence=0.1) is True

    def test_invalid_raises_by_default(self):
        with pytest.raises(InvalidActionError, match="validation failed"):
            validate_model_output(action=5, confidence=0.5)

    def test_invalid_returns_false_when_raise_disabled(self):
        result = validate_model_output(action=5, confidence=0.5, raise_on_error=False)
        assert result is False

    def test_boundary_confidence_values(self):
        assert validate_model_output(action=1, confidence=0.0) is True
        assert validate_model_output(action=1, confidence=1.0) is True
