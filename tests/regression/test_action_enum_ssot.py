"""
Test Action Enum SSOT Compliance
================================
Regression tests to ensure the Action enum remains consistent with the
trained model output mapping.

Contract: CTR-ACTION-001
SSOT Location: src/core/contracts/action_contract.py

CRITICAL: The model produces output with shape (3,) = [P(SELL), P(HOLD), P(BUY)]
         argmax produces: 0=SELL, 1=HOLD, 2=BUY
         This mapping is IMMUTABLE and corresponds to the trained model.
"""

import subprocess
import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.contracts import (
    Action,
    ACTION_COUNT,
    ACTION_NAMES,
    VALID_ACTIONS,
    ACTION_SELL,
    ACTION_HOLD,
    ACTION_BUY,
)


class TestActionEnumValues:
    """Test that Action enum values match the trained model output mapping."""

    def test_action_sell_is_zero(self):
        """
        SELL action MUST be 0.

        Rationale: The trained PPO model outputs probabilities as [P(SELL), P(HOLD), P(BUY)].
        When argmax returns 0, it means SELL has the highest probability.
        """
        assert Action.SELL == 0, f"SELL should be 0, got {Action.SELL}"
        assert Action.SELL.value == 0, f"SELL.value should be 0, got {Action.SELL.value}"
        assert ACTION_SELL == 0, f"ACTION_SELL constant should be 0, got {ACTION_SELL}"

    def test_action_hold_is_one(self):
        """
        HOLD action MUST be 1.

        Rationale: The trained PPO model outputs probabilities as [P(SELL), P(HOLD), P(BUY)].
        When argmax returns 1, it means HOLD has the highest probability.
        """
        assert Action.HOLD == 1, f"HOLD should be 1, got {Action.HOLD}"
        assert Action.HOLD.value == 1, f"HOLD.value should be 1, got {Action.HOLD.value}"
        assert ACTION_HOLD == 1, f"ACTION_HOLD constant should be 1, got {ACTION_HOLD}"

    def test_action_buy_is_two(self):
        """
        BUY action MUST be 2.

        Rationale: The trained PPO model outputs probabilities as [P(SELL), P(HOLD), P(BUY)].
        When argmax returns 2, it means BUY has the highest probability.
        """
        assert Action.BUY == 2, f"BUY should be 2, got {Action.BUY}"
        assert Action.BUY.value == 2, f"BUY.value should be 2, got {Action.BUY.value}"
        assert ACTION_BUY == 2, f"ACTION_BUY constant should be 2, got {ACTION_BUY}"

    def test_action_count_is_three(self):
        """
        There MUST be exactly 3 actions: SELL, HOLD, BUY.

        Rationale: The PPO model was trained with a discrete action space of size 3.
        Adding or removing actions would break model compatibility.
        """
        assert ACTION_COUNT == 3, f"ACTION_COUNT should be 3, got {ACTION_COUNT}"
        assert len(Action) == 3, f"Action enum should have 3 members, got {len(Action)}"
        assert len(VALID_ACTIONS) == 3, f"VALID_ACTIONS should have 3 elements, got {len(VALID_ACTIONS)}"

    def test_action_values_are_contiguous(self):
        """
        Action values MUST be contiguous integers starting from 0.

        Rationale: The model uses these values as array indices.
        Non-contiguous values would cause index errors.
        """
        action_values = sorted([a.value for a in Action])
        expected = [0, 1, 2]
        assert action_values == expected, (
            f"Action values must be contiguous [0, 1, 2], got {action_values}"
        )

    def test_action_names_mapping(self):
        """
        ACTION_NAMES constant MUST map correctly to action values.

        Rationale: This mapping is used for logging and debugging.
        Incorrect mapping would cause confusion in production monitoring.
        """
        assert ACTION_NAMES[0] == "SELL", f"ACTION_NAMES[0] should be 'SELL', got {ACTION_NAMES[0]}"
        assert ACTION_NAMES[1] == "HOLD", f"ACTION_NAMES[1] should be 'HOLD', got {ACTION_NAMES[1]}"
        assert ACTION_NAMES[2] == "BUY", f"ACTION_NAMES[2] should be 'BUY', got {ACTION_NAMES[2]}"

        # Verify bidirectional consistency
        for action in Action:
            assert ACTION_NAMES[action.value] == action.name, (
                f"Mismatch: ACTION_NAMES[{action.value}]={ACTION_NAMES[action.value]} "
                f"but Action.{action.name}.value={action.value}"
            )


def _search_pattern_in_files(search_dirs: list, pattern: str, file_glob: str = "*.py") -> list:
    """
    Search for a regex pattern in files using Python's built-in capabilities.

    This is a cross-platform alternative to grep.

    Args:
        search_dirs: List of directories to search
        pattern: Regex pattern to search for
        file_glob: File pattern to match (default: *.py)

    Returns:
        List of tuples (filepath, line_number, line_content)
    """
    import re
    import glob

    results = []
    compiled_pattern = re.compile(pattern)

    for search_dir in search_dirs:
        search_path = Path(search_dir)
        if not search_path.exists():
            continue

        # Use glob to find all Python files
        for filepath in search_path.rglob(file_glob):
            # Skip pycache directories
            if "__pycache__" in str(filepath):
                continue

            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        if compiled_pattern.search(line):
                            results.append((str(filepath), line_num, line.strip()))
            except (IOError, OSError):
                continue

    return results


class TestActionEnumUniqueness:
    """Test that Action class definition is unique in the codebase."""

    def test_no_duplicate_action_class_definitions(self):
        """
        There MUST be only ONE 'class Action(IntEnum)' definition in src/ and services/.

        Rationale: Multiple Action class definitions would cause import conflicts
        and potentially different action mappings in different modules.

        SSOT Location: src/core/contracts/action_contract.py
        """
        search_dirs = [
            str(PROJECT_ROOT / "src"),
            str(PROJECT_ROOT / "services"),
        ]

        # Search for Action(IntEnum) class definitions
        pattern = r"class\s+Action\s*\(\s*IntEnum\s*\)"
        results = _search_pattern_in_files(search_dirs, pattern)

        # Filter to only actual class definitions (not test files, not comments)
        actual_definitions = [
            (filepath, line_num, content)
            for filepath, line_num, content in results
            if "test_" not in filepath.lower()
            and not content.strip().startswith("#")
        ]

        expected_file = "action_contract.py"

        if len(actual_definitions) == 0:
            pytest.fail(
                f"No Action(IntEnum) class found! Expected in src/core/contracts/{expected_file}"
            )
        elif len(actual_definitions) > 1:
            pytest.fail(
                f"Multiple Action(IntEnum) definitions found (SSOT violation):\n"
                f"Expected only in: src/core/contracts/{expected_file}\n"
                f"Found in:\n" + "\n".join(
                    f"  - {fp}:{ln}: {c}" for fp, ln, c in actual_definitions
                )
            )
        else:
            # Verify it's in the expected location
            filepath = actual_definitions[0][0]
            assert expected_file in filepath, (
                f"Action(IntEnum) should be defined in {expected_file}, "
                f"but found in: {filepath}"
            )


class TestActionEnumMethods:
    """Test Action enum conversion methods work correctly."""

    def test_from_int_valid_values(self):
        """Action.from_int should correctly convert valid integers."""
        assert Action.from_int(0) == Action.SELL
        assert Action.from_int(1) == Action.HOLD
        assert Action.from_int(2) == Action.BUY

    def test_from_int_invalid_values(self):
        """Action.from_int should raise InvalidActionError for invalid integers."""
        from src.core.contracts import InvalidActionError

        with pytest.raises(InvalidActionError):
            Action.from_int(-1)

        with pytest.raises(InvalidActionError):
            Action.from_int(3)

        with pytest.raises(InvalidActionError):
            Action.from_int(100)

    def test_from_model_output_argmax(self):
        """Action.from_model_output should correctly identify the action with highest probability."""
        # SELL has highest probability
        assert Action.from_model_output([0.8, 0.1, 0.1]) == Action.SELL

        # HOLD has highest probability
        assert Action.from_model_output([0.1, 0.8, 0.1]) == Action.HOLD

        # BUY has highest probability
        assert Action.from_model_output([0.1, 0.1, 0.8]) == Action.BUY

    def test_to_position_mapping(self):
        """Action.to_position should map to correct position values."""
        assert Action.SELL.to_position() == -1, "SELL should map to position -1"
        assert Action.HOLD.to_position() == 0, "HOLD should map to position 0"
        assert Action.BUY.to_position() == 1, "BUY should map to position 1"
