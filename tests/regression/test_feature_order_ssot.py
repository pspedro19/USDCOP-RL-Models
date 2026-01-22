"""
Test Feature Order SSOT Compliance
==================================
Regression tests to ensure the feature order remains consistent with the
trained model observation space.

Contract: CTR-FEATURE-001
SSOT Location: src/core/contracts/feature_contract.py

CRITICAL: The model was trained with a 15-dimensional observation space.
         The feature order is IMMUTABLE and must match exactly:
         [0-12]: 13 market features
         [13]: position
         [14]: time_normalized
"""

import subprocess
import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.contracts import (
    FEATURE_ORDER,
    OBSERVATION_DIM,
    FEATURE_CONTRACT,
    FEATURE_SPECS,
)


class TestFeatureOrderStructure:
    """Test that feature order structure matches the trained model."""

    def test_feature_order_has_exactly_15_elements(self):
        """
        FEATURE_ORDER MUST have exactly 15 elements.

        Rationale: The PPO model was trained with observation_space.shape = (15,).
        Adding or removing features would cause dimension mismatch errors.
        """
        assert len(FEATURE_ORDER) == 15, (
            f"FEATURE_ORDER should have exactly 15 elements, got {len(FEATURE_ORDER)}\n"
            f"Current features: {FEATURE_ORDER}"
        )

    def test_observation_dim_matches_feature_count(self):
        """
        OBSERVATION_DIM MUST equal len(FEATURE_ORDER).

        Rationale: These constants are used in different parts of the codebase.
        A mismatch would cause silent errors in observation building.
        """
        assert OBSERVATION_DIM == len(FEATURE_ORDER), (
            f"OBSERVATION_DIM ({OBSERVATION_DIM}) != len(FEATURE_ORDER) ({len(FEATURE_ORDER)})"
        )
        assert OBSERVATION_DIM == 15, (
            f"OBSERVATION_DIM should be 15, got {OBSERVATION_DIM}"
        )

    def test_no_session_progress_exists(self):
        """
        'session_progress' feature MUST NOT exist in FEATURE_ORDER.

        Rationale: This feature was deprecated in favor of 'time_normalized'.
        The model was trained without session_progress.
        """
        deprecated_features = ["session_progress"]
        for feature in deprecated_features:
            assert feature not in FEATURE_ORDER, (
                f"Deprecated feature '{feature}' found in FEATURE_ORDER. "
                f"This was replaced by 'time_normalized'."
            )

    def test_time_normalized_is_last_feature(self):
        """
        'time_normalized' MUST be the last feature (index 14).

        Rationale: State features (position, time_normalized) are appended
        after market features. time_normalized must be at index 14.
        """
        assert FEATURE_ORDER[-1] == "time_normalized", (
            f"Last feature should be 'time_normalized', got '{FEATURE_ORDER[-1]}'"
        )
        assert FEATURE_ORDER[14] == "time_normalized", (
            f"Feature at index 14 should be 'time_normalized', got '{FEATURE_ORDER[14]}'"
        )

    def test_position_is_second_to_last(self):
        """
        'position' MUST be at index 13 (second to last).

        Rationale: State features are ordered as [position, time_normalized].
        Position must be at index 13.
        """
        assert FEATURE_ORDER[13] == "position", (
            f"Feature at index 13 should be 'position', got '{FEATURE_ORDER[13]}'"
        )
        assert FEATURE_ORDER[-2] == "position", (
            f"Second to last feature should be 'position', got '{FEATURE_ORDER[-2]}'"
        )

    def test_first_feature_is_returns_1h(self):
        """
        First feature should be 'log_ret_5m'.

        Rationale: The feature order starts with price returns,
        followed by technical indicators, then macro features, then state.
        """
        # Note: Based on the actual contract, first feature is log_ret_5m
        expected_first = "log_ret_5m"
        assert FEATURE_ORDER[0] == expected_first, (
            f"First feature should be '{expected_first}', got '{FEATURE_ORDER[0]}'"
        )

    def test_no_duplicate_features(self):
        """
        FEATURE_ORDER MUST NOT contain duplicate feature names.

        Rationale: Each feature must be unique to avoid overwrites
        and ensure correct observation building.
        """
        unique_features = set(FEATURE_ORDER)
        assert len(unique_features) == len(FEATURE_ORDER), (
            f"Duplicate features found in FEATURE_ORDER!\n"
            f"Total: {len(FEATURE_ORDER)}, Unique: {len(unique_features)}\n"
            f"Duplicates: {[f for f in FEATURE_ORDER if FEATURE_ORDER.count(f) > 1]}"
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

    results = []
    compiled_pattern = re.compile(pattern)

    for search_dir in search_dirs:
        search_path = Path(search_dir)
        if not search_path.exists():
            continue

        for filepath in search_path.rglob(file_glob):
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


class TestFeatureOrderConsistency:
    """Test that feature order is consistent across the codebase."""

    def test_no_duplicate_feature_order_definitions(self):
        """
        There MUST be only ONE FEATURE_ORDER definition in src/core/contracts/.

        Rationale: Multiple definitions would cause import conflicts
        and potentially different feature orders in different modules.

        SSOT Location: src/core/contracts/feature_contract.py
        """
        search_dir = str(PROJECT_ROOT / "src" / "core" / "contracts")

        if not Path(search_dir).exists():
            pytest.skip(f"Directory {search_dir} does not exist")

        # Search for FEATURE_ORDER definitions with Final type hint
        pattern = r"FEATURE_ORDER\s*:\s*Final"
        results = _search_pattern_in_files([search_dir], pattern)

        # Filter to actual definitions
        actual_definitions = [
            (filepath, line_num, content)
            for filepath, line_num, content in results
            if not content.strip().startswith("#")
        ]

        if len(actual_definitions) > 1:
            pytest.fail(
                f"Multiple FEATURE_ORDER definitions found (SSOT violation):\n"
                + "\n".join(f"  - {fp}:{ln}: {c}" for fp, ln, c in actual_definitions)
            )

    def test_no_session_progress_in_codebase(self):
        """
        'session_progress' MUST NOT be used as a feature name in FEATURE_ORDER.

        Note: session_progress may still be used as a parameter name in some
        adapters/builders for backwards compatibility, as long as it maps to
        'time_normalized' internally.

        Rationale: The feature name 'session_progress' was deprecated and
        replaced by 'time_normalized' in the SSOT contracts.
        """
        # This test verifies that session_progress is NOT in FEATURE_ORDER
        # (we already test this in test_no_session_progress_exists above)

        # Additionally, check that no file defines session_progress as a
        # feature in a feature order list
        search_dirs = [
            str(PROJECT_ROOT / "src" / "core" / "contracts"),
        ]

        # Look for session_progress in feature order definitions
        pattern = r'["\']session_progress["\']'
        results = _search_pattern_in_files(search_dirs, pattern)

        # Filter to only feature order/feature spec definitions
        violations = [
            (filepath, line_num, content)
            for filepath, line_num, content in results
            if "FEATURE_ORDER" in content or "FEATURE_SPECS" in content
            and not content.strip().startswith("#")
        ]

        if violations:
            pytest.fail(
                f"'session_progress' found in feature contract definitions:\n"
                + "\n".join(f"  - {fp}:{ln}: {c}" for fp, ln, c in violations)
            )


class TestFeatureSpecs:
    """Test that feature specifications are complete and correct."""

    def test_all_features_have_specs(self):
        """Every feature in FEATURE_ORDER must have a corresponding spec."""
        for feature in FEATURE_ORDER:
            assert feature in FEATURE_SPECS, (
                f"Feature '{feature}' is in FEATURE_ORDER but missing from FEATURE_SPECS"
            )

    def test_spec_indices_match_order(self):
        """Feature spec indices must match their position in FEATURE_ORDER."""
        for idx, feature in enumerate(FEATURE_ORDER):
            spec = FEATURE_SPECS.get(feature)
            if spec:
                assert spec.index == idx, (
                    f"Feature '{feature}' has spec.index={spec.index} "
                    f"but is at position {idx} in FEATURE_ORDER"
                )

    def test_state_features_not_normalized(self):
        """State features (position, time_normalized) should not require normalization."""
        state_features = ["position", "time_normalized"]
        for feature in state_features:
            spec = FEATURE_SPECS.get(feature)
            if spec:
                assert not spec.requires_normalization, (
                    f"State feature '{feature}' should not require normalization"
                )


class TestFeatureContract:
    """Test the FeatureContract class methods."""

    def test_contract_observation_dim(self):
        """Contract observation_dim should match OBSERVATION_DIM constant."""
        assert FEATURE_CONTRACT.observation_dim == OBSERVATION_DIM

    def test_contract_feature_order(self):
        """Contract feature_order should match FEATURE_ORDER constant."""
        assert FEATURE_CONTRACT.feature_order == FEATURE_ORDER

    def test_get_feature_index(self):
        """get_feature_index should return correct indices."""
        assert FEATURE_CONTRACT.get_feature_index("log_ret_5m") == 0
        assert FEATURE_CONTRACT.get_feature_index("position") == 13
        assert FEATURE_CONTRACT.get_feature_index("time_normalized") == 14

    def test_get_feature_index_invalid(self):
        """get_feature_index should raise error for unknown features."""
        from src.core.contracts import FeatureContractError

        with pytest.raises(FeatureContractError):
            FEATURE_CONTRACT.get_feature_index("unknown_feature")

        with pytest.raises(FeatureContractError):
            FEATURE_CONTRACT.get_feature_index("session_progress")
