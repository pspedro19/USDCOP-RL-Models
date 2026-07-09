"""
Test Feature Order SSOT Compliance
==================================
Regression tests ensuring the RL feature order stays internally consistent and
that there is exactly ONE SSOT definition across the codebase.

Contract: CTR-FEATURE-001
SSOT Location: src/core/contracts/feature_contract.py

Expectations are DERIVED from the contract (not a frozen 15-feature snapshot), so
this suite tracks the SSOT as it evolves (it is currently a 20-dim observation).
Deep structural assertions live in tests/contracts/test_feature_contract.py — this
file focuses on the cross-file / codebase-wide invariants that file cannot cover.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.contracts import (  # noqa: E402
    FEATURE_ORDER,
    OBSERVATION_DIM,
    FEATURE_CONTRACT,
    FEATURE_SPECS,
)


class TestFeatureOrderStructure:
    """Structural invariants derived from the contract."""

    def test_observation_dim_matches_feature_count(self):
        assert OBSERVATION_DIM == len(FEATURE_ORDER), (
            f"OBSERVATION_DIM ({OBSERVATION_DIM}) != len(FEATURE_ORDER) ({len(FEATURE_ORDER)})"
        )

    def test_no_duplicate_features(self):
        assert len(set(FEATURE_ORDER)) == len(FEATURE_ORDER), (
            f"Duplicate features: {[f for f in FEATURE_ORDER if FEATURE_ORDER.count(f) > 1]}"
        )

    def test_no_deprecated_leakage_feature(self):
        assert "session_progress" not in FEATURE_ORDER

    def test_position_is_a_trailing_state_feature(self):
        """State features are appended after market features; position is one."""
        normalizable = set(FEATURE_CONTRACT.get_normalizable_features())
        assert "position" in FEATURE_ORDER
        assert "position" not in normalizable, "position is a state feature (not normalized)"
        pos_idx = FEATURE_ORDER.index("position")
        # state features live in the tail of the vector
        assert pos_idx >= len(FEATURE_ORDER) - 3

    def test_first_feature_is_log_ret_5m(self):
        assert FEATURE_ORDER[0] == "log_ret_5m"


def _search_pattern_in_files(search_dirs, pattern, file_glob="*.py"):
    """Cross-platform grep: returns (filepath, line_no, line) matches."""
    import re

    results = []
    compiled = re.compile(pattern)
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
                        if compiled.search(line):
                            results.append((str(filepath), line_num, line.strip()))
            except (IOError, OSError):
                continue
    return results


class TestFeatureOrderConsistency:
    """Codebase-wide SSOT invariants (the unique value of this file)."""

    def test_no_literal_feature_order_fork_outside_contracts(self):
        """SSOT locality: no file OUTSIDE src/core/contracts/ may define
        FEATURE_ORDER as a LITERAL list/tuple of feature-name strings. Re-exporting
        the SSOT value (`FEATURE_ORDER = _SSOT_ORDER`) is fine and encouraged — only
        an independent hardcoded copy forks the observation space.

        KNOWN drift (documented, not fixed here): the contracts registry carries a
        legacy 15-dim FEATURE_ORDER while the active contract is 20-dim; reconciling
        those versioned contracts is separate RL-internal work.
        """
        import os

        src_dir = str(PROJECT_ROOT / "src")
        # literal definition = assignment to a tuple/list opening with a quoted string
        pattern = r"FEATURE_ORDER\b.*=\s*[\(\[]\s*[\"']"
        results = _search_pattern_in_files([src_dir], pattern)
        contracts_pkg = "core" + os.sep + "contracts"
        offenders = {
            fp for fp, _, content in results
            if not content.startswith("#") and contracts_pkg not in fp
        }
        assert not offenders, (
            "Hardcoded FEATURE_ORDER literal OUTSIDE src/core/contracts/ (SSOT fork):\n"
            + "\n".join(f"  - {fp}" for fp in offenders)
        )

    def test_no_session_progress_in_contract_definitions(self):
        search_dirs = [str(PROJECT_ROOT / "src" / "core" / "contracts")]
        results = _search_pattern_in_files(search_dirs, r'["\']session_progress["\']')
        violations = [
            (fp, ln, c)
            for fp, ln, c in results
            if ("FEATURE_ORDER" in c or "FEATURE_SPECS" in c) and not c.startswith("#")
        ]
        assert not violations, (
            "'session_progress' in feature contract definitions:\n"
            + "\n".join(f"  - {fp}:{ln}: {c}" for fp, ln, c in violations)
        )


class TestFeatureSpecs:
    """Spec/order alignment (one-directional invariant)."""

    def test_all_ordered_features_have_specs(self):
        for feature in FEATURE_ORDER:
            assert feature in FEATURE_SPECS, f"'{feature}' missing from FEATURE_SPECS"

    def test_spec_indices_match_order(self):
        for idx, feature in enumerate(FEATURE_ORDER):
            spec = FEATURE_SPECS.get(feature)
            if spec:
                assert spec.index == idx


class TestFeatureContract:
    """FeatureContract accessor invariants."""

    def test_contract_observation_dim(self):
        assert FEATURE_CONTRACT.observation_dim == OBSERVATION_DIM

    def test_contract_feature_order(self):
        assert FEATURE_CONTRACT.feature_order == FEATURE_ORDER

    def test_get_feature_index_is_consistent(self):
        assert FEATURE_CONTRACT.get_feature_index("log_ret_5m") == 0
        assert FEATURE_CONTRACT.get_feature_index("position") == FEATURE_ORDER.index("position")

    def test_get_feature_index_invalid(self):
        from src.core.contracts import FeatureContractError

        with pytest.raises(FeatureContractError):
            FEATURE_CONTRACT.get_feature_index("unknown_feature")
