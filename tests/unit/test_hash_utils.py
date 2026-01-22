"""
Unit tests for hash utilities.

Tests Phase 1.1: MLflow Hash Logging (CTR-HASH-001)

Author: Trading Team
Version: 1.0.0
Created: 2026-01-17
"""

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from utils.hash_utils import (
    compute_file_hash,
    compute_json_hash,
    compute_string_hash,
    compute_feature_order_hash,
    HashResult,
)


class TestComputeFileHash:
    """Tests for compute_file_hash function."""

    def test_computes_correct_hash_for_binary_file(self, tmp_path):
        """Should compute correct SHA256 hash for binary file."""
        # Create test file
        test_file = tmp_path / "test.bin"
        content = b"Hello, World!"
        test_file.write_bytes(content)

        # Compute expected hash
        expected = hashlib.sha256(content).hexdigest()

        # Test
        result = compute_file_hash(test_file)

        assert result.full_hash == expected
        assert result.short_hash == expected[:16]
        assert result.algorithm == "sha256"
        assert result.source_type == "file"

    def test_raises_for_nonexistent_file(self):
        """Should raise FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            compute_file_hash("/nonexistent/path/file.txt")

    def test_returns_hash_result_object(self, tmp_path):
        """Should return HashResult with all fields populated."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = compute_file_hash(test_file)

        assert isinstance(result, HashResult)
        assert len(result.full_hash) == 64  # SHA256 = 64 hex chars
        assert len(result.short_hash) == 16
        assert result.source_path == str(test_file)


class TestComputeJsonHash:
    """Tests for compute_json_hash function."""

    def test_computes_consistent_hash_regardless_of_key_order(self, tmp_path):
        """Should compute same hash for same content with different key order."""
        # Create two JSON files with same content but different key order
        data1 = {"b": 2, "a": 1, "c": 3}
        data2 = {"a": 1, "c": 3, "b": 2}

        file1 = tmp_path / "test1.json"
        file2 = tmp_path / "test2.json"

        file1.write_text(json.dumps(data1))
        file2.write_text(json.dumps(data2))

        result1 = compute_json_hash(file1)
        result2 = compute_json_hash(file2)

        # Both should have same hash due to canonical sorting
        assert result1.full_hash == result2.full_hash

    def test_computes_correct_hash_for_norm_stats(self, tmp_path):
        """Should compute correct hash for norm_stats-like JSON."""
        norm_stats = {
            "rsi_9": {"mean": 50.0, "std": 15.0, "min": 0.0, "max": 100.0},
            "atr_pct": {"mean": 0.01, "std": 0.005, "min": 0.0, "max": 0.1},
            "_metadata": {"version": "v1", "created_at": "2026-01-17"},
        }

        test_file = tmp_path / "norm_stats.json"
        test_file.write_text(json.dumps(norm_stats))

        result = compute_json_hash(test_file)

        # Compute expected hash
        canonical = json.dumps(norm_stats, sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256(canonical.encode()).hexdigest()

        assert result.full_hash == expected
        assert result.source_type == "json"

    def test_raises_for_invalid_json(self, tmp_path):
        """Should raise for invalid JSON file."""
        test_file = tmp_path / "invalid.json"
        test_file.write_text("not valid json {")

        with pytest.raises(json.JSONDecodeError):
            compute_json_hash(test_file)


class TestComputeStringHash:
    """Tests for compute_string_hash function."""

    def test_computes_correct_hash_for_string(self):
        """Should compute correct SHA256 hash for string."""
        content = "test string"
        expected = hashlib.sha256(content.encode()).hexdigest()

        result = compute_string_hash(content)

        assert result.full_hash == expected
        assert result.source_type == "string"

    def test_different_strings_produce_different_hashes(self):
        """Should produce different hashes for different strings."""
        result1 = compute_string_hash("string1")
        result2 = compute_string_hash("string2")

        assert result1.full_hash != result2.full_hash


class TestComputeFeatureOrderHash:
    """Tests for compute_feature_order_hash function."""

    def test_computes_hash_for_feature_list(self):
        """Should compute hash for list of features."""
        features = ["rsi_9", "atr_pct", "adx_14", "position", "time_normalized"]

        result = compute_feature_order_hash(features)

        # Compute expected
        expected = hashlib.sha256(",".join(features).encode()).hexdigest()

        assert result.full_hash == expected
        assert result.source_type == "feature_order"

    def test_computes_hash_for_feature_tuple(self):
        """Should compute same hash for tuple as list."""
        features_list = ["a", "b", "c"]
        features_tuple = ("a", "b", "c")

        result1 = compute_feature_order_hash(features_list)
        result2 = compute_feature_order_hash(features_tuple)

        assert result1.full_hash == result2.full_hash

    def test_order_matters(self):
        """Feature order should affect hash."""
        features1 = ["a", "b", "c"]
        features2 = ["b", "a", "c"]

        result1 = compute_feature_order_hash(features1)
        result2 = compute_feature_order_hash(features2)

        assert result1.full_hash != result2.full_hash

    def test_contract_feature_order_produces_consistent_hash(self):
        """Should produce consistent hash for FEATURE_ORDER contract."""
        # The canonical 15-feature order from CTR-FEAT-001
        feature_order = (
            "log_ret_5m", "log_ret_1h", "log_ret_4h",
            "rsi_9", "atr_pct", "adx_14",
            "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
            "brent_change_1d", "rate_spread", "usdmxn_change_1d",
            "position", "time_normalized"
        )

        result = compute_feature_order_hash(feature_order)

        # Should always produce the same hash for the same order
        expected_string = ",".join(feature_order)
        expected_hash = hashlib.sha256(expected_string.encode()).hexdigest()

        assert result.full_hash == expected_hash
        assert len(result.short_hash) == 16


class TestHashResult:
    """Tests for HashResult dataclass."""

    def test_to_dict_returns_all_fields(self):
        """Should return dictionary with all fields."""
        result = HashResult(
            full_hash="abc123" * 10 + "abcd",  # 64 chars
            short_hash="abc123" + "def456ab",
            algorithm="sha256",
            source_path="/path/to/file",
            source_type="file",
        )

        as_dict = result.to_dict()

        assert "full_hash" in as_dict
        assert "short_hash" in as_dict
        assert "algorithm" in as_dict
        assert "source_path" in as_dict
        assert "source_type" in as_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
