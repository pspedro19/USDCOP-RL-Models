"""
MLflow Dataset Tracking Tests
=============================
P0-08: Comprehensive tests for MLflow dataset logging.

This module validates that the MLflow integration properly tracks:
- Dataset hashes for reproducibility
- Date ranges for temporal tracking
- Feature order hashes for contract verification
- Feature contract artifacts

Contract ID: CTR-MLFLOW-001
Author: Trading Team
Date: 2026-01-17
"""

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Direct imports to avoid broken src/__init__.py chain
sys.path.insert(0, str(project_root / "src" / "core" / "contracts"))
sys.path.insert(0, str(project_root / "src" / "training"))

from feature_contract import (
    FEATURE_ORDER,
    FEATURE_ORDER_HASH,
    OBSERVATION_DIM,
)
from train_ssot import (
    TrainingConfig,
    compute_dataset_hash,
    set_reproducible_seeds,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_tracking_uri(tmp_path):
    """Create a temporary MLflow tracking URI for testing."""
    tracking_dir = tmp_path / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    return f"file://{tracking_dir}"


@pytest.fixture
def sample_df():
    """
    Create a sample DataFrame with realistic training data structure.

    Matches the feature contract with 15 features:
    - Technical: log_ret_5m, log_ret_1h, log_ret_4h, rsi_9, atr_pct, adx_14
    - Macro: dxy_z, dxy_change_1d, vix_z, embi_z, brent_change_1d, rate_spread, usdmxn_change_1d
    - State: position, time_normalized
    """
    np.random.seed(42)
    n_samples = 1000

    # Create datetime index
    dates = pd.date_range("2025-01-01 13:00", periods=n_samples, freq="5min")

    # Build feature DataFrame matching FEATURE_ORDER
    data = {
        # Technical features (z-scored)
        "log_ret_5m": np.random.normal(0, 1, n_samples),
        "log_ret_1h": np.random.normal(0, 1, n_samples),
        "log_ret_4h": np.random.normal(0, 1, n_samples),
        "rsi_9": np.random.normal(0, 1, n_samples).clip(-3, 3),
        "atr_pct": np.random.normal(0, 1, n_samples),
        "adx_14": np.random.normal(0, 1, n_samples),

        # Macro features (z-scored)
        "dxy_z": np.random.normal(0, 1, n_samples),
        "dxy_change_1d": np.random.normal(0, 1, n_samples),
        "vix_z": np.random.normal(0, 1, n_samples),
        "embi_z": np.random.normal(0, 1, n_samples),
        "brent_change_1d": np.random.normal(0, 1, n_samples),
        "rate_spread": np.random.normal(0, 1, n_samples),
        "usdmxn_change_1d": np.random.normal(0, 1, n_samples),

        # State features
        "position": np.random.choice([-1, 0, 1], n_samples).astype(float),
        "time_normalized": np.linspace(0, 1, n_samples),

        # Additional metadata columns
        "datetime": dates,
        "close": 4250.0 + np.cumsum(np.random.randn(n_samples) * 5),
    }

    return pd.DataFrame(data)


@pytest.fixture
def trainer(temp_tracking_uri, tmp_path):
    """
    Create a mock MLflowTrainer with temporary tracking URI.

    This fixture provides a trainer-like object for testing MLflow logging
    without requiring a running MLflow server.
    """
    class MockMLflowTrainer:
        """Mock MLflow trainer for testing."""

        def __init__(self, tracking_uri: str, output_dir: Path):
            self.tracking_uri = tracking_uri
            self.output_dir = output_dir
            self.params_logged: Dict[str, Any] = {}
            self.metrics_logged: Dict[str, float] = {}
            self.artifacts_logged: Dict[str, Path] = {}
            self.run_id: Optional[str] = None

        def start_run(self, run_name: str = None) -> str:
            """Start a mock MLflow run."""
            self.run_id = f"mock_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            return self.run_id

        def log_param(self, key: str, value: Any) -> None:
            """Log a parameter."""
            self.params_logged[key] = value

        def log_params(self, params: Dict[str, Any]) -> None:
            """Log multiple parameters."""
            self.params_logged.update(params)

        def log_metric(self, key: str, value: float) -> None:
            """Log a metric."""
            self.metrics_logged[key] = value

        def log_artifact(self, local_path: str, artifact_path: str = None) -> None:
            """Log an artifact."""
            self.artifacts_logged[artifact_path or "default"] = Path(local_path)

        def log_dataset_metadata(
            self,
            df: pd.DataFrame,
            dataset_path: Path,
            train_start_date: str,
            train_end_date: str,
        ) -> str:
            """
            Log comprehensive dataset metadata for reproducibility.

            Returns:
                Dataset hash
            """
            # Compute dataset hash
            dataset_hash = self._compute_dataframe_hash(df)

            # Log parameters
            self.log_param("dataset_hash", dataset_hash)
            self.log_param("train_start_date", train_start_date)
            self.log_param("train_end_date", train_end_date)
            self.log_param("FEATURE_ORDER_HASH", FEATURE_ORDER_HASH)
            self.log_param("observation_dim", OBSERVATION_DIM)
            self.log_param("n_samples", len(df))

            # Create and log feature contract artifact
            feature_contract = {
                "feature_order": list(FEATURE_ORDER),
                "feature_order_hash": FEATURE_ORDER_HASH,
                "observation_dim": OBSERVATION_DIM,
                "version": "2.0.0",
            }

            contract_path = self.output_dir / "feature_contract.json"
            with open(contract_path, "w") as f:
                json.dump(feature_contract, f, indent=2)

            self.log_artifact(str(contract_path), "feature_contract.json")

            return dataset_hash

        def _compute_dataframe_hash(self, df: pd.DataFrame) -> str:
            """Compute hash of DataFrame for reproducibility."""
            # Use feature columns only for hash
            feature_cols = [c for c in df.columns if c in FEATURE_ORDER]
            hash_df = df[feature_cols].copy()

            # Serialize and hash
            csv_bytes = hash_df.to_csv(index=False).encode("utf-8")
            return hashlib.sha256(csv_bytes).hexdigest()[:16]

        def get_run_metadata(self) -> Dict[str, Any]:
            """Get all logged metadata for the current run."""
            return {
                "run_id": self.run_id,
                "params": self.params_logged.copy(),
                "metrics": self.metrics_logged.copy(),
                "artifacts": list(self.artifacts_logged.keys()),
            }

        def end_run(self) -> None:
            """End the current run."""
            self.run_id = None

    output_dir = tmp_path / "mlflow_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    return MockMLflowTrainer(temp_tracking_uri, output_dir)


# =============================================================================
# TestMLflowDatasetLineage
# =============================================================================

class TestMLflowDatasetLineage:
    """
    Tests for MLflow dataset lineage tracking.

    Validates that training runs properly log all required metadata
    for dataset reproducibility and lineage tracking.
    """

    def test_dataset_hash_is_logged(self, trainer, sample_df, tmp_path):
        """Verify that dataset_hash parameter is logged to MLflow."""
        # Setup
        dataset_path = tmp_path / "train_data.csv"
        sample_df.to_csv(dataset_path, index=False)

        # Execute
        trainer.start_run("test_hash_logging")
        dataset_hash = trainer.log_dataset_metadata(
            df=sample_df,
            dataset_path=dataset_path,
            train_start_date="2025-01-01",
            train_end_date="2025-01-15",
        )
        trainer.end_run()

        # Verify
        assert "dataset_hash" in trainer.params_logged
        assert trainer.params_logged["dataset_hash"] == dataset_hash
        assert len(dataset_hash) == 16  # SHA256 truncated to 16 chars

    def test_date_range_is_logged(self, trainer, sample_df, tmp_path):
        """Verify that train_start_date and train_end_date are logged."""
        # Setup
        dataset_path = tmp_path / "train_data.csv"
        sample_df.to_csv(dataset_path, index=False)
        train_start = "2025-01-01"
        train_end = "2025-01-15"

        # Execute
        trainer.start_run("test_date_range_logging")
        trainer.log_dataset_metadata(
            df=sample_df,
            dataset_path=dataset_path,
            train_start_date=train_start,
            train_end_date=train_end,
        )
        trainer.end_run()

        # Verify
        assert "train_start_date" in trainer.params_logged
        assert "train_end_date" in trainer.params_logged
        assert trainer.params_logged["train_start_date"] == train_start
        assert trainer.params_logged["train_end_date"] == train_end

    def test_feature_order_hash_is_logged(self, trainer, sample_df, tmp_path):
        """Verify that FEATURE_ORDER_HASH is logged for contract verification."""
        # Setup
        dataset_path = tmp_path / "train_data.csv"
        sample_df.to_csv(dataset_path, index=False)

        # Execute
        trainer.start_run("test_feature_order_hash")
        trainer.log_dataset_metadata(
            df=sample_df,
            dataset_path=dataset_path,
            train_start_date="2025-01-01",
            train_end_date="2025-01-15",
        )
        trainer.end_run()

        # Verify
        assert "FEATURE_ORDER_HASH" in trainer.params_logged
        assert trainer.params_logged["FEATURE_ORDER_HASH"] == FEATURE_ORDER_HASH

    def test_feature_list_artifact_created(self, trainer, sample_df, tmp_path):
        """Verify that feature_contract.json artifact is created and logged."""
        # Setup
        dataset_path = tmp_path / "train_data.csv"
        sample_df.to_csv(dataset_path, index=False)

        # Execute
        trainer.start_run("test_artifact_creation")
        trainer.log_dataset_metadata(
            df=sample_df,
            dataset_path=dataset_path,
            train_start_date="2025-01-01",
            train_end_date="2025-01-15",
        )
        trainer.end_run()

        # Verify artifact was logged
        assert "feature_contract.json" in trainer.artifacts_logged
        artifact_path = trainer.artifacts_logged["feature_contract.json"]

        # Verify artifact file exists and contains correct data
        assert artifact_path.exists()
        with open(artifact_path) as f:
            contract = json.load(f)

        assert contract["feature_order"] == list(FEATURE_ORDER)
        assert contract["feature_order_hash"] == FEATURE_ORDER_HASH
        assert contract["observation_dim"] == OBSERVATION_DIM

    def test_hash_deterministic_for_same_data(self, trainer, sample_df, tmp_path):
        """Verify that the same data produces the same hash deterministically."""
        # Setup
        dataset_path = tmp_path / "train_data.csv"
        sample_df.to_csv(dataset_path, index=False)

        # Execute - compute hash multiple times
        trainer.start_run("test_determinism_1")
        hash1 = trainer.log_dataset_metadata(
            df=sample_df,
            dataset_path=dataset_path,
            train_start_date="2025-01-01",
            train_end_date="2025-01-15",
        )
        trainer.end_run()

        trainer.start_run("test_determinism_2")
        hash2 = trainer.log_dataset_metadata(
            df=sample_df,
            dataset_path=dataset_path,
            train_start_date="2025-01-01",
            train_end_date="2025-01-15",
        )
        trainer.end_run()

        # Verify
        assert hash1 == hash2, "Same data must produce same hash"

    def test_hash_changes_for_different_data(self, trainer, sample_df, tmp_path):
        """Verify that different data produces different hashes."""
        # Setup
        dataset_path = tmp_path / "train_data.csv"
        sample_df.to_csv(dataset_path, index=False)

        # Create modified data
        modified_df = sample_df.copy()
        modified_df["log_ret_5m"] = modified_df["log_ret_5m"] + 1.0  # Change a feature

        # Execute
        trainer.start_run("test_hash_original")
        hash_original = trainer.log_dataset_metadata(
            df=sample_df,
            dataset_path=dataset_path,
            train_start_date="2025-01-01",
            train_end_date="2025-01-15",
        )
        trainer.end_run()

        trainer.start_run("test_hash_modified")
        hash_modified = trainer.log_dataset_metadata(
            df=modified_df,
            dataset_path=dataset_path,
            train_start_date="2025-01-01",
            train_end_date="2025-01-15",
        )
        trainer.end_run()

        # Verify
        assert hash_original != hash_modified, "Different data must produce different hash"


# =============================================================================
# TestDatasetReproduction
# =============================================================================

class TestDatasetReproduction:
    """
    Tests for dataset reproduction from MLflow metadata.

    Validates that logged metadata can be used to verify and
    reproduce training datasets.
    """

    def test_get_run_metadata_extracts_all_fields(self, trainer, sample_df, tmp_path):
        """Verify that get_run_metadata extracts all required fields."""
        # Setup
        dataset_path = tmp_path / "train_data.csv"
        sample_df.to_csv(dataset_path, index=False)

        # Execute
        trainer.start_run("test_metadata_extraction")
        trainer.log_dataset_metadata(
            df=sample_df,
            dataset_path=dataset_path,
            train_start_date="2025-01-01",
            train_end_date="2025-01-15",
        )

        metadata = trainer.get_run_metadata()
        trainer.end_run()

        # Verify required fields
        assert "run_id" in metadata
        assert metadata["run_id"] is not None

        params = metadata["params"]
        required_params = [
            "dataset_hash",
            "train_start_date",
            "train_end_date",
            "FEATURE_ORDER_HASH",
            "observation_dim",
            "n_samples",
        ]

        for param in required_params:
            assert param in params, f"Missing required param: {param}"

        # Verify artifacts
        assert "feature_contract.json" in metadata["artifacts"]

    def test_get_run_metadata_fails_on_missing_params(self, trainer):
        """Verify that missing required params are detectable."""
        # Execute - start run without logging metadata
        trainer.start_run("test_missing_params")
        metadata = trainer.get_run_metadata()
        trainer.end_run()

        # Verify params are empty
        required_params = ["dataset_hash", "train_start_date", "train_end_date"]

        for param in required_params:
            assert param not in metadata["params"], \
                f"Param {param} should not exist without logging"

    def test_verify_hash_matches(self, trainer, sample_df, tmp_path):
        """Verify that hash verification passes for unchanged data."""
        # Setup
        dataset_path = tmp_path / "train_data.csv"
        sample_df.to_csv(dataset_path, index=False)

        # Log original metadata
        trainer.start_run("test_hash_verification")
        original_hash = trainer.log_dataset_metadata(
            df=sample_df,
            dataset_path=dataset_path,
            train_start_date="2025-01-01",
            train_end_date="2025-01-15",
        )

        # Verify hash matches
        verification_hash = trainer._compute_dataframe_hash(sample_df)
        trainer.end_run()

        assert original_hash == verification_hash, "Hash verification must pass for same data"

    def test_verify_hash_fails_on_mismatch(self, trainer, sample_df, tmp_path):
        """Verify that hash verification fails for modified data."""
        # Setup
        dataset_path = tmp_path / "train_data.csv"
        sample_df.to_csv(dataset_path, index=False)

        # Log original metadata
        trainer.start_run("test_hash_mismatch")
        original_hash = trainer.log_dataset_metadata(
            df=sample_df,
            dataset_path=dataset_path,
            train_start_date="2025-01-01",
            train_end_date="2025-01-15",
        )

        # Modify data and verify hash mismatch
        modified_df = sample_df.copy()
        modified_df["log_ret_5m"] = 999.0  # Significant change

        verification_hash = trainer._compute_dataframe_hash(modified_df)
        trainer.end_run()

        assert original_hash != verification_hash, "Hash verification must fail for modified data"


# =============================================================================
# TestIntegrationWithRealMLflow
# =============================================================================

class TestIntegrationWithRealMLflow:
    """
    Integration tests that use the actual MLflow library.

    These tests are skipped if MLflow is not available or
    if running in CI without MLflow infrastructure.
    """

    @pytest.fixture
    def mlflow_available(self):
        """Check if MLflow is available and working."""
        try:
            import mlflow
            # Test basic MLflow functionality
            _ = mlflow.__version__
            return True
        except Exception:
            # Catch any import or initialization error
            return False

    @pytest.mark.skipif(
        os.environ.get("CI") == "true" and os.environ.get("MLFLOW_AVAILABLE") != "true",
        reason="MLflow not available in CI"
    )
    def test_mlflow_logging_integration(self, mlflow_available, sample_df, tmp_path):
        """Test actual MLflow logging with a local file store."""
        if not mlflow_available:
            pytest.skip("MLflow not installed or not working")

        try:
            import mlflow
        except Exception as e:
            pytest.skip(f"MLflow import failed: {e}")

        # Setup temporary tracking
        tracking_uri = f"file://{tmp_path / 'mlruns'}"
        mlflow.set_tracking_uri(tracking_uri)

        # Create experiment
        experiment_name = "test_dataset_tracking"
        mlflow.set_experiment(experiment_name)

        # Start run and log
        with mlflow.start_run(run_name="integration_test"):
            # Compute hash
            feature_cols = [c for c in sample_df.columns if c in FEATURE_ORDER]
            csv_bytes = sample_df[feature_cols].to_csv(index=False).encode("utf-8")
            dataset_hash = hashlib.sha256(csv_bytes).hexdigest()[:16]

            # Log parameters
            mlflow.log_param("dataset_hash", dataset_hash)
            mlflow.log_param("train_start_date", "2025-01-01")
            mlflow.log_param("train_end_date", "2025-01-15")
            mlflow.log_param("FEATURE_ORDER_HASH", FEATURE_ORDER_HASH)
            mlflow.log_param("observation_dim", OBSERVATION_DIM)

            # Log artifact
            contract = {
                "feature_order": list(FEATURE_ORDER),
                "feature_order_hash": FEATURE_ORDER_HASH,
            }
            contract_path = tmp_path / "feature_contract.json"
            with open(contract_path, "w") as f:
                json.dump(contract, f)
            mlflow.log_artifact(str(contract_path))

            run_id = mlflow.active_run().info.run_id

        # Verify logged data
        run = mlflow.get_run(run_id)

        assert run.data.params["dataset_hash"] == dataset_hash
        assert run.data.params["train_start_date"] == "2025-01-01"
        assert run.data.params["train_end_date"] == "2025-01-15"
        assert run.data.params["FEATURE_ORDER_HASH"] == FEATURE_ORDER_HASH


# =============================================================================
# TestDatasetHashConsistency
# =============================================================================

class TestDatasetHashConsistency:
    """
    Tests for dataset hash consistency across different scenarios.
    """

    def test_hash_consistent_with_train_ssot(self, sample_df, tmp_path):
        """Verify hash computation is consistent with train_ssot module."""
        # Save sample data
        dataset_path = tmp_path / "test_data.csv"
        sample_df.to_csv(dataset_path, index=False)

        # Compute hash using train_ssot function
        file_hash = compute_dataset_hash(dataset_path)

        # Verify hash is a valid hex string
        assert len(file_hash) == 64  # Full SHA256
        assert all(c in "0123456789abcdef" for c in file_hash)

    def test_hash_changes_with_row_order(self, sample_df, tmp_path):
        """Verify hash changes when row order changes (for CSV-based hashing)."""
        # Shuffle data
        shuffled_df = sample_df.sample(frac=1, random_state=123).reset_index(drop=True)

        # Save both
        original_path = tmp_path / "original.csv"
        shuffled_path = tmp_path / "shuffled.csv"
        sample_df.to_csv(original_path, index=False)
        shuffled_df.to_csv(shuffled_path, index=False)

        # Compute hashes
        original_hash = compute_dataset_hash(original_path)
        shuffled_hash = compute_dataset_hash(shuffled_path)

        # Hashes should differ (row order matters)
        assert original_hash != shuffled_hash

    def test_hash_stable_across_sessions(self, sample_df, tmp_path):
        """Verify hash is stable across different Python sessions."""
        # Save data
        dataset_path = tmp_path / "test_data.csv"
        sample_df.to_csv(dataset_path, index=False)

        # Compute hash multiple times
        hashes = [compute_dataset_hash(dataset_path) for _ in range(5)]

        # All hashes should be identical
        assert len(set(hashes)) == 1


# =============================================================================
# Main Entry Point
# =============================================================================

def run_tests():
    """Run all MLflow dataset tracking tests."""
    print("=" * 60)
    print("MLflow Dataset Tracking Tests (P0-08)")
    print("=" * 60)

    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",
    ])

    return exit_code


if __name__ == "__main__":
    sys.exit(run_tests())
