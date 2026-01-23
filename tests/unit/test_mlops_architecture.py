"""
MLOps Architecture Tests
========================

Tests for validating the "MLflow-First + DVC-Tracked" architectural principle.

Tests:
- Artifact storage policy
- Lineage service
- Promotion gate validators
- Policy compliance

@version 1.0.0
@principle MLflow-First + DVC-Tracked
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestArtifactPolicy:
    """Test artifact storage policy."""

    def test_artifact_types_defined(self):
        """Verify all artifact types are defined."""
        from src.ml_workflow.artifact_policy import ArtifactType

        expected_types = [
            "dataset", "model", "metrics", "config",
            "checkpoint", "backtest", "forecast", "lineage"
        ]

        for type_name in expected_types:
            assert hasattr(ArtifactType, type_name.upper()), f"Missing type: {type_name}"

    def test_storage_backends_defined(self):
        """Verify storage backends are defined."""
        from src.ml_workflow.artifact_policy import StorageBackend

        expected_backends = ["mlflow", "minio", "postgresql", "dvc", "local"]

        for backend in expected_backends:
            assert hasattr(StorageBackend, backend.upper()), f"Missing backend: {backend}"

    def test_model_primary_backend_is_mlflow(self):
        """Verify models use MLflow as primary backend (MLflow-First)."""
        from src.ml_workflow.artifact_policy import (
            ArtifactType,
            StorageBackend,
            ARTIFACT_STORAGE_RULES,
        )

        model_rule = ARTIFACT_STORAGE_RULES[ArtifactType.MODEL]
        assert model_rule.primary_backend == StorageBackend.MLFLOW, \
            "Models must use MLflow as primary backend"

    def test_dataset_primary_backend_is_dvc(self):
        """Verify datasets use DVC as primary backend (DVC-Tracked)."""
        from src.ml_workflow.artifact_policy import (
            ArtifactType,
            StorageBackend,
            ARTIFACT_STORAGE_RULES,
        )

        dataset_rule = ARTIFACT_STORAGE_RULES[ArtifactType.DATASET]
        assert dataset_rule.primary_backend == StorageBackend.DVC, \
            "Datasets must use DVC as primary backend"

    def test_model_storage_is_mandatory(self):
        """Verify model storage is mandatory."""
        from src.ml_workflow.artifact_policy import (
            ArtifactType,
            ARTIFACT_STORAGE_RULES,
        )

        model_rule = ARTIFACT_STORAGE_RULES[ArtifactType.MODEL]
        assert model_rule.is_mandatory is True, "Model storage must be mandatory"

    def test_dataset_storage_is_mandatory(self):
        """Verify dataset storage is mandatory."""
        from src.ml_workflow.artifact_policy import (
            ArtifactType,
            ARTIFACT_STORAGE_RULES,
        )

        dataset_rule = ARTIFACT_STORAGE_RULES[ArtifactType.DATASET]
        assert dataset_rule.is_mandatory is True, "Dataset storage must be mandatory"

    def test_policy_singleton(self):
        """Verify artifact policy is singleton."""
        from src.ml_workflow.artifact_policy import get_artifact_policy

        policy1 = get_artifact_policy()
        policy2 = get_artifact_policy()

        assert policy1 is policy2, "Policy should be singleton"

    def test_model_location_uri_format(self):
        """Verify model location returns MLflow URI."""
        from src.ml_workflow.artifact_policy import (
            ArtifactPolicy,
            ArtifactType,
        )

        policy = ArtifactPolicy()
        location = policy.get_artifact_location(
            ArtifactType.MODEL,
            model_name="test-model",
            version="v1"
        )

        assert location.primary_uri.startswith("models:/"), \
            "Model URI should start with models:/"

    def test_dataset_location_uri_format(self):
        """Verify dataset location returns DVC URI."""
        from src.ml_workflow.artifact_policy import (
            ArtifactPolicy,
            ArtifactType,
        )

        policy = ArtifactPolicy()
        location = policy.get_artifact_location(
            ArtifactType.DATASET,
            dataset_name="test-dataset",
            version="v1"
        )

        assert location.primary_uri.startswith("dvc://"), \
            "Dataset URI should start with dvc://"

    def test_policy_compliance_validation(self):
        """Test policy compliance validation."""
        from src.ml_workflow.artifact_policy import (
            ArtifactPolicy,
            ArtifactType,
        )

        policy = ArtifactPolicy()

        # Valid MLflow location
        is_valid, violations = policy.validate_policy_compliance(
            ArtifactType.MODEL,
            "models:/test-model/1"
        )
        assert is_valid, f"Should be valid: {violations}"

        # Invalid location (local path)
        is_valid, violations = policy.validate_policy_compliance(
            ArtifactType.MODEL,
            "/local/path/model.pkl"
        )
        assert not is_valid, "Local path should violate policy"
        assert len(violations) > 0


class TestLineageService:
    """Test unified lineage service."""

    def test_lineage_service_singleton(self):
        """Verify lineage service is singleton."""
        from src.ml_workflow.lineage_service import get_lineage_service

        service1 = get_lineage_service()
        service2 = get_lineage_service()

        assert service1 is service2, "Service should be singleton"

    def test_record_dataset(self):
        """Test recording dataset lineage."""
        from src.ml_workflow.lineage_service import LineageService

        service = LineageService()

        node = service.record_dataset(
            name="test-dataset",
            path="data/test.parquet",
            dvc_tag="dataset-v1.0.0",
            content_hash="abc123",
            version="v1.0.0"
        )

        assert node.name == "test-dataset"
        assert node.dvc_tag == "dataset-v1.0.0"
        assert node.content_hash == "abc123"

    def test_record_complete_lineage(self):
        """Test recording complete lineage."""
        from src.ml_workflow.lineage_service import LineageService

        service = LineageService()

        record = service.record_complete_lineage(
            pipeline="forecasting",
            dataset_path="data/forecasting/train.parquet",
            dataset_hash="abc123",
            dataset_dvc_tag="dataset-v1.0.0",
            mlflow_run_id="run123",
            model_name="xgboost-h5",
            model_version="v1",
            model_uri="models:/xgboost-h5/1",
            feature_config_hash="feat123",
        )

        assert record.pipeline == "forecasting"
        assert record.dataset_dvc_tag == "dataset-v1.0.0"
        assert record.mlflow_run_id == "run123"
        assert record.model_uri == "models:/xgboost-h5/1"

    def test_lineage_record_hash(self):
        """Test lineage record hash is deterministic."""
        from src.ml_workflow.lineage_service import LineageRecord

        record1 = LineageRecord(
            record_id="test1",
            pipeline="rl",
            dataset_hash="abc123",
            feature_config_hash="feat123",
            training_config_hash="train123",
            model_hash="model123",
        )

        record2 = LineageRecord(
            record_id="test2",
            pipeline="rl",
            dataset_hash="abc123",
            feature_config_hash="feat123",
            training_config_hash="train123",
            model_hash="model123",
        )

        assert record1.compute_lineage_hash() == record2.compute_lineage_hash(), \
            "Same content should produce same hash"


class TestPromotionGate:
    """Test promotion gate validators."""

    def test_default_gate_has_mlflow_first_validator(self):
        """Verify default gate includes MLflow-First validator."""
        from src.ml_workflow.promotion_gate import (
            create_default_gate,
            MLflowFirstValidator,
        )

        gate = create_default_gate()

        validator_types = [type(v) for v in gate._validators]
        assert MLflowFirstValidator in validator_types, \
            "Default gate must include MLflowFirstValidator"

    def test_default_gate_has_dvc_tracked_validator(self):
        """Verify default gate includes DVC-Tracked validator."""
        from src.ml_workflow.promotion_gate import (
            create_default_gate,
            DVCTrackedValidator,
        )

        gate = create_default_gate()

        validator_types = [type(v) for v in gate._validators]
        assert DVCTrackedValidator in validator_types, \
            "Default gate must include DVCTrackedValidator"

    def test_default_config_enforces_mlflow_first(self):
        """Verify default config enforces MLflow-First."""
        from src.ml_workflow.promotion_gate import DEFAULT_GATE_CONFIG

        assert DEFAULT_GATE_CONFIG.get("enforce_mlflow_first") is True, \
            "Default config must enforce MLflow-First"

    def test_default_config_enforces_dvc_tracking(self):
        """Verify default config enforces DVC tracking."""
        from src.ml_workflow.promotion_gate import DEFAULT_GATE_CONFIG

        assert DEFAULT_GATE_CONFIG.get("enforce_dvc_tracking") is True, \
            "Default config must enforce DVC tracking"

    def test_mlflow_first_validator_rejects_local_path(self):
        """Test MLflow-First validator rejects local paths."""
        from src.ml_workflow.promotion_gate import MLflowFirstValidator

        validator = MLflowFirstValidator()

        # Create mock model with local path
        mock_model = Mock()
        mock_model.mlflow_run_id = None
        mock_model.model_uri = "/local/path/model.pkl"
        mock_model.s3_uri = None

        issues = validator.validate(mock_model, None, {"enforce_mlflow_first": True})

        assert len(issues) > 0, "Should have validation issues"
        assert any("mlflow_run_id" in i.message.lower() for i in issues)

    def test_mlflow_first_validator_accepts_mlflow_uri(self):
        """Test MLflow-First validator accepts MLflow URIs."""
        from src.ml_workflow.promotion_gate import MLflowFirstValidator

        validator = MLflowFirstValidator()

        # Create mock model with MLflow URI
        mock_model = Mock()
        mock_model.mlflow_run_id = "abc123"
        mock_model.model_uri = "models:/test-model/1"

        issues = validator.validate(mock_model, None, {"enforce_mlflow_first": True})

        # Should not have blocking errors for MLflow URI
        errors = [i for i in issues if i.severity.value == "error"]
        assert len(errors) == 0, f"Should not have errors: {errors}"


class TestMinioBucketStructure:
    """Test MinIO bucket structure."""

    def test_bucket_structure_defined(self):
        """Verify bucket structure is defined."""
        from src.ml_workflow.artifact_policy import MINIO_BUCKET_STRUCTURE

        expected_buckets = [
            "datasets",
            "models",
            "mlflow-artifacts",
            "backtest-results",
            "forecasts",
        ]

        for bucket in expected_buckets:
            assert bucket in MINIO_BUCKET_STRUCTURE, f"Missing bucket: {bucket}"

    def test_datasets_bucket_has_versioning(self):
        """Verify datasets bucket has versioning enabled."""
        from src.ml_workflow.artifact_policy import MINIO_BUCKET_STRUCTURE

        datasets_config = MINIO_BUCKET_STRUCTURE["datasets"]
        assert datasets_config["versioning"] is True, \
            "Datasets bucket must have versioning"

    def test_models_bucket_has_versioning(self):
        """Verify models bucket has versioning enabled."""
        from src.ml_workflow.artifact_policy import MINIO_BUCKET_STRUCTURE

        models_config = MINIO_BUCKET_STRUCTURE["models"]
        assert models_config["versioning"] is True, \
            "Models bucket must have versioning"


class TestArchitecturePrinciple:
    """Test the overall MLflow-First + DVC-Tracked principle."""

    def test_principle_documented(self):
        """Verify principle is documented in artifact policy."""
        from src.ml_workflow.artifact_policy import ArtifactPolicy

        policy = ArtifactPolicy()
        policy_dict = policy.to_dict()

        assert policy_dict["principle"] == "MLflow-First + DVC-Tracked", \
            "Principle must be documented"

    def test_principle_in_module_docstring(self):
        """Verify principle is in module docstring."""
        from src.ml_workflow import artifact_policy

        assert "MLflow-First" in artifact_policy.__doc__, \
            "MLflow-First should be in module docstring"
        assert "DVC-Tracked" in artifact_policy.__doc__, \
            "DVC-Tracked should be in module docstring"

    def test_ml_workflow_exports_principle_components(self):
        """Verify ml_workflow exports principle components."""
        from src.ml_workflow import (
            ArtifactPolicy,
            get_artifact_policy,
            enforce_mlflow_first,
            enforce_dvc_tracking,
            LineageService,
            get_lineage_service,
            MLflowFirstValidator,
            DVCTrackedValidator,
        )

        # All should be importable
        assert ArtifactPolicy is not None
        assert get_artifact_policy is not None
        assert enforce_mlflow_first is not None
        assert enforce_dvc_tracking is not None
        assert LineageService is not None
        assert get_lineage_service is not None
        assert MLflowFirstValidator is not None
        assert DVCTrackedValidator is not None


class TestDatabaseMigration:
    """Test lineage database schema."""

    def test_migration_file_exists(self):
        """Verify migration file exists."""
        migration_path = PROJECT_ROOT / "database" / "migrations" / "025_lineage_tables.sql"
        assert migration_path.exists(), "Lineage migration file should exist"

    def test_migration_creates_ml_schema(self):
        """Verify migration creates ml schema."""
        migration_path = PROJECT_ROOT / "database" / "migrations" / "025_lineage_tables.sql"

        with open(migration_path) as f:
            content = f.read()

        assert "CREATE SCHEMA IF NOT EXISTS ml" in content, \
            "Migration should create ml schema"

    def test_migration_creates_lineage_tables(self):
        """Verify migration creates required tables."""
        migration_path = PROJECT_ROOT / "database" / "migrations" / "025_lineage_tables.sql"

        with open(migration_path) as f:
            content = f.read()

        required_tables = [
            "ml.lineage_nodes",
            "ml.lineage_edges",
            "ml.lineage_records",
            "ml.model_promotion_audit",
            "ml.artifact_storage_audit",
        ]

        for table in required_tables:
            assert table in content, f"Migration should create {table}"

    def test_migration_has_compliance_columns(self):
        """Verify migration includes compliance tracking columns."""
        migration_path = PROJECT_ROOT / "database" / "migrations" / "025_lineage_tables.sql"

        with open(migration_path) as f:
            content = f.read()

        assert "mlflow_first_compliant" in content, \
            "Should track MLflow-First compliance"
        assert "dvc_tracked_compliant" in content, \
            "Should track DVC-Tracked compliance"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
