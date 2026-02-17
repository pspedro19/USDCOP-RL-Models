"""
SSOT Lineage Integration - Bridge between pipeline_ssot.yaml and MLOps infrastructure
======================================================================================

This module integrates the new unified SSOT (pipeline_ssot.yaml) with existing
MLOps components:
- LineageTracker: Add SSOT version and config hash tracking
- ModelRegistry: Register models with SSOT compliance verification
- Feature Store: Validate feature consistency with SSOT
- DVC: Track SSOT config changes

Contract: CTR-SSOT-LINEAGE-001
Version: 1.0.0
Date: 2026-02-03

Usage:
    from src.ml_workflow.ssot_lineage_integration import (
        SSOTLineageIntegration,
        create_ssot_lineage_record,
        validate_ssot_compliance,
    )

    # Create lineage record with SSOT tracking
    integration = SSOTLineageIntegration()
    lineage = integration.create_lineage_record(
        run_id="training_v1",
        stage="L3_training",
        model_path=Path("models/ppo_v1/final_model.zip"),
        dataset_path=Path("data/pipeline/07_output/5min/DS_production_train.parquet"),
    )

    # Validate SSOT compliance before deployment
    compliance = integration.validate_compliance(model_id="ppo_v1")
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SSOTLineageRecord:
    """
    Extended lineage record with SSOT tracking.

    Adds pipeline_ssot.yaml version and compliance checks to standard lineage.
    """
    # Standard lineage fields
    run_id: str
    stage: str  # L2_dataset, L3_training, L4_backtest, L5_inference
    created_at: datetime = field(default_factory=datetime.now)

    # SSOT tracking (NEW)
    ssot_version: Optional[str] = None
    ssot_config_hash: Optional[str] = None
    ssot_based_on_model: Optional[str] = None

    # Feature tracking
    feature_order_hash: Optional[str] = None
    observation_dim: Optional[int] = None
    market_features_count: Optional[int] = None

    # Artifact hashes
    dataset_hash: Optional[str] = None
    norm_stats_hash: Optional[str] = None
    model_hash: Optional[str] = None
    config_hash: Optional[str] = None

    # DVC tracking
    dvc_tag: Optional[str] = None
    dvc_commit: Optional[str] = None

    # MLflow tracking
    mlflow_run_id: Optional[str] = None
    mlflow_experiment_id: Optional[str] = None

    # Artifact paths
    dataset_path: Optional[str] = None
    norm_stats_path: Optional[str] = None
    model_path: Optional[str] = None

    # Training/Backtest parity (CRITICAL)
    training_backtest_parity_verified: bool = False
    parity_issues: List[str] = field(default_factory=list)

    # Parent lineage
    parent_run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "stage": self.stage,
            "created_at": self.created_at.isoformat(),
            "ssot": {
                "version": self.ssot_version,
                "config_hash": self.ssot_config_hash,
                "based_on_model": self.ssot_based_on_model,
            },
            "features": {
                "order_hash": self.feature_order_hash,
                "observation_dim": self.observation_dim,
                "market_features_count": self.market_features_count,
            },
            "hashes": {
                "dataset": self.dataset_hash,
                "norm_stats": self.norm_stats_hash,
                "model": self.model_hash,
                "config": self.config_hash,
            },
            "dvc": {
                "tag": self.dvc_tag,
                "commit": self.dvc_commit,
            },
            "mlflow": {
                "run_id": self.mlflow_run_id,
                "experiment_id": self.mlflow_experiment_id,
            },
            "artifacts": {
                "dataset": self.dataset_path,
                "norm_stats": self.norm_stats_path,
                "model": self.model_path,
            },
            "parity": {
                "verified": self.training_backtest_parity_verified,
                "issues": self.parity_issues,
            },
            "parent_run_id": self.parent_run_id,
        }

    def to_mlflow_params(self) -> Dict[str, str]:
        """Convert to MLflow-compatible params."""
        params = {
            "ssot_version": self.ssot_version or "",
            "ssot_config_hash": self.ssot_config_hash or "",
            "ssot_based_on_model": self.ssot_based_on_model or "",
            "feature_order_hash": self.feature_order_hash or "",
            "observation_dim": str(self.observation_dim or 0),
            "dataset_hash": self.dataset_hash or "",
            "norm_stats_hash": self.norm_stats_hash or "",
            "model_hash": self.model_hash or "",
            "dvc_tag": self.dvc_tag or "",
            "training_backtest_parity": str(self.training_backtest_parity_verified),
        }
        return {k: v for k, v in params.items() if v}


@dataclass
class SSOTComplianceReport:
    """Report of SSOT compliance validation."""
    is_compliant: bool
    ssot_version: str
    checked_at: datetime = field(default_factory=datetime.now)

    # Compliance checks
    feature_order_match: bool = False
    observation_dim_match: bool = False
    training_backtest_parity: bool = False
    norm_stats_from_train_only: bool = False
    anti_leakage_verified: bool = False

    # Details
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# MAIN INTEGRATION CLASS
# =============================================================================

class SSOTLineageIntegration:
    """
    Integration layer between pipeline_ssot.yaml and MLOps infrastructure.

    Provides:
    - SSOT-aware lineage record creation
    - Compliance validation against SSOT
    - Integration with existing LineageTracker and ModelRegistry
    - DVC tag generation for SSOT versions
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize with SSOT configuration.

        Args:
            config_path: Path to pipeline_ssot.yaml (uses default if None)
        """
        from src.config.pipeline_config import load_pipeline_config
        self.config = load_pipeline_config(str(config_path) if config_path else None)
        self._ssot_hash = self._compute_ssot_hash()

        logger.info(
            f"SSOTLineageIntegration initialized: "
            f"v{self.config.version}, hash={self._ssot_hash[:8]}"
        )

    def _compute_ssot_hash(self) -> str:
        """Compute hash of SSOT configuration for tracking changes."""
        # Hash based on key configuration values
        key_config = {
            "version": self.config.version,
            "feature_order": list(self.config.get_feature_order()),
            "observation_dim": self.config.get_observation_dim(),
            "training": {
                "transaction_cost_bps": self.config.environment.transaction_cost_bps,
                "thresholds": [
                    self.config.environment.threshold_short,
                    self.config.environment.threshold_long,
                ],
                "stop_loss_pct": self.config.environment.stop_loss_pct,
                "take_profit_pct": self.config.environment.take_profit_pct,
            },
            "backtest": {
                "transaction_cost_bps": self.config.backtest.transaction_cost_bps,
                "thresholds": [
                    self.config.backtest.threshold_short,
                    self.config.backtest.threshold_long,
                ],
                "stop_loss_pct": self.config.backtest.stop_loss_pct,
                "take_profit_pct": self.config.backtest.take_profit_pct,
            },
        }

        config_str = json.dumps(key_config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def create_lineage_record(
        self,
        run_id: str,
        stage: str,
        model_path: Optional[Path] = None,
        dataset_path: Optional[Path] = None,
        norm_stats_path: Optional[Path] = None,
        mlflow_run_id: Optional[str] = None,
        dvc_tag: Optional[str] = None,
        parent_run_id: Optional[str] = None,
    ) -> SSOTLineageRecord:
        """
        Create a lineage record with full SSOT tracking.

        Args:
            run_id: Unique identifier for this run
            stage: Pipeline stage (L2_dataset, L3_training, L4_backtest, L5_inference)
            model_path: Path to model file
            dataset_path: Path to dataset file
            norm_stats_path: Path to normalization stats
            mlflow_run_id: MLflow run ID if available
            dvc_tag: DVC tag if available
            parent_run_id: Parent lineage run ID

        Returns:
            SSOTLineageRecord with all tracking information
        """
        # Compute feature order hash
        feature_order = self.config.get_feature_order()
        feature_hash = hashlib.md5(",".join(feature_order).encode()).hexdigest()

        # Validate training/backtest parity
        parity_issues = self.config.validate_training_backtest_parity()

        # Compute artifact hashes
        dataset_hash = self._compute_file_hash(dataset_path) if dataset_path else None
        norm_stats_hash = self._compute_file_hash(norm_stats_path) if norm_stats_path else None
        model_hash = self._compute_file_hash(model_path) if model_path else None

        record = SSOTLineageRecord(
            run_id=run_id,
            stage=stage,
            ssot_version=self.config.version,
            ssot_config_hash=self._ssot_hash,
            ssot_based_on_model=self.config.based_on_model,
            feature_order_hash=feature_hash,
            observation_dim=self.config.get_observation_dim(),
            market_features_count=len(self.config.get_market_features()),
            dataset_hash=dataset_hash,
            norm_stats_hash=norm_stats_hash,
            model_hash=model_hash,
            config_hash=self._ssot_hash,
            dvc_tag=dvc_tag or self._generate_dvc_tag(run_id),
            mlflow_run_id=mlflow_run_id,
            dataset_path=str(dataset_path) if dataset_path else None,
            norm_stats_path=str(norm_stats_path) if norm_stats_path else None,
            model_path=str(model_path) if model_path else None,
            training_backtest_parity_verified=len(parity_issues) == 0,
            parity_issues=parity_issues,
            parent_run_id=parent_run_id,
        )

        logger.info(
            f"Created lineage record: {run_id} (stage={stage}, "
            f"ssot_v={self.config.version}, parity_ok={record.training_backtest_parity_verified})"
        )

        return record

    def validate_compliance(
        self,
        model_path: Optional[Path] = None,
        dataset_path: Optional[Path] = None,
        norm_stats_path: Optional[Path] = None,
        expected_feature_order: Optional[List[str]] = None,
    ) -> SSOTComplianceReport:
        """
        Validate compliance with SSOT configuration.

        Checks:
        1. Feature order matches SSOT
        2. Observation dimension matches
        3. Training/backtest parity
        4. Normalization stats computed from training data only
        5. Anti-leakage guarantees

        Args:
            model_path: Path to model to validate
            dataset_path: Path to dataset to validate
            norm_stats_path: Path to norm stats to validate
            expected_feature_order: Expected feature order (from model/dataset)

        Returns:
            SSOTComplianceReport with validation results
        """
        issues = []
        warnings = []

        # Check 1: Feature order
        ssot_feature_order = list(self.config.get_feature_order())
        feature_order_match = True

        if expected_feature_order:
            if expected_feature_order != ssot_feature_order:
                feature_order_match = False
                issues.append(
                    f"Feature order mismatch: expected {len(expected_feature_order)} features, "
                    f"SSOT has {len(ssot_feature_order)}"
                )

        # Check 2: Observation dimension
        observation_dim_match = True
        expected_dim = self.config.get_observation_dim()

        if expected_feature_order:
            actual_dim = len(expected_feature_order)
            if actual_dim != expected_dim:
                observation_dim_match = False
                issues.append(
                    f"Observation dim mismatch: expected {expected_dim}, got {actual_dim}"
                )

        # Check 3: Training/backtest parity
        parity_issues = self.config.validate_training_backtest_parity()
        training_backtest_parity = len(parity_issues) == 0

        if not training_backtest_parity:
            issues.extend([f"Parity: {i}" for i in parity_issues])

        # Check 4: Norm stats from training only
        norm_stats_from_train_only = True
        if norm_stats_path and norm_stats_path.exists():
            with open(norm_stats_path) as f:
                norm_stats = json.load(f)

            meta = norm_stats.get("_meta", {})
            if meta.get("pipeline_stage") != "L2" and "train" not in str(meta.get("train_end_date", "")):
                norm_stats_from_train_only = False
                warnings.append("Norm stats may not be computed from training data only")

        # Check 5: Anti-leakage (basic check)
        anti_leakage_verified = True
        preprocessing = self.config.get_preprocessing_config()
        macro_shift = preprocessing.get("macro", {}).get("anti_leakage", {}).get("shift_days", 0)

        if macro_shift < 1:
            anti_leakage_verified = False
            warnings.append("Macro anti-leakage shift is less than 1 day")

        # Overall compliance
        is_compliant = (
            feature_order_match and
            observation_dim_match and
            training_backtest_parity and
            norm_stats_from_train_only and
            anti_leakage_verified
        )

        report = SSOTComplianceReport(
            is_compliant=is_compliant,
            ssot_version=self.config.version,
            feature_order_match=feature_order_match,
            observation_dim_match=observation_dim_match,
            training_backtest_parity=training_backtest_parity,
            norm_stats_from_train_only=norm_stats_from_train_only,
            anti_leakage_verified=anti_leakage_verified,
            issues=issues,
            warnings=warnings,
        )

        logger.info(
            f"Compliance check: {'PASSED' if is_compliant else 'FAILED'} "
            f"({len(issues)} issues, {len(warnings)} warnings)"
        )

        return report

    def integrate_with_model_registry(
        self,
        lineage: SSOTLineageRecord,
        model_registry: "ModelRegistry",
        model_metadata: Dict[str, Any],
    ) -> str:
        """
        Register model with SSOT compliance information.

        Args:
            lineage: Lineage record with SSOT tracking
            model_registry: ModelRegistry instance
            model_metadata: Additional model metadata

        Returns:
            Registered model ID
        """
        # Add SSOT tracking to metadata
        enhanced_metadata = {
            **model_metadata,
            "ssot_version": lineage.ssot_version,
            "ssot_config_hash": lineage.ssot_config_hash,
            "ssot_based_on_model": lineage.ssot_based_on_model,
            "feature_order_hash": lineage.feature_order_hash,
            "training_backtest_parity_verified": lineage.training_backtest_parity_verified,
        }

        # Register with model registry
        model_id = model_registry.register_model(
            model_path=Path(lineage.model_path) if lineage.model_path else None,
            norm_stats_path=Path(lineage.norm_stats_path) if lineage.norm_stats_path else None,
            **enhanced_metadata,
        )

        logger.info(f"Registered model {model_id} with SSOT compliance tracking")
        return model_id

    def integrate_with_dvc(self, lineage: SSOTLineageRecord) -> Optional[str]:
        """
        Create DVC tag for this lineage record.

        Args:
            lineage: Lineage record

        Returns:
            DVC tag if created, None otherwise
        """
        import subprocess

        tag = lineage.dvc_tag or self._generate_dvc_tag(lineage.run_id)

        try:
            # Create DVC tag with metadata
            subprocess.run(
                ["dvc", "tag", tag, "-m", f"SSOT v{lineage.ssot_version} - {lineage.stage}"],
                check=True,
                capture_output=True,
            )
            logger.info(f"Created DVC tag: {tag}")
            return tag
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Failed to create DVC tag: {e}")
            return None

    def _generate_dvc_tag(self, run_id: str) -> str:
        """Generate a DVC tag name for a run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"ssot_v{self.config.version}_{run_id}_{timestamp}"

    def _compute_file_hash(self, file_path: Optional[Path]) -> Optional[str]:
        """Compute SHA256 hash of a file."""
        if file_path is None or not Path(file_path).exists():
            return None

        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]

    def get_ssot_summary(self) -> Dict[str, Any]:
        """Get summary of current SSOT configuration."""
        return {
            "version": self.config.version,
            "config_hash": self._ssot_hash[:8],
            "based_on_model": self.config.based_on_model,
            "observation_dim": self.config.get_observation_dim(),
            "market_features": len(self.config.get_market_features()),
            "state_features": len(self.config.get_state_features()),
            "training": {
                "transaction_cost_bps": self.config.environment.transaction_cost_bps,
                "thresholds": [
                    self.config.environment.threshold_short,
                    self.config.environment.threshold_long,
                ],
            },
            "backtest": {
                "transaction_cost_bps": self.config.backtest.transaction_cost_bps,
                "thresholds": [
                    self.config.backtest.threshold_short,
                    self.config.backtest.threshold_long,
                ],
            },
            "parity_verified": len(self.config.validate_training_backtest_parity()) == 0,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_ssot_lineage_record(
    run_id: str,
    stage: str,
    **kwargs,
) -> SSOTLineageRecord:
    """
    Convenience function to create SSOT lineage record.

    Args:
        run_id: Unique run identifier
        stage: Pipeline stage
        **kwargs: Additional arguments passed to create_lineage_record

    Returns:
        SSOTLineageRecord
    """
    integration = SSOTLineageIntegration()
    return integration.create_lineage_record(run_id=run_id, stage=stage, **kwargs)


def validate_ssot_compliance(**kwargs) -> SSOTComplianceReport:
    """
    Convenience function to validate SSOT compliance.

    Args:
        **kwargs: Arguments passed to validate_compliance

    Returns:
        SSOTComplianceReport
    """
    integration = SSOTLineageIntegration()
    return integration.validate_compliance(**kwargs)


def get_ssot_config_hash() -> str:
    """Get hash of current SSOT configuration."""
    integration = SSOTLineageIntegration()
    return integration._ssot_hash


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "SSOTLineageIntegration",
    "SSOTLineageRecord",
    "SSOTComplianceReport",
    "create_ssot_lineage_record",
    "validate_ssot_compliance",
    "get_ssot_config_hash",
]
