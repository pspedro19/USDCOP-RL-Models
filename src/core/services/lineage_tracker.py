"""
Lineage Tracker Service
=======================

Unified service for tracking artifacts, hashes, and lineage across the ML pipeline.
Implements GAPs 2, 4, 5: dvc_tag logging, config_hash XCom, artifact collection.

Design Patterns:
- Facade Pattern: Unified interface for MLflow, DVC, and hashing
- Builder Pattern: Construct lineage records incrementally
- Observer Pattern: Notify on lineage events

SOLID Principles:
- Single Responsibility: Only handles lineage tracking
- Interface Segregation: Separate interfaces for different consumers
- Dependency Inversion: Depends on abstractions (MLflow, DVC clients)

Author: Trading Team
Date: 2026-01-17
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Union
import shutil

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols (Interface Segregation)
# =============================================================================

class ArtifactStore(Protocol):
    """Protocol for artifact storage backends."""

    def log_artifact(self, local_path: Path, artifact_path: str) -> str:
        """Log artifact and return URI."""
        ...

    def download_artifact(self, artifact_uri: str, dest_path: Path) -> Path:
        """Download artifact to local path."""
        ...


class MetadataStore(Protocol):
    """Protocol for metadata storage."""

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters."""
        ...

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics."""
        ...

    def get_run_params(self, run_id: str) -> Dict[str, Any]:
        """Get parameters for a run."""
        ...


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LineageRecord:
    """
    Complete lineage record for an experiment/training run.

    Tracks all artifacts, hashes, and metadata for full reproducibility.
    """

    # Identifiers
    run_id: str
    experiment_name: str
    experiment_version: str
    created_at: datetime = field(default_factory=datetime.now)

    # Hashes for integrity
    config_hash: Optional[str] = None
    dataset_hash: Optional[str] = None
    norm_stats_hash: Optional[str] = None
    feature_order_hash: Optional[str] = None
    model_hash: Optional[str] = None

    # DVC tracking
    dvc_tag: Optional[str] = None
    dvc_commit: Optional[str] = None

    # MLflow tracking
    mlflow_run_id: Optional[str] = None
    mlflow_experiment_id: Optional[str] = None

    # Artifact paths
    config_path: Optional[str] = None
    dataset_path: Optional[str] = None
    norm_stats_path: Optional[str] = None
    model_path: Optional[str] = None

    # Artifact URIs (for download)
    config_uri: Optional[str] = None
    norm_stats_uri: Optional[str] = None
    model_uri: Optional[str] = None

    # Pipeline stage
    stage: str = "unknown"  # L1_features, L3_training, L5_inference

    # Parent lineage (for chaining)
    parent_run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "experiment_version": self.experiment_version,
            "created_at": self.created_at.isoformat(),
            "hashes": {
                "config": self.config_hash,
                "dataset": self.dataset_hash,
                "norm_stats": self.norm_stats_hash,
                "feature_order": self.feature_order_hash,
                "model": self.model_hash,
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
                "config": self.config_uri,
                "norm_stats": self.norm_stats_uri,
                "model": self.model_uri,
            },
            "stage": self.stage,
            "parent_run_id": self.parent_run_id,
        }

    def to_mlflow_params(self) -> Dict[str, str]:
        """Convert to MLflow-compatible params dict."""
        params = {}

        # Add all hashes
        if self.config_hash:
            params["config_hash"] = self.config_hash
        if self.dataset_hash:
            params["dataset_hash"] = self.dataset_hash
        if self.norm_stats_hash:
            params["norm_stats_hash"] = self.norm_stats_hash
        if self.feature_order_hash:
            params["feature_order_hash"] = self.feature_order_hash

        # Add DVC info
        if self.dvc_tag:
            params["dvc_tag"] = self.dvc_tag
        if self.dvc_commit:
            params["dvc_commit"] = self.dvc_commit

        # Add paths
        if self.config_path:
            params["config_path"] = self.config_path
        if self.dataset_path:
            params["dataset_path"] = self.dataset_path

        # Add stage info
        params["pipeline_stage"] = self.stage
        if self.parent_run_id:
            params["parent_run_id"] = self.parent_run_id

        return params

    def to_xcom(self) -> Dict[str, Any]:
        """Convert to XCom-compatible dict for Airflow."""
        return {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "dataset_hash": self.dataset_hash,
            "feature_order_hash": self.feature_order_hash,
            "dvc_tag": self.dvc_tag,
            "mlflow_run_id": self.mlflow_run_id,
            "stage": self.stage,
        }

    @classmethod
    def from_xcom(cls, xcom_data: Dict[str, Any], experiment_name: str) -> "LineageRecord":
        """Create from XCom data."""
        return cls(
            run_id=xcom_data.get("run_id", ""),
            experiment_name=experiment_name,
            experiment_version="from_xcom",
            config_hash=xcom_data.get("config_hash"),
            dataset_hash=xcom_data.get("dataset_hash"),
            feature_order_hash=xcom_data.get("feature_order_hash"),
            dvc_tag=xcom_data.get("dvc_tag"),
            mlflow_run_id=xcom_data.get("mlflow_run_id"),
            stage=xcom_data.get("stage", "unknown"),
        )

    def save(self, path: Path) -> None:
        """Save lineage record to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "LineageRecord":
        """Load lineage record from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            run_id=data["run_id"],
            experiment_name=data["experiment_name"],
            experiment_version=data["experiment_version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            config_hash=data["hashes"].get("config"),
            dataset_hash=data["hashes"].get("dataset"),
            norm_stats_hash=data["hashes"].get("norm_stats"),
            feature_order_hash=data["hashes"].get("feature_order"),
            model_hash=data["hashes"].get("model"),
            dvc_tag=data["dvc"].get("tag"),
            dvc_commit=data["dvc"].get("commit"),
            mlflow_run_id=data["mlflow"].get("run_id"),
            mlflow_experiment_id=data["mlflow"].get("experiment_id"),
            config_uri=data["artifacts"].get("config"),
            norm_stats_uri=data["artifacts"].get("norm_stats"),
            model_uri=data["artifacts"].get("model"),
            stage=data.get("stage", "unknown"),
            parent_run_id=data.get("parent_run_id"),
        )


# =============================================================================
# Lineage Tracker (Builder Pattern)
# =============================================================================

class LineageTrackerBuilder:
    """Builder for constructing LineageRecord incrementally."""

    def __init__(self, run_id: str, experiment_name: str, experiment_version: str):
        self._record = LineageRecord(
            run_id=run_id,
            experiment_name=experiment_name,
            experiment_version=experiment_version,
        )

    def with_config_hash(self, config_hash: str) -> "LineageTrackerBuilder":
        """Add config hash."""
        self._record.config_hash = config_hash
        return self

    def with_dataset_hash(self, dataset_hash: str) -> "LineageTrackerBuilder":
        """Add dataset hash."""
        self._record.dataset_hash = dataset_hash
        return self

    def with_norm_stats_hash(self, norm_stats_hash: str) -> "LineageTrackerBuilder":
        """Add normalization stats hash."""
        self._record.norm_stats_hash = norm_stats_hash
        return self

    def with_feature_order_hash(self, feature_order_hash: str) -> "LineageTrackerBuilder":
        """Add feature order hash."""
        self._record.feature_order_hash = feature_order_hash
        return self

    def with_model_hash(self, model_hash: str) -> "LineageTrackerBuilder":
        """Add model hash."""
        self._record.model_hash = model_hash
        return self

    def with_dvc_info(self, tag: str, commit: Optional[str] = None) -> "LineageTrackerBuilder":
        """Add DVC tracking info."""
        self._record.dvc_tag = tag
        self._record.dvc_commit = commit
        return self

    def with_mlflow_info(self, run_id: str, experiment_id: Optional[str] = None) -> "LineageTrackerBuilder":
        """Add MLflow tracking info."""
        self._record.mlflow_run_id = run_id
        self._record.mlflow_experiment_id = experiment_id
        return self

    def with_artifact_paths(
        self,
        config_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        norm_stats_path: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> "LineageTrackerBuilder":
        """Add artifact paths."""
        self._record.config_path = config_path
        self._record.dataset_path = dataset_path
        self._record.norm_stats_path = norm_stats_path
        self._record.model_path = model_path
        return self

    def with_artifact_uris(
        self,
        config_uri: Optional[str] = None,
        norm_stats_uri: Optional[str] = None,
        model_uri: Optional[str] = None,
    ) -> "LineageTrackerBuilder":
        """Add artifact URIs."""
        self._record.config_uri = config_uri
        self._record.norm_stats_uri = norm_stats_uri
        self._record.model_uri = model_uri
        return self

    def with_stage(self, stage: str) -> "LineageTrackerBuilder":
        """Set pipeline stage."""
        self._record.stage = stage
        return self

    def with_parent(self, parent_run_id: str) -> "LineageTrackerBuilder":
        """Set parent run for chaining."""
        self._record.parent_run_id = parent_run_id
        return self

    def build(self) -> LineageRecord:
        """Build the LineageRecord."""
        return self._record


# =============================================================================
# Lineage Tracker Service (Facade Pattern)
# =============================================================================

class LineageTracker:
    """
    Unified service for tracking lineage across the ML pipeline.

    Provides a facade over MLflow, DVC, and file hashing to ensure
    complete traceability from data to deployment.

    Example:
        tracker = LineageTracker(project_root=Path("."))

        # Track experiment
        lineage = tracker.track_experiment(
            experiment_name="baseline_ppo_v1",
            config_path=Path("config/experiments/baseline.yaml"),
            dataset_path=Path("data/processed/train.parquet"),
        )

        # Log to MLflow
        tracker.log_to_mlflow(lineage)

        # Get artifacts for model version
        artifacts = tracker.get_model_artifacts("model_v1.0.0")
    """

    def __init__(
        self,
        project_root: Path,
        mlflow_tracking_uri: Optional[str] = None,
    ):
        """
        Initialize lineage tracker.

        Args:
            project_root: Project root directory
            mlflow_tracking_uri: MLflow tracking server URI
        """
        self.project_root = Path(project_root)
        self.mlflow_tracking_uri = mlflow_tracking_uri

        # Import contracts
        from src.core.contracts import FEATURE_ORDER_HASH

        self.feature_order_hash = FEATURE_ORDER_HASH

        # Initialize MLflow if available
        try:
            import mlflow
            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
            self._mlflow = mlflow
            self._mlflow_available = True
        except ImportError:
            self._mlflow = None
            self._mlflow_available = False

    def compute_hash(self, path: Path) -> str:
        """Compute SHA256 hash of a file."""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]

    def compute_json_hash(self, path: Path) -> str:
        """Compute hash of JSON file (normalized)."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        normalized = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def track_experiment(
        self,
        experiment_name: str,
        experiment_version: str,
        run_id: str,
        config_path: Optional[Path] = None,
        dataset_path: Optional[Path] = None,
        norm_stats_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
        dvc_tag: Optional[str] = None,
        stage: str = "L3_training",
    ) -> LineageRecord:
        """
        Track all lineage for an experiment.

        Computes hashes, collects paths, and builds complete lineage record.

        Args:
            experiment_name: Name of experiment
            experiment_version: Version string
            run_id: Unique run identifier
            config_path: Path to experiment config YAML
            dataset_path: Path to training dataset
            norm_stats_path: Path to normalization statistics
            model_path: Path to trained model
            dvc_tag: DVC tag for dataset
            stage: Pipeline stage (L1_features, L3_training, L5_inference)

        Returns:
            Complete LineageRecord
        """
        builder = LineageTrackerBuilder(run_id, experiment_name, experiment_version)
        builder.with_stage(stage)
        builder.with_feature_order_hash(self.feature_order_hash)

        # Compute hashes and add paths
        if config_path and Path(config_path).exists():
            builder.with_config_hash(self.compute_json_hash(Path(config_path)))
            builder.with_artifact_paths(config_path=str(config_path))

        if dataset_path and Path(dataset_path).exists():
            builder.with_dataset_hash(self.compute_hash(Path(dataset_path)))
            builder.with_artifact_paths(dataset_path=str(dataset_path))

        if norm_stats_path and Path(norm_stats_path).exists():
            builder.with_norm_stats_hash(self.compute_json_hash(Path(norm_stats_path)))
            builder.with_artifact_paths(norm_stats_path=str(norm_stats_path))

        if model_path and Path(model_path).exists():
            builder.with_model_hash(self.compute_hash(Path(model_path)))
            builder.with_artifact_paths(model_path=str(model_path))

        if dvc_tag:
            builder.with_dvc_info(tag=dvc_tag)

        return builder.build()

    def log_to_mlflow(
        self,
        lineage: LineageRecord,
        log_artifacts: bool = True,
    ) -> None:
        """
        Log lineage record to MLflow.

        Args:
            lineage: LineageRecord to log
            log_artifacts: Whether to log artifact files
        """
        if not self._mlflow_available:
            logger.warning("MLflow not available, skipping lineage logging")
            return

        # Log params
        self._mlflow.log_params(lineage.to_mlflow_params())

        # Log artifacts
        if log_artifacts:
            if lineage.config_path and Path(lineage.config_path).exists():
                self._mlflow.log_artifact(lineage.config_path, artifact_path="config")

            if lineage.norm_stats_path and Path(lineage.norm_stats_path).exists():
                self._mlflow.log_artifact(lineage.norm_stats_path, artifact_path="config")

        # Update lineage with MLflow info
        if self._mlflow.active_run():
            lineage.mlflow_run_id = self._mlflow.active_run().info.run_id
            lineage.mlflow_experiment_id = self._mlflow.active_run().info.experiment_id

    def get_model_artifacts(
        self,
        run_id: str,
        dest_dir: Path,
    ) -> Dict[str, Path]:
        """
        Download all artifacts for a model run.

        Implements GAP 8: Get experiment_config.yaml, dvc_tag, norm_stats.json.

        Args:
            run_id: MLflow run ID
            dest_dir: Destination directory for artifacts

        Returns:
            Dict mapping artifact type to local path
        """
        if not self._mlflow_available:
            raise RuntimeError("MLflow not available")

        from mlflow.tracking import MlflowClient
        client = MlflowClient()

        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        artifacts = {}

        # Get run info
        run = client.get_run(run_id)
        params = run.data.params

        # Download artifacts
        artifact_uri = run.info.artifact_uri

        try:
            # Config
            config_path = client.download_artifacts(run_id, "config", dest_dir)
            if Path(config_path).exists():
                artifacts["config"] = Path(config_path)
        except Exception as e:
            logger.warning(f"Could not download config: {e}")

        # Get DVC tag from params
        artifacts["dvc_tag"] = params.get("dvc_tag")
        artifacts["dataset_hash"] = params.get("dataset_hash")
        artifacts["feature_order_hash"] = params.get("feature_order_hash")

        return artifacts

    def download_baseline_artifacts(
        self,
        baseline_experiment_name: str,
        dest_dir: Path,
    ) -> Dict[str, Any]:
        """
        Download baseline experiment artifacts for comparison.

        Implements GAP Q6: Auto-download baseline artifacts for diff.

        This method:
        1. Finds the latest successful run for the baseline experiment
        2. Downloads config, norm_stats, and model artifacts
        3. Gets DVC info for dataset checkout
        4. Returns comparison-ready artifact paths and metadata

        Args:
            baseline_experiment_name: Name of the baseline experiment (e.g., 'baseline_full_macro')
            dest_dir: Destination directory for downloaded artifacts

        Returns:
            Dict with:
                - artifacts: Dict of artifact type -> local path
                - params: Dict of run parameters (hashes, dvc_tag, etc.)
                - run_id: MLflow run ID
                - run_name: Run name
                - metrics: Dict of logged metrics

        Raises:
            RuntimeError: If MLflow not available or no runs found
        """
        if not self._mlflow_available:
            raise RuntimeError("MLflow not available - cannot download baseline artifacts")

        from mlflow.tracking import MlflowClient
        client = MlflowClient()

        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Find baseline experiment
        experiment = client.get_experiment_by_name(baseline_experiment_name)
        if experiment is None:
            # Try with prefix
            experiments = client.search_experiments(
                filter_string=f"name LIKE '%{baseline_experiment_name}%'"
            )
            if not experiments:
                raise RuntimeError(f"Baseline experiment not found: {baseline_experiment_name}")
            experiment = experiments[0]

        logger.info(f"Found baseline experiment: {experiment.name} (ID: {experiment.experiment_id})")

        # Find the latest successful run
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        if not runs:
            raise RuntimeError(f"No successful runs found for baseline: {baseline_experiment_name}")

        baseline_run = runs[0]
        run_id = baseline_run.info.run_id
        run_name = baseline_run.info.run_name or baseline_run.data.tags.get("mlflow.runName", run_id)

        logger.info(f"Downloading artifacts from baseline run: {run_name} (ID: {run_id})")

        result = {
            "run_id": run_id,
            "run_name": run_name,
            "artifacts": {},
            "params": dict(baseline_run.data.params),
            "metrics": dict(baseline_run.data.metrics),
        }

        # Download config artifacts
        try:
            config_dest = dest_dir / "config"
            config_path = client.download_artifacts(run_id, "config", str(dest_dir))
            if Path(config_path).exists():
                result["artifacts"]["config_dir"] = Path(config_path)

                # Find specific files
                config_dir = Path(config_path)
                for yaml_file in config_dir.glob("*.yaml"):
                    result["artifacts"]["experiment_config"] = yaml_file
                for json_file in config_dir.glob("norm_stats*.json"):
                    result["artifacts"]["norm_stats"] = json_file
        except Exception as e:
            logger.warning(f"Could not download config artifacts: {e}")

        # Download model artifacts
        try:
            model_path = client.download_artifacts(run_id, "model", str(dest_dir))
            if Path(model_path).exists():
                result["artifacts"]["model_dir"] = Path(model_path)
        except Exception as e:
            logger.warning(f"Could not download model artifacts: {e}")

        # Extract key metadata for comparison
        result["comparison_metadata"] = {
            "feature_order_hash": result["params"].get("feature_order_hash"),
            "dataset_hash": result["params"].get("dataset_hash"),
            "norm_stats_hash": result["params"].get("norm_stats_hash"),
            "config_hash": result["params"].get("config_hash"),
            "dvc_tag": result["params"].get("dvc_tag"),
            "observation_dim": result["params"].get("observation_dim"),
        }

        # Key metrics for comparison
        result["comparison_metrics"] = {
            "sharpe_ratio": result["metrics"].get("sharpe_ratio"),
            "max_drawdown": result["metrics"].get("max_drawdown"),
            "total_return": result["metrics"].get("total_return"),
            "win_rate": result["metrics"].get("win_rate"),
        }

        logger.info(f"Downloaded baseline artifacts to: {dest_dir}")
        logger.info(f"  Config: {result['artifacts'].get('experiment_config')}")
        logger.info(f"  Norm stats: {result['artifacts'].get('norm_stats')}")
        logger.info(f"  Model: {result['artifacts'].get('model_dir')}")

        return result

    def compute_experiment_diff(
        self,
        baseline_artifacts: Dict[str, Any],
        treatment_artifacts: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute diff between baseline and treatment experiment artifacts.

        Args:
            baseline_artifacts: Result from download_baseline_artifacts for baseline
            treatment_artifacts: Result from download_baseline_artifacts for treatment

        Returns:
            Dict with detailed diff information
        """
        diff = {
            "hash_comparison": {},
            "metric_comparison": {},
            "config_diff": {},
            "compatible": True,
            "warnings": [],
        }

        # Compare hashes
        baseline_meta = baseline_artifacts.get("comparison_metadata", {})
        treatment_meta = treatment_artifacts.get("comparison_metadata", {})

        for key in ["feature_order_hash", "dataset_hash", "norm_stats_hash"]:
            baseline_val = baseline_meta.get(key)
            treatment_val = treatment_meta.get(key)
            diff["hash_comparison"][key] = {
                "baseline": baseline_val,
                "treatment": treatment_val,
                "match": baseline_val == treatment_val,
            }

            # Feature order MUST match for valid comparison
            if key == "feature_order_hash" and baseline_val != treatment_val:
                diff["compatible"] = False
                diff["warnings"].append(
                    f"CRITICAL: Feature order hash mismatch - results not comparable"
                )

        # Compare metrics
        baseline_metrics = baseline_artifacts.get("comparison_metrics", {})
        treatment_metrics = treatment_artifacts.get("comparison_metrics", {})

        for key in ["sharpe_ratio", "max_drawdown", "total_return", "win_rate"]:
            baseline_val = baseline_metrics.get(key)
            treatment_val = treatment_metrics.get(key)

            if baseline_val is not None and treatment_val is not None:
                diff_val = treatment_val - baseline_val
                pct_diff = (diff_val / abs(baseline_val) * 100) if baseline_val != 0 else 0

                diff["metric_comparison"][key] = {
                    "baseline": baseline_val,
                    "treatment": treatment_val,
                    "difference": diff_val,
                    "percent_change": pct_diff,
                    "treatment_better": self._is_treatment_better(key, diff_val),
                }

        return diff

    def _is_treatment_better(self, metric: str, diff: float) -> bool:
        """Determine if treatment is better based on metric type."""
        # Lower is better for these metrics
        lower_is_better = {"max_drawdown"}

        if metric in lower_is_better:
            return diff < 0
        else:
            return diff > 0

    def validate_lineage_chain(
        self,
        current_lineage: LineageRecord,
        parent_lineage: LineageRecord,
    ) -> List[str]:
        """
        Validate that lineage chain is consistent.

        Checks that hashes match between parent and current stages.

        Args:
            current_lineage: Current stage lineage
            parent_lineage: Previous stage lineage

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Feature order must match
        if current_lineage.feature_order_hash != parent_lineage.feature_order_hash:
            errors.append(
                f"Feature order hash mismatch: "
                f"current={current_lineage.feature_order_hash}, "
                f"parent={parent_lineage.feature_order_hash}"
            )

        # If parent had dataset hash, current should match
        if parent_lineage.dataset_hash and current_lineage.dataset_hash:
            if current_lineage.dataset_hash != parent_lineage.dataset_hash:
                errors.append(
                    f"Dataset hash mismatch: "
                    f"current={current_lineage.dataset_hash}, "
                    f"parent={parent_lineage.dataset_hash}"
                )

        # Norm stats should match for inference
        if current_lineage.stage == "L5_inference":
            if current_lineage.norm_stats_hash != parent_lineage.norm_stats_hash:
                errors.append(
                    f"Norm stats hash mismatch in inference: "
                    f"current={current_lineage.norm_stats_hash}, "
                    f"training={parent_lineage.norm_stats_hash}"
                )

        return errors


# =============================================================================
# Factory Function
# =============================================================================

def create_lineage_tracker(
    project_root: Optional[Path] = None,
    mlflow_tracking_uri: Optional[str] = None,
) -> LineageTracker:
    """
    Factory function to create LineageTracker.

    Args:
        project_root: Project root (auto-detected if None)
        mlflow_tracking_uri: MLflow server URI

    Returns:
        Configured LineageTracker instance
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent.parent

    return LineageTracker(
        project_root=project_root,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )


__all__ = [
    "LineageTracker",
    "LineageRecord",
    "LineageTrackerBuilder",
    "create_lineage_tracker",
]
