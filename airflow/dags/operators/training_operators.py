"""
Training Operators - Reusable Operators for ML Training DAGs
=============================================================
Professional operators implementing SOLID principles for ML training workflows.

SOLID Principles:
- Single Responsibility: Each operator does one thing
- Open/Closed: Extensible via inheritance
- Liskov Substitution: All operators are interchangeable
- Interface Segregation: Small, focused interfaces
- Dependency Inversion: Depends on abstractions (XCom, configs)

Design Patterns:
- Template Method: Base operator with hook methods
- Strategy Pattern: Pluggable validation strategies
- Observer Pattern: MLflow tracking as observer

Author: Trading Team
Version: 1.0.0
Created: 2025-01-12
"""

import json
import logging
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# Add project root to path for SSOT imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# SSOT import for hash utilities
from src.utils.hash_utils import compute_file_hash as _compute_file_hash_ssot, compute_json_hash as _compute_json_hash_ssot
from typing import Any, Dict, List, Optional, Callable

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES - Immutable Value Objects
# =============================================================================

@dataclass(frozen=True)
class TrainingArtifact:
    """Immutable artifact reference passed via XCom"""
    artifact_type: str  # "dataset", "norm_stats", "contract", "model"
    path: str
    hash: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingArtifact":
        return cls(**data)


@dataclass
class StageResult:
    """Result of a pipeline stage"""
    success: bool
    stage_name: str
    duration_seconds: float
    artifacts: List[TrainingArtifact]
    metrics: Dict[str, Any]
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "stage_name": self.stage_name,
            "duration_seconds": self.duration_seconds,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "metrics": self.metrics,
            "error_message": self.error_message,
        }


# =============================================================================
# MLFLOW INTEGRATION - Observer Pattern
# =============================================================================

class MLflowTracker:
    """
    MLflow integration for experiment tracking.

    Implements Observer Pattern - observes training stages and logs metrics.

    Usage:
        tracker = MLflowTracker(experiment_name="ppo_usdcop")
        with tracker.start_run(run_name="training_001"):
            tracker.log_params({"lr": 0.0003})
            tracker.log_metrics({"reward": 123.5})
            tracker.log_artifact(model_path)
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        enabled: bool = True
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.enabled = enabled
        self._run = None
        self._mlflow = None

        if self.enabled:
            self._init_mlflow()

    def _init_mlflow(self) -> None:
        """Initialize MLflow with lazy import"""
        try:
            import mlflow
            self._mlflow = mlflow

            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)

            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
            mlflow.set_experiment(self.experiment_name)

            logger.info(f"MLflow initialized: experiment={self.experiment_name}")

        except ImportError:
            logger.warning("MLflow not installed. Tracking disabled.")
            self.enabled = False

    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for MLflow run"""
        return _MLflowRunContext(self, run_name, tags)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters"""
        if self.enabled and self._mlflow and self._run:
            for key, value in params.items():
                self._mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics"""
        if self.enabled and self._mlflow and self._run:
            for key, value in metrics.items():
                self._mlflow.log_metric(key, value, step=step)

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        """Log artifact file"""
        if self.enabled and self._mlflow and self._run:
            self._mlflow.log_artifact(path, artifact_path)

    def log_dict(self, data: Dict[str, Any], filename: str) -> None:
        """Log dictionary as JSON artifact"""
        if self.enabled and self._mlflow and self._run:
            self._mlflow.log_dict(data, filename)

    def set_tag(self, key: str, value: str) -> None:
        """Set run tag"""
        if self.enabled and self._mlflow and self._run:
            self._mlflow.set_tag(key, value)


class _MLflowRunContext:
    """Context manager for MLflow runs"""

    def __init__(self, tracker: MLflowTracker, run_name: str, tags: Optional[Dict[str, str]]):
        self.tracker = tracker
        self.run_name = run_name
        self.tags = tags or {}

    def __enter__(self):
        if self.tracker.enabled and self.tracker._mlflow:
            self.tracker._run = self.tracker._mlflow.start_run(run_name=self.run_name)
            for key, value in self.tags.items():
                self.tracker._mlflow.set_tag(key, value)
        return self.tracker

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tracker.enabled and self.tracker._mlflow and self.tracker._run:
            if exc_type:
                self.tracker.set_tag("status", "FAILED")
                self.tracker.set_tag("error", str(exc_val))
            else:
                self.tracker.set_tag("status", "SUCCESS")
            self.tracker._mlflow.end_run()
            self.tracker._run = None
        return False


# =============================================================================
# BASE TRAINING OPERATOR - Template Method Pattern
# =============================================================================

class BaseTrainingOperator(BaseOperator, ABC):
    """
    Base class for training operators using Template Method pattern.

    Subclasses implement:
    - _validate_inputs(): Validate XCom inputs
    - _execute_stage(): Main stage logic
    - _create_artifacts(): Create output artifacts
    """

    template_fields = ['version', 'experiment_name']
    ui_color = '#e8f4ea'
    ui_fgcolor = '#1a5928'

    @apply_defaults
    def __init__(
        self,
        version: str,
        experiment_name: str,
        project_root: str = "/opt/airflow",
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_enabled: bool = True,
        on_failure_callback: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.version = version
        self.experiment_name = experiment_name
        self.project_root = Path(project_root)
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_enabled = mlflow_enabled
        self._on_failure_callback = on_failure_callback

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Template method - defines the skeleton of the operation.

        Steps:
        1. Initialize MLflow tracker
        2. Validate inputs
        3. Execute stage logic
        4. Create and register artifacts
        5. Push results to XCom
        """
        start_time = time.time()
        stage_name = self.__class__.__name__

        # Initialize MLflow
        tracker = MLflowTracker(
            experiment_name=self.experiment_name,
            tracking_uri=self.mlflow_tracking_uri,
            enabled=self.mlflow_enabled,
        )

        try:
            # Step 1: Validate inputs
            logger.info(f"[{stage_name}] Validating inputs...")
            self._validate_inputs(context)

            # Step 2: Execute stage with MLflow tracking
            run_name = f"{stage_name}_{self.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            with tracker.start_run(run_name=run_name, tags={"version": self.version}):
                # Log stage start
                tracker.log_params({
                    "stage": stage_name,
                    "version": self.version,
                    "project_root": str(self.project_root),
                })

                logger.info(f"[{stage_name}] Executing stage logic...")
                result = self._execute_stage(context, tracker)

                # Step 3: Create artifacts
                logger.info(f"[{stage_name}] Creating artifacts...")
                artifacts = self._create_artifacts(context, result)

                # Log metrics to MLflow
                duration = time.time() - start_time
                tracker.log_metrics({
                    "duration_seconds": duration,
                    "success": 1.0,
                })

            # Step 4: Create stage result
            stage_result = StageResult(
                success=True,
                stage_name=stage_name,
                duration_seconds=duration,
                artifacts=artifacts,
                metrics=result.get("metrics", {}),
            )

            # Step 5: Push to XCom
            self._push_artifacts_to_xcom(context, artifacts)
            context['ti'].xcom_push(key=f'{stage_name}_result', value=stage_result.to_dict())

            logger.info(f"[{stage_name}] Completed in {duration:.1f}s")
            return stage_result.to_dict()

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[{stage_name}] Failed after {duration:.1f}s: {e}")

            # Log failure to MLflow
            tracker.log_metrics({"success": 0.0, "duration_seconds": duration})

            # Call failure callback if provided
            if self._on_failure_callback:
                self._on_failure_callback(context, e)

            raise

    @abstractmethod
    def _validate_inputs(self, context: Dict[str, Any]) -> None:
        """Validate inputs from XCom. Raises on failure."""
        pass

    @abstractmethod
    def _execute_stage(
        self,
        context: Dict[str, Any],
        tracker: MLflowTracker
    ) -> Dict[str, Any]:
        """Execute main stage logic. Returns result dict."""
        pass

    @abstractmethod
    def _create_artifacts(
        self,
        context: Dict[str, Any],
        result: Dict[str, Any]
    ) -> List[TrainingArtifact]:
        """Create output artifacts from result."""
        pass

    def _push_artifacts_to_xcom(
        self,
        context: Dict[str, Any],
        artifacts: List[TrainingArtifact]
    ) -> None:
        """Push artifacts to XCom for downstream tasks"""
        ti = context['ti']
        for artifact in artifacts:
            ti.xcom_push(key=f'{artifact.artifact_type}_artifact', value=artifact.to_dict())

    def _pull_artifact_from_xcom(
        self,
        context: Dict[str, Any],
        artifact_type: str,
        task_id: str
    ) -> Optional[TrainingArtifact]:
        """Pull artifact from XCom"""
        ti = context['ti']
        data = ti.xcom_pull(key=f'{artifact_type}_artifact', task_ids=task_id)
        if data:
            return TrainingArtifact.from_dict(data)
        return None


# =============================================================================
# HASH UTILITIES - Delegates to SSOT src.utils.hash_utils
# =============================================================================

def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file. SSOT: Delegates to src.utils.hash_utils"""
    return _compute_file_hash_ssot(file_path).full_hash


def compute_json_hash(file_path: Path) -> str:
    """Compute SHA256 hash of JSON content. SSOT: Delegates to src.utils.hash_utils"""
    return _compute_json_hash_ssot(file_path).full_hash
