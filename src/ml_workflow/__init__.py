"""
ML Workflow Module - Disciplined ML experiment tracking.
Contrato: CTR-008

Provides:
- Experiment tracking (MLWorkflowTracker)
- Dynamic contract generation (ContractFactory)
- Model registration (ModelRegistry)
- Artifact storage policy (MLflow-First + DVC-Tracked)
- Unified lineage tracking
- Promotion gate for model deployment

Principle: MLflow-First + DVC-Tracked
- Models: MLflow Model Registry (SSOT)
- Datasets: DVC versioned
- Metrics: MLflow + PostgreSQL
- Lineage: Unified in PostgreSQL

NOTE: Training pipeline has been moved to src/training/engine.py
      Use: from src.training import TrainingEngine, run_training
"""

from .experiment_tracker import MLWorkflowTracker, ExperimentLog

# Dynamic contract factory
from .dynamic_contract_factory import (
    DynamicFeatureContract,
    NormStatsCalculator,
    ContractFactory,
    ContractRegistry,
    create_contract_from_training,
    get_contract,
)

# Artifact policy (MLflow-First + DVC-Tracked)
from .artifact_policy import (
    ArtifactType,
    StorageBackend,
    ModelStage,
    ArtifactPolicy,
    ArtifactLocation,
    get_artifact_policy,
    enforce_mlflow_first,
    enforce_dvc_tracking,
)

# Lineage service
from .lineage_service import (
    LineageService,
    LineageRecord,
    LineageNode,
    LineageEdge,
    get_lineage_service,
)

# Promotion gate
from .promotion_gate import (
    PromotionGate,
    ValidationResult,
    ValidationIssue,
    MLflowFirstValidator,
    DVCTrackedValidator,
    create_default_gate,
    create_strict_gate,
)

# Re-export training from new location for backwards compatibility
# DEPRECATED: Import directly from src.training instead
from src.training import (
    TrainingEngine,
    TrainingRequest,
    run_training,
)

__all__ = [
    # Experiment tracking
    "MLWorkflowTracker",
    "ExperimentLog",
    # Dynamic contracts
    "DynamicFeatureContract",
    "NormStatsCalculator",
    "ContractFactory",
    "ContractRegistry",
    "create_contract_from_training",
    "get_contract",
    # Artifact policy
    "ArtifactType",
    "StorageBackend",
    "ModelStage",
    "ArtifactPolicy",
    "ArtifactLocation",
    "get_artifact_policy",
    "enforce_mlflow_first",
    "enforce_dvc_tracking",
    # Lineage
    "LineageService",
    "LineageRecord",
    "LineageNode",
    "LineageEdge",
    "get_lineage_service",
    # Promotion gate
    "PromotionGate",
    "ValidationResult",
    "ValidationIssue",
    "MLflowFirstValidator",
    "DVCTrackedValidator",
    "create_default_gate",
    "create_strict_gate",
    # Training (DEPRECATED - use src.training directly)
    "TrainingEngine",
    "TrainingRequest",
    "run_training",
]
