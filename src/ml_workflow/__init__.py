"""
ML Workflow Module - Disciplined ML experiment tracking.
Contrato: CTR-008

Provides:
- Experiment tracking (MLWorkflowTracker)
- Dynamic contract generation (ContractFactory)
- Model registration (ModelRegistry)

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
    # Training (DEPRECATED - use src.training directly)
    "TrainingEngine",
    "TrainingRequest",
    "run_training",
]
