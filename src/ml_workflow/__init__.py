"""
ML Workflow Module - Disciplined ML experiment tracking.
Contrato: CTR-008

Provides:
- Experiment tracking (MLWorkflowTracker)
- Dynamic contract generation (ContractFactory)
- End-to-end training pipeline (TrainingPipeline)
- Model registration (ModelRegistry)
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

# Training pipeline
from .training_pipeline import (
    TrainingConfig,
    TrainingPipeline,
    PipelineResult,
    run_training,
    FEATURE_ORDER,
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
    # Training pipeline
    "TrainingConfig",
    "TrainingPipeline",
    "PipelineResult",
    "run_training",
    "FEATURE_ORDER",
]
