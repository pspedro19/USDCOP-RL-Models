"""
Services module - Feature building, calculation, inference, and pipeline services.

Components:
- FeatureBuilder: Production feature builder
- FeatureBuilderRefactored: SOLID-compliant version with design patterns
- InferenceService: Model inference with proper DI (ApplicationContext)
- DVCService: Automatic dataset versioning with DVC/MinIO
- LineageTracker: Unified artifact/hash tracking across pipeline
- ContractValidator: L1→L3→L5 contract validation

Usage:
    # Feature building (production)
    from src.core.services import FeatureBuilder

    # Inference with DI (recommended)
    from src.core.services import InferenceService
    from src.core.container import ApplicationContext

    context = ApplicationContext.create_production(config)
    service = InferenceService(context)
    result = service.run_inference(market_data, position=0.0, bar_idx=30)

    # DVC versioning
    from src.core.services import DVCService, DVCTag
    dvc = DVCService(project_root=Path("."))
    result = dvc.add_and_push(dataset_path, tag=DVCTag.for_experiment("exp_v1"))

    # Lineage tracking
    from src.core.services import LineageTracker
    tracker = LineageTracker(project_root=Path("."))
    lineage = tracker.track_experiment(...)

    # Contract validation
    from src.core.services import ContractValidator
    validator = ContractValidator()
    result = validator.validate_l5_inference(model_feature_order_hash=hash)
"""

from .feature_builder import FeatureBuilder, create_feature_builder

# DVC Service for dataset versioning (GAP 1)
from .dvc_service import (
    DVCService,
    DVCTag,
    DVCResult,
    create_dvc_service,
)

# Lineage Tracker for artifact/hash tracking (GAP 2, 4, 5)
from .lineage_tracker import (
    LineageTracker,
    LineageRecord,
    LineageTrackerBuilder,
    create_lineage_tracker,
)

# Contract Validator for L1→L3→L5 validation (GAP 3, 10)
from .contract_validator import (
    ContractValidator,
    ValidationResult,
    ValidationError,
    ValidationSeverity,
    PipelineStage,
    ContractValidationError,
    validate_contract,
    create_contract_validator,
)

# SOLID refactored version (optional, Phase 6)
# Note: FeatureBuilderRefactored is conditionally imported to avoid ImportError
# if the module is not yet implemented
try:
    from .feature_builder_refactored import FeatureBuilderRefactored
except ImportError:
    FeatureBuilderRefactored = None  # Not implemented yet

# Inference service with proper DI (P0-5)
from .inference_service import (
    InferenceService,
    InferenceServiceResult,
    InferenceRequestedEvent,
    InferenceCompletedEvent,
    TradeSignalGeneratedEvent,
    create_inference_service,
    create_inference_service_from_config,
)

__all__ = [
    # Feature builders
    'FeatureBuilder',
    'create_feature_builder',
    'FeatureBuilderRefactored',

    # Inference service (DI pattern)
    'InferenceService',
    'InferenceServiceResult',
    'InferenceRequestedEvent',
    'InferenceCompletedEvent',
    'TradeSignalGeneratedEvent',
    'create_inference_service',
    'create_inference_service_from_config',

    # DVC Service (GAP 1)
    'DVCService',
    'DVCTag',
    'DVCResult',
    'create_dvc_service',

    # Lineage Tracker (GAP 2, 4, 5)
    'LineageTracker',
    'LineageRecord',
    'LineageTrackerBuilder',
    'create_lineage_tracker',

    # Contract Validator (GAP 3, 10)
    'ContractValidator',
    'ValidationResult',
    'ValidationError',
    'ValidationSeverity',
    'PipelineStage',
    'ContractValidationError',
    'validate_contract',
    'create_contract_validator',
]
