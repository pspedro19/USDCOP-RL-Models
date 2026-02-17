"""Contract definitions for the USD/COP RL Trading System."""

# Action Contract
from .action_contract import (
    Action,
    InvalidActionError,
    validate_model_output,
    MODEL_OUTPUT_CONTRACT,
    ModelOutputContract,
    ACTION_CONTRACT_VERSION,
    VALID_ACTIONS,
    ACTION_NAMES,
    ACTION_COUNT,
    ACTION_PROBS_DIM,
    ACTION_SELL,
    ACTION_HOLD,
    ACTION_BUY,
)

# Feature Contract
from .feature_contract import (
    FEATURE_ORDER,
    OBSERVATION_DIM,
    FEATURE_CONTRACT_VERSION,
    FEATURE_ORDER_HASH,
    FEATURE_SPECS,
    FEATURE_CONTRACT,
    FeatureType,
    FeatureUnit,
    FeatureSpec,
    FeatureContract,
    FeatureContractError,
    validate_feature_vector,
    get_feature_index,
    get_feature_names,
    features_dict_to_array,
)

# Model Input Contract
from .model_input_contract import (
    MODEL_INPUT_CONTRACT,
    ModelInputContract,
    ModelInputError,
    ObservationValidator,
    validate_model_input,
)

# Model Metadata Contract
from .model_metadata_contract import (
    ModelMetadata,
    validate_model_metadata,
)

# Training Run Contract
from .training_run_contract import (
    TRAINING_RUN_CONTRACT,
    TrainingRunContract,
    TrainingRunValidator,
    TrainingContractError,
)

# Norm Stats Contract
from .norm_stats_contract import (
    NORM_STATS_CONTRACT_VERSION,
    FeatureStats,
    NormStatsMetadata,
    NormStatsContract,
    NormStatsContractError,
    load_norm_stats,
    save_norm_stats,
)

# Storage Contracts
from .storage_contracts import (
    STORAGE_CONTRACT_VERSION,
    EXPERIMENTS_BUCKET,
    PRODUCTION_BUCKET,
    DVC_BUCKET,
    LineageRecord,
    DatasetSnapshot,
    ModelSnapshot,
    BacktestSnapshot,
    ABComparisonSnapshot,
    compute_content_hash,
    compute_schema_hash,
    compute_json_hash,
    parse_s3_uri,
    build_s3_uri,
)

# Experiment Contract (from YAML SSOT)
from .experiment_contract import (
    EXPERIMENT_CONTRACT_VERSION,
    ExperimentContract,
    ExperimentContractError,
    load_experiment_contract,
)

# Promotion Contract (L4 -> Dashboard)
from .promotion_contract import (
    PROMOTION_CONTRACT_VERSION,
    PromotionRecommendation,
    PromotionStatus,
    PromotionContractError,
    BacktestMetrics,
    CriteriaResult,
    PromotionProposal,
    evaluate_criteria,
    determine_recommendation,
)

# Production Contract (for L1/L5)
from .production_contract import (
    PRODUCTION_CONTRACT_VERSION,
    ProductionContract,
    ProductionContractError,
    get_production_contract,
    promote_model_to_production,
    archive_production_model,
)

__all__ = [
    # Action Contract
    "Action",
    "InvalidActionError",
    "validate_model_output",
    "MODEL_OUTPUT_CONTRACT",
    "ModelOutputContract",
    "ACTION_CONTRACT_VERSION",
    "VALID_ACTIONS",
    "ACTION_NAMES",
    "ACTION_COUNT",
    "ACTION_PROBS_DIM",
    "ACTION_SELL",
    "ACTION_HOLD",
    "ACTION_BUY",
    # Feature Contract
    "FEATURE_ORDER",
    "OBSERVATION_DIM",
    "FEATURE_CONTRACT_VERSION",
    "FEATURE_ORDER_HASH",
    "FEATURE_SPECS",
    "FEATURE_CONTRACT",
    "FeatureType",
    "FeatureUnit",
    "FeatureSpec",
    "FeatureContract",
    "FeatureContractError",
    "validate_feature_vector",
    "get_feature_index",
    "get_feature_names",
    "features_dict_to_array",
    # Model Input Contract
    "MODEL_INPUT_CONTRACT",
    "ModelInputContract",
    "ModelInputError",
    "ObservationValidator",
    "validate_model_input",
    # Model Metadata Contract
    "ModelMetadata",
    "validate_model_metadata",
    # Training Run Contract
    "TRAINING_RUN_CONTRACT",
    "TrainingRunContract",
    "TrainingRunValidator",
    "TrainingContractError",
    # Norm Stats Contract
    "NORM_STATS_CONTRACT_VERSION",
    "FeatureStats",
    "NormStatsMetadata",
    "NormStatsContract",
    "NormStatsContractError",
    "load_norm_stats",
    "save_norm_stats",
    # Storage Contracts
    "STORAGE_CONTRACT_VERSION",
    "EXPERIMENTS_BUCKET",
    "PRODUCTION_BUCKET",
    "DVC_BUCKET",
    "LineageRecord",
    "DatasetSnapshot",
    "ModelSnapshot",
    "BacktestSnapshot",
    "ABComparisonSnapshot",
    "compute_content_hash",
    "compute_schema_hash",
    "compute_json_hash",
    "parse_s3_uri",
    "build_s3_uri",
    # Experiment Contract
    "EXPERIMENT_CONTRACT_VERSION",
    "ExperimentContract",
    "ExperimentContractError",
    "load_experiment_contract",
    # Promotion Contract
    "PROMOTION_CONTRACT_VERSION",
    "PromotionRecommendation",
    "PromotionStatus",
    "PromotionContractError",
    "BacktestMetrics",
    "CriteriaResult",
    "PromotionProposal",
    "evaluate_criteria",
    "determine_recommendation",
    # Production Contract
    "PRODUCTION_CONTRACT_VERSION",
    "ProductionContract",
    "ProductionContractError",
    "get_production_contract",
    "promote_model_to_production",
    "archive_production_model",
]
