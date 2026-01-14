"""
Pipeline Contracts Module
=========================
Pydantic models defining contracts between pipeline stages.

Layers:
    L0: Data Acquisition (raw OHLCV, macro)
    L1: Feature Calculation
    L2: Preprocessing Pipeline (fusion, cleaning, resampling, datasets)
    L3: Model Training
    L4: Validation/Backtesting
    L5: Multi-Model Inference

Contract Modules:
    - l0_data_contracts: L0 data acquisition contracts
    - l1_feature_contracts: L1 feature calculation contracts
    - l2_preprocessing_contracts: L2 pipeline contracts
    - l3_training_contracts: L3 training pipeline contracts
    - l5_inference_contracts: L5 inference contracts
    - backtest_contracts: L3/L4 backtest contracts

Contract Coverage: 100% (All layers have formal Pydantic contracts)
"""

# =============================================================================
# L0 DATA CONTRACTS
# =============================================================================
from .l0_data_contracts import (
    # Enums
    L0XComKeys,
    DataSourceType,
    AcquisitionStatus,
    TradingDayStatus,
    # Input contracts
    TwelveDataBar,
    TwelveDataResponse,
    FREDSeriesValue,
    MacroAPIResponse,
    # Output contracts
    OHLCVRecord as L0OHLCVRecord,
    MacroIndicatorRecord as L0MacroRecord,
    # Task contracts
    OHLCVAcquisitionResult,
    MacroAcquisitionResult,
    MacroMergeResult,
    DataValidationResult,
    BackupResult,
    GapDetectionResult,
    # Config contracts
    TwelveDataConfig,
    FREDConfig,
    MarketHoursConfig,
    # Quality contracts
    L0QualityReport,
    # Factories
    create_ohlcv_acquisition_result,
    create_macro_acquisition_result,
)

# =============================================================================
# L1 FEATURE CONTRACTS
# =============================================================================
from .l1_feature_contracts import (
    # Enums
    L1XComKeys,
    FeatureCalculationStatus,
    FeatureType,
    # Feature definitions
    FeatureDefinition,
    CORE_FEATURE_DEFINITIONS,
    FeatureContract as L1FeatureContract,
    FEATURE_CONTRACT,
    # Input contracts
    OHLCVInput,
    OHLCVBatchInput,
    MacroInput,
    MacroZScoreConfig,
    # Output contracts
    CalculatedFeatures,
    FeatureCalculationResult,
    FeatureValidationResult,
    # Sensor contracts
    NewBarDetection,
    # Quality contracts
    L1QualityReport,
    # Factories
    create_feature_calculation_result,
    create_calculated_features_from_dict,
)

# =============================================================================
# L2 PREPROCESSING CONTRACTS
# =============================================================================
from .l2_preprocessing_contracts import (
    # Enums
    DatasetType,
    TimeframeType,
    NormalizationMethod,
    ValidationStatus,
    L2XComKeys,
    # Input contracts
    OHLCVRecord,
    OHLCVBatch,
    MacroIndicatorRecord,
    MacroBatch,
    # Feature contracts
    NormalizationStats,
    FeatureSpec,
    FeatureContract as L2FeatureContract,
    # Output contracts
    DatasetQualityChecks,
    DatasetMetadata,
    DatasetBatch,
    ValidationResult as L2ValidationResult,
    # Task contracts
    ExportResult,
    FusionResult,
    CleaningResult,
    ResamplingResult,
    DatasetBuildResult,
    PipelineSummary as L2PipelineSummary,
    # Factory
    create_feature_contract,
)

# =============================================================================
# L3 TRAINING CONTRACTS
# =============================================================================
from .l3_training_contracts import (
    # Enums
    L3XComKeys,
    TrainingStatus,
    ModelStatus,
    DatasetSplit,
    # Config contracts
    TrainingConfig,
    PPOHyperparameters,
    EnvironmentConfig,
    # Dataset contracts
    DatasetInfo,
    DatasetValidationResult as L3DatasetValidationResult,
    DatasetSplitInfo,
    # Normalization contracts
    FeatureNormStats,
    NormStatsMetadata,
    NormalizationStats as L3NormalizationStats,
    NormStatsResult,
    # Contract contracts
    FeatureContract,
    ContractResult,
    # Training contracts
    TrainingMetrics,
    TrainingResult,
    # Registration contracts
    ModelRegistration,
    RegistrationResult,
    # Backtest contracts
    BacktestConfig as L3BacktestConfig,
    BacktestMetrics as L3BacktestMetrics,
    BacktestResult as L3BacktestResult,
    # Summary contracts
    PipelineSummary as L3PipelineSummary,
    L3QualityReport,
    # Factories
    create_training_result,
    create_pipeline_summary,
)

# =============================================================================
# L5 INFERENCE CONTRACTS
# =============================================================================
from .l5_inference_contracts import (
    # Enums
    L5XComKeys,
    SignalAction,
    SystemStatus,
    ModelType,
    TradeStatus,
    # Observation contracts
    ObservationContract,
    OBSERVATION_CONTRACT,
    MarketFeatures,
    StateFeatures,
    FullObservation,
    # Model contracts
    ModelConfig,
    ModelLoadResult,
    # Inference contracts
    InferenceResult,
    BatchInferenceResult,
    # Trade contracts
    TradeSignal,
    ExecutedTrade,
    # Risk contracts
    RiskLimits,
    RiskValidationResult,
    # Monitoring contracts
    ModelHealth,
    L5HealthReport,
    # Task contracts
    SystemReadinessResult,
    ModelLoadSummary,
    ObservationBuildResult,
    InferenceSummary,
    # Factories
    create_inference_result,
    create_full_observation,
)

# =============================================================================
# BACKTEST CONTRACTS (L4)
# =============================================================================
from .backtest_contracts import (
    # Enums
    BacktestPeriodType,
    BacktestStatus,
    AlertSeverity,
    ValidationResult,
    # Base
    BaseContract,
    TimestampedContract,
    # Input Contracts
    BacktestRequest,
    ValidationThresholds,
    ModelComparisonRequest,
    # Output Contracts
    TradeRecord,
    BacktestMetrics,
    BacktestResult,
    ValidationCheckResult,
    ValidationReport,
    ModelComparisonResult,
    # Alert Contracts
    Alert,
    # Pipeline Context
    PipelineContext,
    # Factory Configs
    BacktestConfig,
    StrategyConfig,
)

# =============================================================================
# EXPORTS
# =============================================================================
__all__ = [
    # =========================================================================
    # L0 Data Acquisition
    # =========================================================================
    "L0XComKeys",
    "DataSourceType",
    "AcquisitionStatus",
    "TradingDayStatus",
    "TwelveDataBar",
    "TwelveDataResponse",
    "FREDSeriesValue",
    "MacroAPIResponse",
    "L0OHLCVRecord",
    "L0MacroRecord",
    "OHLCVAcquisitionResult",
    "MacroAcquisitionResult",
    "MacroMergeResult",
    "DataValidationResult",
    "BackupResult",
    "GapDetectionResult",
    "TwelveDataConfig",
    "FREDConfig",
    "MarketHoursConfig",
    "L0QualityReport",
    "create_ohlcv_acquisition_result",
    "create_macro_acquisition_result",

    # =========================================================================
    # L1 Feature Calculation
    # =========================================================================
    "L1XComKeys",
    "FeatureCalculationStatus",
    "FeatureType",
    "FeatureDefinition",
    "CORE_FEATURE_DEFINITIONS",
    "L1FeatureContract",
    "FEATURE_CONTRACT",
    "OHLCVInput",
    "OHLCVBatchInput",
    "MacroInput",
    "MacroZScoreConfig",
    "CalculatedFeatures",
    "FeatureCalculationResult",
    "FeatureValidationResult",
    "NewBarDetection",
    "L1QualityReport",
    "create_feature_calculation_result",
    "create_calculated_features_from_dict",

    # =========================================================================
    # L2 Preprocessing
    # =========================================================================
    "DatasetType",
    "TimeframeType",
    "NormalizationMethod",
    "ValidationStatus",
    "L2XComKeys",
    "OHLCVRecord",
    "OHLCVBatch",
    "MacroIndicatorRecord",
    "MacroBatch",
    "NormalizationStats",
    "FeatureSpec",
    "L2FeatureContract",
    "DatasetQualityChecks",
    "DatasetMetadata",
    "DatasetBatch",
    "L2ValidationResult",
    "ExportResult",
    "FusionResult",
    "CleaningResult",
    "ResamplingResult",
    "DatasetBuildResult",
    "L2PipelineSummary",
    "create_feature_contract",

    # =========================================================================
    # L3 Training
    # =========================================================================
    "L3XComKeys",
    "TrainingStatus",
    "ModelStatus",
    "DatasetSplit",
    "TrainingConfig",
    "PPOHyperparameters",
    "EnvironmentConfig",
    "DatasetInfo",
    "L3DatasetValidationResult",
    "DatasetSplitInfo",
    "FeatureNormStats",
    "NormStatsMetadata",
    "L3NormalizationStats",
    "NormStatsResult",
    "FeatureContract",
    "ContractResult",
    "TrainingMetrics",
    "TrainingResult",
    "ModelRegistration",
    "RegistrationResult",
    "L3BacktestConfig",
    "L3BacktestMetrics",
    "L3BacktestResult",
    "L3PipelineSummary",
    "L3QualityReport",
    "create_training_result",
    "create_pipeline_summary",

    # =========================================================================
    # L5 Inference
    # =========================================================================
    "L5XComKeys",
    "SignalAction",
    "SystemStatus",
    "ModelType",
    "TradeStatus",
    "ObservationContract",
    "OBSERVATION_CONTRACT",
    "MarketFeatures",
    "StateFeatures",
    "FullObservation",
    "ModelConfig",
    "ModelLoadResult",
    "InferenceResult",
    "BatchInferenceResult",
    "TradeSignal",
    "ExecutedTrade",
    "RiskLimits",
    "RiskValidationResult",
    "ModelHealth",
    "L5HealthReport",
    "SystemReadinessResult",
    "ModelLoadSummary",
    "ObservationBuildResult",
    "InferenceSummary",
    "create_inference_result",
    "create_full_observation",

    # =========================================================================
    # L4 Backtest (Legacy)
    # =========================================================================
    "BacktestPeriodType",
    "BacktestStatus",
    "AlertSeverity",
    "ValidationResult",
    "BaseContract",
    "TimestampedContract",
    "BacktestRequest",
    "ValidationThresholds",
    "ModelComparisonRequest",
    "TradeRecord",
    "BacktestMetrics",
    "BacktestResult",
    "ValidationCheckResult",
    "ValidationReport",
    "ModelComparisonResult",
    "Alert",
    "PipelineContext",
    "BacktestConfig",
    "StrategyConfig",
]
