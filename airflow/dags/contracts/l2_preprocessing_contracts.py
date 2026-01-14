"""
L2 Preprocessing Pipeline Contracts
====================================

Pydantic contracts for Layer 2 (Preprocessing Pipeline) data flows.
Ensures type safety and validation between L0/L1 inputs and L3 outputs.

Contract: CTR-L2-001

Architecture:
    L0 (Data Acquisition) → L2 (Preprocessing) → L3 (Training)

Data Flow:
    OHLCV + Macro → Fusion → Cleaning → Resampling → RL Datasets

Version: 1.0.0
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# =============================================================================
# ENUMS
# =============================================================================


class DatasetType(str, Enum):
    """Available RL dataset types."""
    DS1_MINIMAL = "RL_DS1_MINIMAL"
    DS2_TECHNICAL_MTF = "RL_DS2_TECHNICAL_MTF"
    DS3_MACRO_CORE = "RL_DS3_MACRO_CORE"  # Primary production dataset
    DS4_COST_AWARE = "RL_DS4_COST_AWARE"
    DS5_REGIME = "RL_DS5_REGIME"
    DS6_CARRY_TRADE = "RL_DS6_CARRY_TRADE"
    DS7_COMMODITY_BASKET = "RL_DS7_COMMODITY_BASKET"
    DS8_RISK_SENTIMENT = "RL_DS8_RISK_SENTIMENT"
    DS9_FED_WATCH = "RL_DS9_FED_WATCH"
    DS10_FLOWS_FUNDAMENTALS = "RL_DS10_FLOWS_FUNDAMENTALS"


class TimeframeType(str, Enum):
    """Supported timeframes for datasets."""
    FIVE_MIN = "5min"
    FIFTEEN_MIN = "15min"
    DAILY = "daily"


class ValidationStatus(str, Enum):
    """Validation result status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class NormalizationMethod(str, Enum):
    """Feature normalization methods."""
    FIXED_ZSCORE = "fixed_zscore"
    ROLLING_ZSCORE = "rolling_zscore"
    MIN_MAX = "min_max"
    NONE = "none"


# =============================================================================
# INPUT CONTRACTS
# =============================================================================


class OHLCVRecord(BaseModel):
    """Single OHLCV bar record from L0.

    Contract: CTR-L2-INPUT-001
    Source: usdcop_m5_ohlcv table
    """
    model_config = ConfigDict(frozen=True)

    time: dt.datetime = Field(..., description="Bar timestamp (UTC)")
    symbol: str = Field(default="USD/COP", description="Trading symbol")
    open: Decimal = Field(..., ge=0, description="Open price")
    high: Decimal = Field(..., ge=0, description="High price")
    low: Decimal = Field(..., ge=0, description="Low price")
    close: Decimal = Field(..., ge=0, description="Close price")
    volume: Optional[int] = Field(default=None, ge=0, description="Volume (often 0 for FX)")
    source: str = Field(default="twelvedata", description="Data source")

    @model_validator(mode="after")
    def validate_ohlc_consistency(self) -> "OHLCVRecord":
        """Ensure high >= low and high >= open/close, low <= open/close."""
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) must be >= low ({self.low})")
        if self.high < self.open or self.high < self.close:
            raise ValueError("high must be >= open and close")
        if self.low > self.open or self.low > self.close:
            raise ValueError("low must be <= open and close")
        return self


class OHLCVBatch(BaseModel):
    """Batch of OHLCV records for pipeline processing.

    Contract: CTR-L2-INPUT-002
    """
    model_config = ConfigDict(frozen=True)

    records: List[OHLCVRecord] = Field(..., min_length=1)
    symbol: str = Field(default="USD/COP")
    timeframe: str = Field(default="5min")
    start_time: dt.datetime
    end_time: dt.datetime
    record_count: int

    @model_validator(mode="after")
    def validate_batch(self) -> "OHLCVBatch":
        """Validate batch consistency."""
        if len(self.records) != self.record_count:
            raise ValueError(
                f"record_count ({self.record_count}) doesn't match "
                f"actual records ({len(self.records)})"
            )
        if self.end_time < self.start_time:
            raise ValueError("end_time must be >= start_time")
        return self


class MacroIndicatorRecord(BaseModel):
    """Single macro indicator record from L0.

    Contract: CTR-L2-INPUT-003
    Source: macro_indicators_daily table
    """
    model_config = ConfigDict(frozen=True)

    date: dt.date = Field(..., description="Indicator date")

    # Dollar & FX
    dxy: Optional[float] = Field(default=None, description="Dollar Index")
    usdmxn: Optional[float] = Field(default=None, description="USD/MXN rate")
    usdclp: Optional[float] = Field(default=None, description="USD/CLP rate")

    # Volatility & Risk
    vix: Optional[float] = Field(default=None, ge=0, description="VIX Index")
    embi: Optional[float] = Field(default=None, description="EMBI Spread")

    # Commodities
    brent: Optional[float] = Field(default=None, ge=0, description="Brent Crude")
    wti: Optional[float] = Field(default=None, ge=0, description="WTI Crude")
    coffee: Optional[float] = Field(default=None, ge=0, description="Coffee price")
    gold: Optional[float] = Field(default=None, ge=0, description="Gold price")

    # Interest Rates (%)
    fed_funds: Optional[float] = Field(default=None, description="Fed Funds Rate")
    treasury_2y: Optional[float] = Field(default=None, description="US 2Y Treasury")
    treasury_10y: Optional[float] = Field(default=None, description="US 10Y Treasury")
    ibr: Optional[float] = Field(default=None, description="Colombia IBR rate")

    # Data quality
    source: str = Field(default="manual", description="Data source")
    is_complete: bool = Field(default=False, description="All fields populated")

    @model_validator(mode="after")
    def validate_critical_fields(self) -> "MacroIndicatorRecord":
        """At least critical fields should be present for production data."""
        critical = ["dxy", "vix", "embi", "treasury_10y", "treasury_2y"]
        present = sum(1 for f in critical if getattr(self, f) is not None)
        # Warning level: at least 3 critical fields
        if present < 3:
            # Don't raise, just mark as incomplete
            object.__setattr__(self, "is_complete", False)
        return self


class MacroBatch(BaseModel):
    """Batch of macro indicator records.

    Contract: CTR-L2-INPUT-004
    """
    model_config = ConfigDict(frozen=True)

    records: List[MacroIndicatorRecord] = Field(..., min_length=1)
    start_date: dt.date
    end_date: dt.date
    record_count: int
    completeness_pct: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def validate_batch(self) -> "MacroBatch":
        """Validate batch consistency."""
        if len(self.records) != self.record_count:
            raise ValueError(
                f"record_count ({self.record_count}) doesn't match "
                f"actual records ({len(self.records)})"
            )
        return self


# =============================================================================
# FEATURE CONTRACTS
# =============================================================================


class NormalizationStats(BaseModel):
    """Normalization statistics for a single feature.

    Contract: CTR-L2-NORM-001
    """
    model_config = ConfigDict(frozen=True)

    mean: float = Field(..., description="Feature mean from training period")
    std: float = Field(..., gt=0, description="Feature std (must be positive)")
    min_val: Optional[float] = Field(default=None, description="Observed minimum")
    max_val: Optional[float] = Field(default=None, description="Observed maximum")
    clip_range: Tuple[float, float] = Field(
        default=(-5.0, 5.0),
        description="Z-score clipping range"
    )


class FeatureSpec(BaseModel):
    """Specification for a single feature.

    Contract: CTR-L2-FEAT-001
    """
    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Feature name (e.g., 'rsi_9')")
    category: Literal[
        "returns", "momentum", "volatility", "trend",
        "macro_zscore", "macro_change", "state"
    ] = Field(..., description="Feature category")
    requires: List[str] = Field(
        default_factory=list,
        description="Required input columns"
    )
    normalization: NormalizationMethod = Field(
        default=NormalizationMethod.FIXED_ZSCORE
    )
    norm_stats: Optional[NormalizationStats] = Field(default=None)
    valid_range: Tuple[float, float] = Field(
        default=(-10.0, 10.0),
        description="Expected range after normalization"
    )
    allow_nan: bool = Field(
        default=False,
        description="Whether NaN is allowed in output"
    )


class FeatureContract(BaseModel):
    """Feature Contract - 15 dimensions.

    Contract: CTR-L2-FEAT
    SSOT for feature ordering and specification.
    """
    model_config = ConfigDict(frozen=True)

    version: str = Field(default="current")
    observation_dim: int = Field(default=15)

    feature_order: Tuple[str, ...] = Field(
        default=(
            # Returns (3)
            "log_ret_5m", "log_ret_1h", "log_ret_4h",
            # Technical (3)
            "rsi_9", "atr_pct", "adx_14",
            # Macro Z-scores (4)
            "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
            # Macro Changes (3)
            "brent_change_1d", "rate_spread", "usdmxn_change_1d",
            # State (2)
            "position", "time_normalized"
        )
    )

    features: Dict[str, FeatureSpec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_dimensions(self) -> "FeatureContract":
        """Ensure feature order matches observation dimension."""
        if len(self.feature_order) != self.observation_dim:
            raise ValueError(
                f"feature_order length ({len(self.feature_order)}) "
                f"must match observation_dim ({self.observation_dim})"
            )
        return self


# =============================================================================
# OUTPUT CONTRACTS
# =============================================================================


class DatasetQualityChecks(BaseModel):
    """Quality check results for a dataset.

    Contract: CTR-L2-QC-001
    """
    model_config = ConfigDict(frozen=True)

    no_nan_rows: bool = Field(..., description="No NaN values in dataset")
    temporal_ordered: bool = Field(..., description="Timestamps are sorted")
    no_duplicates: bool = Field(..., description="No duplicate timestamps")
    feature_ranges_valid: bool = Field(..., description="Features within bounds")
    min_rows_satisfied: bool = Field(..., description="Meets minimum row count")
    warmup_stripped: bool = Field(..., description="Warmup bars removed")

    @property
    def all_passed(self) -> bool:
        """Check if all quality checks passed."""
        return all([
            self.no_nan_rows,
            self.temporal_ordered,
            self.no_duplicates,
            self.feature_ranges_valid,
            self.min_rows_satisfied,
            self.warmup_stripped
        ])


class DatasetMetadata(BaseModel):
    """Metadata for a generated RL dataset.

    Contract: CTR-L2-OUTPUT-001
    """
    model_config = ConfigDict(frozen=True)

    name: DatasetType = Field(..., description="Dataset identifier")
    version: str = Field(default="v3.0", description="Dataset version")
    timeframe: TimeframeType = Field(default=TimeframeType.FIVE_MIN)

    # Shape
    row_count: int = Field(..., ge=0, description="Number of rows")
    column_count: int = Field(..., ge=0, description="Number of columns")
    feature_count: int = Field(..., ge=0, description="Number of features")

    # Date range
    start_date: dt.date = Field(..., description="First data point date")
    end_date: dt.date = Field(..., description="Last data point date")

    # Features
    feature_names: List[str] = Field(..., description="Ordered feature names")

    # Normalization
    normalization_method: NormalizationMethod = Field(
        default=NormalizationMethod.FIXED_ZSCORE
    )
    norm_stats_source: str = Field(
        default="training_period_2020_03_to_2025_10"
    )

    # Quality
    quality_checks: DatasetQualityChecks

    # Storage
    file_path: Optional[str] = Field(default=None)
    file_size_mb: Optional[float] = Field(default=None, ge=0)

    # Timestamps
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)


class DatasetBatch(BaseModel):
    """Collection of generated datasets from a pipeline run.

    Contract: CTR-L2-OUTPUT-002
    """
    model_config = ConfigDict(frozen=True)

    datasets_5min: List[DatasetMetadata] = Field(default_factory=list)
    datasets_15min: List[DatasetMetadata] = Field(default_factory=list)
    datasets_daily: List[DatasetMetadata] = Field(default_factory=list)

    primary_dataset: Optional[DatasetMetadata] = Field(
        default=None,
        description="Primary training dataset (typically DS3_MACRO_CORE)"
    )

    total_datasets: int = Field(..., ge=0)
    total_rows: int = Field(..., ge=0)
    total_size_mb: float = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_totals(self) -> "DatasetBatch":
        """Validate total counts match."""
        actual_total = (
            len(self.datasets_5min) +
            len(self.datasets_15min) +
            len(self.datasets_daily)
        )
        if actual_total != self.total_datasets:
            raise ValueError(
                f"total_datasets ({self.total_datasets}) doesn't match "
                f"actual count ({actual_total})"
            )
        return self


# =============================================================================
# PIPELINE CONTRACTS
# =============================================================================


class ExportResult(BaseModel):
    """Result of database export task.

    Contract: CTR-L2-TASK-001
    XCom key: export_results
    """
    model_config = ConfigDict(frozen=True)

    ohlcv_rows_exported: int = Field(..., ge=0)
    macro_rows_exported: int = Field(..., ge=0)
    ohlcv_file_path: str
    macro_file_path: str
    export_timestamp: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    export_duration_seconds: float = Field(..., ge=0)


class FusionResult(BaseModel):
    """Result of data fusion task.

    Contract: CTR-L2-TASK-002
    XCom key: fusion_results
    """
    model_config = ConfigDict(frozen=True)

    output_rows: int = Field(..., ge=0)
    ohlcv_rows_input: int = Field(..., ge=0)
    macro_rows_input: int = Field(..., ge=0)
    merge_method: Literal["asof", "left", "inner"] = Field(default="asof")
    output_file_path: str
    fusion_duration_seconds: float = Field(..., ge=0)


class CleaningResult(BaseModel):
    """Result of data cleaning task.

    Contract: CTR-L2-TASK-003
    XCom key: cleaning_results
    """
    model_config = ConfigDict(frozen=True)

    rows_input: int = Field(..., ge=0)
    rows_output: int = Field(..., ge=0)
    rows_removed: int = Field(..., ge=0)
    outliers_detected: int = Field(default=0, ge=0)
    nan_rows_removed: int = Field(default=0, ge=0)
    duplicate_rows_removed: int = Field(default=0, ge=0)
    output_file_path: str
    cleaning_duration_seconds: float = Field(..., ge=0)


class ResamplingResult(BaseModel):
    """Result of resampling task.

    Contract: CTR-L2-TASK-004
    XCom key: resampling_results
    """
    model_config = ConfigDict(frozen=True)

    input_rows: int = Field(..., ge=0)
    output_5min_rows: int = Field(default=0, ge=0)
    output_15min_rows: int = Field(default=0, ge=0)
    output_daily_rows: int = Field(default=0, ge=0)
    timeframes_generated: List[TimeframeType]
    output_directory: str
    resampling_duration_seconds: float = Field(..., ge=0)


class DatasetBuildResult(BaseModel):
    """Result of RL dataset builder task.

    Contract: CTR-L2-TASK-005
    XCom key: dataset_build_results
    """
    model_config = ConfigDict(frozen=True)

    datasets_generated: int = Field(..., ge=0)
    primary_dataset: str = Field(default="RL_DS3_MACRO_CORE")
    dataset_metadata: List[DatasetMetadata]
    output_directory: str
    build_duration_seconds: float = Field(..., ge=0)

    # Feature contract used
    feature_contract_version: str = Field(default="current")
    observation_dim: int = Field(default=15)


class ValidationResult(BaseModel):
    """Result of validation task.

    Contract: CTR-L2-TASK-006
    XCom key: validation_results
    """
    model_config = ConfigDict(frozen=True)

    status: ValidationStatus
    datasets_validated: int = Field(..., ge=0)
    passed_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    warning_count: int = Field(default=0, ge=0)

    issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    validation_metrics: Dict[str, Any] = Field(default_factory=dict)
    validation_duration_seconds: float = Field(..., ge=0)

    @property
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return self.status == ValidationStatus.PASSED


class PipelineSummary(BaseModel):
    """Summary of complete L2 pipeline execution.

    Contract: CTR-L2-SUMMARY-001
    XCom key: pipeline_summary
    """
    model_config = ConfigDict(frozen=True)

    dag_run_id: str
    execution_date: dt.datetime

    # Task results
    export_result: ExportResult
    fusion_result: FusionResult
    cleaning_result: CleaningResult
    resampling_result: ResamplingResult
    dataset_build_result: DatasetBuildResult
    validation_result: ValidationResult

    # Aggregated metrics
    total_duration_seconds: float = Field(..., ge=0)
    total_rows_processed: int = Field(..., ge=0)
    total_datasets_generated: int = Field(..., ge=0)
    total_size_mb: float = Field(..., ge=0)

    # Status
    success: bool
    error_message: Optional[str] = Field(default=None)

    # Timestamps
    started_at: dt.datetime
    completed_at: dt.datetime


# =============================================================================
# XCOM COMMUNICATION CONTRACTS
# =============================================================================


class L2XComKeys:
    """Standard XCom keys for L2 pipeline tasks.

    Contract: CTR-L2-XCOM-001
    Usage:
        ti.xcom_push(key=L2XComKeys.EXPORT, value=export_result.model_dump())
        result = ExportResult(**ti.xcom_pull(key=L2XComKeys.EXPORT))
    """
    EXPORT = "export_results"
    FUSION = "fusion_results"
    CLEANING = "cleaning_results"
    RESAMPLING = "resampling_results"
    DATASET_BUILD = "dataset_build_results"
    VALIDATION = "validation_results"
    PIPELINE_SUMMARY = "pipeline_summary"


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_feature_contract() -> FeatureContract:
    """Create feature contract with full specifications.

    Returns:
        FeatureContract with all feature specs populated.
    """
    features = {
        "log_ret_5m": FeatureSpec(
            name="log_ret_5m",
            category="returns",
            requires=["close"],
            normalization=NormalizationMethod.NONE,
            valid_range=(-0.1, 0.1)
        ),
        "log_ret_1h": FeatureSpec(
            name="log_ret_1h",
            category="returns",
            requires=["close"],
            normalization=NormalizationMethod.NONE,
            valid_range=(-0.2, 0.2)
        ),
        "log_ret_4h": FeatureSpec(
            name="log_ret_4h",
            category="returns",
            requires=["close"],
            normalization=NormalizationMethod.NONE,
            valid_range=(-0.3, 0.3)
        ),
        "rsi_9": FeatureSpec(
            name="rsi_9",
            category="momentum",
            requires=["close"],
            normalization=NormalizationMethod.FIXED_ZSCORE,
            valid_range=(0, 100)
        ),
        "atr_pct": FeatureSpec(
            name="atr_pct",
            category="volatility",
            requires=["high", "low", "close"],
            normalization=NormalizationMethod.FIXED_ZSCORE,
            valid_range=(0, 10)
        ),
        "adx_14": FeatureSpec(
            name="adx_14",
            category="trend",
            requires=["high", "low", "close"],
            normalization=NormalizationMethod.FIXED_ZSCORE,
            valid_range=(0, 100)
        ),
        "dxy_z": FeatureSpec(
            name="dxy_z",
            category="macro_zscore",
            requires=["dxy"],
            normalization=NormalizationMethod.ROLLING_ZSCORE,
            norm_stats=NormalizationStats(mean=100.21, std=5.60),
            valid_range=(-4, 4)
        ),
        "dxy_change_1d": FeatureSpec(
            name="dxy_change_1d",
            category="macro_change",
            requires=["dxy"],
            normalization=NormalizationMethod.NONE,
            valid_range=(-0.1, 0.1)
        ),
        "vix_z": FeatureSpec(
            name="vix_z",
            category="macro_zscore",
            requires=["vix"],
            normalization=NormalizationMethod.ROLLING_ZSCORE,
            norm_stats=NormalizationStats(mean=21.16, std=7.89),
            valid_range=(-4, 4)
        ),
        "embi_z": FeatureSpec(
            name="embi_z",
            category="macro_zscore",
            requires=["embi"],
            normalization=NormalizationMethod.ROLLING_ZSCORE,
            norm_stats=NormalizationStats(mean=322.01, std=62.68),
            valid_range=(-4, 4)
        ),
        "brent_change_1d": FeatureSpec(
            name="brent_change_1d",
            category="macro_change",
            requires=["brent"],
            normalization=NormalizationMethod.NONE,
            valid_range=(-0.1, 0.1)
        ),
        "rate_spread": FeatureSpec(
            name="rate_spread",
            category="macro_change",
            requires=["treasury_10y", "treasury_2y"],
            normalization=NormalizationMethod.FIXED_ZSCORE,
            valid_range=(-3, 3)
        ),
        "usdmxn_change_1d": FeatureSpec(
            name="usdmxn_change_1d",
            category="macro_change",
            requires=["usdmxn"],
            normalization=NormalizationMethod.NONE,
            valid_range=(-0.1, 0.1)
        ),
        "position": FeatureSpec(
            name="position",
            category="state",
            requires=[],
            normalization=NormalizationMethod.NONE,
            valid_range=(-1, 1)
        ),
        "time_normalized": FeatureSpec(
            name="time_normalized",
            category="state",
            requires=[],
            normalization=NormalizationMethod.NONE,
            valid_range=(0, 1)
        ),
    }

    return FeatureContract(features=features)


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "DatasetType",
    "TimeframeType",
    "ValidationStatus",
    "NormalizationMethod",
    # Input contracts
    "OHLCVRecord",
    "OHLCVBatch",
    "MacroIndicatorRecord",
    "MacroBatch",
    # Feature contracts
    "NormalizationStats",
    "FeatureSpec",
    "FeatureContract",
    # Output contracts
    "DatasetQualityChecks",
    "DatasetMetadata",
    "DatasetBatch",
    # Task contracts
    "ExportResult",
    "FusionResult",
    "CleaningResult",
    "ResamplingResult",
    "DatasetBuildResult",
    "ValidationResult",
    "PipelineSummary",
    # XCom
    "L2XComKeys",
    # Factory
    "create_feature_contract",
]
