"""
Services Package
Contract: CTR-SVC

Business logic services following Single Responsibility Principle.

Services:
- BacktestFeatureBuilder: Builds features for historical backtest periods
- L1NRTDataService: NRT data layer - maintains inference_ready_nrt
- L5NRTInferenceService: NRT inference layer - pure model.predict()
"""

from src.services.backtest_feature_builder import (
    BacktestFeatureBuilder,
    FeatureBuildConfig,
)

# NRT Services (lazy import to avoid asyncpg dependency if not needed)
try:
    from src.services.l1_nrt_data_service import (
        L1NRTDataService,
        L1NRTConfig,
        L1NRTError,
        NormStatsHashMismatchError,
    )
except ImportError:
    L1NRTDataService = None  # type: ignore
    L1NRTConfig = None  # type: ignore
    L1NRTError = None  # type: ignore
    NormStatsHashMismatchError = None  # type: ignore

try:
    from src.services.l5_nrt_inference_service import (
        L5NRTInferenceService,
        L5NRTConfig,
    )
except ImportError:
    L5NRTInferenceService = None  # type: ignore
    L5NRTConfig = None  # type: ignore

__all__ = [
    # Backtest
    "BacktestFeatureBuilder",
    "FeatureBuildConfig",
    # NRT L1
    "L1NRTDataService",
    "L1NRTConfig",
    "L1NRTError",
    "NormStatsHashMismatchError",
    # NRT L5
    "L5NRTInferenceService",
    "L5NRTConfig",
]
