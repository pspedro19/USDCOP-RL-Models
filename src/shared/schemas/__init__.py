"""
Shared Schema System
====================

Single Source of Truth for API contracts between:
- Backend (Pydantic models)
- Frontend (TypeScript/Zod schemas)

Architecture:
    shared/schemas/          - Schema definitions (SSOT)
        core.py             - Core domain models
        api.py              - API request/response models
        trading.py          - Trading domain models
        features.py         - Feature contracts

    Generated outputs:
        - services/inference_api/models/  (Pydantic - backend)
        - usdcop-trading-dashboard/types/generated/  (TypeScript)

Usage:
    # Python
    from shared.schemas import TradeSchema, SignalSchema

    # Generate TypeScript
    python -m shared.schemas.codegen --output ../dashboard/types/generated/

Contract: CTR-SHARED-001
Version: 1.0.0
"""

from .core import (
    # Enums
    SignalType,
    TradeSide,
    TradeStatus,
    OrderSide,
    MarketStatus,
    DataSource,
    # Core models
    BaseSchema,
    TimestampedSchema,
)

from .trading import (
    # Trade models
    TradeSchema,
    TradeMetadataSchema,
    TradeSummarySchema,
    # Signal models
    SignalSchema,
    # Market models
    CandlestickSchema,
    MarketContextSchema,
)

from .features import (
    # Feature contracts (new names)
    FeatureOrderType,
    NamedFeatures,
    FeatureSnapshotSchema,
    NormalizationStatsSchema,
    ObservationSchema,
    OBSERVATION_DIM,
    FEATURE_ORDER,
    validate_observation,
)

from .api import (
    # Request models
    BacktestRequestSchema,
    InferenceRequestSchema,
    ReplayLoadRequestSchema,
    # Response models
    ApiResponseSchema,
    ApiMetadataSchema,
    BacktestResponseSchema,
    HealthResponseSchema,
    ErrorResponseSchema,
    ProgressUpdateSchema,
    ReplayLoadResponseSchema,
    ModelInfoSchema,
    ModelsResponseSchema,
    # Wrappers
    create_api_response,
)

__all__ = [
    # Core
    "SignalType",
    "TradeSide",
    "TradeStatus",
    "OrderSide",
    "MarketStatus",
    "DataSource",
    "BaseSchema",
    "TimestampedSchema",
    # Trading
    "TradeSchema",
    "TradeMetadataSchema",
    "TradeSummarySchema",
    "SignalSchema",
    "CandlestickSchema",
    "MarketContextSchema",
    # Features
    "FeatureOrderType",
    "NamedFeatures",
    "FeatureSnapshotSchema",
    "NormalizationStatsSchema",
    "ObservationSchema",
    "OBSERVATION_DIM",
    "FEATURE_ORDER",
    "validate_observation",
    # API
    "BacktestRequestSchema",
    "InferenceRequestSchema",
    "ReplayLoadRequestSchema",
    "ApiResponseSchema",
    "ApiMetadataSchema",
    "BacktestResponseSchema",
    "HealthResponseSchema",
    "ErrorResponseSchema",
    "ProgressUpdateSchema",
    "ReplayLoadResponseSchema",
    "ModelInfoSchema",
    "ModelsResponseSchema",
    "create_api_response",
]
