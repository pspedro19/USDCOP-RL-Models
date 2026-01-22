"""
SSOT API Router
===============

Exposes Single Source of Truth values for frontend consumption.
Allows frontend to validate its local SSOT against backend at runtime.

Endpoints:
- GET /ssot - Complete SSOT summary
- GET /ssot/features - Feature contract
- GET /ssot/actions - Action contract
- GET /ssot/validate - Validate frontend SSOT against backend

Author: Trading Team
Date: 2026-01-18
"""

import hashlib
import json
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Import SSOT from contracts
from src.core.contracts import (
    FEATURE_ORDER,
    FEATURE_ORDER_HASH,
    OBSERVATION_DIM,
)
from src.core.contracts.action_contract import (
    Action,
    ACTION_COUNT,
    ACTION_NAMES,
    VALID_ACTIONS,
    ACTION_CONTRACT_VERSION,
)
from src.core.contracts.feature_contract import (
    FEATURE_CONTRACT_VERSION,
    FEATURE_SPECS,
)
from src.core.constants import (
    RSI_PERIOD,
    ATR_PERIOD,
    ADX_PERIOD,
    WARMUP_BARS,
    CLIP_MIN,
    CLIP_MAX,
    THRESHOLD_LONG,
    THRESHOLD_SHORT,
    MIN_CONFIDENCE_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    MAX_POSITION_SIZE,
    DEFAULT_STOP_LOSS_PCT,
    MAX_DRAWDOWN_PCT,
    TRADING_TIMEZONE,
    TRADING_START_HOUR,
    TRADING_END_HOUR,
    UTC_OFFSET_BOGOTA,
)

router = APIRouter(prefix="/ssot", tags=["ssot"])


# =============================================================================
# Response Models
# =============================================================================

class FeatureContractResponse(BaseModel):
    """Feature contract SSOT response."""
    version: str
    feature_order: List[str]
    feature_order_hash: str
    observation_dim: int
    market_features_count: int
    state_features_count: int
    feature_specs: Dict[str, Any]


class ActionContractResponse(BaseModel):
    """Action contract SSOT response."""
    version: str
    actions: Dict[str, int]
    action_count: int
    action_names: Dict[int, str]
    valid_actions: List[int]


class IndicatorsResponse(BaseModel):
    """Indicator periods SSOT response."""
    rsi_period: int
    atr_period: int
    adx_period: int
    warmup_bars: int


class NormalizationResponse(BaseModel):
    """Normalization constants SSOT response."""
    clip_min: float
    clip_max: float


class ThresholdsResponse(BaseModel):
    """Action thresholds SSOT response."""
    long: float
    short: float


class RiskResponse(BaseModel):
    """Risk management constants SSOT response."""
    min_confidence: float
    high_confidence: float
    max_position_size: float
    default_stop_loss_pct: float
    max_drawdown_pct: float


class MarketHoursResponse(BaseModel):
    """Market hours SSOT response."""
    timezone: str
    start_hour: int
    end_hour: int
    utc_offset: int


class SSOTResponse(BaseModel):
    """Complete SSOT response."""
    feature_contract: FeatureContractResponse
    action_contract: ActionContractResponse
    indicators: IndicatorsResponse
    normalization: NormalizationResponse
    thresholds: ThresholdsResponse
    risk: RiskResponse
    market_hours: MarketHoursResponse
    ssot_hash: str


class ValidationRequest(BaseModel):
    """Request to validate frontend SSOT."""
    feature_order_hash: str
    observation_dim: int
    action_count: int


class ValidationResponse(BaseModel):
    """Validation result response."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    backend_hash: str


# =============================================================================
# Helper Functions
# =============================================================================

def compute_ssot_hash() -> str:
    """Compute hash of entire SSOT for quick validation."""
    ssot_data = {
        "feature_order": list(FEATURE_ORDER),
        "observation_dim": OBSERVATION_DIM,
        "action_count": ACTION_COUNT,
        "rsi_period": RSI_PERIOD,
        "atr_period": ATR_PERIOD,
        "adx_period": ADX_PERIOD,
    }
    json_str = json.dumps(ssot_data, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


# =============================================================================
# Endpoints
# =============================================================================

@router.get("", response_model=SSOTResponse)
async def get_ssot():
    """
    Get complete SSOT summary.

    Returns all Single Source of Truth values that frontend should use.
    Frontend can cache this and validate periodically.
    """
    return SSOTResponse(
        feature_contract=FeatureContractResponse(
            version=FEATURE_CONTRACT_VERSION,
            feature_order=list(FEATURE_ORDER),
            feature_order_hash=FEATURE_ORDER_HASH,
            observation_dim=OBSERVATION_DIM,
            market_features_count=OBSERVATION_DIM - 2,  # minus state features
            state_features_count=2,
            feature_specs={
                name: {
                    "index": spec.index,
                    "clip_min": spec.clip_min,
                    "clip_max": spec.clip_max,
                    "unit": spec.unit,
                }
                for name, spec in FEATURE_SPECS.items()
            },
        ),
        action_contract=ActionContractResponse(
            version=ACTION_CONTRACT_VERSION,
            actions={
                "SELL": Action.SELL,
                "HOLD": Action.HOLD,
                "BUY": Action.BUY,
            },
            action_count=ACTION_COUNT,
            action_names={int(k): v for k, v in ACTION_NAMES.items()},
            valid_actions=list(VALID_ACTIONS),
        ),
        indicators=IndicatorsResponse(
            rsi_period=RSI_PERIOD,
            atr_period=ATR_PERIOD,
            adx_period=ADX_PERIOD,
            warmup_bars=WARMUP_BARS,
        ),
        normalization=NormalizationResponse(
            clip_min=CLIP_MIN,
            clip_max=CLIP_MAX,
        ),
        thresholds=ThresholdsResponse(
            long=THRESHOLD_LONG,
            short=THRESHOLD_SHORT,
        ),
        risk=RiskResponse(
            min_confidence=MIN_CONFIDENCE_THRESHOLD,
            high_confidence=HIGH_CONFIDENCE_THRESHOLD,
            max_position_size=MAX_POSITION_SIZE,
            default_stop_loss_pct=DEFAULT_STOP_LOSS_PCT,
            max_drawdown_pct=MAX_DRAWDOWN_PCT,
        ),
        market_hours=MarketHoursResponse(
            timezone=TRADING_TIMEZONE,
            start_hour=TRADING_START_HOUR,
            end_hour=TRADING_END_HOUR,
            utc_offset=UTC_OFFSET_BOGOTA,
        ),
        ssot_hash=compute_ssot_hash(),
    )


@router.get("/features", response_model=FeatureContractResponse)
async def get_feature_contract():
    """Get feature contract SSOT only."""
    return FeatureContractResponse(
        version=FEATURE_CONTRACT_VERSION,
        feature_order=list(FEATURE_ORDER),
        feature_order_hash=FEATURE_ORDER_HASH,
        observation_dim=OBSERVATION_DIM,
        market_features_count=OBSERVATION_DIM - 2,
        state_features_count=2,
        feature_specs={
            name: {
                "index": spec.index,
                "clip_min": spec.clip_min,
                "clip_max": spec.clip_max,
                "unit": spec.unit,
            }
            for name, spec in FEATURE_SPECS.items()
        },
    )


@router.get("/actions", response_model=ActionContractResponse)
async def get_action_contract():
    """Get action contract SSOT only."""
    return ActionContractResponse(
        version=ACTION_CONTRACT_VERSION,
        actions={
            "SELL": Action.SELL,
            "HOLD": Action.HOLD,
            "BUY": Action.BUY,
        },
        action_count=ACTION_COUNT,
        action_names={int(k): v for k, v in ACTION_NAMES.items()},
        valid_actions=list(VALID_ACTIONS),
    )


@router.post("/validate", response_model=ValidationResponse)
async def validate_frontend_ssot(request: ValidationRequest):
    """
    Validate frontend SSOT against backend.

    Frontend should call this on startup to ensure its local
    SSOT matches the backend.
    """
    errors = []
    warnings = []

    # Validate feature order hash
    if request.feature_order_hash != FEATURE_ORDER_HASH:
        errors.append(
            f"Feature order hash mismatch: frontend={request.feature_order_hash}, "
            f"backend={FEATURE_ORDER_HASH}"
        )

    # Validate observation dim
    if request.observation_dim != OBSERVATION_DIM:
        errors.append(
            f"Observation dim mismatch: frontend={request.observation_dim}, "
            f"backend={OBSERVATION_DIM}"
        )

    # Validate action count
    if request.action_count != ACTION_COUNT:
        errors.append(
            f"Action count mismatch: frontend={request.action_count}, "
            f"backend={ACTION_COUNT}"
        )

    return ValidationResponse(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        backend_hash=compute_ssot_hash(),
    )
