"""
Replay Router - Feature Replay Endpoint for Week 2
===================================================
Provides endpoint to replay features from a specific timestamp for debugging
and audit purposes.

Contract: CTR-API-002
Author: Trading Team
Version: 1.0.0
Created: 2025-01-17
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import asyncpg
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..config import get_settings, FEATURE_ORDER

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/replay", tags=["replay"])
settings = get_settings()


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class ReplayRequest(BaseModel):
    """Request model for feature replay."""

    timestamp: datetime = Field(
        ...,
        description="Timestamp to replay features from (ISO 8601 format)",
        examples=["2025-01-15T14:30:00Z"]
    )
    symbol: str = Field(
        default="USDCOP",
        description="Trading symbol (default: USDCOP)"
    )
    model_id: Optional[str] = Field(
        default=None,
        description="Optional model ID to filter by specific model inference"
    )

    model_config = {"protected_namespaces": ()}


class ReplayResponse(BaseModel):
    """Response model for feature replay."""

    timestamp: datetime = Field(
        ...,
        description="Timestamp of the replayed features"
    )
    features: Dict[str, float] = Field(
        ...,
        description="Feature values keyed by feature name"
    )
    prediction: Optional[float] = Field(
        default=None,
        description="Model prediction value if available"
    )
    signal: Optional[str] = Field(
        default=None,
        description="Trading signal (LONG, SHORT, HOLD)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (model_id, source, etc.)"
    )

    model_config = {"protected_namespaces": ()}


class FeatureSnapshot(BaseModel):
    """Internal model for feature snapshot from database."""

    version: Optional[str] = None
    timestamp: Optional[str] = None
    bar_idx: Optional[int] = None
    raw_features: Optional[Dict[str, float]] = None
    normalized_features: Optional[Dict[str, float]] = None


# =============================================================================
# FEATURE READER SERVICE
# =============================================================================


class FeatureReader:
    """
    Service to read features from various sources.

    Priority order:
    1. trades_history.features_snapshot (JSONB)
    2. dw.fact_rl_inference (materialized view)
    3. inference_features_5m (materialized view)
    """

    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        self._pool = db_pool

    async def _get_connection(self) -> asyncpg.Connection:
        """Get database connection."""
        if self._pool:
            return await self._pool.acquire()

        return await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
        )

    async def _release_connection(self, conn: asyncpg.Connection) -> None:
        """Release database connection."""
        if self._pool:
            await self._pool.release(conn)
        else:
            await conn.close()

    async def get_features_at_timestamp(
        self,
        timestamp: datetime,
        symbol: str = "USDCOP",
        model_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get features at a specific timestamp.

        Args:
            timestamp: Target timestamp
            symbol: Trading symbol
            model_id: Optional model ID filter

        Returns:
            Dict with features, prediction, signal, and metadata
        """
        conn = await self._get_connection()

        try:
            # Try trades_history first (has features_snapshot)
            result = await self._get_from_trades_history(
                conn, timestamp, symbol, model_id
            )

            if result:
                return result

            # Fallback to fact_rl_inference
            result = await self._get_from_fact_inference(
                conn, timestamp, symbol
            )

            if result:
                return result

            # Final fallback to inference_features_5m
            result = await self._get_from_inference_features(
                conn, timestamp, symbol
            )

            return result

        finally:
            await self._release_connection(conn)

    async def _get_from_trades_history(
        self,
        conn: asyncpg.Connection,
        timestamp: datetime,
        symbol: str,
        model_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Get features from trades_history table."""
        try:
            # Build query with optional model_id filter
            query = """
                SELECT
                    timestamp,
                    signal,
                    confidence as prediction,
                    features_snapshot,
                    model_id,
                    model_hash
                FROM trades_history
                WHERE timestamp >= $1 - INTERVAL '5 minutes'
                  AND timestamp <= $1 + INTERVAL '5 minutes'
            """
            params = [timestamp]

            if model_id:
                query += " AND model_id = $2"
                params.append(model_id)

            query += " ORDER BY ABS(EXTRACT(EPOCH FROM (timestamp - $1))) LIMIT 1"

            row = await conn.fetchrow(query, *params)

            if not row or not row['features_snapshot']:
                return None

            # Parse features_snapshot JSONB
            snapshot = row['features_snapshot']

            # Handle both old and new schema versions
            if isinstance(snapshot, dict):
                if 'normalized_features' in snapshot:
                    features = snapshot['normalized_features']
                elif 'raw_features' in snapshot:
                    features = snapshot['raw_features']
                else:
                    # Flat structure - features directly in snapshot
                    features = {
                        k: v for k, v in snapshot.items()
                        if k not in ('version', 'timestamp', 'bar_idx')
                    }
            else:
                return None

            return {
                "timestamp": row['timestamp'],
                "features": features,
                "prediction": float(row['prediction']) if row['prediction'] else None,
                "signal": row['signal'],
                "metadata": {
                    "source": "trades_history",
                    "model_id": row['model_id'],
                    "model_hash": row['model_hash'],
                    "snapshot_version": snapshot.get('version'),
                }
            }

        except Exception as e:
            logger.warning(f"Error querying trades_history: {e}")
            return None

    async def _get_from_fact_inference(
        self,
        conn: asyncpg.Connection,
        timestamp: datetime,
        symbol: str,
    ) -> Optional[Dict[str, Any]]:
        """Get features from dw.fact_rl_inference table."""
        try:
            query = """
                SELECT
                    timestamp,
                    signal,
                    confidence,
                    log_ret_5m, log_ret_1h, log_ret_4h,
                    rsi_9, atr_pct, adx_14,
                    dxy_z, dxy_change_1d, vix_z, embi_z,
                    brent_change_1d, rate_spread, usdmxn_change_1d,
                    position, time_normalized,
                    model_id
                FROM dw.fact_rl_inference
                WHERE timestamp >= $1 - INTERVAL '5 minutes'
                  AND timestamp <= $1 + INTERVAL '5 minutes'
                ORDER BY ABS(EXTRACT(EPOCH FROM (timestamp - $1)))
                LIMIT 1
            """

            row = await conn.fetchrow(query, timestamp)

            if not row:
                return None

            # Build features dict from columns
            features = {}
            for feature_name in FEATURE_ORDER:
                if feature_name in dict(row):
                    value = row[feature_name]
                    features[feature_name] = float(value) if value is not None else 0.0

            return {
                "timestamp": row['timestamp'],
                "features": features,
                "prediction": float(row['confidence']) if row['confidence'] else None,
                "signal": row['signal'],
                "metadata": {
                    "source": "dw.fact_rl_inference",
                    "model_id": row['model_id'],
                }
            }

        except Exception as e:
            logger.warning(f"Error querying dw.fact_rl_inference: {e}")
            return None

    async def _get_from_inference_features(
        self,
        conn: asyncpg.Connection,
        timestamp: datetime,
        symbol: str,
    ) -> Optional[Dict[str, Any]]:
        """Get features from inference_features_5m materialized view."""
        try:
            query = """
                SELECT *
                FROM inference_features_5m
                WHERE timestamp >= $1 - INTERVAL '5 minutes'
                  AND timestamp <= $1 + INTERVAL '5 minutes'
                ORDER BY ABS(EXTRACT(EPOCH FROM (timestamp - $1)))
                LIMIT 1
            """

            row = await conn.fetchrow(query, timestamp)

            if not row:
                return None

            # Build features dict from row
            row_dict = dict(row)
            features = {}

            for feature_name in FEATURE_ORDER:
                if feature_name in row_dict:
                    value = row_dict[feature_name]
                    features[feature_name] = float(value) if value is not None else 0.0

            return {
                "timestamp": row_dict.get('timestamp', timestamp),
                "features": features,
                "prediction": None,
                "signal": None,
                "metadata": {
                    "source": "inference_features_5m",
                }
            }

        except Exception as e:
            logger.warning(f"Error querying inference_features_5m: {e}")
            return None


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post("/features", response_model=ReplayResponse)
async def replay_features(
    request: ReplayRequest,
    req: Request,
) -> ReplayResponse:
    """
    Replay features from a specific timestamp.

    This endpoint retrieves the feature vector that was used (or would have been
    used) for model inference at the specified timestamp.

    Use cases:
    - Debugging trade decisions
    - Auditing model behavior
    - Validating feature calculations
    - Reproducing historical predictions

    Args:
        request: ReplayRequest with timestamp, symbol, and optional model_id

    Returns:
        ReplayResponse with features, prediction, signal, and metadata

    Raises:
        404: No features found for the specified timestamp
        500: Internal server error
    """
    logger.info(
        f"Replay request: timestamp={request.timestamp}, "
        f"symbol={request.symbol}, model_id={request.model_id}"
    )

    # Get database pool from app state if available
    db_pool = getattr(req.app.state, 'db_pool', None)

    # Create feature reader
    reader = FeatureReader(db_pool=db_pool)

    try:
        result = await reader.get_features_at_timestamp(
            timestamp=request.timestamp,
            symbol=request.symbol,
            model_id=request.model_id,
        )

        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"No features found for timestamp {request.timestamp.isoformat()}"
            )

        return ReplayResponse(
            timestamp=result["timestamp"],
            features=result["features"],
            prediction=result["prediction"],
            signal=result["signal"],
            metadata=result["metadata"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error replaying features: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to replay features: {str(e)}"
        )


@router.get("/features/{timestamp}")
async def replay_features_get(
    timestamp: str,
    symbol: str = "USDCOP",
    model_id: Optional[str] = None,
    req: Request = None,
) -> ReplayResponse:
    """
    GET endpoint for feature replay (convenience endpoint).

    Args:
        timestamp: ISO 8601 timestamp string
        symbol: Trading symbol (default: USDCOP)
        model_id: Optional model ID filter

    Returns:
        ReplayResponse with features
    """
    try:
        ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid timestamp format: {e}. Use ISO 8601 format."
        )

    request = ReplayRequest(
        timestamp=ts,
        symbol=symbol,
        model_id=model_id,
    )

    return await replay_features(request, req)
