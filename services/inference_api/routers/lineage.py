"""
Lineage Router - Trade and Model Lineage API for Week 2
=======================================================
Provides complete lineage tracking for trades and models, enabling
full audit trails and reproducibility.

Contract: CTR-API-003
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
router = APIRouter(prefix="/lineage", tags=["lineage"])
settings = get_settings()


# =============================================================================
# LINEAGE MODELS
# =============================================================================


class FeatureLineage(BaseModel):
    """Feature lineage information."""

    feature_order: List[str] = Field(
        default_factory=list,
        description="Order of features in observation vector"
    )
    values: Dict[str, float] = Field(
        default_factory=dict,
        description="Feature values keyed by name"
    )
    source: Optional[str] = Field(
        default=None,
        description="Feature source (l1_pipeline, feast, fallback)"
    )
    snapshot_version: Optional[str] = Field(
        default=None,
        description="Version of feature snapshot schema"
    )


class ModelLineageInfo(BaseModel):
    """Model information for lineage tracking."""

    model_id: str = Field(..., description="Model identifier")
    model_version: Optional[str] = Field(default=None, description="Model version")
    model_hash: Optional[str] = Field(default=None, description="SHA256 hash of model file")
    observation_dim: Optional[int] = Field(default=None, description="Input observation dimension")
    action_space: Optional[int] = Field(default=None, description="Action space size")

    model_config = {"protected_namespaces": ()}


class DatasetLineage(BaseModel):
    """Dataset lineage information."""

    training_dataset_id: Optional[int] = Field(default=None, description="Training dataset ID")
    training_start_date: Optional[str] = Field(default=None, description="Training data start date")
    training_end_date: Optional[str] = Field(default=None, description="Training data end date")


class NormStatsLineage(BaseModel):
    """Normalization statistics lineage."""

    norm_stats_hash: Optional[str] = Field(
        default=None,
        description="SHA256 hash of norm_stats.json file"
    )
    norm_stats_path: Optional[str] = Field(
        default=None,
        description="Path to norm_stats.json"
    )


class TradeLineage(BaseModel):
    """Complete lineage for a single trade."""

    trade_id: int = Field(..., description="Trade ID")
    timestamp: datetime = Field(..., description="Trade timestamp")
    signal: str = Field(..., description="Trading signal (LONG, SHORT, HOLD)")
    prediction: Optional[float] = Field(default=None, description="Model prediction/confidence")
    features: FeatureLineage = Field(..., description="Feature lineage")
    model: ModelLineageInfo = Field(..., description="Model lineage")
    dataset: Optional[DatasetLineage] = Field(default=None, description="Dataset lineage")
    norm_stats_hash: Optional[str] = Field(default=None, description="Norm stats hash")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    model_config = {"protected_namespaces": ()}


class ModelLineage(BaseModel):
    """Complete lineage for a model."""

    model_id: str = Field(..., description="Model identifier")
    model_version: Optional[str] = Field(default=None, description="Model version")
    model_hash: Optional[str] = Field(default=None, description="Model file hash")
    model_path: Optional[str] = Field(default=None, description="Model file path")
    status: Optional[str] = Field(default=None, description="Model status")

    # Training info
    observation_dim: Optional[int] = Field(default=None, description="Input dimension")
    action_space: Optional[int] = Field(default=None, description="Action space size")
    feature_order: Optional[List[str]] = Field(default=None, description="Feature order")

    # Dataset lineage
    dataset: Optional[DatasetLineage] = Field(default=None, description="Training dataset info")

    # Norm stats lineage
    norm_stats_hash: Optional[str] = Field(default=None, description="Norm stats hash")

    # Performance metrics
    validation_metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Validation metrics"
    )
    test_sharpe: Optional[float] = Field(default=None, description="Test Sharpe ratio")
    test_max_drawdown: Optional[float] = Field(default=None, description="Test max drawdown")
    test_win_rate: Optional[float] = Field(default=None, description="Test win rate")

    # Timestamps
    created_at: Optional[datetime] = Field(default=None, description="Registration time")
    deployed_at: Optional[datetime] = Field(default=None, description="Deployment time")
    retired_at: Optional[datetime] = Field(default=None, description="Retirement time")

    # Trade statistics
    trade_count: Optional[int] = Field(default=None, description="Number of trades with this model")
    first_trade: Optional[datetime] = Field(default=None, description="First trade timestamp")
    last_trade: Optional[datetime] = Field(default=None, description="Last trade timestamp")

    # Config hash
    config_hash: Optional[str] = Field(default=None, description="Config hash")

    model_config = {"protected_namespaces": ()}


# =============================================================================
# LINEAGE SERVICE
# =============================================================================


class LineageService:
    """Service to retrieve lineage information from database."""

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

    async def get_trade_lineage(self, trade_id: int) -> Optional[TradeLineage]:
        """
        Get complete lineage for a trade.

        Args:
            trade_id: Trade ID

        Returns:
            TradeLineage with full lineage information
        """
        conn = await self._get_connection()

        try:
            # Query trade with model join
            query = """
                SELECT
                    t.id,
                    t.timestamp,
                    t.signal,
                    t.confidence,
                    t.features_snapshot,
                    t.features_source,
                    t.model_id,
                    t.model_hash,
                    mr.model_version,
                    mr.observation_dim,
                    mr.action_space,
                    mr.norm_stats_hash,
                    mr.training_dataset_id,
                    mr.training_start_date,
                    mr.training_end_date,
                    mr.feature_order
                FROM trades_history t
                LEFT JOIN model_registry mr ON t.model_id = mr.model_id
                WHERE t.id = $1
            """

            row = await conn.fetchrow(query, trade_id)

            if not row:
                return None

            # Parse features from snapshot
            snapshot = row['features_snapshot'] or {}
            features_values = {}
            snapshot_version = None

            if isinstance(snapshot, dict):
                snapshot_version = snapshot.get('version')
                if 'normalized_features' in snapshot:
                    features_values = snapshot['normalized_features']
                elif 'raw_features' in snapshot:
                    features_values = snapshot['raw_features']
                else:
                    features_values = {
                        k: v for k, v in snapshot.items()
                        if k not in ('version', 'timestamp', 'bar_idx')
                        and isinstance(v, (int, float))
                    }

            # Build feature lineage
            feature_lineage = FeatureLineage(
                feature_order=FEATURE_ORDER,
                values=features_values,
                source=row['features_source'],
                snapshot_version=snapshot_version,
            )

            # Build model lineage
            model_lineage = ModelLineageInfo(
                model_id=row['model_id'] or 'unknown',
                model_version=row['model_version'],
                model_hash=row['model_hash'],
                observation_dim=row['observation_dim'],
                action_space=row['action_space'],
            )

            # Build dataset lineage
            dataset_lineage = None
            if row['training_dataset_id'] or row['training_start_date']:
                dataset_lineage = DatasetLineage(
                    training_dataset_id=row['training_dataset_id'],
                    training_start_date=(
                        row['training_start_date'].isoformat()
                        if row['training_start_date'] else None
                    ),
                    training_end_date=(
                        row['training_end_date'].isoformat()
                        if row['training_end_date'] else None
                    ),
                )

            return TradeLineage(
                trade_id=row['id'],
                timestamp=row['timestamp'],
                signal=row['signal'] or 'UNKNOWN',
                prediction=float(row['confidence']) if row['confidence'] else None,
                features=feature_lineage,
                model=model_lineage,
                dataset=dataset_lineage,
                norm_stats_hash=row['norm_stats_hash'],
                metadata={
                    "feature_order_from_registry": row['feature_order'],
                }
            )

        finally:
            await self._release_connection(conn)

    async def get_model_lineage(self, model_id: str) -> Optional[ModelLineage]:
        """
        Get complete lineage for a model.

        Args:
            model_id: Model identifier

        Returns:
            ModelLineage with full model information
        """
        conn = await self._get_connection()

        try:
            # Query model registry
            query = """
                SELECT
                    model_id,
                    model_version,
                    model_path,
                    model_hash,
                    norm_stats_hash,
                    config_hash,
                    observation_dim,
                    action_space,
                    feature_order,
                    training_dataset_id,
                    training_start_date,
                    training_end_date,
                    validation_metrics,
                    test_sharpe,
                    test_max_drawdown,
                    test_win_rate,
                    created_at,
                    deployed_at,
                    retired_at,
                    status
                FROM model_registry
                WHERE model_id = $1
            """

            row = await conn.fetchrow(query, model_id)

            if not row:
                return None

            # Get trade statistics for this model
            trade_stats_query = """
                SELECT
                    COUNT(*) as trade_count,
                    MIN(timestamp) as first_trade,
                    MAX(timestamp) as last_trade
                FROM trades_history
                WHERE model_id = $1
            """

            trade_stats = await conn.fetchrow(trade_stats_query, model_id)

            # Build dataset lineage
            dataset_lineage = None
            if row['training_dataset_id'] or row['training_start_date']:
                dataset_lineage = DatasetLineage(
                    training_dataset_id=row['training_dataset_id'],
                    training_start_date=(
                        row['training_start_date'].isoformat()
                        if row['training_start_date'] else None
                    ),
                    training_end_date=(
                        row['training_end_date'].isoformat()
                        if row['training_end_date'] else None
                    ),
                )

            # Parse feature_order
            feature_order = row['feature_order']
            if isinstance(feature_order, str):
                import json
                feature_order = json.loads(feature_order)

            return ModelLineage(
                model_id=row['model_id'],
                model_version=row['model_version'],
                model_hash=row['model_hash'],
                model_path=row['model_path'],
                status=row['status'],
                observation_dim=row['observation_dim'],
                action_space=row['action_space'],
                feature_order=feature_order,
                dataset=dataset_lineage,
                norm_stats_hash=row['norm_stats_hash'],
                validation_metrics=row['validation_metrics'],
                test_sharpe=float(row['test_sharpe']) if row['test_sharpe'] else None,
                test_max_drawdown=float(row['test_max_drawdown']) if row['test_max_drawdown'] else None,
                test_win_rate=float(row['test_win_rate']) if row['test_win_rate'] else None,
                created_at=row['created_at'],
                deployed_at=row['deployed_at'],
                retired_at=row['retired_at'],
                trade_count=trade_stats['trade_count'] if trade_stats else 0,
                first_trade=trade_stats['first_trade'] if trade_stats else None,
                last_trade=trade_stats['last_trade'] if trade_stats else None,
                config_hash=row['config_hash'],
            )

        finally:
            await self._release_connection(conn)


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.get("/trade/{trade_id}", response_model=TradeLineage)
async def get_trade_lineage(
    trade_id: int,
    req: Request,
) -> TradeLineage:
    """
    Get complete lineage for a trade.

    Returns full audit trail including:
    - Trade details (timestamp, signal, prediction)
    - Feature snapshot with values and source
    - Model information (version, hash, dimensions)
    - Training dataset information
    - Normalization stats hash

    Use cases:
    - Audit trade decisions
    - Debug model behavior
    - Validate feature calculations
    - Compliance and governance

    Args:
        trade_id: Trade ID to retrieve lineage for

    Returns:
        TradeLineage with complete lineage information

    Raises:
        404: Trade not found
        500: Internal server error
    """
    logger.info(f"Getting lineage for trade_id={trade_id}")

    db_pool = getattr(req.app.state, 'db_pool', None)
    service = LineageService(db_pool=db_pool)

    try:
        lineage = await service.get_trade_lineage(trade_id)

        if not lineage:
            raise HTTPException(
                status_code=404,
                detail=f"Trade with id {trade_id} not found"
            )

        return lineage

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting trade lineage: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get trade lineage: {str(e)}"
        )


@router.get("/model/{model_id}", response_model=ModelLineage)
async def get_model_lineage(
    model_id: str,
    req: Request,
) -> ModelLineage:
    """
    Get complete lineage for a model.

    Returns full model information including:
    - Model metadata (version, hash, path)
    - Configuration (observation_dim, action_space, feature_order)
    - Training dataset information
    - Performance metrics
    - Deployment history
    - Trade statistics

    Use cases:
    - Model governance
    - Performance tracking
    - Audit and compliance
    - Rollback decisions

    Args:
        model_id: Model identifier (e.g., "ppo_primary")

    Returns:
        ModelLineage with complete model information

    Raises:
        404: Model not found
        500: Internal server error
    """
    logger.info(f"Getting lineage for model_id={model_id}")

    db_pool = getattr(req.app.state, 'db_pool', None)
    service = LineageService(db_pool=db_pool)

    try:
        lineage = await service.get_model_lineage(model_id)

        if not lineage:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_id}' not found in registry"
            )

        return lineage

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model lineage: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model lineage: {str(e)}"
        )


@router.get("/models")
async def list_models_with_lineage(
    req: Request,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List all models with basic lineage information.

    Args:
        status: Optional filter by status (registered, deployed, retired)

    Returns:
        List of models with basic lineage info
    """
    db_pool = getattr(req.app.state, 'db_pool', None)

    try:
        if db_pool:
            conn = await db_pool.acquire()
        else:
            conn = await asyncpg.connect(
                host=settings.postgres_host,
                port=settings.postgres_port,
                database=settings.postgres_db,
                user=settings.postgres_user,
                password=settings.postgres_password,
            )

        try:
            query = """
                SELECT
                    model_id,
                    model_version,
                    model_hash,
                    norm_stats_hash,
                    observation_dim,
                    status,
                    created_at,
                    deployed_at
                FROM model_registry
            """
            params = []

            if status:
                query += " WHERE status = $1"
                params.append(status)

            query += " ORDER BY created_at DESC"

            rows = await conn.fetch(query, *params)

            return [
                {
                    "model_id": row['model_id'],
                    "model_version": row['model_version'],
                    "model_hash": row['model_hash'],
                    "norm_stats_hash": row['norm_stats_hash'],
                    "observation_dim": row['observation_dim'],
                    "status": row['status'],
                    "created_at": row['created_at'].isoformat() if row['created_at'] else None,
                    "deployed_at": row['deployed_at'].isoformat() if row['deployed_at'] else None,
                }
                for row in rows
            ]

        finally:
            if db_pool:
                await db_pool.release(conn)
            else:
                await conn.close()

    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list models: {str(e)}"
        )
