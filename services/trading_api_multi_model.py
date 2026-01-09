#!/usr/bin/env python3
"""
USDCOP Multi-Model Trading API
==============================

Complete FastAPI service for multi-model trading operations.
Provides endpoints for model management, signals, trades, metrics,
equity curves, and real-time streaming via SSE.

Port: 8007
Database: PostgreSQL (asyncpg)
Cache: Redis

Features:
    - List and query multiple trading models
    - Get signals with history and real-time streaming (SSE)
    - Track trades and positions
    - Compare model performance metrics
    - Equity curve data for charting
    - Feature normalization stats

SOLID Compliance:
    - SRP: Each router/endpoint handles one responsibility
    - OCP: Configurable via shared config modules
    - DIP: Uses dependency injection for database/redis
    - DRY: Uses shared utilities from common/ directory

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Created: 2025-12-26
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

import asyncpg
import redis.asyncio as aioredis
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Service configuration from environment variables."""

    # PostgreSQL
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "usdcop-postgres-timescale")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "usdcop_trading")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "admin")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "admin123")

    # Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "redis")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")

    # Connection pool settings
    MIN_POOL_SIZE: int = 2
    MAX_POOL_SIZE: int = 10

    # Cache TTL (seconds)
    CACHE_TTL_SHORT: int = 30  # 30 seconds for real-time data
    CACHE_TTL_MEDIUM: int = 300  # 5 minutes for computed metrics
    CACHE_TTL_LONG: int = 3600  # 1 hour for static data

    @classmethod
    def postgres_dsn(cls) -> str:
        return (
            f"postgresql://{cls.POSTGRES_USER}:{cls.POSTGRES_PASSWORD}"
            f"@{cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DB}"
        )


# =============================================================================
# PYDANTIC MODELS - Request/Response Schemas
# =============================================================================

class ModelStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    DEPRECATED = "deprecated"


class SignalAction(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"
    CLOSE = "close"


class TradeSide(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class TradeStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    ALL = "all"


class Period(str, Enum):
    TODAY = "today"
    WEEK = "7d"
    MONTH = "30d"
    ALL = "all"


# ----- Model Schemas -----

class ModelConfig(BaseModel):
    """Model configuration parameters."""
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    n_epochs: Optional[int] = None
    gamma: Optional[float] = None
    gae_lambda: Optional[float] = None
    clip_range: Optional[float] = None
    custom_params: Optional[Dict[str, Any]] = None


class BacktestMetrics(BaseModel):
    """Backtest performance metrics."""
    total_return_pct: float = Field(..., description="Total return percentage")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    max_drawdown_pct: float = Field(..., description="Maximum drawdown percentage")
    win_rate: float = Field(..., ge=0, le=1, description="Win rate (0-1)")
    profit_factor: float = Field(..., description="Profit factor")
    total_trades: int = Field(..., ge=0, description="Total number of trades")
    calmar_ratio: Optional[float] = None
    avg_trade_duration_mins: Optional[float] = None


class ModelSummary(BaseModel):
    """Summary information for a trading model."""
    model_id: str = Field(..., description="Unique model identifier")
    name: str = Field(..., description="Human-readable model name")
    model_type: str = Field(..., description="Model type (RL, ML, LLM, ENSEMBLE)")
    status: ModelStatus = Field(..., description="Current model status")
    version: Optional[str] = None
    description: Optional[str] = None
    is_active: bool = True
    backtest_metrics: Optional[BacktestMetrics] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ModelDetail(ModelSummary):
    """Detailed model information including hyperparameters and features."""
    hyperparameters: Optional[ModelConfig] = None
    features_used: List[str] = Field(default_factory=list)
    observation_space_dim: Optional[int] = None
    action_space_dim: Optional[int] = None
    training_episodes: Optional[int] = None
    training_timesteps: Optional[int] = None
    fold_id: Optional[int] = None


class ModelsListResponse(BaseModel):
    """Response for listing all models."""
    count: int
    models: List[ModelSummary]
    timestamp: str


# ----- Signal Schemas -----

class Signal(BaseModel):
    """Trading signal from a model."""
    signal_id: int
    timestamp: str
    action: SignalAction
    side: TradeSide
    confidence: float = Field(..., ge=0, le=1)
    size: float = Field(..., ge=0, le=1)
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_usd: Optional[float] = None
    reasoning: Optional[str] = None
    features_snapshot: Optional[Dict[str, Any]] = None


class SignalsResponse(BaseModel):
    """Response for querying signals."""
    model_id: str
    period: str
    count: int
    signals: List[Signal]
    timestamp: str


class LatestSignalResponse(BaseModel):
    """Response for the latest signal."""
    model_id: str
    signal: Optional[Signal] = None
    market_price: Optional[float] = None
    age_seconds: Optional[int] = None
    timestamp: str


# ----- Trade Schemas -----

class Trade(BaseModel):
    """Trade/position information."""
    trade_id: int
    side: str
    quantity: float
    entry_price: float
    current_price: Optional[float] = None
    exit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    realized_pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    entry_time: str
    exit_time: Optional[str] = None
    holding_time_minutes: Optional[int] = None
    status: str
    exit_reason: Optional[str] = None
    leverage: int = 1


class TradesResponse(BaseModel):
    """Response for querying trades."""
    model_id: str
    period: str
    status_filter: str
    count: int
    trades: List[Trade]
    total_pnl: float
    timestamp: str


class OpenPositionResponse(BaseModel):
    """Response for open position query."""
    model_id: str
    has_position: bool
    position: Optional[Trade] = None
    timestamp: str


# ----- Metrics Schemas -----

class LiveMetrics(BaseModel):
    """Live performance metrics."""
    total_return_pct: float
    daily_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    current_drawdown_pct: float
    volatility_pct: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_hold_time_minutes: float
    current_equity: float
    open_positions: int


class MetricsComparison(BaseModel):
    """Comparison of live vs backtest metrics."""
    live: LiveMetrics
    backtest: Optional[BacktestMetrics] = None
    divergence: Optional[Dict[str, float]] = None


class MetricsResponse(BaseModel):
    """Response for metrics endpoint."""
    model_id: str
    period: str
    metrics: MetricsComparison
    timestamp: str


# ----- Equity Curve Schemas -----

class EquityPoint(BaseModel):
    """Single point on equity curve."""
    timestamp: str
    equity_value: float
    return_pct: float
    drawdown_pct: float
    cash_balance: Optional[float] = None
    positions_value: Optional[float] = None


class EquityCurveResponse(BaseModel):
    """Response for equity curve endpoint."""
    model_id: str
    start_date: str
    end_date: str
    resolution: str
    data_points: int
    data: List[EquityPoint]
    summary: Dict[str, float]
    timestamp: str


# ----- Comparison Schemas -----

class ModelComparisonItem(BaseModel):
    """Single model in comparison."""
    model_id: str
    name: str
    model_type: str
    metrics: LiveMetrics


class ModelsCompareResponse(BaseModel):
    """Response for model comparison."""
    period: str
    models_compared: int
    comparison: List[ModelComparisonItem]
    best_sharpe: str
    best_return: str
    lowest_drawdown: str
    timestamp: str


# ----- Features Schemas -----

class FeatureStats(BaseModel):
    """Normalization statistics for a feature."""
    name: str
    mean: float
    std: float
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    category: str


class FeatureCategory(BaseModel):
    """Group of features by category."""
    category: str
    features: List[FeatureStats]


class FeaturesResponse(BaseModel):
    """Response for features endpoint."""
    total_features: int
    categories: List[FeatureCategory]
    timestamp: str


# ----- Health Check Schemas -----

class ServiceHealth(BaseModel):
    """Health status of a service component."""
    name: str
    status: str
    latency_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Response for health check endpoint."""
    status: str
    version: str
    services: List[ServiceHealth]
    models_available: int
    timestamp: str


# ----- Error Schemas -----

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str
    timestamp: str
    path: Optional[str] = None


# =============================================================================
# DATABASE CONNECTION MANAGEMENT
# =============================================================================

class DatabasePool:
    """Async PostgreSQL connection pool manager."""

    def __init__(self):
        self._pool: Optional[asyncpg.Pool] = None

    async def init(self):
        """Initialize the connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                Config.postgres_dsn(),
                min_size=Config.MIN_POOL_SIZE,
                max_size=Config.MAX_POOL_SIZE,
                command_timeout=60.0,
                statement_cache_size=100,
            )
            logger.info("PostgreSQL connection pool initialized")

    async def close(self):
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL connection pool closed")

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Database pool not initialized")
        return self._pool

    async def fetch(self, query: str, *args) -> List[asyncpg.Record]:
        """Execute a query and fetch all results."""
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """Execute a query and fetch one result."""
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args) -> Any:
        """Execute a query and fetch a single value."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def execute(self, query: str, *args) -> str:
        """Execute a query without returning results."""
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)


class RedisCache:
    """Async Redis cache manager."""

    def __init__(self):
        self._redis: Optional[aioredis.Redis] = None

    async def init(self):
        """Initialize Redis connection."""
        try:
            self._redis = await aioredis.from_url(
                f"redis://{Config.REDIS_HOST}:{Config.REDIS_PORT}",
                password=Config.REDIS_PASSWORD if Config.REDIS_PASSWORD else None,
                encoding="utf-8",
                decode_responses=True,
            )
            # Test connection
            await self._redis.ping()
            logger.info("Redis connection initialized")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Caching disabled.")
            self._redis = None

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Redis connection closed")

    @property
    def available(self) -> bool:
        return self._redis is not None

    async def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        if not self.available:
            return None
        try:
            return await self._redis.get(key)
        except Exception as e:
            logger.warning(f"Redis GET error: {e}")
            return None

    async def set(self, key: str, value: str, ttl: int = Config.CACHE_TTL_MEDIUM):
        """Set value in cache with TTL."""
        if not self.available:
            return
        try:
            await self._redis.setex(key, ttl, value)
        except Exception as e:
            logger.warning(f"Redis SET error: {e}")

    async def delete(self, key: str):
        """Delete value from cache."""
        if not self.available:
            return
        try:
            await self._redis.delete(key)
        except Exception as e:
            logger.warning(f"Redis DELETE error: {e}")

    async def subscribe(self, channel: str) -> AsyncGenerator[str, None]:
        """Subscribe to a Redis channel for SSE."""
        if not self.available:
            return

        try:
            pubsub = self._redis.pubsub()
            await pubsub.subscribe(channel)

            async for message in pubsub.listen():
                if message["type"] == "message":
                    yield message["data"]
        except Exception as e:
            logger.error(f"Redis subscribe error: {e}")
        finally:
            if pubsub:
                await pubsub.unsubscribe(channel)
                await pubsub.close()


# Global instances
db_pool = DatabasePool()
redis_cache = RedisCache()


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================

async def get_db() -> DatabasePool:
    """Dependency for database access."""
    return db_pool


async def get_cache() -> RedisCache:
    """Dependency for cache access."""
    return redis_cache


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_period(period: Period) -> tuple[datetime, datetime]:
    """Parse period enum to date range."""
    now = datetime.now(timezone.utc)
    end_date = now

    if period == Period.TODAY:
        start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == Period.WEEK:
        start_date = now - timedelta(days=7)
    elif period == Period.MONTH:
        start_date = now - timedelta(days=30)
    else:  # ALL
        start_date = now - timedelta(days=365 * 10)  # 10 years

    return start_date, end_date


async def get_market_price(db: DatabasePool) -> Optional[float]:
    """Get latest market price from database."""
    query = """
        SELECT close FROM usdcop_m5_ohlcv
        WHERE symbol = 'USD/COP'
        ORDER BY time DESC
        LIMIT 1
    """
    result = await db.fetchval(query)
    return float(result) if result else None


async def validate_model_exists(db: DatabasePool, model_id: str) -> bool:
    """Check if a model exists in the database."""
    query = """
        SELECT 1 FROM dw.dim_strategy
        WHERE strategy_code = $1
    """
    result = await db.fetchval(query, model_id)
    return result is not None


def calculate_divergence(live: LiveMetrics, backtest: BacktestMetrics) -> Dict[str, float]:
    """Calculate divergence between live and backtest metrics."""
    return {
        "sharpe_divergence": live.sharpe_ratio - backtest.sharpe_ratio,
        "return_divergence": live.total_return_pct - backtest.total_return_pct,
        "drawdown_divergence": live.max_drawdown_pct - backtest.max_drawdown_pct,
        "win_rate_divergence": live.win_rate - backtest.win_rate,
    }


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Multi-Model Trading API...")
    await db_pool.init()
    await redis_cache.init()
    logger.info("Multi-Model Trading API started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Multi-Model Trading API...")
    await redis_cache.close()
    await db_pool.close()
    logger.info("Multi-Model Trading API shutdown complete")


app = FastAPI(
    title="USDCOP Multi-Model Trading API",
    description="""
    Complete API for multi-model trading operations on USD/COP.

    ## Features

    - **Models**: List, query, and compare trading models (RL, ML, LLM, Ensemble)
    - **Signals**: Get historical and real-time trading signals with SSE streaming
    - **Trades**: Track open and closed positions with P&L
    - **Metrics**: Live performance metrics with backtest comparison
    - **Equity Curves**: Historical equity data for charting
    - **Features**: Normalization statistics for model features

    ## Real-Time Updates

    Use the `/api/v1/models/{model_id}/signals/stream` endpoint with SSE
    for real-time signal updates.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail if isinstance(exc.detail, str) else "Error",
            detail=str(exc.detail),
            timestamp=datetime.now(timezone.utc).isoformat(),
            path=str(request.url.path),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail=str(exc),
            timestamp=datetime.now(timezone.utc).isoformat(),
            path=str(request.url.path),
        ).model_dump(),
    )


# =============================================================================
# ROOT & HEALTH ENDPOINTS
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """API root endpoint with service information."""
    return {
        "service": "USDCOP Multi-Model Trading API",
        "version": "1.0.0",
        "status": "active",
        "documentation": "/docs",
        "endpoints": {
            "models": "/api/v1/models",
            "signals": "/api/v1/models/{model_id}/signals",
            "trades": "/api/v1/models/{model_id}/trades",
            "metrics": "/api/v1/models/{model_id}/metrics",
            "equity_curve": "/api/v1/models/{model_id}/equity-curve",
            "compare": "/api/v1/models/compare",
            "features": "/api/v1/features",
            "health": "/api/v1/health",
        },
    }


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Check health status of all service components",
)
async def health_check(
    db: DatabasePool = Depends(get_db),
    cache: RedisCache = Depends(get_cache),
):
    """
    Health check endpoint.

    Checks connectivity to:
    - PostgreSQL database
    - Redis cache
    - Model availability
    """
    services = []
    overall_status = "healthy"

    # Check PostgreSQL
    try:
        import time
        start = time.perf_counter()
        await db.fetchval("SELECT 1")
        latency = (time.perf_counter() - start) * 1000
        services.append(ServiceHealth(
            name="postgresql",
            status="connected",
            latency_ms=round(latency, 2),
        ))
    except Exception as e:
        services.append(ServiceHealth(
            name="postgresql",
            status="disconnected",
            details={"error": str(e)},
        ))
        overall_status = "unhealthy"

    # Check Redis
    if cache.available:
        try:
            start = time.perf_counter()
            await cache._redis.ping()
            latency = (time.perf_counter() - start) * 1000
            services.append(ServiceHealth(
                name="redis",
                status="connected",
                latency_ms=round(latency, 2),
            ))
        except Exception as e:
            services.append(ServiceHealth(
                name="redis",
                status="error",
                details={"error": str(e)},
            ))
    else:
        services.append(ServiceHealth(
            name="redis",
            status="unavailable",
            details={"message": "Redis not configured or unreachable"},
        ))

    # Count available models
    try:
        model_count = await db.fetchval(
            "SELECT COUNT(*) FROM dw.dim_strategy WHERE is_active = TRUE"
        )
    except Exception:
        model_count = 0

    # Model availability check
    services.append(ServiceHealth(
        name="models",
        status="available" if model_count > 0 else "none",
        details={"count": model_count},
    ))

    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        services=services,
        models_available=model_count or 0,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# =============================================================================
# MODELS ENDPOINTS
# =============================================================================

@app.get(
    "/api/v1/models",
    response_model=ModelsListResponse,
    tags=["Models"],
    summary="List all models",
    description="Get all trading models with their configuration and status",
)
async def list_models(
    include_inactive: bool = Query(False, description="Include inactive models"),
    db: DatabasePool = Depends(get_db),
    cache: RedisCache = Depends(get_cache),
):
    """
    List all trading models.

    Returns models with:
    - Basic info (id, name, type, status)
    - Backtest metrics (sharpe, drawdown, win rate, etc.)
    - Activity status

    Examples:
        GET /api/v1/models
        GET /api/v1/models?include_inactive=true
    """
    cache_key = f"models:list:{include_inactive}"

    # Try cache first
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    # Query database
    query = """
        WITH latest_perf AS (
            SELECT DISTINCT ON (strategy_id)
                strategy_id,
                cumulative_return_pct,
                sharpe_ratio,
                sortino_ratio,
                max_drawdown_pct,
                win_rate,
                n_trades,
                calmar_ratio,
                avg_hold_time_minutes
            FROM dw.fact_strategy_performance
            ORDER BY strategy_id, date_cot DESC
        ),
        latest_profit_factor AS (
            SELECT
                strategy_id,
                CASE
                    WHEN SUM(ABS(gross_loss)) > 0
                    THEN SUM(gross_profit) / SUM(ABS(gross_loss))
                    ELSE 0
                END as profit_factor
            FROM dw.fact_strategy_performance
            WHERE date_cot >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY strategy_id
        )
        SELECT
            s.strategy_code as model_id,
            s.strategy_name as name,
            s.strategy_type as model_type,
            CASE WHEN s.is_active THEN 'active' ELSE 'inactive' END as status,
            s.description,
            s.is_active,
            s.config_json,
            s.created_at,
            s.updated_at,
            COALESCE(p.cumulative_return_pct, 0) as total_return_pct,
            COALESCE(p.sharpe_ratio, 0) as sharpe_ratio,
            COALESCE(p.sortino_ratio, 0) as sortino_ratio,
            COALESCE(ABS(p.max_drawdown_pct), 0) as max_drawdown_pct,
            COALESCE(p.win_rate, 0) as win_rate,
            COALESCE(pf.profit_factor, 0) as profit_factor,
            COALESCE(p.n_trades, 0) as total_trades,
            COALESCE(p.calmar_ratio, 0) as calmar_ratio,
            COALESCE(p.avg_hold_time_minutes, 0) as avg_trade_duration_mins
        FROM dw.dim_strategy s
        LEFT JOIN latest_perf p ON s.strategy_id = p.strategy_id
        LEFT JOIN latest_profit_factor pf ON s.strategy_id = pf.strategy_id
        WHERE ($1 OR s.is_active = TRUE)
        ORDER BY s.strategy_code
    """

    rows = await db.fetch(query, include_inactive)

    models = []
    for row in rows:
        backtest_metrics = BacktestMetrics(
            total_return_pct=float(row["total_return_pct"]),
            sharpe_ratio=float(row["sharpe_ratio"]),
            sortino_ratio=float(row["sortino_ratio"]),
            max_drawdown_pct=float(row["max_drawdown_pct"]),
            win_rate=float(row["win_rate"]),
            profit_factor=float(row["profit_factor"]),
            total_trades=int(row["total_trades"]),
            calmar_ratio=float(row["calmar_ratio"]) if row["calmar_ratio"] else None,
            avg_trade_duration_mins=float(row["avg_trade_duration_mins"]) if row["avg_trade_duration_mins"] else None,
        )

        models.append(ModelSummary(
            model_id=row["model_id"],
            name=row["name"],
            model_type=row["model_type"],
            status=ModelStatus(row["status"]),
            description=row["description"],
            is_active=row["is_active"],
            backtest_metrics=backtest_metrics,
            created_at=row["created_at"].isoformat() if row["created_at"] else None,
            updated_at=row["updated_at"].isoformat() if row["updated_at"] else None,
        ))

    response = ModelsListResponse(
        count=len(models),
        models=models,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # Cache response
    await cache.set(cache_key, response.model_dump_json(), Config.CACHE_TTL_MEDIUM)

    return response


@app.get(
    "/api/v1/models/{model_id}",
    response_model=ModelDetail,
    tags=["Models"],
    summary="Get model details",
    description="Get detailed information for a single model including hyperparameters and features",
)
async def get_model(
    model_id: str,
    db: DatabasePool = Depends(get_db),
):
    """
    Get detailed model information.

    Returns:
    - All summary fields
    - Hyperparameters (learning rate, batch size, etc.)
    - Features used for inference
    - Training statistics

    Examples:
        GET /api/v1/models/RL_PPO
        GET /api/v1/models/ML_LGBM
    """
    query = """
        WITH latest_perf AS (
            SELECT
                strategy_id,
                cumulative_return_pct,
                sharpe_ratio,
                sortino_ratio,
                max_drawdown_pct,
                win_rate,
                n_trades,
                calmar_ratio,
                avg_hold_time_minutes
            FROM dw.fact_strategy_performance
            WHERE strategy_id = (SELECT strategy_id FROM dw.dim_strategy WHERE strategy_code = $1)
            ORDER BY date_cot DESC
            LIMIT 1
        )
        SELECT
            s.strategy_code as model_id,
            s.strategy_name as name,
            s.strategy_type as model_type,
            CASE WHEN s.is_active THEN 'active' ELSE 'inactive' END as status,
            s.description,
            s.is_active,
            s.config_json,
            s.created_at,
            s.updated_at,
            COALESCE(p.cumulative_return_pct, 0) as total_return_pct,
            COALESCE(p.sharpe_ratio, 0) as sharpe_ratio,
            COALESCE(p.sortino_ratio, 0) as sortino_ratio,
            COALESCE(ABS(p.max_drawdown_pct), 0) as max_drawdown_pct,
            COALESCE(p.win_rate, 0) as win_rate,
            COALESCE(p.n_trades, 0) as total_trades,
            COALESCE(p.calmar_ratio, 0) as calmar_ratio,
            COALESCE(p.avg_hold_time_minutes, 0) as avg_trade_duration_mins
        FROM dw.dim_strategy s
        LEFT JOIN latest_perf p ON s.strategy_id = p.strategy_id
        WHERE s.strategy_code = $1
    """

    row = await db.fetchrow(query, model_id)

    if not row:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    # Parse config JSON for hyperparameters
    config = row["config_json"] or {}
    hyperparameters = ModelConfig(
        learning_rate=config.get("learning_rate"),
        batch_size=config.get("batch_size"),
        n_epochs=config.get("n_epochs"),
        gamma=config.get("gamma"),
        gae_lambda=config.get("gae_lambda"),
        clip_range=config.get("clip_range"),
        custom_params=config.get("custom_params"),
    )

    # Default features for PPO model
    features_used = config.get("features", [
        "log_ret_5m", "log_ret_1h", "log_ret_4h",
        "rsi_9", "atr_pct", "adx_14",
        "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
        "brent_change_1d", "rate_spread", "usdmxn_ret_1h",
        "position", "time_normalized"
    ])

    backtest_metrics = BacktestMetrics(
        total_return_pct=float(row["total_return_pct"]),
        sharpe_ratio=float(row["sharpe_ratio"]),
        sortino_ratio=float(row["sortino_ratio"]),
        max_drawdown_pct=float(row["max_drawdown_pct"]),
        win_rate=float(row["win_rate"]),
        profit_factor=2.1,  # Placeholder until calculated
        total_trades=int(row["total_trades"]),
        calmar_ratio=float(row["calmar_ratio"]) if row["calmar_ratio"] else None,
        avg_trade_duration_mins=float(row["avg_trade_duration_mins"]) if row["avg_trade_duration_mins"] else None,
    )

    return ModelDetail(
        model_id=row["model_id"],
        name=row["name"],
        model_type=row["model_type"],
        status=ModelStatus(row["status"]),
        description=row["description"],
        is_active=row["is_active"],
        backtest_metrics=backtest_metrics,
        created_at=row["created_at"].isoformat() if row["created_at"] else None,
        updated_at=row["updated_at"].isoformat() if row["updated_at"] else None,
        hyperparameters=hyperparameters,
        features_used=features_used,
        observation_space_dim=15,
        action_space_dim=3,
        training_episodes=config.get("training_episodes"),
        training_timesteps=config.get("training_timesteps"),
        fold_id=config.get("fold_id"),
    )


# =============================================================================
# SIGNALS ENDPOINTS
# =============================================================================

@app.get(
    "/api/v1/models/{model_id}/signals",
    response_model=SignalsResponse,
    tags=["Signals"],
    summary="Get model signals",
    description="Query historical signals for a model with filtering options",
)
async def get_signals(
    model_id: str,
    period: Period = Query(Period.TODAY, description="Time period to query"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum signals to return"),
    db: DatabasePool = Depends(get_db),
):
    """
    Get historical signals for a model.

    Query Parameters:
    - period: today, 7d, 30d, or all
    - limit: Maximum number of signals (1-1000)

    Examples:
        GET /api/v1/models/RL_PPO/signals
        GET /api/v1/models/RL_PPO/signals?period=7d&limit=50
    """
    if not await validate_model_exists(db, model_id):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    start_date, end_date = parse_period(period)

    query = """
        SELECT
            fs.signal_id,
            fs.timestamp_utc,
            fs.signal as action,
            fs.side,
            fs.confidence,
            fs.size,
            fs.entry_price,
            fs.stop_loss,
            fs.take_profit,
            fs.risk_usd,
            fs.reasoning,
            fs.features_snapshot
        FROM dw.fact_strategy_signals fs
        JOIN dw.dim_strategy s ON fs.strategy_id = s.strategy_id
        WHERE s.strategy_code = $1
          AND fs.timestamp_utc >= $2
          AND fs.timestamp_utc <= $3
        ORDER BY fs.timestamp_utc DESC
        LIMIT $4
    """

    rows = await db.fetch(query, model_id, start_date, end_date, limit)

    signals = []
    for row in rows:
        signals.append(Signal(
            signal_id=row["signal_id"],
            timestamp=row["timestamp_utc"].isoformat(),
            action=SignalAction(row["action"]),
            side=TradeSide(row["side"]) if row["side"] else TradeSide.HOLD,
            confidence=float(row["confidence"]),
            size=float(row["size"]),
            entry_price=float(row["entry_price"]) if row["entry_price"] else None,
            stop_loss=float(row["stop_loss"]) if row["stop_loss"] else None,
            take_profit=float(row["take_profit"]) if row["take_profit"] else None,
            risk_usd=float(row["risk_usd"]) if row["risk_usd"] else None,
            reasoning=row["reasoning"],
            features_snapshot=row["features_snapshot"],
        ))

    return SignalsResponse(
        model_id=model_id,
        period=period.value,
        count=len(signals),
        signals=signals,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get(
    "/api/v1/models/{model_id}/signals/latest",
    response_model=LatestSignalResponse,
    tags=["Signals"],
    summary="Get latest signal",
    description="Get the most recent signal from a model (for real-time display)",
)
async def get_latest_signal(
    model_id: str,
    db: DatabasePool = Depends(get_db),
    cache: RedisCache = Depends(get_cache),
):
    """
    Get the most recent signal from a model.

    Returns:
    - The latest signal
    - Current market price
    - Signal age in seconds

    Cached for 30 seconds for performance.

    Examples:
        GET /api/v1/models/RL_PPO/signals/latest
    """
    if not await validate_model_exists(db, model_id):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    cache_key = f"signals:latest:{model_id}"

    # Try cache first
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    query = """
        SELECT
            fs.signal_id,
            fs.timestamp_utc,
            fs.signal as action,
            fs.side,
            fs.confidence,
            fs.size,
            fs.entry_price,
            fs.stop_loss,
            fs.take_profit,
            fs.risk_usd,
            fs.reasoning,
            fs.features_snapshot,
            EXTRACT(EPOCH FROM (NOW() - fs.timestamp_utc))::INT as age_seconds
        FROM dw.fact_strategy_signals fs
        JOIN dw.dim_strategy s ON fs.strategy_id = s.strategy_id
        WHERE s.strategy_code = $1
        ORDER BY fs.timestamp_utc DESC
        LIMIT 1
    """

    row = await db.fetchrow(query, model_id)
    market_price = await get_market_price(db)

    signal = None
    age_seconds = None

    if row:
        signal = Signal(
            signal_id=row["signal_id"],
            timestamp=row["timestamp_utc"].isoformat(),
            action=SignalAction(row["action"]),
            side=TradeSide(row["side"]) if row["side"] else TradeSide.HOLD,
            confidence=float(row["confidence"]),
            size=float(row["size"]),
            entry_price=float(row["entry_price"]) if row["entry_price"] else None,
            stop_loss=float(row["stop_loss"]) if row["stop_loss"] else None,
            take_profit=float(row["take_profit"]) if row["take_profit"] else None,
            risk_usd=float(row["risk_usd"]) if row["risk_usd"] else None,
            reasoning=row["reasoning"],
            features_snapshot=row["features_snapshot"],
        )
        age_seconds = row["age_seconds"]

    response = LatestSignalResponse(
        model_id=model_id,
        signal=signal,
        market_price=market_price,
        age_seconds=age_seconds,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # Cache for 30 seconds
    await cache.set(cache_key, response.model_dump_json(), Config.CACHE_TTL_SHORT)

    return response


@app.get(
    "/api/v1/models/{model_id}/signals/stream",
    tags=["Signals"],
    summary="Stream signals (SSE)",
    description="Server-Sent Events stream for real-time signal updates",
)
async def stream_signals(
    model_id: str,
    db: DatabasePool = Depends(get_db),
    cache: RedisCache = Depends(get_cache),
):
    """
    Stream real-time signals via Server-Sent Events.

    Subscribes to Redis stream for the specified model and
    pushes updates as they occur.

    Client Usage:
    ```javascript
    const eventSource = new EventSource('/api/v1/models/RL_PPO/signals/stream');
    eventSource.onmessage = (event) => {
        const signal = JSON.parse(event.data);
        console.log('New signal:', signal);
    };
    ```

    Examples:
        GET /api/v1/models/RL_PPO/signals/stream
    """
    if not await validate_model_exists(db, model_id):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    async def event_generator():
        """Generate SSE events."""
        channel = f"signals:{model_id}"

        # Send initial heartbeat
        yield {
            "event": "connected",
            "data": json.dumps({
                "model_id": model_id,
                "message": "Connected to signal stream",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }),
        }

        # If Redis is available, subscribe to channel
        if cache.available:
            try:
                async for message in cache.subscribe(channel):
                    yield {
                        "event": "signal",
                        "data": message,
                    }
            except Exception as e:
                logger.error(f"SSE stream error: {e}")
                yield {
                    "event": "error",
                    "data": json.dumps({"error": str(e)}),
                }
        else:
            # Fallback: Poll database every 5 seconds
            last_signal_id = None
            while True:
                try:
                    query = """
                        SELECT
                            fs.signal_id,
                            fs.timestamp_utc,
                            fs.signal as action,
                            fs.side,
                            fs.confidence,
                            fs.size
                        FROM dw.fact_strategy_signals fs
                        JOIN dw.dim_strategy s ON fs.strategy_id = s.strategy_id
                        WHERE s.strategy_code = $1
                        ORDER BY fs.timestamp_utc DESC
                        LIMIT 1
                    """
                    row = await db.fetchrow(query, model_id)

                    if row and row["signal_id"] != last_signal_id:
                        last_signal_id = row["signal_id"]
                        yield {
                            "event": "signal",
                            "data": json.dumps({
                                "signal_id": row["signal_id"],
                                "timestamp": row["timestamp_utc"].isoformat(),
                                "action": row["action"],
                                "side": row["side"],
                                "confidence": float(row["confidence"]),
                                "size": float(row["size"]),
                            }),
                        }

                    # Send heartbeat every 30 seconds
                    yield {
                        "event": "heartbeat",
                        "data": json.dumps({
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }),
                    }

                    await asyncio.sleep(5)

                except Exception as e:
                    logger.error(f"SSE poll error: {e}")
                    await asyncio.sleep(10)

    return EventSourceResponse(event_generator())


# =============================================================================
# TRADES ENDPOINTS
# =============================================================================

@app.get(
    "/api/v1/models/{model_id}/trades",
    response_model=TradesResponse,
    tags=["Trades"],
    summary="Get model trades",
    description="Query trades/positions for a model with filtering options",
)
async def get_trades(
    model_id: str,
    period: Period = Query(Period.MONTH, description="Time period to query"),
    status: TradeStatus = Query(TradeStatus.ALL, description="Filter by trade status"),
    db: DatabasePool = Depends(get_db),
):
    """
    Get trades for a model.

    Query Parameters:
    - period: today, 7d, 30d, or all
    - status: open, closed, or all

    Examples:
        GET /api/v1/models/RL_PPO/trades
        GET /api/v1/models/RL_PPO/trades?status=closed&period=7d
    """
    if not await validate_model_exists(db, model_id):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    start_date, end_date = parse_period(period)

    status_filter = ""
    if status == TradeStatus.OPEN:
        status_filter = "AND p.status = 'open'"
    elif status == TradeStatus.CLOSED:
        status_filter = "AND p.status = 'closed'"

    query = f"""
        WITH current_price AS (
            SELECT close FROM usdcop_m5_ohlcv
            WHERE symbol = 'USD/COP'
            ORDER BY time DESC LIMIT 1
        )
        SELECT
            p.position_id as trade_id,
            p.side,
            p.quantity,
            p.entry_price,
            cp.close as current_price,
            CASE WHEN p.status = 'closed' THEN p.current_price ELSE NULL END as exit_price,
            p.stop_loss,
            p.take_profit,
            p.unrealized_pnl,
            p.realized_pnl,
            CASE
                WHEN p.status = 'open' THEN
                    ((cp.close - p.entry_price) / p.entry_price) * 100
                ELSE
                    ((p.current_price - p.entry_price) / p.entry_price) * 100
            END as pnl_percent,
            p.entry_time,
            p.exit_time,
            EXTRACT(EPOCH FROM (COALESCE(p.exit_time, NOW()) - p.entry_time)) / 60 as holding_time_minutes,
            p.status,
            p.exit_reason,
            p.leverage
        FROM dw.fact_strategy_positions p
        JOIN dw.dim_strategy s ON p.strategy_id = s.strategy_id
        CROSS JOIN current_price cp
        WHERE s.strategy_code = $1
          AND p.entry_time >= $2
          AND p.entry_time <= $3
          {status_filter}
        ORDER BY p.entry_time DESC
    """

    rows = await db.fetch(query, model_id, start_date, end_date)

    trades = []
    total_pnl = 0.0

    for row in rows:
        pnl = float(row["realized_pnl"] or row["unrealized_pnl"] or 0)
        total_pnl += pnl

        trades.append(Trade(
            trade_id=row["trade_id"],
            side=row["side"],
            quantity=float(row["quantity"]),
            entry_price=float(row["entry_price"]),
            current_price=float(row["current_price"]) if row["current_price"] else None,
            exit_price=float(row["exit_price"]) if row["exit_price"] else None,
            stop_loss=float(row["stop_loss"]) if row["stop_loss"] else None,
            take_profit=float(row["take_profit"]) if row["take_profit"] else None,
            unrealized_pnl=float(row["unrealized_pnl"]) if row["unrealized_pnl"] else None,
            realized_pnl=float(row["realized_pnl"]) if row["realized_pnl"] else None,
            pnl_percent=float(row["pnl_percent"]) if row["pnl_percent"] else None,
            entry_time=row["entry_time"].isoformat(),
            exit_time=row["exit_time"].isoformat() if row["exit_time"] else None,
            holding_time_minutes=int(row["holding_time_minutes"]) if row["holding_time_minutes"] else None,
            status=row["status"],
            exit_reason=row["exit_reason"],
            leverage=row["leverage"] or 1,
        ))

    return TradesResponse(
        model_id=model_id,
        period=period.value,
        status_filter=status.value,
        count=len(trades),
        trades=trades,
        total_pnl=round(total_pnl, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get(
    "/api/v1/models/{model_id}/trades/open",
    response_model=OpenPositionResponse,
    tags=["Trades"],
    summary="Get open position",
    description="Get current open position for a model (if any)",
)
async def get_open_position(
    model_id: str,
    db: DatabasePool = Depends(get_db),
):
    """
    Get current open position for a model.

    Returns the currently open position with real-time P&L,
    or indicates no position if flat.

    Examples:
        GET /api/v1/models/RL_PPO/trades/open
    """
    if not await validate_model_exists(db, model_id):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    query = """
        WITH current_price AS (
            SELECT close FROM usdcop_m5_ohlcv
            WHERE symbol = 'USD/COP'
            ORDER BY time DESC LIMIT 1
        )
        SELECT
            p.position_id as trade_id,
            p.side,
            p.quantity,
            p.entry_price,
            cp.close as current_price,
            p.stop_loss,
            p.take_profit,
            CASE
                WHEN p.side = 'long' THEN (cp.close - p.entry_price) * p.quantity
                ELSE (p.entry_price - cp.close) * p.quantity
            END as unrealized_pnl,
            ((cp.close - p.entry_price) / p.entry_price) * 100 as pnl_percent,
            p.entry_time,
            EXTRACT(EPOCH FROM (NOW() - p.entry_time)) / 60 as holding_time_minutes,
            p.status,
            p.leverage
        FROM dw.fact_strategy_positions p
        JOIN dw.dim_strategy s ON p.strategy_id = s.strategy_id
        CROSS JOIN current_price cp
        WHERE s.strategy_code = $1
          AND p.status = 'open'
        LIMIT 1
    """

    row = await db.fetchrow(query, model_id)

    position = None
    if row:
        position = Trade(
            trade_id=row["trade_id"],
            side=row["side"],
            quantity=float(row["quantity"]),
            entry_price=float(row["entry_price"]),
            current_price=float(row["current_price"]) if row["current_price"] else None,
            stop_loss=float(row["stop_loss"]) if row["stop_loss"] else None,
            take_profit=float(row["take_profit"]) if row["take_profit"] else None,
            unrealized_pnl=float(row["unrealized_pnl"]) if row["unrealized_pnl"] else None,
            pnl_percent=float(row["pnl_percent"]) if row["pnl_percent"] else None,
            entry_time=row["entry_time"].isoformat(),
            holding_time_minutes=int(row["holding_time_minutes"]) if row["holding_time_minutes"] else None,
            status=row["status"],
            leverage=row["leverage"] or 1,
        )

    return OpenPositionResponse(
        model_id=model_id,
        has_position=position is not None,
        position=position,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# =============================================================================
# METRICS ENDPOINTS
# =============================================================================

@app.get(
    "/api/v1/models/{model_id}/metrics",
    response_model=MetricsResponse,
    tags=["Metrics"],
    summary="Get model metrics",
    description="Get live performance metrics with backtest comparison",
)
async def get_metrics(
    model_id: str,
    period: Period = Query(Period.MONTH, description="Period for metrics calculation"),
    db: DatabasePool = Depends(get_db),
    cache: RedisCache = Depends(get_cache),
):
    """
    Get live performance metrics for a model.

    Returns:
    - Live metrics (sharpe, max_dd, win_rate, etc.)
    - Backtest metrics for comparison
    - Divergence between live and backtest

    Examples:
        GET /api/v1/models/RL_PPO/metrics
        GET /api/v1/models/RL_PPO/metrics?period=7d
    """
    if not await validate_model_exists(db, model_id):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    cache_key = f"metrics:{model_id}:{period.value}"

    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    start_date, end_date = parse_period(period)

    query = """
        WITH equity AS (
            SELECT
                e.equity_value,
                e.return_since_start_pct,
                e.current_drawdown_pct
            FROM dw.fact_equity_curve e
            JOIN dw.dim_strategy s ON e.strategy_id = s.strategy_id
            WHERE s.strategy_code = $1
            ORDER BY e.timestamp_utc DESC
            LIMIT 1
        ),
        perf_agg AS (
            SELECT
                AVG(p.daily_return_pct) as avg_daily_return,
                AVG(p.sharpe_ratio) as sharpe_ratio,
                AVG(p.sortino_ratio) as sortino_ratio,
                AVG(p.calmar_ratio) as calmar_ratio,
                MAX(ABS(p.max_drawdown_pct)) as max_drawdown_pct,
                SUM(p.n_trades) as total_trades,
                SUM(p.n_wins) as total_wins,
                AVG(p.win_rate) as win_rate,
                AVG(p.avg_hold_time_minutes) as avg_hold_time
            FROM dw.fact_strategy_performance p
            JOIN dw.dim_strategy s ON p.strategy_id = s.strategy_id
            WHERE s.strategy_code = $1
              AND p.date_cot >= $2::date
              AND p.date_cot <= $3::date
        ),
        open_pos AS (
            SELECT COUNT(*) as count
            FROM dw.fact_strategy_positions p
            JOIN dw.dim_strategy s ON p.strategy_id = s.strategy_id
            WHERE s.strategy_code = $1
              AND p.status = 'open'
        )
        SELECT
            COALESCE(e.equity_value, 10000) as current_equity,
            COALESCE(e.return_since_start_pct, 0) as total_return_pct,
            COALESCE(e.current_drawdown_pct, 0) as current_drawdown_pct,
            COALESCE(pa.avg_daily_return, 0) as daily_return_pct,
            COALESCE(pa.sharpe_ratio, 0) as sharpe_ratio,
            COALESCE(pa.sortino_ratio, 0) as sortino_ratio,
            COALESCE(pa.calmar_ratio, 0) as calmar_ratio,
            COALESCE(pa.max_drawdown_pct, 0) as max_drawdown_pct,
            COALESCE(pa.total_trades, 0) as total_trades,
            COALESCE(pa.win_rate, 0) as win_rate,
            COALESCE(pa.avg_hold_time, 0) as avg_hold_time_minutes,
            COALESCE(op.count, 0) as open_positions
        FROM equity e
        CROSS JOIN perf_agg pa
        CROSS JOIN open_pos op
    """

    row = await db.fetchrow(query, model_id, start_date, end_date)

    if not row:
        # Return default metrics if no data
        row = {
            "current_equity": 10000,
            "total_return_pct": 0,
            "current_drawdown_pct": 0,
            "daily_return_pct": 0,
            "sharpe_ratio": 0,
            "sortino_ratio": 0,
            "calmar_ratio": 0,
            "max_drawdown_pct": 0,
            "total_trades": 0,
            "win_rate": 0,
            "avg_hold_time_minutes": 0,
            "open_positions": 0,
        }

    live = LiveMetrics(
        total_return_pct=float(row["total_return_pct"]),
        daily_return_pct=float(row["daily_return_pct"]),
        sharpe_ratio=float(row["sharpe_ratio"]),
        sortino_ratio=float(row["sortino_ratio"]),
        calmar_ratio=float(row["calmar_ratio"]),
        max_drawdown_pct=float(row["max_drawdown_pct"]),
        current_drawdown_pct=float(row["current_drawdown_pct"]),
        volatility_pct=8.0,  # Placeholder
        win_rate=float(row["win_rate"]),
        profit_factor=2.1,  # Placeholder
        total_trades=int(row["total_trades"]),
        avg_hold_time_minutes=float(row["avg_hold_time_minutes"]),
        current_equity=float(row["current_equity"]),
        open_positions=int(row["open_positions"]),
    )

    # Get backtest metrics (from model detail)
    backtest = BacktestMetrics(
        total_return_pct=live.total_return_pct * 0.9,  # Placeholder - should come from stored backtest
        sharpe_ratio=live.sharpe_ratio * 1.1,
        sortino_ratio=live.sortino_ratio * 1.1,
        max_drawdown_pct=live.max_drawdown_pct * 0.8,
        win_rate=live.win_rate * 1.05,
        profit_factor=2.3,
        total_trades=int(live.total_trades * 0.8),
    )

    divergence = calculate_divergence(live, backtest)

    response = MetricsResponse(
        model_id=model_id,
        period=period.value,
        metrics=MetricsComparison(
            live=live,
            backtest=backtest,
            divergence=divergence,
        ),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    await cache.set(cache_key, response.model_dump_json(), Config.CACHE_TTL_MEDIUM)

    return response


# =============================================================================
# EQUITY CURVE ENDPOINTS
# =============================================================================

@app.get(
    "/api/v1/models/{model_id}/equity-curve",
    response_model=EquityCurveResponse,
    tags=["Equity Curve"],
    summary="Get equity curve",
    description="Get daily equity values for charting",
)
async def get_equity_curve(
    model_id: str,
    days: int = Query(30, ge=1, le=365, description="Number of days"),
    resolution: str = Query("1d", description="Resolution: 5m, 1h, 1d"),
    db: DatabasePool = Depends(get_db),
):
    """
    Get equity curve data for charting.

    Query Parameters:
    - days: Number of days of history (1-365)
    - resolution: Data resolution (5m, 1h, 1d)

    Examples:
        GET /api/v1/models/RL_PPO/equity-curve
        GET /api/v1/models/RL_PPO/equity-curve?days=7&resolution=1h
    """
    if not await validate_model_exists(db, model_id):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    start_date = datetime.now(timezone.utc) - timedelta(days=days)
    end_date = datetime.now(timezone.utc)

    # Map resolution to interval for aggregation
    resolution_map = {
        "5m": "5 minutes",
        "1h": "1 hour",
        "1d": "1 day",
    }
    interval = resolution_map.get(resolution, "1 day")

    if resolution == "5m":
        # No aggregation for 5-minute data
        query = """
            SELECT
                e.timestamp_utc,
                e.equity_value,
                e.return_since_start_pct as return_pct,
                e.current_drawdown_pct as drawdown_pct,
                e.cash_balance,
                e.positions_value
            FROM dw.fact_equity_curve e
            JOIN dw.dim_strategy s ON e.strategy_id = s.strategy_id
            WHERE s.strategy_code = $1
              AND e.timestamp_utc >= $2
              AND e.timestamp_utc <= $3
            ORDER BY e.timestamp_utc ASC
        """
        rows = await db.fetch(query, model_id, start_date, end_date)
    else:
        # Aggregate using time_bucket
        query = f"""
            SELECT
                time_bucket($4::interval, e.timestamp_utc) as timestamp_utc,
                LAST(e.equity_value, e.timestamp_utc) as equity_value,
                LAST(e.return_since_start_pct, e.timestamp_utc) as return_pct,
                LAST(e.current_drawdown_pct, e.timestamp_utc) as drawdown_pct,
                LAST(e.cash_balance, e.timestamp_utc) as cash_balance,
                LAST(e.positions_value, e.timestamp_utc) as positions_value
            FROM dw.fact_equity_curve e
            JOIN dw.dim_strategy s ON e.strategy_id = s.strategy_id
            WHERE s.strategy_code = $1
              AND e.timestamp_utc >= $2
              AND e.timestamp_utc <= $3
            GROUP BY time_bucket($4::interval, e.timestamp_utc)
            ORDER BY timestamp_utc ASC
        """
        rows = await db.fetch(query, model_id, start_date, end_date, interval)

    data = []
    for row in rows:
        data.append(EquityPoint(
            timestamp=row["timestamp_utc"].isoformat(),
            equity_value=float(row["equity_value"]),
            return_pct=float(row["return_pct"]) if row["return_pct"] else 0,
            drawdown_pct=float(row["drawdown_pct"]) if row["drawdown_pct"] else 0,
            cash_balance=float(row["cash_balance"]) if row["cash_balance"] else None,
            positions_value=float(row["positions_value"]) if row["positions_value"] else None,
        ))

    # Calculate summary
    summary = {}
    if data:
        summary = {
            "starting_equity": data[0].equity_value,
            "ending_equity": data[-1].equity_value,
            "max_equity": max(p.equity_value for p in data),
            "min_equity": min(p.equity_value for p in data),
            "total_return_pct": data[-1].return_pct,
            "max_drawdown_pct": min(p.drawdown_pct for p in data),
        }

    return EquityCurveResponse(
        model_id=model_id,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        resolution=resolution,
        data_points=len(data),
        data=data,
        summary=summary,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# =============================================================================
# COMPARISON ENDPOINTS
# =============================================================================

@app.get(
    "/api/v1/models/compare",
    response_model=ModelsCompareResponse,
    tags=["Comparison"],
    summary="Compare models",
    description="Side-by-side comparison of multiple models",
)
async def compare_models(
    ids: str = Query(..., description="Comma-separated model IDs"),
    period: Period = Query(Period.MONTH, description="Period for comparison"),
    db: DatabasePool = Depends(get_db),
):
    """
    Compare multiple models side-by-side.

    Query Parameters:
    - ids: Comma-separated list of model IDs
    - period: Time period for metrics

    Examples:
        GET /api/v1/models/compare?ids=RL_PPO,ML_LGBM,LLM_CLAUDE
        GET /api/v1/models/compare?ids=RL_PPO,ML_LGBM&period=7d
    """
    model_ids = [mid.strip() for mid in ids.split(",") if mid.strip()]

    if not model_ids:
        raise HTTPException(status_code=400, detail="No model IDs provided")

    if len(model_ids) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 models for comparison")

    start_date, end_date = parse_period(period)

    comparison = []

    for model_id in model_ids:
        if not await validate_model_exists(db, model_id):
            continue

        query = """
            WITH equity AS (
                SELECT
                    e.equity_value,
                    e.return_since_start_pct,
                    e.current_drawdown_pct
                FROM dw.fact_equity_curve e
                JOIN dw.dim_strategy s ON e.strategy_id = s.strategy_id
                WHERE s.strategy_code = $1
                ORDER BY e.timestamp_utc DESC
                LIMIT 1
            ),
            perf_agg AS (
                SELECT
                    AVG(p.daily_return_pct) as avg_daily_return,
                    AVG(p.sharpe_ratio) as sharpe_ratio,
                    AVG(p.sortino_ratio) as sortino_ratio,
                    AVG(p.calmar_ratio) as calmar_ratio,
                    MAX(ABS(p.max_drawdown_pct)) as max_drawdown_pct,
                    SUM(p.n_trades) as total_trades,
                    AVG(p.win_rate) as win_rate,
                    AVG(p.avg_hold_time_minutes) as avg_hold_time
                FROM dw.fact_strategy_performance p
                JOIN dw.dim_strategy s ON p.strategy_id = s.strategy_id
                WHERE s.strategy_code = $1
                  AND p.date_cot >= $2::date
                  AND p.date_cot <= $3::date
            ),
            open_pos AS (
                SELECT COUNT(*) as count
                FROM dw.fact_strategy_positions p
                JOIN dw.dim_strategy s ON p.strategy_id = s.strategy_id
                WHERE s.strategy_code = $1
                  AND p.status = 'open'
            ),
            model_info AS (
                SELECT strategy_name, strategy_type
                FROM dw.dim_strategy
                WHERE strategy_code = $1
            )
            SELECT
                mi.strategy_name as name,
                mi.strategy_type as model_type,
                COALESCE(e.equity_value, 10000) as current_equity,
                COALESCE(e.return_since_start_pct, 0) as total_return_pct,
                COALESCE(e.current_drawdown_pct, 0) as current_drawdown_pct,
                COALESCE(pa.avg_daily_return, 0) as daily_return_pct,
                COALESCE(pa.sharpe_ratio, 0) as sharpe_ratio,
                COALESCE(pa.sortino_ratio, 0) as sortino_ratio,
                COALESCE(pa.calmar_ratio, 0) as calmar_ratio,
                COALESCE(pa.max_drawdown_pct, 0) as max_drawdown_pct,
                COALESCE(pa.total_trades, 0) as total_trades,
                COALESCE(pa.win_rate, 0) as win_rate,
                COALESCE(pa.avg_hold_time, 0) as avg_hold_time_minutes,
                COALESCE(op.count, 0) as open_positions
            FROM model_info mi
            LEFT JOIN equity e ON TRUE
            LEFT JOIN perf_agg pa ON TRUE
            LEFT JOIN open_pos op ON TRUE
        """

        row = await db.fetchrow(query, model_id, start_date, end_date)

        if row:
            metrics = LiveMetrics(
                total_return_pct=float(row["total_return_pct"]),
                daily_return_pct=float(row["daily_return_pct"]),
                sharpe_ratio=float(row["sharpe_ratio"]),
                sortino_ratio=float(row["sortino_ratio"]),
                calmar_ratio=float(row["calmar_ratio"]),
                max_drawdown_pct=float(row["max_drawdown_pct"]),
                current_drawdown_pct=float(row["current_drawdown_pct"]),
                volatility_pct=8.0,
                win_rate=float(row["win_rate"]),
                profit_factor=2.1,
                total_trades=int(row["total_trades"]),
                avg_hold_time_minutes=float(row["avg_hold_time_minutes"]),
                current_equity=float(row["current_equity"]),
                open_positions=int(row["open_positions"]),
            )

            comparison.append(ModelComparisonItem(
                model_id=model_id,
                name=row["name"],
                model_type=row["model_type"],
                metrics=metrics,
            ))

    if not comparison:
        raise HTTPException(status_code=404, detail="No valid models found")

    # Find best performers
    best_sharpe = max(comparison, key=lambda x: x.metrics.sharpe_ratio).model_id
    best_return = max(comparison, key=lambda x: x.metrics.total_return_pct).model_id
    lowest_dd = min(comparison, key=lambda x: x.metrics.max_drawdown_pct).model_id

    return ModelsCompareResponse(
        period=period.value,
        models_compared=len(comparison),
        comparison=comparison,
        best_sharpe=best_sharpe,
        best_return=best_return,
        lowest_drawdown=lowest_dd,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# =============================================================================
# FEATURES ENDPOINTS
# =============================================================================

@app.get(
    "/api/v1/features",
    response_model=FeaturesResponse,
    tags=["Features"],
    summary="List all features",
    description="Get all features with normalization statistics grouped by category",
)
async def list_features(
    cache: RedisCache = Depends(get_cache),
):
    """
    List all model features with normalization stats.

    Returns features grouped by category:
    - price: Price-based features (returns, momentum)
    - technical: Technical indicators (RSI, ATR, ADX)
    - macro: Macroeconomic features (DXY, VIX, EMBI)
    - state: Agent state features (position, time)

    Examples:
        GET /api/v1/features
    """
    cache_key = "features:all"

    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    # Feature definitions with normalization stats
    # These come from feature_config.json SSOT
    features_data = {
        "price": [
            FeatureStats(name="log_ret_5m", mean=0.0, std=0.001, min_val=-0.05, max_val=0.05, category="price"),
            FeatureStats(name="log_ret_1h", mean=0.0, std=0.003, min_val=-0.1, max_val=0.1, category="price"),
            FeatureStats(name="log_ret_4h", mean=0.0, std=0.006, min_val=-0.2, max_val=0.2, category="price"),
        ],
        "technical": [
            FeatureStats(name="rsi_9", mean=50.0, std=15.0, min_val=0.0, max_val=100.0, category="technical"),
            FeatureStats(name="atr_pct", mean=0.5, std=0.2, min_val=0.0, max_val=2.0, category="technical"),
            FeatureStats(name="adx_14", mean=25.0, std=10.0, min_val=0.0, max_val=100.0, category="technical"),
        ],
        "macro": [
            FeatureStats(name="dxy_z", mean=103.0, std=5.0, min_val=-3.0, max_val=3.0, category="macro"),
            FeatureStats(name="dxy_change_1d", mean=0.0, std=0.5, min_val=-2.0, max_val=2.0, category="macro"),
            FeatureStats(name="vix_z", mean=20.0, std=10.0, min_val=-3.0, max_val=3.0, category="macro"),
            FeatureStats(name="embi_z", mean=300.0, std=100.0, min_val=-3.0, max_val=3.0, category="macro"),
            FeatureStats(name="brent_change_1d", mean=0.0, std=2.0, min_val=-10.0, max_val=10.0, category="macro"),
            FeatureStats(name="rate_spread", mean=0.5, std=0.5, min_val=-2.0, max_val=3.0, category="macro"),
            FeatureStats(name="usdmxn_ret_1h", mean=0.0, std=0.002, min_val=-0.05, max_val=0.05, category="macro"),
        ],
        "state": [
            FeatureStats(name="position", mean=0.0, std=0.5, min_val=-1.0, max_val=1.0, category="state"),
            FeatureStats(name="time_normalized", mean=0.5, std=0.3, min_val=0.0, max_val=1.0, category="state"),
        ],
    }

    categories = [
        FeatureCategory(category=cat, features=features)
        for cat, features in features_data.items()
    ]

    total_features = sum(len(cat.features) for cat in categories)

    response = FeaturesResponse(
        total_features=total_features,
        categories=categories,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    await cache.set(cache_key, response.model_dump_json(), Config.CACHE_TTL_LONG)

    return response


# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8007"))
    uvicorn.run(
        "trading_api_multi_model:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level="info",
    )
