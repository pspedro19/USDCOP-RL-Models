"""
BI API - Business Intelligence Data Warehouse API
==================================================
FastAPI service for querying the Kimball Data Warehouse.

Port: 8007
Purpose: Expose DWH data to dashboards, BI tools, and analytics applications.

Endpoints:
- /health - Health check
- /api/bi/bars - OHLCV bars with dimension enrichment
- /api/bi/l0/acquisition-runs - L0 pipeline metrics
- /api/bi/l1/quality-daily - L1 daily quality metrics
- /api/bi/l2/indicators - Technical indicators
- /api/bi/l2/winsorization - Winsorization stats
- /api/bi/l2/hod-baseline - Hour-of-day baselines
- /api/bi/l3/forward-ic - Forward information coefficient
- /api/bi/l3/leakage - Feature leakage tests
- /api/bi/l3/correlation - Feature correlation matrix
- /api/bi/l4/obs-stats - RL observation statistics
- /api/bi/l4/cost-model - Cost model statistics
- /api/bi/l4/episodes - Episode metadata
- /api/bi/l5/signals - Trading signals
- /api/bi/l5/inference-latency - Inference latency metrics
- /api/bi/l6/backtest/trades - Backtest trades
- /api/bi/l6/backtest/perf-daily - Daily performance
- /api/bi/l6/backtest/summary - Backtest summary
- /api/bi/dimensions/* - Dimension tables
- /api/bi/health-check - DWH health check
"""

import os
import logging
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# PostgreSQL configuration
POSTGRES_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'usdcop-postgres-timescale'),
    'port': int(os.getenv('POSTGRES_PORT', 5432)),
    'database': os.getenv('POSTGRES_DB', 'usdcop_trading'),
    'user': os.getenv('POSTGRES_USER', 'admin'),
    'password': os.getenv('POSTGRES_PASSWORD', 'admin123')
}

# FastAPI app
app = FastAPI(
    title="BI API - Data Warehouse",
    description="Business Intelligence API for querying Kimball DWH",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# DATABASE CONNECTION
# ==============================================================================

def get_db_connection():
    """Get PostgreSQL connection."""
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")

# ==============================================================================
# PYDANTIC MODELS
# ==============================================================================

class ResponseEnvelope(BaseModel):
    """Standard response envelope."""
    ok: bool = True
    data: Any = None
    meta: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, str]] = None

class OHLCVBar(BaseModel):
    """OHLCV bar model."""
    ts_utc: datetime
    ts_cot: Optional[datetime] = None
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    source: Optional[str] = None

class AcquisitionRun(BaseModel):
    """L0 acquisition run model."""
    run_id: str
    symbol: str
    source: str
    execution_date: date
    fetch_mode: str
    date_range_start: datetime
    date_range_end: datetime
    rows_fetched: int
    rows_inserted: int
    stale_rate_pct: Optional[float] = None
    coverage_pct: Optional[float] = None
    duration_sec: int
    quality_passed: bool

class QualityMetrics(BaseModel):
    """L1 quality metrics model."""
    date_cot: date
    symbol: str
    total_episodes: int
    accepted_episodes: int
    rejected_episodes: int
    grid_300s_ok: bool
    repeated_ohlc_rate_pct: Optional[float] = None
    coverage_pct: Optional[float] = None
    status_passed: bool

class IndicatorValue(BaseModel):
    """L2 indicator value model."""
    ts_utc: datetime
    symbol: str
    indicator_name: str
    indicator_family: str
    value: float
    signal: Optional[str] = None

class BacktestTrade(BaseModel):
    """L6 backtest trade model."""
    trade_id: str
    run_id: str
    side: str
    entry_time: datetime
    exit_time: datetime
    duration_bars: int
    entry_px: float
    exit_px: float
    pnl: float
    pnl_pct: Optional[float] = None
    pnl_bps: Optional[float] = None

class BacktestSummary(BaseModel):
    """L6 backtest summary model."""
    run_id: str
    split: str
    total_return: Optional[float] = None
    cagr: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    total_trades: Optional[int] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None

# ==============================================================================
# HEALTH CHECK
# ==============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        conn.close()

        return {
            "status": "healthy",
            "service": "bi-api",
            "version": "1.0.0",
            "database": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "bi-api",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/bi/health-check")
async def dwh_health_check():
    """DWH health check with table counts."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM dw.health_check")
            results = cur.fetchall()
        conn.close()

        return ResponseEnvelope(
            ok=True,
            data=[dict(row) for row in results]
        )
    except Exception as e:
        logger.error(f"DWH health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# L0 ENDPOINTS - Raw Data
# ==============================================================================

@app.get("/api/bi/bars", response_model=ResponseEnvelope)
async def get_bars(
    symbol: str = Query("USD/COP", description="Symbol code"),
    from_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    to_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(200, ge=1, le=1000, description="Page size")
):
    """
    Get OHLCV bars with dimension enrichment.

    Returns bars with symbol and time dimension data.
    """
    try:
        # Parse dates
        if not from_date:
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')

        offset = (page - 1) * page_size

        conn = get_db_connection()
        with conn.cursor() as cur:
            # Count total
            cur.execute("""
                SELECT COUNT(*)
                FROM dw.fact_bar_5m fb
                JOIN dw.dim_symbol ds ON fb.symbol_id = ds.symbol_id
                JOIN dw.dim_time_5m dt ON fb.time_id = dt.time_id
                WHERE ds.symbol_code = %s
                AND dt.date_cot >= %s AND dt.date_cot <= %s
            """, (symbol, from_date, to_date))
            total = cur.fetchone()['count']

            # Get data
            cur.execute("""
                SELECT
                    fb.ts_utc,
                    dt.ts_cot,
                    ds.symbol_code as symbol,
                    fb.open, fb.high, fb.low, fb.close, fb.volume,
                    src.source_name as source
                FROM dw.fact_bar_5m fb
                JOIN dw.dim_symbol ds ON fb.symbol_id = ds.symbol_id
                JOIN dw.dim_time_5m dt ON fb.time_id = dt.time_id
                LEFT JOIN dw.dim_source src ON fb.source_id = src.source_id
                WHERE ds.symbol_code = %s
                AND dt.date_cot >= %s AND dt.date_cot <= %s
                ORDER BY fb.ts_utc DESC
                LIMIT %s OFFSET %s
            """, (symbol, from_date, to_date, page_size, offset))

            rows = cur.fetchall()

        conn.close()

        return ResponseEnvelope(
            ok=True,
            data=[dict(row) for row in rows],
            meta={
                "count": total,
                "page": page,
                "page_size": page_size,
                "total_pages": (total + page_size - 1) // page_size
            }
        )

    except Exception as e:
        logger.error(f"Error fetching bars: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/bi/l0/acquisition-runs", response_model=ResponseEnvelope)
async def get_acquisition_runs(
    date: Optional[str] = Query(None, description="Date (YYYY-MM-DD)"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200)
):
    """Get L0 acquisition runs."""
    try:
        offset = (page - 1) * page_size

        conn = get_db_connection()
        with conn.cursor() as cur:
            where_clause = ""
            params = []

            if date:
                where_clause = "WHERE execution_date = %s"
                params.append(date)

            # Count
            cur.execute(f"""
                SELECT COUNT(*)
                FROM dw.fact_l0_acquisition fa
                {where_clause}
            """, params)
            total = cur.fetchone()['count']

            # Data
            params_data = params + [page_size, offset]
            cur.execute(f"""
                SELECT
                    fa.run_id,
                    ds.symbol_code as symbol,
                    src.source_name as source,
                    fa.execution_date,
                    fa.fetch_mode,
                    fa.date_range_start,
                    fa.date_range_end,
                    fa.rows_fetched,
                    fa.rows_inserted,
                    fa.stale_rate_pct,
                    fa.coverage_pct,
                    fa.duration_sec,
                    fa.quality_passed
                FROM dw.fact_l0_acquisition fa
                JOIN dw.dim_symbol ds ON fa.symbol_id = ds.symbol_id
                JOIN dw.dim_source src ON fa.source_id = src.source_id
                {where_clause}
                ORDER BY fa.execution_date DESC, fa.created_at DESC
                LIMIT %s OFFSET %s
            """, params_data)

            rows = cur.fetchall()

        conn.close()

        return ResponseEnvelope(
            ok=True,
            data=[dict(row) for row in rows],
            meta={"count": total, "page": page, "page_size": page_size}
        )

    except Exception as e:
        logger.error(f"Error fetching acquisition runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# L1 ENDPOINTS - Quality
# ==============================================================================

@app.get("/api/bi/l1/quality-daily", response_model=ResponseEnvelope)
async def get_quality_daily(
    date: Optional[str] = Query(None, description="Date (YYYY-MM-DD)"),
    symbol: str = Query("USD/COP"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200)
):
    """Get L1 daily quality metrics."""
    try:
        offset = (page - 1) * page_size

        conn = get_db_connection()
        with conn.cursor() as cur:
            where_clauses = ["ds.symbol_code = %s"]
            params = [symbol]

            if date:
                where_clauses.append("fq.date_cot = %s")
                params.append(date)

            where_clause = " AND ".join(where_clauses)

            # Count
            cur.execute(f"""
                SELECT COUNT(*)
                FROM dw.fact_l1_quality fq
                JOIN dw.dim_symbol ds ON fq.symbol_id = ds.symbol_id
                WHERE {where_clause}
            """, params)
            total = cur.fetchone()['count']

            # Data
            params_data = params + [page_size, offset]
            cur.execute(f"""
                SELECT
                    fq.date_cot,
                    ds.symbol_code as symbol,
                    fq.total_episodes,
                    fq.accepted_episodes,
                    fq.rejected_episodes,
                    fq.grid_300s_ok,
                    fq.repeated_ohlc_rate_pct,
                    fq.coverage_pct,
                    fq.status_passed
                FROM dw.fact_l1_quality fq
                JOIN dw.dim_symbol ds ON fq.symbol_id = ds.symbol_id
                WHERE {where_clause}
                ORDER BY fq.date_cot DESC
                LIMIT %s OFFSET %s
            """, params_data)

            rows = cur.fetchall()

        conn.close()

        return ResponseEnvelope(
            ok=True,
            data=[dict(row) for row in rows],
            meta={"count": total, "page": page, "page_size": page_size}
        )

    except Exception as e:
        logger.error(f"Error fetching quality metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# L2 ENDPOINTS - Indicators
# ==============================================================================

@app.get("/api/bi/l2/indicators", response_model=ResponseEnvelope)
async def get_indicators(
    indicator: str = Query(..., description="Indicator name (e.g., RSI, MACD)"),
    symbol: str = Query("USD/COP"),
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(200, ge=1, le=1000)
):
    """Get technical indicator values."""
    try:
        if not from_date:
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')

        offset = (page - 1) * page_size

        conn = get_db_connection()
        with conn.cursor() as cur:
            # Count
            cur.execute("""
                SELECT COUNT(*)
                FROM dw.fact_indicator_5m fi
                JOIN dw.dim_symbol ds ON fi.symbol_id = ds.symbol_id
                JOIN dw.dim_indicator di ON fi.indicator_id = di.indicator_id
                JOIN dw.dim_time_5m dt ON fi.time_id = dt.time_id
                WHERE ds.symbol_code = %s
                AND di.indicator_name = %s
                AND dt.date_cot >= %s AND dt.date_cot <= %s
            """, (symbol, indicator, from_date, to_date))
            total = cur.fetchone()['count']

            # Data
            cur.execute("""
                SELECT
                    fi.ts_utc,
                    ds.symbol_code as symbol,
                    di.indicator_name,
                    di.indicator_family,
                    fi.indicator_value as value,
                    fi.signal
                FROM dw.fact_indicator_5m fi
                JOIN dw.dim_symbol ds ON fi.symbol_id = ds.symbol_id
                JOIN dw.dim_indicator di ON fi.indicator_id = di.indicator_id
                JOIN dw.dim_time_5m dt ON fi.time_id = dt.time_id
                WHERE ds.symbol_code = %s
                AND di.indicator_name = %s
                AND dt.date_cot >= %s AND dt.date_cot <= %s
                ORDER BY fi.ts_utc DESC
                LIMIT %s OFFSET %s
            """, (symbol, indicator, from_date, to_date, page_size, offset))

            rows = cur.fetchall()

        conn.close()

        return ResponseEnvelope(
            ok=True,
            data=[dict(row) for row in rows],
            meta={"count": total, "page": page, "page_size": page_size}
        )

    except Exception as e:
        logger.error(f"Error fetching indicators: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# L6 ENDPOINTS - Backtesting
# ==============================================================================

@app.get("/api/bi/l6/backtest/trades", response_model=ResponseEnvelope)
async def get_backtest_trades(
    run_id: str = Query(..., description="Backtest run ID"),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=500)
):
    """Get backtest trades."""
    try:
        offset = (page - 1) * page_size

        conn = get_db_connection()
        with conn.cursor() as cur:
            # Count
            cur.execute("""
                SELECT COUNT(*)
                FROM dw.fact_trade ft
                JOIN dw.dim_backtest_run dr ON ft.run_sk = dr.run_sk
                WHERE dr.run_id = %s
            """, (run_id,))
            total = cur.fetchone()['count']

            # Data
            cur.execute("""
                SELECT
                    ft.trade_id,
                    dr.run_id,
                    ft.side,
                    ft.entry_time,
                    ft.exit_time,
                    ft.duration_bars,
                    ft.entry_px,
                    ft.exit_px,
                    ft.pnl,
                    ft.pnl_pct,
                    ft.pnl_bps
                FROM dw.fact_trade ft
                JOIN dw.dim_backtest_run dr ON ft.run_sk = dr.run_sk
                WHERE dr.run_id = %s
                ORDER BY ft.entry_time
                LIMIT %s OFFSET %s
            """, (run_id, page_size, offset))

            rows = cur.fetchall()

        conn.close()

        return ResponseEnvelope(
            ok=True,
            data=[dict(row) for row in rows],
            meta={"count": total, "page": page, "page_size": page_size}
        )

    except Exception as e:
        logger.error(f"Error fetching trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/bi/l6/backtest/summary", response_model=ResponseEnvelope)
async def get_backtest_summary(
    run_id: Optional[str] = Query(None, description="Backtest run ID"),
    split: Optional[str] = Query(None, description="Split (train/val/test)")
):
    """Get backtest summary statistics."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            where_clauses = []
            params = []

            if run_id:
                where_clauses.append("dr.run_id = %s")
                params.append(run_id)

            if split:
                where_clauses.append("fps.split = %s")
                params.append(split)

            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

            cur.execute(f"""
                SELECT
                    dr.run_id,
                    fps.split,
                    fps.total_return,
                    fps.cagr,
                    fps.sharpe_ratio,
                    fps.sortino_ratio,
                    fps.max_drawdown,
                    fps.total_trades,
                    fps.win_rate,
                    fps.profit_factor
                FROM dw.fact_perf_summary fps
                JOIN dw.dim_backtest_run dr ON fps.run_sk = dr.run_sk
                WHERE {where_clause}
                ORDER BY dr.execution_date DESC
            """, params)

            rows = cur.fetchall()

        conn.close()

        return ResponseEnvelope(
            ok=True,
            data=[dict(row) for row in rows]
        )

    except Exception as e:
        logger.error(f"Error fetching backtest summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# DIMENSION ENDPOINTS
# ==============================================================================

@app.get("/api/bi/dimensions/symbols")
async def get_symbols():
    """Get all symbols."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    symbol_id,
                    symbol_code,
                    base_currency,
                    quote_currency,
                    symbol_type,
                    exchange,
                    is_active
                FROM dw.dim_symbol
                WHERE is_active = TRUE
                ORDER BY symbol_code
            """)
            rows = cur.fetchall()
        conn.close()

        return ResponseEnvelope(ok=True, data=[dict(row) for row in rows])

    except Exception as e:
        logger.error(f"Error fetching symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/bi/dimensions/indicators")
async def get_indicators_list():
    """Get all indicators."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    indicator_id,
                    indicator_name,
                    indicator_family,
                    params,
                    interpretation
                FROM dw.dim_indicator
                ORDER BY indicator_family, indicator_name
            """)
            rows = cur.fetchall()
        conn.close()

        return ResponseEnvelope(ok=True, data=[dict(row) for row in rows])

    except Exception as e:
        logger.error(f"Error fetching indicators: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    port = int(os.getenv("BI_API_PORT", 8007))
    uvicorn.run(
        "bi_api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
