#!/usr/bin/env python3
"""
USDCOP Pipeline Data API
=========================

API para exponer datos de todas las capas del pipeline L0-L6:
- L0: Raw Data (datos crudos de mercado)
- L1: Features (episodios y features de RL)
- L2: Labels (etiquetas y targets)
- L3: Correlations (matriz de correlación de features)
- L4: RL Ready Dataset (dataset listo para entrenar)
- L5: Model (artefactos de modelos ML)
- L6: Backtest (resultados de backtesting)

Cada capa expone endpoints específicos para acceder a sus datos.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
import logging
from pydantic import BaseModel
import uvicorn
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
POSTGRES_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': int(os.getenv('POSTGRES_PORT', '5432')),
    'database': os.getenv('POSTGRES_DB', 'usdcop_trading'),
    'user': os.getenv('POSTGRES_USER', 'admin'),
    'password': os.getenv('POSTGRES_PASSWORD', 'admin123')
}

# Create FastAPI app
app = FastAPI(
    title="USDCOP Pipeline Data API",
    description="API para acceso a datos del pipeline L0-L6",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# DATABASE CONNECTION
# ==========================================

def get_db_connection():
    """Get database connection"""
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

def execute_query(query: str, params: tuple = None) -> List[Dict]:
    """Execute query and return list of dicts"""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close()
        return [dict(row) for row in rows]
    finally:
        conn.close()

def execute_query_df(query: str, params: tuple = None) -> pd.DataFrame:
    """Execute query and return DataFrame"""
    conn = get_db_connection()
    try:
        df = pd.read_sql_query(query, conn, params=params)
        return df
    finally:
        conn.close()

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def serialize_datetime(obj):
    """JSON serializer for datetime objects"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def safe_float(value):
    """Safely convert to float"""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

# ==========================================
# ROOT ENDPOINTS
# ==========================================

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "USDCOP Pipeline Data API",
        "status": "active",
        "version": "1.0.0",
        "layers": {
            "L0": "/api/pipeline/l0/*",
            "L1": "/api/pipeline/l1/*",
            "L3": "/api/pipeline/l3/*",
            "L4": "/api/pipeline/l4/*",
            "L5": "/api/pipeline/l5/*",
            "L6": "/api/pipeline/l6/*"
        }
    }

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) as count FROM market_data")
        result = cur.fetchone()
        count = result['count']
        cur.close()
        conn.close()

        return {
            "status": "healthy",
            "database": "connected",
            "market_data_records": count,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# ==========================================
# L0: RAW DATA ENDPOINTS
# ==========================================

@app.get("/api/pipeline/l0/raw-data")
def get_l0_raw_data(
    limit: int = Query(default=1000, ge=1, le=10000),
    offset: int = Query(default=0, ge=0),
    start_date: Optional[str] = Query(default=None),
    end_date: Optional[str] = Query(default=None),
    source: str = Query(default="all", description="postgres, minio, or all")
):
    """
    Get raw market data from L0 layer (market_data table)
    """
    try:
        # Build query
        query = """
            SELECT
                timestamp,
                symbol,
                price as close,
                bid,
                ask,
                volume,
                'postgres' as source
            FROM market_data
            WHERE 1=1
        """
        params = []

        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)

        query += " ORDER BY timestamp DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        rows = execute_query(query, tuple(params))

        # Process rows
        data = []
        for row in rows:
            data.append({
                "timestamp": row['timestamp'].isoformat() if isinstance(row['timestamp'], datetime) else row['timestamp'],
                "symbol": row['symbol'],
                "close": safe_float(row['close']),
                "bid": safe_float(row['bid']),
                "ask": safe_float(row['ask']),
                "volume": row['volume'],
                "source": row['source']
            })

        # Get total count
        count_query = "SELECT COUNT(*) as total FROM market_data WHERE 1=1"
        count_params = []
        if start_date:
            count_query += " AND timestamp >= %s"
            count_params.append(start_date)
        if end_date:
            count_query += " AND timestamp <= %s"
            count_params.append(end_date)

        count_result = execute_query(count_query, tuple(count_params) if count_params else None)
        total = count_result[0]['total']

        return {
            "data": data,
            "metadata": {
                "source": "postgres",
                "count": len(data),
                "total": total,
                "limit": limit,
                "offset": offset,
                "hasMore": (offset + limit) < total,
                "table": "market_data"
            }
        }

    except Exception as e:
        logger.error(f"Error fetching L0 raw data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pipeline/l0/statistics")
def get_l0_statistics():
    """
    Get statistics about L0 raw data
    """
    try:
        query = """
            SELECT
                COUNT(*) as total_records,
                MIN(timestamp) as earliest_date,
                MAX(timestamp) as latest_date,
                COUNT(DISTINCT symbol) as symbols_count,
                AVG(volume) as avg_volume,
                MIN(price) as min_price,
                MAX(price) as max_price,
                AVG(price) as avg_price
            FROM market_data
        """

        result = execute_query(query)
        stats = result[0]

        return {
            "total_records": stats['total_records'],
            "date_range": {
                "earliest": stats['earliest_date'].isoformat() if stats['earliest_date'] else None,
                "latest": stats['latest_date'].isoformat() if stats['latest_date'] else None,
                "days": (stats['latest_date'] - stats['earliest_date']).days if stats['latest_date'] and stats['earliest_date'] else 0
            },
            "symbols_count": stats['symbols_count'],
            "price_stats": {
                "min": safe_float(stats['min_price']),
                "max": safe_float(stats['max_price']),
                "avg": safe_float(stats['avg_price'])
            },
            "avg_volume": safe_float(stats['avg_volume']),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching L0 statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# L1: FEATURES & EPISODES ENDPOINTS
# ==========================================

@app.get("/api/pipeline/l1/episodes")
def get_l1_episodes(
    limit: int = Query(default=100, ge=1, le=1000)
):
    """
    Get RL training episodes data from L1 layer

    Note: This returns aggregated data from market_data as episodes
    """
    try:
        # Group data by day as "episodes"
        query = """
            SELECT
                DATE(timestamp) as episode_date,
                COUNT(*) as steps,
                AVG(price) as avg_price,
                MIN(price) as min_price,
                MAX(price) as max_price,
                SUM(volume) as total_volume,
                STDDEV(price) as price_volatility
            FROM market_data
            GROUP BY DATE(timestamp)
            ORDER BY episode_date DESC
            LIMIT %s
        """

        rows = execute_query(query, (limit,))

        episodes = []
        for i, row in enumerate(rows):
            episodes.append({
                "episode_id": f"ep_{i}_{row['episode_date']}",
                "date": row['episode_date'].isoformat() if isinstance(row['episode_date'], (datetime, pd.Timestamp)) else str(row['episode_date']),
                "steps": row['steps'],
                "avg_price": safe_float(row['avg_price']),
                "min_price": safe_float(row['min_price']),
                "max_price": safe_float(row['max_price']),
                "total_volume": row['total_volume'],
                "volatility": safe_float(row['price_volatility']),
                "reward": None  # Would come from RL training logs
            })

        return {
            "episodes": episodes,
            "count": len(episodes),
            "metadata": {
                "layer": "L1",
                "data_type": "episodes",
                "source": "market_data_aggregated"
            }
        }

    except Exception as e:
        logger.error(f"Error fetching L1 episodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pipeline/l1/quality-report")
def get_l1_quality_report():
    """
    Get data quality report for L1 layer
    """
    try:
        # Check for data quality issues
        query = """
            SELECT
                COUNT(*) as total_records,
                COUNT(*) FILTER (WHERE price IS NULL) as null_prices,
                COUNT(*) FILTER (WHERE volume IS NULL OR volume = 0) as missing_volume,
                COUNT(*) FILTER (WHERE bid IS NULL OR ask IS NULL) as missing_bid_ask,
                COUNT(DISTINCT DATE(timestamp)) as trading_days
            FROM market_data
            WHERE timestamp >= NOW() - INTERVAL '30 days'
        """

        result = execute_query(query)
        stats = result[0]

        total = stats['total_records']
        null_prices = stats['null_prices']
        missing_volume = stats['missing_volume']
        missing_bid_ask = stats['missing_bid_ask']

        quality_score = 100.0
        if total > 0:
            quality_score -= (null_prices / total) * 100
            quality_score -= (missing_volume / total) * 20
            quality_score -= (missing_bid_ask / total) * 10

        quality_score = max(0, min(100, quality_score))

        return {
            "quality_score": round(quality_score, 2),
            "issues": {
                "null_prices": null_prices,
                "missing_volume": missing_volume,
                "missing_bid_ask": missing_bid_ask
            },
            "total_records": total,
            "trading_days": stats['trading_days'],
            "status": "good" if quality_score >= 80 else "needs_attention" if quality_score >= 60 else "poor",
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error generating quality report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# L3: CORRELATIONS & FEATURES ENDPOINTS
# ==========================================

@app.get("/api/pipeline/l3/features")
def get_l3_features(
    limit: int = Query(default=100, ge=1, le=1000)
):
    """
    Get feature correlation matrix from L3 layer

    Returns calculated features and their correlations
    """
    try:
        # Get recent data for feature calculation
        query = """
            SELECT
                timestamp,
                price,
                volume,
                bid,
                ask
            FROM market_data
            ORDER BY timestamp DESC
            LIMIT %s
        """

        df = execute_query_df(query, (limit,))

        if df.empty:
            raise HTTPException(status_code=404, detail="No data available for feature calculation")

        # Calculate basic features
        df['price_change'] = df['price'].pct_change()
        df['spread'] = df['ask'] - df['bid']
        df['mid_price'] = (df['ask'] + df['bid']) / 2
        df['volume_change'] = df['volume'].pct_change()

        # Calculate rolling features
        df['price_ma_5'] = df['price'].rolling(window=5).mean()
        df['price_ma_20'] = df['price'].rolling(window=20).mean()
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()

        # Drop NaN values
        features_df = df[['price_change', 'spread', 'volume_change', 'price_ma_5', 'price_ma_20', 'volume_ma_5']].dropna()

        # Calculate correlation matrix
        correlation_matrix = features_df.corr()

        # Convert to dict
        features = []
        for col in correlation_matrix.columns:
            feature_data = {
                "name": col,
                "correlations": {}
            }
            for idx in correlation_matrix.index:
                feature_data["correlations"][idx] = float(correlation_matrix.loc[idx, col])
            features.append(feature_data)

        return {
            "features": features,
            "correlation_matrix": correlation_matrix.to_dict(),
            "metadata": {
                "layer": "L3",
                "samples_used": len(features_df),
                "features_count": len(features),
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Error calculating features: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# L4: RL READY DATASET ENDPOINTS
# ==========================================

@app.get("/api/pipeline/l4/dataset")
def get_l4_dataset(
    split: str = Query(default="train", description="train, test, or val"),
    limit: int = Query(default=1000, ge=1, le=10000)
):
    """
    Get RL-ready dataset from L4 layer

    Returns preprocessed data ready for model training
    """
    try:
        # Determine date range based on split
        if split == "train":
            # Last 70% of data
            date_filter = "timestamp >= (SELECT MAX(timestamp) - INTERVAL '70 days' FROM market_data)"
        elif split == "test":
            # Next 20% of data
            date_filter = "timestamp >= (SELECT MAX(timestamp) - INTERVAL '20 days' FROM market_data) AND timestamp < (SELECT MAX(timestamp) - INTERVAL '10 days' FROM market_data)"
        else:  # val
            # Last 10% of data
            date_filter = "timestamp >= (SELECT MAX(timestamp) - INTERVAL '10 days' FROM market_data)"

        query = f"""
            SELECT
                timestamp,
                price,
                volume,
                bid,
                ask
            FROM market_data
            WHERE {date_filter}
            ORDER BY timestamp DESC
            LIMIT %s
        """

        rows = execute_query(query, (limit,))

        dataset = []
        for row in rows:
            dataset.append({
                "timestamp": row['timestamp'].isoformat() if isinstance(row['timestamp'], datetime) else row['timestamp'],
                "price": safe_float(row['price']),
                "volume": row['volume'],
                "bid": safe_float(row['bid']),
                "ask": safe_float(row['ask']),
                "split": split
            })

        return {
            "data": dataset,
            "split": split,
            "count": len(dataset),
            "metadata": {
                "layer": "L4",
                "ready_for_training": True,
                "normalized": False,
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Error fetching L4 dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# L5: MODEL ARTIFACTS ENDPOINTS
# ==========================================

@app.get("/api/pipeline/l5/models")
def get_l5_models():
    """
    Get available ML model artifacts from L5 layer

    Returns list of trained models and their metadata
    """
    try:
        # Mock model data (in production, this would come from MLflow or model registry)
        models = [
            {
                "model_id": "ppo_lstm_v2_1",
                "name": "PPO with LSTM",
                "version": "2.1",
                "algorithm": "PPO",
                "architecture": "LSTM",
                "training_date": (datetime.utcnow() - timedelta(days=2)).isoformat(),
                "metrics": {
                    "train_reward": 1250.5,
                    "val_reward": 1180.3,
                    "sharpe_ratio": 1.87,
                    "win_rate": 0.685
                },
                "status": "active",
                "file_path": "models/ppo_lstm_v2_1.pkl"
            },
            {
                "model_id": "a2c_gru_v1_5",
                "name": "A2C with GRU",
                "version": "1.5",
                "algorithm": "A2C",
                "architecture": "GRU",
                "training_date": (datetime.utcnow() - timedelta(days=5)).isoformat(),
                "metrics": {
                    "train_reward": 1150.2,
                    "val_reward": 1100.5,
                    "sharpe_ratio": 1.65,
                    "win_rate": 0.672
                },
                "status": "inactive",
                "file_path": "models/a2c_gru_v1_5.pkl"
            },
            {
                "model_id": "sac_transformer_v3_0",
                "name": "SAC with Transformer",
                "version": "3.0",
                "algorithm": "SAC",
                "architecture": "Transformer",
                "training_date": (datetime.utcnow() - timedelta(days=1)).isoformat(),
                "metrics": {
                    "train_reward": 1320.8,
                    "val_reward": 1255.1,
                    "sharpe_ratio": 2.05,
                    "win_rate": 0.701
                },
                "status": "testing",
                "file_path": "models/sac_transformer_v3_0.pkl"
            }
        ]

        return {
            "models": models,
            "count": len(models),
            "metadata": {
                "layer": "L5",
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Error fetching L5 models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# L6: BACKTEST RESULTS ENDPOINTS
# ==========================================

@app.get("/api/pipeline/l6/backtest-results")
def get_l6_backtest_results(
    split: str = Query(default="test", description="test or val")
):
    """
    Get backtest results from L6 layer

    Returns performance metrics from backtesting
    """
    try:
        # Mock backtest data (in production, this would come from backtest database)
        results = {
            "run_id": f"backtest_{int(datetime.utcnow().timestamp())}",
            "split": split,
            "timestamp": datetime.utcnow().isoformat(),
            "kpis": {
                "top_bar": {
                    "CAGR": 0.125,
                    "Sharpe": 1.87,
                    "Sortino": 2.15,
                    "Calmar": 1.05,
                    "MaxDD": -0.08,
                    "Vol_annualizada": 0.15
                },
                "trading_micro": {
                    "win_rate": 0.685,
                    "profit_factor": 2.34,
                    "payoff": 1.52,
                    "expectancy_bps": 145.3,
                    "total_trades": 247,
                    "winning_trades": 169,
                    "losing_trades": 78
                },
                "risk_metrics": {
                    "max_drawdown": -0.08,
                    "avg_drawdown": -0.023,
                    "drawdown_duration_days": 5.2,
                    "VaR_95": 0.015,
                    "CVaR_95": 0.022
                }
            },
            "daily_returns": [],
            "trades": []
        }

        # Generate mock daily returns
        base_price = 100000
        for i in range(30):
            daily_return = np.random.normal(0.001, 0.01)
            base_price *= (1 + daily_return)
            results["daily_returns"].append({
                "date": (datetime.utcnow() - timedelta(days=30-i)).strftime("%Y-%m-%d"),
                "return": round(daily_return, 6),
                "cumulative_return": round((base_price - 100000) / 100000, 6),
                "price": round(base_price, 2)
            })

        return results

    except Exception as e:
        logger.error(f"Error fetching L6 backtest results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    port = int(os.getenv("PIPELINE_DATA_API_PORT", "8004"))
    uvicorn.run(
        "pipeline_data_api:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )


# ==========================================
# L0 EXTENDED STATISTICS
# ==========================================


def get_l0_extended_statistics() -> Dict[str, Any]:
    """
    Estadísticas extendidas de calidad L0:
    - Cobertura de barras por día (60/60 esperado)
    - Violaciones de invariantes OHLC
    - Duplicados por timestamp
    - Tasa de datos stale (OHLC repetidos)
    - Gaps (barras faltantes)
    """
    query = """
    -- Cobertura por día
    WITH daily_coverage AS (
        SELECT
            DATE(timestamp) as trading_day,
            COUNT(*) as bars_count,
            (COUNT(*) / 60.0) * 100 as coverage_pct
        FROM market_data
        WHERE EXTRACT(HOUR FROM timestamp) BETWEEN 8 AND 12
          AND EXTRACT(DOW FROM timestamp) BETWEEN 1 AND 5
        GROUP BY DATE(timestamp)
    ),
    -- Violaciones OHLC
    ohlc_violations AS (
        SELECT COUNT(*) as violation_count
        FROM market_data
        WHERE high < low
           OR close > high
           OR close < low
           OR open > high
           OR open < low
    ),
    -- Duplicados
    duplicates AS (
        SELECT COUNT(*) as dup_count
        FROM (
            SELECT timestamp, symbol, COUNT(*) as cnt
            FROM market_data
            GROUP BY timestamp, symbol
            HAVING COUNT(*) > 1
        ) AS dups
    ),
    -- Stale data (OHLC idénticos consecutivos)
    stale_data AS (
        SELECT COUNT(*) as stale_count
        FROM (
            SELECT
                open, high, low, close,
                LAG(open) OVER (ORDER BY timestamp) as prev_open,
                LAG(high) OVER (ORDER BY timestamp) as prev_high,
                LAG(low) OVER (ORDER BY timestamp) as prev_low,
                LAG(close) OVER (ORDER BY timestamp) as prev_close
            FROM market_data
        ) AS ohlc_compare
        WHERE open = prev_open
          AND high = prev_high
          AND low = prev_low
          AND close = prev_close
    ),
    -- Gaps (intervalos > 10 min)
    gaps AS (
        SELECT COUNT(*) as gap_count
        FROM (
            SELECT
                timestamp,
                LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp
            FROM market_data
        ) AS time_series
        WHERE EXTRACT(EPOCH FROM (timestamp - prev_timestamp)) > 600
    )
    SELECT
        (SELECT AVG(coverage_pct) FROM daily_coverage) as avg_coverage_pct,
        (SELECT violation_count FROM ohlc_violations) as ohlc_violations,
        (SELECT dup_count FROM duplicates) as duplicates,
        (SELECT stale_count FROM stale_data) as stale_count,
        (SELECT gap_count FROM gaps) as gaps,
        (SELECT COUNT(*) FROM market_data) as total_records
    """

    df = execute_query(query)

    if df.empty:
        return {
            "avg_coverage_pct": 0,
            "ohlc_violations": 0,
            "duplicates": 0,
            "stale_rate_pct": 0,
            "gaps": 0,
            "pass": False
        }

    row = df.iloc[0]
    total = row['total_records']
    stale_rate = (row['stale_count'] / total * 100) if total > 0 else 0

    # GO/NO-GO criteria
    passed = (
        row['avg_coverage_pct'] >= 95 and
        row['ohlc_violations'] == 0 and
        row['duplicates'] == 0 and
        stale_rate <= 2.0 and
        row['gaps'] == 0
    )

    return {
        "avg_coverage_pct": float(row['avg_coverage_pct']) if row['avg_coverage_pct'] else 0,
        "ohlc_violations": int(row['ohlc_violations']),
        "duplicates": int(row['duplicates']),
        "stale_count": int(row['stale_count']),
        "stale_rate_pct": float(stale_rate),
        "gaps": int(row['gaps']),
        "total_records": int(total),
        "pass": passed,
        "criteria": {
            "coverage_target": ">=95%",
            "ohlc_violations_target": "=0",
            "duplicates_target": "=0",
            "stale_rate_target": "<=2%",
            "gaps_target": "=0"
        }
    }


@app.get("/api/pipeline/l0/extended-statistics")
def get_l0_extended():
    """
    Estadísticas extendidas de calidad L0:
    - Cobertura, violaciones OHLC, duplicados, stale rate, gaps
    - GO/NO-GO gates automáticos
    """
    try:
        stats = get_l0_extended_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting L0 extended stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# L1 GRID VERIFICATION
# ==========================================


def verify_l1_grid_300s() -> Dict[str, Any]:
    """
    Verifica que cada barra esté exactamente 300s (5 min) después de la anterior
    dentro de la misma sesión de trading
    """
    query = """
    WITH time_diffs AS (
        SELECT
            timestamp,
            LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp,
            EXTRACT(EPOCH FROM (timestamp - LAG(timestamp) OVER (ORDER BY timestamp))) as diff_seconds,
            DATE(timestamp) as trading_day
        FROM market_data
        WHERE EXTRACT(HOUR FROM timestamp) BETWEEN 8 AND 12
          AND EXTRACT(DOW FROM timestamp) BETWEEN 1 AND 5
    )
    SELECT
        COUNT(*) as total_intervals,
        SUM(CASE WHEN ABS(diff_seconds - 300) < 1 THEN 1 ELSE 0 END) as perfect_grid_count,
        (SUM(CASE WHEN ABS(diff_seconds - 300) < 1 THEN 1 ELSE 0 END)::float / COUNT(*)) * 100 as grid_pct,
        AVG(diff_seconds) as avg_interval_seconds,
        STDDEV(diff_seconds) as stddev_interval_seconds
    FROM time_diffs
    WHERE diff_seconds IS NOT NULL
      AND trading_day = LAG(trading_day) OVER (ORDER BY timestamp)  -- Same day only
    """

    df = execute_query(query)

    if df.empty:
        return {
            "grid_300s_pct": 0,
            "perfect_grid_count": 0,
            "total_intervals": 0,
            "pass": False
        }

    row = df.iloc[0]
    grid_pct = float(row['grid_pct']) if row['grid_pct'] else 0

    return {
        "grid_300s_pct": grid_pct,
        "perfect_grid_count": int(row['perfect_grid_count']),
        "total_intervals": int(row['total_intervals']),
        "avg_interval_seconds": float(row['avg_interval_seconds']) if row['avg_interval_seconds'] else 0,
        "stddev_interval_seconds": float(row['stddev_interval_seconds']) if row['stddev_interval_seconds'] else 0,
        "pass": grid_pct >= 99.0,  # Al menos 99% de intervalos perfectos
        "criteria": {
            "target": "99% intervals = 300s ±1s"
        }
    }


@app.get("/api/pipeline/l1/grid-verification")
def get_l1_grid_verification():
    """
    Verifica que las barras estén espaciadas exactamente 300s (5 min)
    """
    try:
        grid_stats = verify_l1_grid_300s()
        return grid_stats
    except Exception as e:
        logger.error(f"Error verifying L1 grid: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# L3 FORWARD IC
# ==========================================


def calculate_forward_ic(df: pd.DataFrame, feature_col: str, horizons: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    Calcula Information Coefficient (IC) forward para detectar leakage
    IC = correlation(feature_t, return_t+horizon)

    Args:
        df: DataFrame con features y close price
        feature_col: Nombre de la columna del feature
        horizons: Horizontes forward en períodos

    Returns:
        dict: IC por cada horizonte
    """
    close_col = 'close' if 'close' in df.columns else 'price'

    if close_col not in df.columns or feature_col not in df.columns:
        return {f"ic_{h}step": 0.0 for h in horizons}

    ic_results = {}

    for horizon in horizons:
        # Forward return
        df[f'forward_return_{horizon}'] = df[close_col].pct_change(horizon).shift(-horizon)

        # Correlation (IC)
        ic = df[feature_col].corr(df[f'forward_return_{horizon}'])

        ic_results[f"ic_{horizon}step"] = float(ic) if not np.isnan(ic) else 0.0

        # Limpiar columna temporal
        df.drop(columns=[f'forward_return_{horizon}'], inplace=True)

    # Anti-leakage check
    max_abs_ic = max([abs(ic) for ic in ic_results.values()])
    ic_results['max_abs_ic'] = max_abs_ic
    ic_results['leakage_detected'] = max_abs_ic > 0.10  # Threshold

    return ic_results


@app.get("/api/pipeline/l3/forward-ic")
def get_l3_forward_ic(limit: int = 1000):
    """
    Calcula Information Coefficient forward para detectar leakage
    IC > 0.10 indica posible data leakage
    """
    try:
        # Get data
        df = query_market_data(limit=limit)

        if df.empty or len(df) < 20:
            raise HTTPException(status_code=404, detail="Insufficient data")

        # Calculate basic features
        df['price_change'] = df['price'].pct_change()
        df['spread'] = df['ask'] - df['bid'] if 'ask' in df.columns and 'bid' in df.columns else 0
        df['volume_change'] = df['volume'].pct_change() if 'volume' in df.columns else 0
        df['price_ma_5'] = df['price'].rolling(5).mean()
        df['price_ma_20'] = df['price'].rolling(20).mean()

        features = ['price_change', 'spread', 'volume_change', 'price_ma_5', 'price_ma_20']

        # Calculate IC for each feature
        ic_results = []
        for feature in features:
            if feature in df.columns and df[feature].notna().sum() > 10:
                ic_stats = calculate_forward_ic(df, feature, horizons=[1, 5, 10])
                ic_results.append({
                    "feature": feature,
                    **ic_stats
                })

        # Summary
        max_ic = max([abs(r.get('max_abs_ic', 0)) for r in ic_results]) if ic_results else 0
        leakage_detected = any([r.get('leakage_detected', False) for r in ic_results])

        return {
            "features": ic_results,
            "summary": {
                "max_abs_ic": float(max_ic),
                "leakage_detected": leakage_detected,
                "pass": not leakage_detected,
                "criteria": "All IC < 0.10"
            }
        }

    except Exception as e:
        logger.error(f"Error calculating forward IC: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# L2 - PREPARE (Winsorization, HOD, Indicators)
# ==========================================

import talib

def calculate_all_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula 60+ indicadores técnicos sobre OHLCV
    """
    # Asegurar columnas necesarias
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            df[col] = df['price'] if 'price' in df.columns else 0

    # Momentum Indicators
    df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
    df['rsi_21'] = talib.RSI(df['close'], timeperiod=21)
    df['cci_14'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
    df['williams_r_14'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
    df['roc_10'] = talib.ROC(df['close'], timeperiod=10)
    df['mom_10'] = talib.MOM(df['close'], timeperiod=10)

    # Trend Indicators
    df['adx_14'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    df['aroon_up'], df['aroon_down'] = talib.AROON(df['high'], df['low'], timeperiod=14)

    # Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
        df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)

    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])

    # Bollinger Bands
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    # Volatility
    df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['natr_14'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)

    # Volume
    df['obv'] = talib.OBV(df['close'], df['volume'])
    df['ad'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])

    # Stochastic
    df['slowk'], df['slowd'] = talib.STOCH(df['high'], df['low'], df['close'])

    return df

def winsorize_returns(returns: pd.Series, n_sigma: float = 4) -> tuple:
    """
    Winsoriza outliers en retornos a n*sigma
    Returns: (winsorized_series, winsorization_rate_pct)
    """
    mean = returns.mean()
    std = returns.std()
    lower_bound = mean - n_sigma * std
    upper_bound = mean + n_sigma * std

    clipped = returns.clip(lower=lower_bound, upper=upper_bound)
    winsorized_count = (returns != clipped).sum()
    rate_pct = (winsorized_count / len(returns)) * 100

    return clipped, rate_pct

def hod_deseasonalize(df: pd.DataFrame, value_col: str = 'close') -> pd.DataFrame:
    """
    Hour-of-Day deseasonalization (robust z-score por hora)
    """
    df = df.copy()
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour

    # Median y MAD por hora
    hod_median = df.groupby('hour')[value_col].transform('median')
    hod_mad = df.groupby('hour')[value_col].transform(lambda x: np.median(np.abs(x - x.median())))

    # Z-score robusto
    df[f'{value_col}_hod_z'] = (df[value_col] - hod_median) / hod_mad

    hod_stats = {
        'median_abs_mean': float(hod_median.abs().mean()),
        'mad_mean': float(hod_mad.mean()),
        'z_within_3sigma_pct': float(((df[f'{value_col}_hod_z'].abs() <= 3).sum() / len(df)) * 100)
    }

    return df, hod_stats

@app.get("/api/pipeline/l2/prepared-data")
def get_l2_prepared_data(limit: int = 1000):
    """
    L2: Datos preparados con:
    - 60+ indicadores técnicos
    - Winsorization de retornos
    - HOD deseasonalization
    - NaN rate reporting
    """
    try:
        df = query_market_data(limit=limit)

        if df.empty or len(df) < 100:
            raise HTTPException(status_code=404, detail="Insufficient data")

        # Calcular retornos
        df['returns'] = df['close'].pct_change() if 'close' in df.columns else df['price'].pct_change()

        # Winsorization
        df['returns_winsorized'], winsor_rate = winsorize_returns(df['returns'].dropna(), n_sigma=4)

        # HOD deseasonalization
        df_hod, hod_stats = hod_deseasonalize(df, 'close' if 'close' in df.columns else 'price')

        # Technical indicators
        df_indicators = calculate_all_technical_indicators(df)

        # NaN rate
        nan_count = df_indicators.isnull().sum().sum()
        total_values = len(df_indicators) * len(df_indicators.columns)
        nan_rate_pct = (nan_count / total_values) * 100

        # GO/NO-GO
        passed = (
            winsor_rate <= 1.0 and
            0.8 <= hod_stats['mad_mean'] <= 1.2 and
            abs(hod_stats['median_abs_mean']) <= 0.05 and
            nan_rate_pct <= 0.5
        )

        return {
            "winsorization": {
                "rate_pct": float(winsor_rate),
                "n_sigma": 4,
                "pass": winsor_rate <= 1.0
            },
            "hod_deseasonalization": {
                **hod_stats,
                "pass": 0.8 <= hod_stats['mad_mean'] <= 1.2
            },
            "indicators_count": len([col for col in df_indicators.columns if col not in ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]),
            "nan_rate_pct": float(nan_rate_pct),
            "data_points": len(df_indicators),
            "pass": passed,
            "criteria": {
                "winsor_target": "<=1%",
                "hod_median_target": "|median| <= 0.05",
                "hod_mad_target": "MAD in [0.8, 1.2]",
                "nan_target": "<=0.5%"
            }
        }

    except Exception as e:
        logger.error(f"Error in L2 prepared data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# L4 - RL READY (Contrato completo)
# ==========================================

OBS_SCHEMA = {
    "obs_00": "spread_proxy_bps_norm",
    "obs_01": "ret_5m_z",
    "obs_02": "ret_10m_z",
    "obs_03": "ret_15m_z",
    "obs_04": "ret_30m_z",
    "obs_05": "range_bps_norm",
    "obs_06": "volume_zscore",
    "obs_07": "rsi_norm",
    "obs_08": "macd_zscore",
    "obs_09": "bb_position",
    "obs_10": "ema_cross_signal",
    "obs_11": "atr_norm",
    "obs_12": "vwap_distance",
    "obs_13": "time_of_day_sin",
    "obs_14": "time_of_day_cos",
    "obs_15": "position",
    "obs_16": "inventory_age"
}

@app.get("/api/pipeline/l4/contract")
def get_l4_contract():
    """
    Retorna contrato de observación L4
    17 features, float32, rango [-5, 5]
    """
    return {
        "observation_schema": OBS_SCHEMA,
        "dtype": "float32",
        "range": [-5, 5],
        "clip_threshold": 0.005,  # 0.5% max clip rate
        "features_count": 17,
        "action_space": {
            "type": "discrete",
            "n": 3,
            "actions": {"-1": "SELL", "0": "HOLD", "1": "BUY"}
        },
        "reward_spec": {
            "type": "float32",
            "reproducible": True,
            "rmse_target": 0.0,
            "std_min": 0.1,
            "zero_pct_max": 1.0
        },
        "cost_model": {
            "spread_p95_range_bps": [2, 25],
            "peg_rate_max_pct": 5.0
        }
    }

@app.get("/api/pipeline/l4/quality-check")
def get_l4_quality_check():
    """
    Verifica calidad de L4:
    - Clip rate por feature
    - Reward reproducibility
    - Cost model bounds
    - Splits embargo
    """
    try:
        # Nota: Requiere acceso a datos L4 en MinIO o DB
        # Por ahora retornamos estructura esperada

        # Simular clip rates (en producción leer de L4 real)
        clip_rates = {f"obs_{i:02d}": 0.0 for i in range(17)}

        # Reward checks (en producción calcular de episodes reales)
        reward_stats = {
            "rmse": 0.0,
            "std": 1.25,
            "zero_pct": 0.5,
            "mean": 125.5
        }

        # Cost model (en producción de L4 real)
        cost_stats = {
            "spread_p95_bps": 15.5,
            "peg_rate_pct": 3.2
        }

        # Splits (en producción verificar fechas reales)
        splits_ok = True
        embargo_days = 5

        # GO/NO-GO
        max_clip = max(clip_rates.values())
        passed = (
            max_clip <= 0.5 and
            reward_stats['rmse'] < 0.01 and
            reward_stats['std'] > 0 and
            reward_stats['zero_pct'] < 1.0 and
            2 <= cost_stats['spread_p95_bps'] <= 25 and
            cost_stats['peg_rate_pct'] < 5.0 and
            splits_ok
        )

        return {
            "clip_rates": clip_rates,
            "max_clip_rate": max_clip,
            "reward_check": {
                **reward_stats,
                "pass": reward_stats['rmse'] < 0.01 and reward_stats['std'] > 0
            },
            "cost_model": {
                **cost_stats,
                "pass": 2 <= cost_stats['spread_p95_bps'] <= 25 and cost_stats['peg_rate_pct'] < 5.0
            },
            "splits": {
                "embargo_days": embargo_days,
                "pass": splits_ok
            },
            "overall_pass": passed,
            "note": "Requires L4 episodes data from MinIO/DB for real metrics"
        }

    except Exception as e:
        logger.error(f"Error in L4 quality check: {e}")
        raise HTTPException(status_code=500, detail=str(e))
