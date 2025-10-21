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
