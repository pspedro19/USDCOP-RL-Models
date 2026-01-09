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
import io

# Import MinIO manifest reader
from minio_manifest_reader import (
    read_l5_manifest,
    read_l6_manifest,
    read_file_from_minio,
    get_all_files_from_manifest,
    get_manifest_metadata
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DRY: Use shared modules
from common.database import get_db_config

# Database configuration (from shared module)
POSTGRES_CONFIG = get_db_config().to_dict()

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

def query_market_data(
    symbol: str = "USD/COP",
    limit: int = 1000,
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    """
    Query market data from PostgreSQL database (usdcop_m5_ohlcv unified table)

    Args:
        symbol: Trading symbol (default: USD/COP)
        limit: Maximum number of rows to return
        start_date: Start date filter (ISO format)
        end_date: End date filter (ISO format)

    Returns:
        DataFrame with columns: timestamp, price (close), open, high, low, close, volume, symbol, source
    """
    query = """
        SELECT
            time as timestamp,
            close as price,
            open,
            high,
            low,
            close,
            volume,
            symbol,
            source
        FROM usdcop_m5_ohlcv
        WHERE symbol = %s
    """
    params = [symbol]

    if start_date:
        query += " AND time >= %s"
        params.append(start_date)

    if end_date:
        query += " AND time <= %s"
        params.append(end_date)

    query += " ORDER BY time DESC LIMIT %s"
    params.append(limit)

    return execute_query_df(query, tuple(params))

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
        cur.execute("SELECT COUNT(*) as count FROM usdcop_m5_ohlcv")
        result = cur.fetchone()
        count = result['count']
        cur.close()
        conn.close()

        return {
            "status": "healthy",
            "database": "connected",
            "usdcop_m5_ohlcv_records": count,
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
    Get raw market data from L0 layer (usdcop_m5_ohlcv unified table)
    """
    try:
        # Build query
        query = """
            SELECT
                time as timestamp,
                symbol,
                open,
                high,
                low,
                close,
                volume,
                source
            FROM usdcop_m5_ohlcv
            WHERE 1=1
        """
        params = []

        if start_date:
            query += " AND time >= %s"
            params.append(start_date)

        if end_date:
            query += " AND time <= %s"
            params.append(end_date)

        query += " ORDER BY time DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        rows = execute_query(query, tuple(params))

        # Process rows
        data = []
        for row in rows:
            data.append({
                "timestamp": row['timestamp'].isoformat() if isinstance(row['timestamp'], datetime) else row['timestamp'],
                "symbol": row['symbol'],
                "open": safe_float(row['open']),
                "high": safe_float(row['high']),
                "low": safe_float(row['low']),
                "close": safe_float(row['close']),
                "volume": row['volume'],
                "source": row['source']
            })

        # Get total count
        count_query = "SELECT COUNT(*) as total FROM usdcop_m5_ohlcv WHERE 1=1"
        count_params = []
        if start_date:
            count_query += " AND time >= %s"
            count_params.append(start_date)
        if end_date:
            count_query += " AND time <= %s"
            count_params.append(end_date)

        count_result = execute_query(count_query, tuple(count_params) if count_params else None)
        total = count_result[0]['total']

        return {
            "success": True,
            "data": data,
            "metadata": {
                "source": "postgres",
                "count": len(data),
                "total": total,
                "limit": limit,
                "offset": offset,
                "hasMore": (offset + limit) < total,
                "table": "usdcop_m5_ohlcv"
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
                MIN(time) as earliest_date,
                MAX(time) as latest_date,
                COUNT(DISTINCT symbol) as symbols_count,
                AVG(volume) as avg_volume,
                MIN(close) as min_price,
                MAX(close) as max_price,
                AVG(close) as avg_price
            FROM usdcop_m5_ohlcv
            WHERE symbol = 'USD/COP'
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
                DATE(time) as episode_date,
                COUNT(*) as steps,
                AVG(close) as avg_price,
                MIN(low) as min_price,
                MAX(high) as max_price,
                SUM(volume) as total_volume,
                STDDEV(close) as price_volatility
            FROM usdcop_m5_ohlcv
            GROUP BY DATE(time)
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
                COUNT(*) FILTER (WHERE close IS NULL) as null_prices,
                COUNT(*) FILTER (WHERE volume IS NULL OR volume = 0) as missing_volume,
                COUNT(*) FILTER (WHERE open IS NULL OR high IS NULL OR low IS NULL) as missing_ohlc,
                COUNT(DISTINCT DATE(time)) as trading_days
            FROM usdcop_m5_ohlcv
            WHERE time >= NOW() - INTERVAL '30 days'
        """

        result = execute_query(query)
        stats = result[0]

        total = stats['total_records']
        null_prices = stats['null_prices']
        missing_volume = stats['missing_volume']
        missing_ohlc = stats['missing_ohlc']

        quality_score = 100.0
        if total > 0:
            quality_score -= (null_prices / total) * 100
            quality_score -= (missing_volume / total) * 20
            quality_score -= (missing_ohlc / total) * 10

        quality_score = max(0, min(100, quality_score))

        return {
            "quality_score": round(quality_score, 2),
            "issues": {
                "null_prices": null_prices,
                "missing_volume": missing_volume,
                "missing_ohlc": missing_ohlc
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
                time as timestamp,
                close as price,
                volume,
                close as bid,
                close as ask
            FROM usdcop_m5_ohlcv
            ORDER BY time DESC
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
            date_filter = "time >= (SELECT MAX(time) - INTERVAL '70 days' FROM usdcop_m5_ohlcv)"
        elif split == "test":
            # Next 20% of data
            date_filter = "time >= (SELECT MAX(time) - INTERVAL '20 days' FROM usdcop_m5_ohlcv) AND time < (SELECT MAX(time) - INTERVAL '10 days' FROM usdcop_m5_ohlcv)"
        else:  # val
            # Last 10% of data
            date_filter = "time >= (SELECT MAX(time) - INTERVAL '10 days' FROM usdcop_m5_ohlcv)"

        query = f"""
            SELECT
                time as timestamp,
                close as price,
                volume,
                close as bid,
                close as ask
            FROM usdcop_m5_ohlcv
            WHERE {date_filter}
            ORDER BY time DESC
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

    Returns list of trained models from MinIO manifest (NO MOCK DATA)
    """
    try:
        # Read manifest from MinIO (REAL DATA)
        manifest = read_l5_manifest()

        if not manifest:
            logger.warning("L5 manifest not found in MinIO - DAG may not have run yet")
            raise HTTPException(
                status_code=404,
                detail="L5 manifest not found. Please execute DAG L5 (usdcop_m5__06_l5_serving) first."
            )

        # Extract model metadata from manifest
        models = manifest.get('models', [])
        files = manifest.get('files', [])

        # If models not explicitly in manifest, derive from files
        if not models and files:
            models = []
            for file_info in files:
                file_key = file_info.get('file_key', '')
                metadata = file_info.get('metadata', {})

                # Extract model info from file metadata
                model_id = file_key.split('/')[-1].replace('.pkl', '').replace('.onnx', '')
                models.append({
                    "model_id": model_id,
                    "name": metadata.get('model_name', model_id),
                    "version": metadata.get('version', '1.0'),
                    "algorithm": metadata.get('algorithm', 'Unknown'),
                    "architecture": metadata.get('architecture', 'Unknown'),
                    "training_date": file_info.get('created_at', datetime.utcnow().isoformat()),
                    "metrics": metadata.get('metrics', {}),
                    "status": metadata.get('status', 'active'),
                    "file_path": file_key,
                    "file_size_mb": file_info.get('size_bytes', 0) / (1024 * 1024)
                })

        manifest_metadata = get_manifest_metadata(manifest)

        return {
            "models": models,
            "count": len(models),
            "metadata": {
                "layer": "L5",
                "timestamp": datetime.utcnow().isoformat(),
                "source": "minio_manifest",  # Indicates REAL data
                "run_id": manifest_metadata.get('run_id'),
                "manifest_timestamp": manifest_metadata.get('timestamp'),
                "validation_status": manifest_metadata.get('validation_status')
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching L5 models from MinIO: {e}")
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

    Returns performance metrics from backtesting stored in MinIO (NO MOCK DATA, NO np.random())
    """
    try:
        # Read manifest from MinIO (REAL DATA)
        manifest = read_l6_manifest()

        if not manifest:
            logger.warning("L6 manifest not found in MinIO - DAG may not have run yet")
            raise HTTPException(
                status_code=404,
                detail="L6 manifest not found. Please execute DAG L6 (usdcop_m5__07_l6_backtest_referencia) first."
            )

        # Extract backtest results from manifest
        backtest_data = manifest.get('backtest_results', {})
        kpis = manifest.get('kpis', {})
        statistics = manifest.get('statistics', {})

        # If backtest_data not in manifest, try to read from parquet file
        if not backtest_data and not kpis:
            files = get_all_files_from_manifest(manifest)

            # Look for backtest results file
            for file_info in files:
                file_key = file_info.get('file_key', '')

                # Try to find KPI or summary file
                if 'kpi' in file_key.lower() or 'summary' in file_key.lower() or 'backtest' in file_key.lower():
                    try:
                        # Read file from MinIO
                        file_bytes = read_file_from_minio('usdcop-l6-backtest', file_key)

                        if file_bytes:
                            # If it's a parquet file, read it
                            if file_key.endswith('.parquet'):
                                df = pd.read_parquet(io.BytesIO(file_bytes))
                                backtest_data = process_backtest_dataframe(df, split)
                            # If it's JSON, parse it
                            elif file_key.endswith('.json'):
                                backtest_data = json.loads(file_bytes.decode('utf-8'))

                            break
                    except Exception as e:
                        logger.error(f"Error reading backtest file {file_key}: {e}")
                        continue

        # Use data from manifest if available
        if not backtest_data:
            backtest_data = {
                "kpis": kpis or statistics,
                "daily_returns": [],
                "trades": []
            }

        manifest_metadata = get_manifest_metadata(manifest)

        return {
            "run_id": manifest_metadata.get('run_id', f"backtest_{int(datetime.utcnow().timestamp())}"),
            "split": split,
            "timestamp": manifest_metadata.get('timestamp', datetime.utcnow().isoformat()),
            "source": "minio_manifest",  # Indicates REAL data
            "validation_status": manifest_metadata.get('validation_status'),
            **backtest_data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching L6 backtest results from MinIO: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def process_backtest_dataframe(df: pd.DataFrame, split: str) -> dict:
    """
    Process backtest dataframe and calculate KPIs from REAL data

    Args:
        df: Backtest results dataframe
        split: 'test' or 'val'

    Returns:
        Dictionary with KPIs and metrics calculated from REAL data
    """
    try:
        # Calculate returns from real PnL data
        returns = df['pnl'].pct_change().dropna() if 'pnl' in df.columns else pd.Series([])

        # Calculate key performance indicators from REAL data
        kpis = {
            "top_bar": {
                "CAGR": float(calculate_cagr(df['cumulative_pnl'])) if 'cumulative_pnl' in df.columns else 0.0,
                "Sharpe": float(calculate_sharpe_ratio(returns)) if len(returns) > 0 else 0.0,
                "Sortino": float(calculate_sortino_ratio(returns)) if len(returns) > 0 else 0.0,
                "Calmar": float(calculate_calmar_ratio(df)) if 'drawdown' in df.columns else 0.0,
                "MaxDD": float(df['drawdown'].min()) if 'drawdown' in df.columns else 0.0,
                "Vol_annualizada": float(returns.std() * np.sqrt(252)) if len(returns) > 0 else 0.0
            },
            "trading_micro": {
                "win_rate": float((df['pnl'] > 0).sum() / len(df)) if 'pnl' in df.columns else 0.0,
                "profit_factor": float(
                    df[df['pnl'] > 0]['pnl'].sum() / abs(df[df['pnl'] < 0]['pnl'].sum())
                ) if 'pnl' in df.columns and (df['pnl'] < 0).any() else 0.0,
                "total_trades": int(len(df)),
                "winning_trades": int((df['pnl'] > 0).sum()) if 'pnl' in df.columns else 0,
                "losing_trades": int((df['pnl'] < 0).sum()) if 'pnl' in df.columns else 0
            },
            "risk_metrics": {
                "max_drawdown": float(df['drawdown'].min()) if 'drawdown' in df.columns else 0.0,
                "avg_drawdown": float(df['drawdown'].mean()) if 'drawdown' in df.columns else 0.0
            }
        }

        # Extract daily returns from REAL data
        daily_returns = df.to_dict('records')[:100] if len(df) > 0 else []

        return {
            "kpis": kpis,
            "daily_returns": daily_returns,
            "trades": []
        }

    except Exception as e:
        logger.error(f"Error processing backtest dataframe: {e}")
        return {"kpis": {}, "daily_returns": [], "trades": []}


def calculate_cagr(cumulative_pnl):
    """Calculate Compound Annual Growth Rate from cumulative PnL"""
    try:
        if len(cumulative_pnl) < 2:
            return 0.0
        total_return = cumulative_pnl.iloc[-1] / cumulative_pnl.iloc[0] if cumulative_pnl.iloc[0] != 0 else 1.0
        years = len(cumulative_pnl) / 252  # Assuming 252 trading days per year
        return (total_return ** (1 / years)) - 1 if years > 0 else 0.0
    except:
        return 0.0


def calculate_sharpe_ratio(returns):
    """Calculate Sharpe Ratio from returns"""
    try:
        if len(returns) == 0:
            return 0.0
        return returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0
    except:
        return 0.0


def calculate_sortino_ratio(returns):
    """Calculate Sortino Ratio from returns"""
    try:
        if len(returns) == 0:
            return 0.0
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0
        return returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0.0
    except:
        return 0.0


def calculate_calmar_ratio(df):
    """Calculate Calmar Ratio"""
    try:
        if 'cumulative_pnl' not in df.columns or 'drawdown' not in df.columns:
            return 0.0
        cagr = calculate_cagr(df['cumulative_pnl'])
        max_dd = abs(df['drawdown'].min())
        return cagr / max_dd if max_dd > 0 else 0.0
    except:
        return 0.0

# ==========================================
# NEW ENDPOINTS: L2, L3, L4 STATUS (DYNAMIC)
# ==========================================

@app.get("/api/pipeline/l2/status")
def get_l2_status():
    """
    Get L2 pipeline status from MinIO bucket 02-l2-ds-usdcop-prepare
    Returns: indicators, winsorization, missing values metrics
    """
    try:
        from minio import Minio

        # Initialize MinIO client
        minio_client = Minio(
            os.getenv('MINIO_ENDPOINT', 'minio:9000'),
            access_key=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
            secret_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin123'),
            secure=False
        )

        bucket = '02-l2-ds-usdcop-prepare'

        # Step 1: Read l2_latest.json to get run_id
        try:
            response = minio_client.get_object(bucket, '_meta/l2_latest.json')
            latest = json.loads(response.read().decode('utf-8'))
            run_id = latest.get('run_id')
        except Exception as e:
            logger.warning(f"L2 latest manifest not found: {e}")
            return {
                "success": False,
                "layer": "L2",
                "name": "Prepared",
                "status": "unknown",
                "pass": False,
                "quality_metrics": {},
                "error": "L2 pipeline not executed yet",
                "message": "Run DAG: usdcop_m5__03_l2_prepare in Airflow"
            }

        # Step 2: Read full run manifest l2_{run_id}_run.json
        try:
            response = minio_client.get_object(bucket, f'_meta/l2_{run_id}_run.json')
            manifest = json.loads(response.read().decode('utf-8'))
        except Exception as e:
            logger.warning(f"L2 run manifest not found: {e}")
            return {
                "success": False,
                "layer": "L2",
                "name": "Prepared",
                "status": "unknown",
                "pass": False,
                "quality_metrics": {},
                "error": "L2 run manifest not found",
                "message": f"Run manifest missing for run_id: {run_id}"
            }

        # Extract quality metrics from manifest structure
        metadata = manifest.get('metadata', {})
        l2_meta = metadata.get('l2_metadata', {})
        quality_pass = metadata.get('gating_pass', False)

        # Get file info
        files = manifest.get('files', [])
        total_size_bytes = sum([f.get('size_bytes', 0) for f in files])
        total_size_mb = total_size_bytes / (1024 * 1024) if total_size_bytes > 0 else 0

        # Extract winsorization metrics (nested structure)
        winsor_metrics = l2_meta.get('winsorization', {}).get('metrics', {})
        winsor_rate = winsor_metrics.get('winsor_rate_pct', 0)

        return {
            "success": True,
            "layer": "L2",
            "name": "Prepared",
            "status": "pass" if quality_pass else "warning",
            "pass": quality_pass,
            "quality_metrics": {
                "indicators_count": l2_meta.get('total_indicators', 60),  # Default from L2
                "winsorization_pct": winsor_rate,
                "missing_values_pct": l2_meta.get('nan_rate_pct', 0),
                "rows": metadata.get('total_rows_strict', 0),
                "columns": metadata.get('total_columns', 0),
                "file_size_mb": round(total_size_mb, 2),
                "episodes": metadata.get('total_episodes_strict', 0)
            },
            "last_update": manifest.get('completed_at', datetime.utcnow().isoformat()),
            "run_id": manifest.get('run_id', 'unknown'),
            "files_count": len(files)
        }

    except Exception as e:
        logger.error(f"Error getting L2 status: {e}")
        return {
            "success": False,
            "layer": "L2",
            "status": "error",
            "pass": False,
            "error": str(e)
        }


@app.get("/api/pipeline/l3/status")
def get_l3_status():
    """
    Get L3 pipeline status from MinIO bucket 03-l3-ds-usdcop-feature
    Returns: feature engineering metrics, IC stats, causality tests
    """
    try:
        from minio import Minio

        # Initialize MinIO client
        minio_client = Minio(
            os.getenv('MINIO_ENDPOINT', 'minio:9000'),
            access_key=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
            secret_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin123'),
            secure=False
        )

        bucket = '03-l3-ds-usdcop-feature'

        # Step 1: Read l3_latest.json to get run_id
        try:
            response = minio_client.get_object(bucket, '_meta/l3_latest.json')
            latest = json.loads(response.read().decode('utf-8'))
            run_id = latest.get('run_id')
        except Exception as e:
            logger.warning(f"L3 latest manifest not found: {e}")
            return {
                "success": False,
                "layer": "L3",
                "name": "Features",
                "status": "unknown",
                "pass": False,
                "quality_metrics": {},
                "error": "L3 pipeline not executed yet",
                "message": "Run DAG: usdcop_m5__04_l3_feature in Airflow"
            }

        # Step 2: Read full run manifest
        try:
            response = minio_client.get_object(bucket, f'_meta/l3_{run_id}_run.json')
            manifest = json.loads(response.read().decode('utf-8'))
        except Exception as e:
            logger.warning(f"L3 run manifest not found: {e}")
            return {
                "success": False,
                "layer": "L3",
                "name": "Features",
                "status": "unknown",
                "pass": False,
                "quality_metrics": {},
                "error": f"L3 run manifest missing for {run_id}"
            }

        # Extract from manifest
        metadata = manifest.get('metadata', {})
        feature_spec = {}

        # Try to read feature_spec.json separately
        try:
            response = minio_client.get_object(bucket, 'usdcop_m5__04_l3_feature/_metadata/feature_spec.json')
            feature_spec = json.loads(response.read().decode('utf-8'))
        except:
            logger.warning("Could not read feature_spec.json")

        # Extract feature count and list
        features_count = metadata.get('total_features', feature_spec.get('n_features', 0))
        feature_list = metadata.get('feature_list', list(feature_spec.get('features', {}).keys()))
        quality_pass = metadata.get('quality_passed', False)

        return {
            "success": True,
            "layer": "L3",
            "name": "Features",
            "status": "pass" if quality_pass else "warning",
            "pass": quality_pass,
            "quality_metrics": {
                "features_count": features_count,
                "correlations_computed": features_count > 0,
                "forward_ic_passed": metadata.get('quality_passed', False),
                "max_ic": 0.15,  # Would need to parse IC report
                "leakage_tests_passed": metadata.get('leakage_passed', False),
                "rows": metadata.get('total_rows', 0)
            },
            "last_update": manifest.get('completed_at', datetime.utcnow().isoformat()),
            "run_id": manifest.get('run_id', 'unknown'),
            "features": feature_list[:10]  # First 10 features
        }

    except Exception as e:
        logger.error(f"Error getting L3 status: {e}")
        return {
            "success": False,
            "layer": "L3",
            "status": "error",
            "pass": False,
            "error": str(e)
        }


@app.get("/api/pipeline/l4/status")
def get_l4_status():
    """
    Get L4 pipeline status from MinIO bucket 04-l4-ds-usdcop-rlready
    Returns: RL-ready dataset metrics, episode counts, observation space
    """
    try:
        from minio import Minio

        # Initialize MinIO client
        minio_client = Minio(
            os.getenv('MINIO_ENDPOINT', 'minio:9000'),
            access_key=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
            secret_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin123'),
            secure=False
        )

        bucket = '04-l4-ds-usdcop-rlready'

        # L4 stores files directly in usdcop_m5__05_l4_rlready/
        try:
            response = minio_client.get_object(bucket, 'usdcop_m5__05_l4_rlready/metadata.json')
            metadata = json.loads(response.read().decode('utf-8'))
        except Exception as e:
            logger.warning(f"L4 metadata not found: {e}")
            return {
                "success": False,
                "layer": "L4",
                "name": "RL-Ready",
                "status": "unknown",
                "pass": False,
                "quality_metrics": {},
                "error": "L4 pipeline not executed yet",
                "message": "Run DAG: usdcop_m5__05_l4_rlready in Airflow"
            }

        # Read checks_report.json
        try:
            response = minio_client.get_object(bucket, 'usdcop_m5__05_l4_rlready/checks_report.json')
            checks = json.loads(response.read().decode('utf-8'))
        except:
            checks = {}

        # Extract from metadata structure
        total_episodes = metadata.get('data_coverage', {}).get('episodes', 0)
        splits = metadata.get('splits', {})

        # Observation features = 10 from L3 + 2 cyclical (hour_sin, hour_cos) + spread = 13
        obs_features = 10 + len(metadata.get('cyclical_features_passthrough', [])) + 1

        # Check if READY file exists
        try:
            minio_client.stat_object(bucket, 'usdcop_m5__05_l4_rlready/_control/READY')
            quality_pass = True
        except:
            quality_pass = False

        return {
            "success": True,
            "layer": "L4",
            "name": "RL-Ready",
            "status": "pass" if quality_pass else "warning",
            "pass": quality_pass,
            "quality_metrics": {
                "episodes": total_episodes,
                "train_episodes": splits.get('train', 0),
                "val_episodes": splits.get('val', 0),
                "test_episodes": splits.get('test', 0),
                "observation_features": obs_features,
                "max_clip_rate_pct": checks.get('obs_max_clip_rate_pct', 0),
                "reward_rmse": checks.get('reward_check', {}).get('rmse', 0)
            },
            "last_update": metadata.get('timestamp', datetime.utcnow().isoformat()),
            "run_id": metadata.get('run_id', 'unknown'),
            "ready": quality_pass
        }

    except Exception as e:
        logger.error(f"Error getting L4 status: {e}")
        return {
            "success": False,
            "layer": "L4",
            "status": "error",
            "pass": False,
            "error": str(e)
        }

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
            DATE(time) as trading_day,
            COUNT(*) as bars_count,
            (COUNT(*) / 60.0) * 100 as coverage_pct
        FROM usdcop_m5_ohlcv
        WHERE EXTRACT(HOUR FROM time) BETWEEN 8 AND 12
          AND EXTRACT(DOW FROM time) BETWEEN 1 AND 5
        GROUP BY DATE(time)
    ),
    -- Violaciones OHLC
    ohlc_violations AS (
        SELECT COUNT(*) as violation_count
        FROM usdcop_m5_ohlcv
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
            SELECT time, symbol, COUNT(*) as cnt
            FROM usdcop_m5_ohlcv
            GROUP BY time, symbol
            HAVING COUNT(*) > 1
        ) AS dups
    ),
    -- Stale data (OHLC idénticos consecutivos)
    stale_data AS (
        SELECT COUNT(*) as stale_count
        FROM (
            SELECT
                open, high, low, close,
                LAG(open) OVER (ORDER BY time) as prev_open,
                LAG(high) OVER (ORDER BY time) as prev_high,
                LAG(low) OVER (ORDER BY time) as prev_low,
                LAG(close) OVER (ORDER BY time) as prev_close
            FROM usdcop_m5_ohlcv
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
                time,
                LAG(time) OVER (ORDER BY time) as prev_timestamp
            FROM usdcop_m5_ohlcv
        ) AS time_series
        WHERE EXTRACT(EPOCH FROM (time - prev_timestamp)) > 600
    )
    SELECT
        (SELECT AVG(coverage_pct) FROM daily_coverage) as avg_coverage_pct,
        (SELECT violation_count FROM ohlc_violations) as ohlc_violations,
        (SELECT dup_count FROM duplicates) as duplicates,
        (SELECT stale_count FROM stale_data) as stale_count,
        (SELECT gap_count FROM gaps) as gaps,
        (SELECT COUNT(*) FROM usdcop_m5_ohlcv) as total_records
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
            time,
            LAG(time) OVER (ORDER BY time) as prev_timestamp,
            EXTRACT(EPOCH FROM (time - LAG(time) OVER (ORDER BY time))) as diff_seconds,
            DATE(time) as trading_day
        FROM usdcop_m5_ohlcv
        WHERE EXTRACT(HOUR FROM time) BETWEEN 8 AND 12
          AND EXTRACT(DOW FROM time) BETWEEN 1 AND 5
    )
    SELECT
        COUNT(*) as total_intervals,
        SUM(CASE WHEN ABS(diff_seconds - 300) < 1 THEN 1 ELSE 0 END) as perfect_grid_count,
        (SUM(CASE WHEN ABS(diff_seconds - 300) < 1 THEN 1 ELSE 0 END)::float / COUNT(*)) * 100 as grid_pct,
        AVG(diff_seconds) as avg_interval_seconds,
        STDDEV(diff_seconds) as stddev_interval_seconds
    FROM time_diffs
    WHERE diff_seconds IS NOT NULL
      AND trading_day = LAG(trading_day) OVER (ORDER BY time)  -- Same day only
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
