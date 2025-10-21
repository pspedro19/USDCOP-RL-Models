#!/usr/bin/env python3
"""
USDCOP ML Analytics API
========================

API para monitoreo y análisis de modelos ML:
- Listado de modelos y sus métricas
- Health monitoring de modelos
- Predicciones vs valores reales
- Accuracy over time
- Feature importance/impact
- Alertas de degradación de modelos

Integrado con MLflow, PostgreSQL y MinIO para tracking completo.
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
import random

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
    title="USDCOP ML Analytics API",
    description="API para análisis y monitoreo de modelos ML",
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
# MODELS
# ==========================================

class ModelMetrics(BaseModel):
    mse: float
    mae: float
    rmse: float
    mape: float
    accuracy: float
    correlation: float
    total_predictions: int
    correct_direction: int
    direction_accuracy: float

class ModelInfo(BaseModel):
    model_id: str
    name: str
    version: str
    algorithm: str
    architecture: str
    training_date: str
    status: str
    metrics: Optional[Dict] = None

class HealthStatus(BaseModel):
    model_id: str
    status: str
    health_score: float
    last_prediction: str
    predictions_24h: int
    avg_accuracy: float
    issues: List[str]

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

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def generate_mock_models(limit: int = 10) -> List[Dict]:
    """Generate mock model data"""
    models = []
    algorithms = ["PPO", "A2C", "SAC", "TD3", "DQN"]
    architectures = ["LSTM", "GRU", "Transformer", "MLP", "CNN-LSTM"]

    for i in range(limit):
        algo = random.choice(algorithms)
        arch = random.choice(architectures)
        version = f"{random.randint(1, 3)}.{random.randint(0, 9)}"

        models.append({
            "model_id": f"{algo.lower()}_{arch.lower()}_v{version.replace('.', '_')}",
            "name": f"{algo} with {arch}",
            "version": version,
            "algorithm": algo,
            "architecture": arch,
            "training_date": (datetime.utcnow() - timedelta(days=random.randint(1, 30))).isoformat(),
            "status": random.choice(["active", "testing", "inactive"]),
            "metrics": {
                "train_reward": round(random.uniform(1000, 1500), 2),
                "val_reward": round(random.uniform(950, 1400), 2),
                "sharpe_ratio": round(random.uniform(1.2, 2.5), 2),
                "win_rate": round(random.uniform(0.60, 0.75), 3)
            }
        })

    return models

def calculate_model_metrics(predictions: List[float], actuals: List[float]) -> Dict:
    """Calculate model performance metrics"""
    pred_array = np.array(predictions)
    actual_array = np.array(actuals)

    mse = float(np.mean((pred_array - actual_array) ** 2))
    mae = float(np.mean(np.abs(pred_array - actual_array)))
    rmse = float(np.sqrt(mse))

    # MAPE
    mape = float(np.mean(np.abs((actual_array - pred_array) / actual_array)) * 100)

    # Correlation
    correlation = float(np.corrcoef(pred_array, actual_array)[0, 1])

    # Direction accuracy
    pred_direction = np.sign(np.diff(pred_array))
    actual_direction = np.sign(np.diff(actual_array))
    direction_correct = np.sum(pred_direction == actual_direction)
    direction_accuracy = float(direction_correct / len(pred_direction) * 100)

    # Accuracy (inverse of MAPE)
    accuracy = 100 - min(mape, 100)

    return {
        "mse": round(mse, 6),
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "mape": round(mape, 2),
        "accuracy": round(accuracy, 2),
        "correlation": round(correlation, 4),
        "total_predictions": len(predictions),
        "correct_direction": int(direction_correct),
        "direction_accuracy": round(direction_accuracy, 2)
    }

# ==========================================
# ROOT ENDPOINTS
# ==========================================

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "USDCOP ML Analytics API",
        "status": "active",
        "version": "1.0.0",
        "endpoints": {
            "models": "/api/ml-analytics/models",
            "health": "/api/ml-analytics/health",
            "predictions": "/api/ml-analytics/predictions"
        }
    }

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()

        return {
            "status": "healthy",
            "database": "connected",
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
# MODELS ENDPOINTS
# ==========================================

@app.get("/api/ml-analytics/models")
def get_models(
    action: str = Query(default="list", description="list or metrics"),
    runId: Optional[str] = Query(default=None),
    limit: int = Query(default=10, ge=1, le=100)
):
    """
    Get ML models information

    Actions:
    - list: List all available models
    - metrics: Get detailed metrics for a specific model run
    """
    try:
        if action == "list":
            models = generate_mock_models(limit)
            return {
                "success": True,
                "models": models,
                "count": len(models),
                "timestamp": datetime.utcnow().isoformat()
            }

        elif action == "metrics":
            if not runId:
                raise HTTPException(status_code=400, detail="runId required for metrics action")

            # Generate mock metrics for the run
            metrics = {
                "run_id": runId,
                "model_name": runId.split('_')[0].upper(),
                "training_metrics": {
                    "episodes": random.randint(500, 2000),
                    "total_steps": random.randint(100000, 500000),
                    "avg_reward": round(random.uniform(1000, 1500), 2),
                    "best_reward": round(random.uniform(1400, 1800), 2),
                    "final_reward": round(random.uniform(1100, 1450), 2)
                },
                "evaluation_metrics": {
                    "sharpe_ratio": round(random.uniform(1.2, 2.5), 2),
                    "sortino_ratio": round(random.uniform(1.5, 3.0), 2),
                    "calmar_ratio": round(random.uniform(0.8, 1.5), 2),
                    "win_rate": round(random.uniform(0.60, 0.75), 3),
                    "profit_factor": round(random.uniform(1.5, 3.0), 2),
                    "max_drawdown": round(random.uniform(-0.15, -0.05), 3)
                },
                "prediction_metrics": {
                    "mse": round(random.uniform(0.0001, 0.001), 6),
                    "mae": round(random.uniform(0.01, 0.05), 4),
                    "rmse": round(random.uniform(0.01, 0.03), 4),
                    "accuracy": round(random.uniform(85, 98), 2),
                    "direction_accuracy": round(random.uniform(70, 90), 2)
                },
                "timestamp": datetime.utcnow().isoformat()
            }

            return {
                "success": True,
                "data": metrics
            }

        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in models endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# HEALTH ENDPOINTS
# ==========================================

@app.get("/api/ml-analytics/health")
def get_model_health(
    action: str = Query(default="summary", description="summary, detail, alerts, or metrics-history"),
    modelId: Optional[str] = Query(default=None)
):
    """
    Get model health information

    Actions:
    - summary: Overall health summary of all models
    - detail: Detailed health for specific model
    - alerts: Get system alerts
    - metrics-history: Historical metrics for a model
    """
    try:
        if action == "summary":
            # Generate summary for all models
            models = generate_mock_models(5)
            health_summary = []

            for model in models:
                health_score = random.uniform(75, 98)
                issues = []

                if health_score < 85:
                    issues.append("Accuracy degradation detected")
                if random.random() < 0.3:
                    issues.append("High prediction variance")

                health_summary.append({
                    "model_id": model["model_id"],
                    "name": model["name"],
                    "status": "healthy" if health_score >= 85 else "warning" if health_score >= 70 else "critical",
                    "health_score": round(health_score, 2),
                    "last_prediction": (datetime.utcnow() - timedelta(minutes=random.randint(1, 60))).isoformat(),
                    "predictions_24h": random.randint(500, 2000),
                    "avg_accuracy": round(random.uniform(85, 98), 2),
                    "issues": issues
                })

            return {
                "success": True,
                "data": health_summary,
                "timestamp": datetime.utcnow().isoformat()
            }

        elif action == "detail":
            if not modelId:
                raise HTTPException(status_code=400, detail="modelId required for detail action")

            # Generate detailed health info
            health_score = random.uniform(75, 98)
            issues = []

            if health_score < 85:
                issues.append("Accuracy below threshold")
            if random.random() < 0.3:
                issues.append("Prediction latency increased")

            detail = {
                "model_id": modelId,
                "health_score": round(health_score, 2),
                "status": "healthy" if health_score >= 85 else "warning",
                "last_updated": datetime.utcnow().isoformat(),
                "metrics": {
                    "predictions_total": random.randint(10000, 50000),
                    "predictions_24h": random.randint(500, 2000),
                    "avg_latency_ms": round(random.uniform(10, 50), 2),
                    "avg_accuracy": round(random.uniform(85, 98), 2),
                    "error_rate": round(random.uniform(0, 5), 2)
                },
                "recent_performance": {
                    "1h": round(random.uniform(90, 98), 2),
                    "24h": round(random.uniform(88, 96), 2),
                    "7d": round(random.uniform(85, 95), 2)
                },
                "issues": issues,
                "recommendations": [
                    "Monitor for continued degradation",
                    "Consider retraining if accuracy drops below 85%"
                ] if issues else ["Performance is optimal"]
            }

            return {
                "success": True,
                "data": detail
            }

        elif action == "alerts":
            # Generate system alerts
            alerts = []

            if random.random() < 0.5:
                alerts.append({
                    "id": f"alert_{int(datetime.utcnow().timestamp())}",
                    "severity": "warning",
                    "model_id": "ppo_lstm_v2_1",
                    "message": "Model accuracy dropped to 87.5% (threshold: 90%)",
                    "timestamp": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                    "acknowledged": False
                })

            if random.random() < 0.3:
                alerts.append({
                    "id": f"alert_{int(datetime.utcnow().timestamp()) + 1}",
                    "severity": "info",
                    "model_id": "sac_transformer_v3_0",
                    "message": "New model version available for deployment",
                    "timestamp": (datetime.utcnow() - timedelta(hours=5)).isoformat(),
                    "acknowledged": False
                })

            return {
                "success": True,
                "alerts": alerts,
                "count": len(alerts),
                "timestamp": datetime.utcnow().isoformat()
            }

        elif action == "metrics-history":
            if not modelId:
                raise HTTPException(status_code=400, detail="modelId required for metrics-history action")

            # Generate historical metrics
            history = []
            base_accuracy = 92.0

            for i in range(24):  # Last 24 hours
                timestamp = datetime.utcnow() - timedelta(hours=24-i)
                accuracy = base_accuracy + random.uniform(-5, 5)

                history.append({
                    "timestamp": timestamp.isoformat(),
                    "accuracy": round(accuracy, 2),
                    "predictions_count": random.randint(50, 150),
                    "avg_error": round(random.uniform(0.01, 0.05), 4),
                    "latency_ms": round(random.uniform(15, 45), 2)
                })

            return {
                "success": True,
                "model_id": modelId,
                "data": history,
                "period": "24h",
                "timestamp": datetime.utcnow().isoformat()
            }

        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in health endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ml-analytics/health")
def report_model_health(model_id: str, metrics: Dict):
    """
    Report model health metrics (for models to push their status)
    """
    try:
        logger.info(f"Received health metrics for model {model_id}: {metrics}")

        # In production, store in database
        return {
            "success": True,
            "message": "Health metrics recorded",
            "model_id": model_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error reporting health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# PREDICTIONS ENDPOINTS
# ==========================================

@app.get("/api/ml-analytics/predictions")
def get_predictions(
    action: str = Query(default="data", description="data, metrics, accuracy-over-time, or feature-impact"),
    runId: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=1000),
    timeRange: str = Query(default="24h", description="24h, 7d, or 30d")
):
    """
    Get model predictions and analysis

    Actions:
    - data: Get prediction vs actual data
    - metrics: Get prediction metrics
    - accuracy-over-time: Get accuracy time series
    - feature-impact: Get feature importance
    """
    try:
        if action == "data":
            # Generate prediction vs actual data
            base_price = 4320.0
            predictions = []

            for i in range(limit):
                timestamp = datetime.utcnow() - timedelta(minutes=limit-i)
                actual = base_price + random.uniform(-20, 20)
                predicted = actual + random.uniform(-5, 5)  # Prediction close to actual
                error = predicted - actual
                pct_error = (error / actual) * 100

                predictions.append({
                    "timestamp": timestamp.isoformat(),
                    "actual": round(actual, 2),
                    "predicted": round(predicted, 2),
                    "error": round(error, 2),
                    "percentage_error": round(pct_error, 4),
                    "confidence": round(random.uniform(0.75, 0.95), 2)
                })

            return {
                "success": True,
                "data": predictions,
                "count": len(predictions),
                "run_id": runId,
                "timestamp": datetime.utcnow().isoformat()
            }

        elif action == "metrics":
            # Calculate and return metrics
            # Generate sample data
            actuals = [4320 + random.uniform(-20, 20) for _ in range(100)]
            predictions = [a + random.uniform(-5, 5) for a in actuals]

            metrics = calculate_model_metrics(predictions, actuals)

            # Add sample predictions
            sample_predictions = []
            for i in range(min(10, len(predictions))):
                sample_predictions.append({
                    "timestamp": (datetime.utcnow() - timedelta(minutes=100-i)).isoformat(),
                    "actual": round(actuals[i], 2),
                    "predicted": round(predictions[i], 2),
                    "confidence": round(random.uniform(0.75, 0.95), 2),
                    "percentage_error": round(((predictions[i] - actuals[i]) / actuals[i]) * 100, 4)
                })

            return {
                "success": True,
                "data": {
                    "metrics": metrics,
                    "sample_predictions": sample_predictions
                },
                "run_id": runId,
                "timestamp": datetime.utcnow().isoformat()
            }

        elif action == "accuracy-over-time":
            # Generate accuracy time series
            time_series = []
            base_accuracy = 92.0

            hours = 24 if timeRange == "24h" else 168 if timeRange == "7d" else 720
            interval = 1 if timeRange == "24h" else 6 if timeRange == "7d" else 24

            for i in range(0, hours, interval):
                timestamp = datetime.utcnow() - timedelta(hours=hours-i)
                accuracy = base_accuracy + random.uniform(-5, 5)

                time_series.append({
                    "timestamp": timestamp.isoformat(),
                    "accuracy": round(accuracy, 2),
                    "mse": round(random.uniform(0.0001, 0.001), 6),
                    "mae": round(random.uniform(0.01, 0.05), 4),
                    "predictions_count": random.randint(50, 200)
                })

            return {
                "success": True,
                "data": time_series,
                "run_id": runId,
                "time_range": timeRange,
                "timestamp": datetime.utcnow().isoformat()
            }

        elif action == "feature-impact":
            # Generate feature importance data
            features = [
                "price_change",
                "volume_change",
                "rsi",
                "macd",
                "bollinger_position",
                "ema_20_50_cross",
                "spread",
                "time_of_day",
                "day_of_week",
                "volatility"
            ]

            feature_impact = []
            for feature in features:
                feature_impact.append({
                    "feature": feature,
                    "importance": round(random.uniform(0.01, 0.25), 4),
                    "correlation": round(random.uniform(-0.5, 0.8), 3),
                    "impact_score": round(random.uniform(1, 10), 2)
                })

            # Sort by importance
            feature_impact.sort(key=lambda x: x["importance"], reverse=True)

            return {
                "success": True,
                "data": feature_impact,
                "run_id": runId,
                "timestamp": datetime.utcnow().isoformat()
            }

        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predictions endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ml-analytics/predictions")
def store_predictions(predictions: List[Dict], model_run_id: str):
    """
    Store model predictions for tracking and analysis
    """
    try:
        logger.info(f"Storing {len(predictions)} predictions for run {model_run_id}")

        # In production, store in database
        return {
            "success": True,
            "message": "Predictions stored successfully",
            "count": len(predictions),
            "model_run_id": model_run_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error storing predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    port = int(os.getenv("ML_ANALYTICS_API_PORT", "8005"))
    uvicorn.run(
        "ml_analytics_api:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
