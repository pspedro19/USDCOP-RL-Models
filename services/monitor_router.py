#!/usr/bin/env python3
"""
Model Monitoring Router
=======================

FastAPI router for model health monitoring endpoints.
Provides drift detection, stuck behavior detection, and rolling Sharpe tracking.

Usage:
    In multi_model_trading_api.py:

    from monitor_router import router as monitor_router
    app.include_router(monitor_router, prefix="/api/monitor", tags=["monitoring"])
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import logging
import sys
from pathlib import Path

# Add src to path for importing monitoring module
SRC_PATH = Path(__file__).parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

logger = logging.getLogger(__name__)

router = APIRouter()

# Global model monitors registry
_model_monitors: Dict[str, Any] = {}

# Model metadata (should match multi_model_trading_api.py)
MODEL_METADATA = {
    "ppo_primary": {"name": "PPO USDCOP Primary (Production)", "algorithm": "PPO", "version": "current", "status": "production", "color": "#10B981"},
    "ppo_secondary": {"name": "PPO USDCOP Secondary", "algorithm": "PPO", "version": "current", "status": "standby", "color": "#6B7280"},
    "ppo_legacy": {"name": "PPO USDCOP Legacy", "algorithm": "PPO", "version": "legacy", "status": "deprecated", "color": "#3B82F6"},
    "sac_baseline": {"name": "SAC Baseline", "algorithm": "SAC", "version": "current", "status": "inactive", "color": "#8B5CF6"},
    "td3_baseline": {"name": "TD3 Baseline", "algorithm": "TD3", "version": "current", "status": "inactive", "color": "#F59E0B"},
    "a2c_baseline": {"name": "A2C Baseline", "algorithm": "A2C", "version": "current", "status": "inactive", "color": "#EF4444"},
    "xgb_primary": {"name": "XGBoost Classifier", "algorithm": "XGBoost", "version": "current", "status": "testing", "color": "#EC4899"},
    "lgbm_primary": {"name": "LightGBM Classifier", "algorithm": "LightGBM", "version": "current", "status": "testing", "color": "#F472B6"},
    "llm_claude": {"name": "LLM Claude Analysis", "algorithm": "LLM", "version": "current", "status": "testing", "color": "#6366F1"},
    "ensemble_primary": {"name": "Ensemble Voter", "algorithm": "Ensemble", "version": "current", "status": "testing", "color": "#14B8A6"}
}


def get_model_monitor(model_id: str):
    """Get or create a ModelMonitor for a specific model."""
    global _model_monitors
    if model_id not in _model_monitors:
        try:
            from monitoring import ModelMonitor
            _model_monitors[model_id] = ModelMonitor(window_size=100)
            logger.info(f"Created ModelMonitor for {model_id}")
        except ImportError as e:
            logger.warning(f"Could not import ModelMonitor: {e}")
            return None
    return _model_monitors.get(model_id)


# ============================================================
# PYDANTIC MODELS
# ============================================================

class ModelHealthThresholds(BaseModel):
    """Thresholds for health status determination"""
    kl_warning: float = 0.3
    kl_critical: float = 0.5
    stuck_ratio: float = 0.9


class ModelHealthItem(BaseModel):
    """Health status for a single model"""
    model_id: str
    action_drift_kl: Optional[float] = None
    stuck_behavior: bool
    rolling_sharpe: float
    actions_recorded: int
    trades_recorded: int
    status: str  # "healthy", "warning", "critical"
    details: str
    timestamp: str
    thresholds: ModelHealthThresholds


class ModelHealthResponse(BaseModel):
    """Response for model health endpoint"""
    timestamp: str
    total_models: int
    healthy_count: int
    warning_count: int
    critical_count: int
    models: List[ModelHealthItem]


# ============================================================
# ENDPOINTS
# ============================================================

@router.get("/health", response_model=ModelHealthResponse)
async def get_model_health():
    """
    Get health status of all active trading models.

    Monitors for:
    - Action drift (KL divergence vs baseline distribution)
    - Stuck behavior (model outputting same action repeatedly)
    - Rolling Sharpe ratio degradation

    Status levels:
    - healthy: KL < 0.3, no stuck behavior, Sharpe >= -1
    - warning: 0.3 <= KL <= 0.5, or Sharpe < -1
    - critical: KL > 0.5 or stuck behavior detected

    Returns aggregated health status across all models.
    """
    try:
        active_model_ids = list(MODEL_METADATA.keys())
        models_health = []
        healthy_count = warning_count = critical_count = 0

        for model_id in active_model_ids:
            monitor = get_model_monitor(model_id)
            if monitor is None:
                health_item = ModelHealthItem(
                    model_id=model_id,
                    action_drift_kl=None,
                    stuck_behavior=False,
                    rolling_sharpe=0.0,
                    actions_recorded=0,
                    trades_recorded=0,
                    status="healthy",
                    details="Monitor not initialized - no data yet",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    thresholds=ModelHealthThresholds()
                )
                healthy_count += 1
            else:
                status = monitor.get_health_status()
                health_item = ModelHealthItem(
                    model_id=model_id,
                    action_drift_kl=status.get('action_drift_kl'),
                    stuck_behavior=status.get('stuck_behavior', False),
                    rolling_sharpe=status.get('rolling_sharpe', 0.0),
                    actions_recorded=status.get('actions_recorded', 0),
                    trades_recorded=status.get('trades_recorded', 0),
                    status=status.get('status', 'healthy'),
                    details=status.get('details', ''),
                    timestamp=status.get('timestamp', datetime.now(timezone.utc).isoformat()),
                    thresholds=ModelHealthThresholds(
                        kl_warning=status.get('thresholds', {}).get('kl_warning', 0.3),
                        kl_critical=status.get('thresholds', {}).get('kl_critical', 0.5),
                        stuck_ratio=status.get('thresholds', {}).get('stuck_ratio', 0.9)
                    )
                )
                if health_item.status == "critical":
                    critical_count += 1
                elif health_item.status == "warning":
                    warning_count += 1
                else:
                    healthy_count += 1
            models_health.append(health_item)

        return ModelHealthResponse(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_models=len(models_health),
            healthy_count=healthy_count,
            warning_count=warning_count,
            critical_count=critical_count,
            models=models_health
        )
    except Exception as e:
        logger.error(f"Error getting model health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_id}/record-action")
async def record_model_action(
    model_id: str,
    action: float = Query(..., description="Action value from model (typically -1 to 1)")
):
    """
    Record a model action for drift detection.

    Call this endpoint after each model inference to track
    the action distribution over time.
    """
    monitor = get_model_monitor(model_id)

    if monitor is None:
        return {
            "recorded": False,
            "reason": "ModelMonitor not available",
            "model_id": model_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    try:
        monitor.record_action(action)
        return {
            "recorded": True,
            "model_id": model_id,
            "action": action,
            "actions_recorded": len(monitor.action_history),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error recording action for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_id}/record-pnl")
async def record_model_pnl(
    model_id: str,
    pnl: float = Query(..., description="P&L value from completed trade")
):
    """
    Record trade PnL for rolling Sharpe calculation.

    Call this endpoint after each trade closes to track
    model performance over time.
    """
    monitor = get_model_monitor(model_id)

    if monitor is None:
        return {
            "recorded": False,
            "reason": "ModelMonitor not available",
            "model_id": model_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    try:
        monitor.record_pnl(pnl)
        return {
            "recorded": True,
            "model_id": model_id,
            "pnl": pnl,
            "trades_recorded": len(monitor.pnl_history),
            "rolling_sharpe": monitor.get_rolling_sharpe(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error recording PnL for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_id}/set-baseline")
async def set_model_baseline(
    model_id: str,
    actions: List[float] = Query(..., description="List of baseline actions from backtest")
):
    """
    Set baseline action distribution from backtest data.

    This establishes the expected action distribution that
    will be used to detect drift via KL divergence.

    Call this once with historical backtest actions when
    deploying a new model version.
    """
    monitor = get_model_monitor(model_id)

    if monitor is None:
        return {
            "set": False,
            "reason": "ModelMonitor not available",
            "model_id": model_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    try:
        monitor.set_baseline(actions)
        return {
            "set": True,
            "model_id": model_id,
            "baseline_samples": len(actions),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error setting baseline for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_id}/health", response_model=ModelHealthItem)
async def get_single_model_health(model_id: str):
    """
    Get health status for a specific model.

    Returns detailed health information including:
    - Action drift KL divergence
    - Stuck behavior detection
    - Rolling Sharpe ratio
    - Sample counts
    """
    monitor = get_model_monitor(model_id)

    if monitor is None:
        return ModelHealthItem(
            model_id=model_id,
            action_drift_kl=None,
            stuck_behavior=False,
            rolling_sharpe=0.0,
            actions_recorded=0,
            trades_recorded=0,
            status="healthy",
            details="Monitor not initialized - no data yet",
            timestamp=datetime.now(timezone.utc).isoformat(),
            thresholds=ModelHealthThresholds()
        )

    try:
        status = monitor.get_health_status()
        return ModelHealthItem(
            model_id=model_id,
            action_drift_kl=status.get('action_drift_kl'),
            stuck_behavior=status.get('stuck_behavior', False),
            rolling_sharpe=status.get('rolling_sharpe', 0.0),
            actions_recorded=status.get('actions_recorded', 0),
            trades_recorded=status.get('trades_recorded', 0),
            status=status.get('status', 'healthy'),
            details=status.get('details', ''),
            timestamp=status.get('timestamp', datetime.now(timezone.utc).isoformat()),
            thresholds=ModelHealthThresholds(
                kl_warning=status.get('thresholds', {}).get('kl_warning', 0.3),
                kl_critical=status.get('thresholds', {}).get('kl_critical', 0.5),
                stuck_ratio=status.get('thresholds', {}).get('stuck_ratio', 0.9)
            )
        )
    except Exception as e:
        logger.error(f"Error getting health for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_id}/reset")
async def reset_model_monitor(model_id: str):
    """
    Reset monitor history for a specific model.

    Clears action and PnL history buffers.
    Useful when retraining or deploying a new model version.
    """
    monitor = get_model_monitor(model_id)

    if monitor is None:
        return {
            "reset": False,
            "reason": "ModelMonitor not available",
            "model_id": model_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    try:
        monitor.reset()
        return {
            "reset": True,
            "model_id": model_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error resetting monitor for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
