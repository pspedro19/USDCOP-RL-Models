"""
Production Inference API
========================

FastAPI-based inference service integrating all MLOps components:
- ONNX Runtime inference
- Risk management
- Feature caching
- Health monitoring
- Metrics exposure

Designed for production deployment with:
- Sub-10ms response times
- Circuit breaker protection
- Comprehensive logging
- Prometheus metrics
"""

import os
import time
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

from mlops.config import MLOpsConfig, get_config, SignalType, ModelConfig
from mlops.inference_engine import InferenceEngine, get_inference_engine, InferenceResult, EnsembleResult
from mlops.risk_manager import RiskManager, get_risk_manager, RiskCheckResult
from mlops.feature_cache import FeatureCache, get_feature_cache
from mlops.drift_monitor import DriftMonitor

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================

class InferenceRequest(BaseModel):
    """Request for model inference."""
    observation: List[float] = Field(..., description="Feature vector for inference")
    model_name: Optional[str] = Field(None, description="Specific model to use")
    use_ensemble: bool = Field(False, description="Use ensemble inference")
    enforce_risk_checks: bool = Field(True, description="Apply risk management checks")
    timestamp: Optional[str] = Field(None, description="Feature timestamp for caching")


class InferenceResponse(BaseModel):
    """Response from model inference."""
    signal: str
    confidence: float
    approved: bool
    risk_status: str
    action_probs: Dict[str, float]
    latency_ms: float
    model_name: str
    timestamp: str
    risk_metrics: Optional[Dict[str, Any]] = None
    position_recommendation: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    environment: str
    timestamp: str
    components: Dict[str, Any]


class RiskSummaryResponse(BaseModel):
    """Risk summary response."""
    timestamp: str
    trading_status: Dict[str, Any]
    daily_stats: Dict[str, Any]
    limits: Dict[str, Any]
    risk_utilization: Dict[str, Any]


class MetricsResponse(BaseModel):
    """Metrics response."""
    inference_stats: Dict[str, Any]
    risk_stats: Dict[str, Any]
    cache_stats: Dict[str, Any]


class TradeResultRequest(BaseModel):
    """Request to update trade result."""
    pnl: float = Field(..., description="Profit/loss in currency")
    pnl_percent: float = Field(..., description="Profit/loss as percentage")
    is_win: bool = Field(..., description="Whether trade was profitable")
    trade_id: Optional[str] = Field(None, description="Trade identifier")


# ============================================================================
# Application Setup
# ============================================================================

def create_inference_app(config: Optional[MLOpsConfig] = None) -> FastAPI:
    """
    Create FastAPI application with all MLOps components.

    Args:
        config: Optional MLOps configuration

    Returns:
        Configured FastAPI application
    """
    config = config or get_config()

    # Application state
    state = {
        "config": config,
        "inference_engine": None,
        "risk_manager": None,
        "feature_cache": None,
        "drift_monitor": None,
        "startup_time": None,
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan manager."""
        # Startup
        logger.info("Starting MLOps Inference Service...")

        try:
            # Initialize components
            state["inference_engine"] = InferenceEngine(config)
            state["risk_manager"] = RiskManager(config)
            state["feature_cache"] = FeatureCache(config)

            # Load models
            if config.models:
                state["inference_engine"].load_models()
            else:
                logger.warning("No models configured. Load models manually.")

            state["startup_time"] = datetime.now().isoformat()

            logger.info("✅ MLOps Inference Service started")

        except Exception as e:
            logger.error(f"Failed to start service: {e}")
            raise

        yield

        # Shutdown
        logger.info("Shutting down MLOps Inference Service...")

    # Create app
    app = FastAPI(
        title="USDCOP Trading Inference API",
        description="Production-grade ML inference service with risk management",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store state
    app.state.mlops = state

    # ========================================================================
    # Dependency Injection
    # ========================================================================

    def get_engine() -> InferenceEngine:
        if app.state.mlops["inference_engine"] is None:
            raise HTTPException(503, "Inference engine not initialized")
        return app.state.mlops["inference_engine"]

    def get_risk_mgr() -> RiskManager:
        if app.state.mlops["risk_manager"] is None:
            raise HTTPException(503, "Risk manager not initialized")
        return app.state.mlops["risk_manager"]

    def get_cache() -> FeatureCache:
        if app.state.mlops["feature_cache"] is None:
            raise HTTPException(503, "Feature cache not initialized")
        return app.state.mlops["feature_cache"]

    # ========================================================================
    # Endpoints
    # ========================================================================

    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint."""
        return {
            "service": "USDCOP Trading Inference API",
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs",
        }

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check(
        engine: InferenceEngine = Depends(get_engine),
        risk_mgr: RiskManager = Depends(get_risk_mgr),
        cache: FeatureCache = Depends(get_cache),
    ):
        """Comprehensive health check."""
        components = {
            "inference_engine": engine.health_check(),
            "risk_manager": {
                "status": "healthy",
                "circuit_breaker": risk_mgr.is_circuit_breaker_active(),
            },
            "feature_cache": cache.health_check(),
        }

        # Determine overall status
        overall_status = "healthy"
        for name, comp in components.items():
            if comp.get("status") == "unhealthy":
                overall_status = "unhealthy"
                break
            elif comp.get("status") == "degraded":
                overall_status = "degraded"

        return HealthResponse(
            status=overall_status,
            environment=config.environment.value,
            timestamp=datetime.now().isoformat(),
            components=components,
        )

    @app.post("/v1/inference", response_model=InferenceResponse, tags=["Inference"])
    async def inference(
        request: InferenceRequest,
        engine: InferenceEngine = Depends(get_engine),
        risk_mgr: RiskManager = Depends(get_risk_mgr),
        cache: FeatureCache = Depends(get_cache),
    ):
        """
        Run model inference with risk management.

        Returns trading signal with confidence and risk approval status.
        """
        start_time = time.perf_counter()

        try:
            # Convert observation to numpy
            observation = np.array(request.observation, dtype=np.float32)

            # Run inference
            if request.use_ensemble and len(engine.models) > 1:
                result = engine.predict_ensemble(observation)
                model_name = "ensemble"
                signal = result.signal
                confidence = result.confidence
                action_probs = result.action_probs
            else:
                result = engine.predict(observation, request.model_name)
                model_name = result.model_name
                signal = result.signal
                confidence = result.confidence
                action_probs = result.action_probs

            # Apply risk checks
            approved = True
            risk_status = "NOT_CHECKED"
            risk_metrics = None
            position_rec = None

            if request.enforce_risk_checks:
                risk_result = risk_mgr.check_signal(signal, confidence)
                approved = risk_result.approved
                risk_status = risk_result.status.value
                signal = risk_result.adjusted_signal
                risk_metrics = risk_result.risk_metrics

                # Get position recommendation for approved trades
                if approved and signal != SignalType.HOLD:
                    position_rec = risk_mgr.get_position_size_recommendation(
                        confidence, 100000  # Default capital
                    )

            # Cache features if timestamp provided
            if request.timestamp:
                cache.set_features(
                    timestamp=request.timestamp,
                    features={f"f{i}": v for i, v in enumerate(request.observation)},
                    source="inference_api"
                )

            total_latency = (time.perf_counter() - start_time) * 1000

            return InferenceResponse(
                signal=signal.value,
                confidence=confidence,
                approved=approved,
                risk_status=risk_status,
                action_probs=action_probs,
                latency_ms=total_latency,
                model_name=model_name,
                timestamp=datetime.now().isoformat(),
                risk_metrics=risk_metrics,
                position_recommendation=position_rec,
            )

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise HTTPException(500, f"Inference failed: {str(e)}")

    @app.post("/v1/inference/batch", tags=["Inference"])
    async def batch_inference(
        requests: List[InferenceRequest],
        engine: InferenceEngine = Depends(get_engine),
    ):
        """Run batch inference on multiple observations."""
        results = []

        for req in requests:
            try:
                observation = np.array(req.observation, dtype=np.float32)
                result = engine.predict(observation, req.model_name)
                results.append({
                    "signal": result.signal.value,
                    "confidence": result.confidence,
                    "latency_ms": result.latency_ms,
                    "success": True,
                })
            except Exception as e:
                results.append({
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "error": str(e),
                    "success": False,
                })

        return {"results": results, "total": len(results)}

    @app.get("/v1/risk/summary", response_model=RiskSummaryResponse, tags=["Risk"])
    async def risk_summary(risk_mgr: RiskManager = Depends(get_risk_mgr)):
        """Get comprehensive risk summary."""
        return risk_mgr.get_risk_summary()

    @app.post("/v1/risk/trade-result", tags=["Risk"])
    async def update_trade_result(
        request: TradeResultRequest,
        risk_mgr: RiskManager = Depends(get_risk_mgr),
    ):
        """Update risk manager with trade result."""
        risk_mgr.update_trade_result(
            pnl=request.pnl,
            pnl_percent=request.pnl_percent,
            is_win=request.is_win,
            trade_id=request.trade_id,
        )
        return {"status": "updated", "timestamp": datetime.now().isoformat()}

    @app.post("/v1/risk/reset", tags=["Risk"])
    async def reset_daily_stats(risk_mgr: RiskManager = Depends(get_risk_mgr)):
        """Reset daily risk statistics."""
        risk_mgr.reset_daily_stats()
        return {"status": "reset", "timestamp": datetime.now().isoformat()}

    @app.get("/v1/metrics", response_model=MetricsResponse, tags=["Monitoring"])
    async def get_metrics(
        engine: InferenceEngine = Depends(get_engine),
        risk_mgr: RiskManager = Depends(get_risk_mgr),
        cache: FeatureCache = Depends(get_cache),
    ):
        """Get service metrics."""
        return MetricsResponse(
            inference_stats=engine.get_stats(),
            risk_stats=risk_mgr.get_risk_summary(),
            cache_stats=cache.get_stats(),
        )

    @app.get("/v1/models", tags=["Models"])
    async def list_models(engine: InferenceEngine = Depends(get_engine)):
        """List loaded models."""
        return {
            "loaded": engine.is_loaded,
            "models": engine.model_names,
            "stats": engine.get_stats(),
        }

    @app.post("/v1/models/load", tags=["Models"])
    async def load_model(
        name: str,
        onnx_path: str,
        observation_dim: int = 45,
        engine: InferenceEngine = Depends(get_engine),
    ):
        """Load a model dynamically."""
        try:
            engine.load_single_model(name, onnx_path, observation_dim)
            return {"status": "loaded", "model": name}
        except Exception as e:
            raise HTTPException(400, f"Failed to load model: {str(e)}")

    @app.get("/v1/cache/stats", tags=["Cache"])
    async def cache_stats(cache: FeatureCache = Depends(get_cache)):
        """Get feature cache statistics."""
        return cache.get_stats()

    @app.get("/v1/cache/latest", tags=["Cache"])
    async def cache_latest(cache: FeatureCache = Depends(get_cache)):
        """Get latest cached features."""
        latest = cache.get_latest()
        if latest:
            return latest.to_dict()
        return {"message": "No cached features"}

    return app


# ============================================================================
# Standalone Application
# ============================================================================

# Default app instance for uvicorn
app = create_inference_app()


def run_server(host: str = "0.0.0.0", port: int = None):
    """Run the inference server."""
    import uvicorn
    if port is None:
        port = int(os.getenv("PORT", "8090"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
