"""
Models Router
=============
Endpoint for listing available models from file system and database registry.
Includes model promotion system (registered -> deployed).

MLOps-3 Features:
- Model reload endpoint for hot-reloading without restart
- Shadow mode support via ModelRouter integration
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel

try:
    from prometheus_client import Counter
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(tags=["models"])

# Prometheus metrics for model reload tracking
if PROMETHEUS_AVAILABLE:
    MODEL_RELOAD_TOTAL = Counter(
        'model_reload_total',
        'Total model reload attempts',
        ['status']
    )
    MODEL_RELOAD_FAILURES = Counter(
        'model_reload_failures_total',
        'Total model reload failures'
    )
else:
    class MockCounter:
        def labels(self, *args, **kwargs):
            return self
        def inc(self, *args, **kwargs):
            pass
    MODEL_RELOAD_TOTAL = MockCounter()
    MODEL_RELOAD_FAILURES = MockCounter()


class ModelInfo(BaseModel):
    """Model information"""
    model_id: str
    display_name: str
    version: Optional[str] = None
    status: str  # "available", "loaded", "not_found"
    db_status: str  # "registered", "deployed" - raw database status
    observation_dim: int
    description: Optional[str] = None

    model_config = {"protected_namespaces": ()}


class ModelsResponse(BaseModel):
    """Response for models listing"""
    models: List[ModelInfo]
    default_model: str
    total: int


class PromoteResponse(BaseModel):
    """Response for model promotion"""
    success: bool
    promoted_model: str
    previous_deployed: Optional[str] = None
    message: str


class RollbackRequest(BaseModel):
    """Request for model rollback"""
    target_version: Optional[int] = None  # None = previous model
    reason: str
    initiated_by: str = "dashboard"


class RollbackResponse(BaseModel):
    """Response for model rollback"""
    success: bool
    previous_model: str
    new_model: str
    rollback_time_ms: float
    message: str


class PromoteRequest(BaseModel):
    """Request for model promotion with validation"""
    target_stage: str  # 'staging' or 'production'
    reason: str
    promoted_by: str = "dashboard"
    checklist: Optional[Dict[str, bool]] = None
    skip_staging_time: bool = False  # For test override


class PromoteRequestResponse(BaseModel):
    """Response for model promotion request"""
    success: bool
    model_id: str
    new_stage: str
    previous_stage: str
    message: str
    metrics_passed: bool
    promoted_at: str


class RollbackTarget(BaseModel):
    """Available rollback target model"""
    model_id: str
    version: str
    archived_at: str
    metrics: Dict[str, float]

    model_config = {"protected_namespaces": ()}


class RollbackTargetsResponse(BaseModel):
    """Response for rollback targets listing"""
    current_production: Optional[Dict[str, Any]]
    available_targets: List[RollbackTarget]
    recommendation: Optional[RollbackTarget]


class ReloadResponse(BaseModel):
    """Response for model reload operation"""
    success: bool
    reloaded_models: List[str]
    failed_models: List[str]
    timestamp: str
    details: Optional[Dict[str, Any]] = None
    message: str


# Model configurations loaded dynamically from database
# This dict is populated at runtime from model_registry table
MODEL_CONFIGS = {}


async def load_model_configs_from_db():
    """Load model configurations from database config.models table (SSOT)"""
    global MODEL_CONFIGS

    try:
        import asyncpg
        from ..config import get_settings

        settings = get_settings()

        conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db,
        )
        try:
            # Query config.models table (SSOT for model configuration)
            rows = await conn.fetch("""
                SELECT
                    model_id,
                    name,
                    algorithm,
                    version,
                    model_path,
                    status,
                    color,
                    description,
                    environment_config
                FROM config.models
                WHERE status = 'active'
                ORDER BY model_id
            """)
        finally:
            await conn.close()

        # Clear existing configs to get fresh data
        MODEL_CONFIGS.clear()

        for row in rows:
            model_id = row['model_id']
            name = row['name'] or model_id
            version = row['version'] or 'v1'
            algorithm = row['algorithm'] or 'PPO'
            db_status = row['status']  # 'active', 'inactive', etc.
            env_config = row['environment_config'] or {}
            obs_dim = env_config.get('observation_space_dim', 15) if isinstance(env_config, dict) else 15

            MODEL_CONFIGS[model_id] = {
                "display_name": name,
                "version": version,
                "algorithm": algorithm,
                "db_status": db_status,
                "observation_dim": obs_dim,
                "description": row['description'] or f"{algorithm} {version} model",
                "color": row['color'] or '#3B82F6',
                "patterns": [row['model_path']] if row['model_path'] else [f"{model_id}*.zip"],
            }

        logger.info(f"Loaded {len(MODEL_CONFIGS)} models from config.models: {list(MODEL_CONFIGS.keys())}")

    except Exception as e:
        logger.warning(f"Could not load models from database: {e}. Using empty config.")
        MODEL_CONFIGS = {}


def find_model_file(model_id: str, models_dir: Path) -> Optional[Path]:
    """Find model file by pattern matching"""
    config = MODEL_CONFIGS.get(model_id, {})
    patterns = config.get("patterns", [f"{model_id}*.zip"])

    for pattern in patterns:
        # Check for subdirectory pattern (e.g., ppo_production/final_model.zip)
        if "/" in pattern:
            full_path = models_dir / pattern
            if full_path.exists():
                return full_path
        else:
            # Glob pattern matching
            matches = list(models_dir.glob(pattern))
            if matches:
                return matches[0]

    return None


@router.get("/models", response_model=ModelsResponse)
async def list_models(request: Request):
    """
    List all available models dynamically from database.

    Returns models found in:
    1. Model registry database (primary source - always fetches fresh)
    2. File system (models/ directory) as validation

    Each model includes:
    - model_id: Unique identifier
    - display_name: Human-readable name
    - status: "available", "loaded", or "not_found"
    - db_status: "registered" or "deployed" (raw database status)
    - observation_dim: Input dimension for the model
    """
    # Always reload from database to get fresh status
    await load_model_configs_from_db()

    inference_engine = request.app.state.inference_engine
    models_dir = Path("/models")  # Docker volume mount point

    # Fallback to local path if not in Docker
    if not models_dir.exists():
        from ..config import get_settings
        settings = get_settings()
        models_dir = settings.project_root / "models"

    models: List[ModelInfo] = []

    for model_id, config in MODEL_CONFIGS.items():
        model_file = find_model_file(model_id, models_dir)
        algorithm = config.get("algorithm", "PPO")

        # Determine runtime status
        if model_id in inference_engine.models:
            status = "loaded"
        elif algorithm == "SYNTHETIC":
            # Synthetic models (demo mode) don't need model files
            status = "available"
        elif model_file and model_file.exists():
            status = "available"
        else:
            status = "not_found"

        # Include all active models from database
        # Models marked 'active' in config.models should be shown
        models.append(ModelInfo(
            model_id=model_id,
            display_name=config["display_name"],
            version=config.get("version"),
            observation_dim=config["observation_dim"],
            description=config.get("description"),
            status=status,
            db_status=config.get("db_status", "active"),
        ))

    # Sort by display name
    models.sort(key=lambda m: m.display_name)

    return ModelsResponse(
        models=models,
        default_model="ppo_primary",
        total=len(models),
    )


@router.get("/models/{model_id}")
async def get_model_info(model_id: str, request: Request):
    """Get detailed info for a specific model"""
    inference_engine = request.app.state.inference_engine
    models_dir = Path("/models")

    if not models_dir.exists():
        from ..config import get_settings
        settings = get_settings()
        models_dir = settings.project_root / "models"

    config = MODEL_CONFIGS.get(model_id)
    if not config:
        return {"error": f"Unknown model: {model_id}", "available_models": list(MODEL_CONFIGS.keys())}

    model_file = find_model_file(model_id, models_dir)

    if model_id in inference_engine.models:
        status = "loaded"
        model_info = inference_engine.model_info.get(model_id, {})
    elif model_file and model_file.exists():
        status = "available"
        model_info = {"path": str(model_file)}
    else:
        status = "not_found"
        model_info = {}

    return {
        "model_id": model_id,
        "display_name": config["display_name"],
        "observation_dim": config["observation_dim"],
        "description": config.get("description"),
        "status": status,
        "file_path": str(model_file) if model_file else None,
        "info": model_info,
    }


@router.post("/models/{model_id}/promote", response_model=PromoteResponse)
async def promote_model(model_id: str, request: Request):
    """
    Promote a model to 'deployed' status.

    Only one model can be deployed at a time.
    The currently deployed model will be demoted to 'registered'.

    Flow:
    ┌────────────┐    promote()    ┌────────────┐
    │ registered │ ──────────────▶ │  deployed  │ (solo 1 activo)
    └────────────┘                 └────────────┘
    """
    global MODEL_CONFIGS

    try:
        import asyncpg
        from ..config import get_settings

        settings = get_settings()

        conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db,
        )

        try:
            # 1. Verify the model exists and is registered
            target = await conn.fetchrow(
                "SELECT model_id, status FROM model_registry WHERE model_id = $1",
                model_id
            )

            if not target:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{model_id}' not found in registry"
                )

            if target['status'] == 'deployed':
                return PromoteResponse(
                    success=True,
                    promoted_model=model_id,
                    previous_deployed=None,
                    message=f"Model '{model_id}' is already deployed"
                )

            if target['status'] not in ('registered', 'deployed'):
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{model_id}' has status '{target['status']}' and cannot be promoted"
                )

            # 2. Find currently deployed model (if any)
            current_deployed = await conn.fetchrow(
                "SELECT model_id FROM model_registry WHERE status = 'deployed'"
            )
            previous_deployed = current_deployed['model_id'] if current_deployed else None

            # 3. Transaction: demote old, promote new
            async with conn.transaction():
                # Demote current deployed to registered
                if previous_deployed:
                    await conn.execute(
                        "UPDATE model_registry SET status = 'registered' WHERE model_id = $1",
                        previous_deployed
                    )
                    logger.info(f"Demoted model '{previous_deployed}' to registered")

                # Promote target to deployed
                await conn.execute(
                    "UPDATE model_registry SET status = 'deployed' WHERE model_id = $1",
                    model_id
                )
                logger.info(f"Promoted model '{model_id}' to deployed")

            # 4. Reload model configs to reflect changes
            MODEL_CONFIGS = {}  # Force reload on next request
            load_model_configs_from_db()

            return PromoteResponse(
                success=True,
                promoted_model=model_id,
                previous_deployed=previous_deployed,
                message=f"Successfully promoted '{model_id}' to deployed" +
                        (f" (demoted '{previous_deployed}')" if previous_deployed else "")
            )

        finally:
            await conn.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to promote model: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to promote model: {str(e)}"
        )


@router.post("/models/reload", response_model=ReloadResponse)
async def reload_models(request: Request, background_tasks: BackgroundTasks):
    """
    Hot reload all models without service restart.

    This endpoint triggers a reload of all loaded models from their
    configured sources (MLflow, filesystem, or database registry).

    MLOps-3: Shadow Mode Support
    - Reloads both champion (production) and shadow (staging) models
    - Updates model versions from MLflow if ModelRouter is in use
    - Safe reload: keeps old model if new load fails

    Use cases:
    - After promoting a model in MLflow
    - After updating model files
    - After changing model registry entries
    - Recovering from model corruption

    Returns:
        ReloadResponse with status of each model reload
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    reloaded = []
    failed = []
    details = {}

    try:
        inference_engine = request.app.state.inference_engine

        # Check if we have a ModelRouter (shadow mode)
        model_router = getattr(request.app.state, 'model_router', None)

        if model_router is not None:
            # Use ModelRouter's reload (handles champion + shadow)
            logger.info("Reloading models via ModelRouter...")

            try:
                reload_result = model_router.reload_models()
                details["model_router"] = reload_result

                if reload_result.get("champion", {}).get("success"):
                    reloaded.append("champion")
                else:
                    failed.append("champion")

                if reload_result.get("shadow", {}).get("success"):
                    reloaded.append("shadow")
                elif model_router.enable_shadow:
                    failed.append("shadow")

            except Exception as e:
                logger.error(f"ModelRouter reload failed: {e}")
                failed.extend(["champion", "shadow"])
                details["model_router_error"] = str(e)

        # Also reload InferenceEngine models
        if hasattr(inference_engine, 'models') and inference_engine.models:
            logger.info("Reloading InferenceEngine models...")

            for model_id in list(inference_engine.models.keys()):
                try:
                    # Remove cached model to force reload
                    old_model = inference_engine.models.pop(model_id, None)

                    # Reload model
                    success = inference_engine.load_model(model_id)

                    if success:
                        reloaded.append(model_id)
                        logger.info(f"Reloaded model: {model_id}")
                    else:
                        failed.append(model_id)
                        # Restore old model if reload failed
                        if old_model is not None:
                            inference_engine.models[model_id] = old_model
                            logger.warning(f"Kept old version of {model_id} after reload failure")

                except Exception as e:
                    logger.error(f"Failed to reload model {model_id}: {e}")
                    failed.append(model_id)
                    details[f"{model_id}_error"] = str(e)

        # Refresh model configs from database
        try:
            load_model_configs_from_db()
            details["config_refresh"] = "success"
        except Exception as e:
            logger.warning(f"Could not refresh model configs: {e}")
            details["config_refresh_error"] = str(e)

        # Update metrics
        if failed:
            MODEL_RELOAD_TOTAL.labels(status="partial").inc()
            MODEL_RELOAD_FAILURES.inc()
        else:
            MODEL_RELOAD_TOTAL.labels(status="success").inc()

        success = len(failed) == 0
        message = (
            f"Reloaded {len(reloaded)} model(s)"
            if success
            else f"Partial reload: {len(reloaded)} succeeded, {len(failed)} failed"
        )

        return ReloadResponse(
            success=success,
            reloaded_models=reloaded,
            failed_models=failed,
            timestamp=timestamp,
            details=details,
            message=message,
        )

    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        MODEL_RELOAD_TOTAL.labels(status="failed").inc()
        MODEL_RELOAD_FAILURES.inc()
        raise HTTPException(
            status_code=500,
            detail=f"Model reload failed: {str(e)}"
        )


@router.get("/models/router/status")
async def get_router_status(request: Request):
    """
    Get ModelRouter status including shadow mode statistics.

    MLOps-3: Shadow Mode Monitoring
    Returns:
    - Champion model info and version
    - Shadow model info and version
    - Agreement rate between models
    - Divergence statistics
    """
    model_router = getattr(request.app.state, 'model_router', None)

    if model_router is None:
        return {
            "enabled": False,
            "message": "ModelRouter not configured. Using standard inference engine.",
        }

    try:
        status = model_router.get_status()
        return {
            "enabled": True,
            **status
        }
    except Exception as e:
        logger.error(f"Error getting router status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get router status: {str(e)}"
        )


# =========================================================================
# Phase 2.2: Rollback & Promote UI Endpoints
# =========================================================================


@router.get("/models/rollback-targets", response_model=RollbackTargetsResponse)
async def get_rollback_targets():
    """
    Get available models for rollback.

    Returns the last 5 archived models that can be used as rollback targets,
    along with their performance metrics.
    """
    try:
        import asyncpg
        from ..config import get_settings

        settings = get_settings()

        conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db,
        )

        try:
            # Get current production model
            current_prod = await conn.fetchrow("""
                SELECT model_id, model_version, status, created_at
                FROM model_registry
                WHERE status = 'deployed'
                LIMIT 1
            """)

            current_production = None
            if current_prod:
                current_production = {
                    "model_id": current_prod['model_id'],
                    "version": current_prod['model_version'],
                    "status": current_prod['status'],
                    "deployed_at": current_prod['created_at'].isoformat() if current_prod['created_at'] else None
                }

            # Get archived models as rollback targets
            archived = await conn.fetch("""
                SELECT
                    mr.model_id,
                    mr.model_version,
                    mr.status,
                    mr.created_at,
                    COALESCE(mm.sharpe_ratio, 0) as sharpe,
                    COALESCE(mm.win_rate, 0) as win_rate,
                    COALESCE(mm.max_drawdown, 0) as max_drawdown,
                    COALESCE(mm.total_trades, 0) as total_trades
                FROM model_registry mr
                LEFT JOIN model_metrics mm ON mr.model_id = mm.model_id
                WHERE mr.status IN ('registered', 'archived')
                ORDER BY mr.created_at DESC
                LIMIT 5
            """)

            available_targets = []
            for row in archived:
                available_targets.append(RollbackTarget(
                    model_id=row['model_id'],
                    version=row['model_version'] or 'v1',
                    archived_at=row['created_at'].isoformat() if row['created_at'] else '',
                    metrics={
                        "sharpe": float(row['sharpe'] or 0),
                        "win_rate": float(row['win_rate'] or 0),
                        "max_drawdown": float(row['max_drawdown'] or 0),
                        "total_trades": float(row['total_trades'] or 0)
                    }
                ))

            recommendation = available_targets[0] if available_targets else None

            return RollbackTargetsResponse(
                current_production=current_production,
                available_targets=available_targets,
                recommendation=recommendation
            )

        finally:
            await conn.close()

    except Exception as e:
        logger.error(f"Failed to get rollback targets: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get rollback targets: {str(e)}"
        )


@router.post("/models/rollback", response_model=RollbackResponse)
async def rollback_model(
    request_body: RollbackRequest,
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Rollback to previous model version.

    Process:
    1. Get current production model
    2. Find previous version (or specified version)
    3. Validate previous model exists and is valid
    4. Atomic swap: demote current, promote previous
    5. Reload inference engine
    6. Notify team
    7. Log to audit
    """
    import time
    start = time.time()

    try:
        import asyncpg
        from ..config import get_settings

        settings = get_settings()

        conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db,
        )

        try:
            # 1. Get current production model
            current = await conn.fetchrow("""
                SELECT model_id, model_version, status
                FROM model_registry
                WHERE status = 'deployed'
                LIMIT 1
            """)

            if not current:
                raise HTTPException(404, "No production model found")

            # 2. Get target version
            if request_body.target_version:
                target = await conn.fetchrow("""
                    SELECT model_id, model_version, status
                    FROM model_registry
                    WHERE model_version = $1 AND status IN ('registered', 'archived')
                    LIMIT 1
                """, str(request_body.target_version))
            else:
                # Get most recent non-deployed model
                target = await conn.fetchrow("""
                    SELECT model_id, model_version, status
                    FROM model_registry
                    WHERE status IN ('registered', 'archived')
                    ORDER BY created_at DESC
                    LIMIT 1
                """)

            if not target:
                raise HTTPException(404, "No rollback target found")

            # 3. Atomic rollback using transaction
            async with conn.transaction():
                # Demote current to archived
                await conn.execute("""
                    UPDATE model_registry
                    SET status = 'archived'
                    WHERE model_id = $1
                """, current['model_id'])

                # Promote target to deployed
                await conn.execute("""
                    UPDATE model_registry
                    SET status = 'deployed'
                    WHERE model_id = $1
                """, target['model_id'])

                # Log rollback event
                await conn.execute("""
                    INSERT INTO model_audit_log
                    (action, from_model, to_model, reason, initiated_by, timestamp)
                    VALUES ('rollback', $1, $2, $3, $4, NOW())
                """, current['model_id'], target['model_id'],
                    request_body.reason, request_body.initiated_by)

            # 4. Reload model configs
            load_model_configs_from_db()

            # 5. Notify team (async background task)
            background_tasks.add_task(
                _notify_rollback,
                current['model_id'],
                target['model_id'],
                request_body.reason,
                request_body.initiated_by
            )

            elapsed = (time.time() - start) * 1000

            logger.warning(
                f"ROLLBACK: {current['model_id']} -> {target['model_id']} "
                f"by {request_body.initiated_by}: {request_body.reason}"
            )

            return RollbackResponse(
                success=True,
                previous_model=current['model_id'],
                new_model=target['model_id'],
                rollback_time_ms=elapsed,
                message=f"Rollback completed in {elapsed:.0f}ms"
            )

        finally:
            await conn.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        raise HTTPException(500, f"Rollback failed: {str(e)}")


@router.post("/models/{model_id}/promote", response_model=PromoteRequestResponse)
async def promote_model_with_validation(
    model_id: str,
    request_body: PromoteRequest,
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Promote a model to staging or production with full validation.

    Validates:
    - Metrics thresholds (Sharpe, Win Rate, Max Drawdown, Total Trades)
    - Staging time requirements (7 days for production)
    - Checklist completion

    Different thresholds for staging vs production:
    - Staging: Sharpe >= 0.5, Win Rate >= 45%, Drawdown <= 15%, Trades >= 50
    - Production: Sharpe >= 1.0, Win Rate >= 50%, Drawdown <= 10%, Trades >= 100
    """
    from datetime import datetime, timezone, timedelta

    PROMOTION_THRESHOLDS = {
        "staging": {
            "min_sharpe": 0.5,
            "min_win_rate": 0.45,
            "max_drawdown": -0.15,
            "min_trades": 50,
        },
        "production": {
            "min_sharpe": 1.0,
            "min_win_rate": 0.50,
            "max_drawdown": -0.10,
            "min_trades": 100,
            "min_staging_days": 7,
        },
    }

    try:
        import asyncpg
        from ..config import get_settings

        settings = get_settings()

        conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db,
        )

        try:
            # Get model info and metrics
            model = await conn.fetchrow("""
                SELECT
                    mr.model_id,
                    mr.model_version,
                    mr.status,
                    mr.created_at,
                    COALESCE(mm.sharpe_ratio, 0) as sharpe,
                    COALESCE(mm.win_rate, 0) as win_rate,
                    COALESCE(mm.max_drawdown, 0) as max_drawdown,
                    COALESCE(mm.total_trades, 0) as total_trades
                FROM model_registry mr
                LEFT JOIN model_metrics mm ON mr.model_id = mm.model_id
                WHERE mr.model_id = $1
                LIMIT 1
            """, model_id)

            if not model:
                raise HTTPException(404, f"Model '{model_id}' not found")

            current_stage = model['status']
            target_stage = request_body.target_stage

            # Validate stage transition
            valid_transitions = {
                'registered': ['staging'],
                'staging': ['production', 'deployed'],
                'deployed': [],  # Already in production
            }

            # Normalize target stage
            if target_stage == 'production':
                target_stage = 'deployed'

            if target_stage not in valid_transitions.get(current_stage, []):
                raise HTTPException(400,
                    f"Cannot promote from '{current_stage}' to '{request_body.target_stage}'. "
                    f"Valid transitions: {valid_transitions.get(current_stage, [])}"
                )

            # Get thresholds
            threshold_key = 'staging' if target_stage == 'staging' else 'production'
            thresholds = PROMOTION_THRESHOLDS[threshold_key]

            # Validate metrics
            metrics_passed = True
            validation_errors = []

            if model['sharpe'] < thresholds['min_sharpe']:
                metrics_passed = False
                validation_errors.append(
                    f"Sharpe ({model['sharpe']:.2f}) < {thresholds['min_sharpe']}"
                )

            if model['win_rate'] < thresholds['min_win_rate']:
                metrics_passed = False
                validation_errors.append(
                    f"Win Rate ({model['win_rate']:.2%}) < {thresholds['min_win_rate']:.0%}"
                )

            if model['max_drawdown'] < thresholds['max_drawdown']:
                metrics_passed = False
                validation_errors.append(
                    f"Max Drawdown ({model['max_drawdown']:.2%}) < {thresholds['max_drawdown']:.0%}"
                )

            if model['total_trades'] < thresholds['min_trades']:
                metrics_passed = False
                validation_errors.append(
                    f"Total Trades ({model['total_trades']}) < {thresholds['min_trades']}"
                )

            # Check staging time for production promotion
            if target_stage == 'deployed' and not request_body.skip_staging_time:
                if current_stage == 'staging' and model['created_at']:
                    days_in_staging = (datetime.now(timezone.utc) - model['created_at'].replace(tzinfo=timezone.utc)).days
                    if days_in_staging < thresholds.get('min_staging_days', 7):
                        metrics_passed = False
                        validation_errors.append(
                            f"Days in staging ({days_in_staging}) < {thresholds['min_staging_days']}"
                        )

            if not metrics_passed:
                raise HTTPException(400,
                    f"Metrics validation failed: {', '.join(validation_errors)}"
                )

            # Perform promotion
            async with conn.transaction():
                # If promoting to production, demote current production
                if target_stage == 'deployed':
                    await conn.execute("""
                        UPDATE model_registry
                        SET status = 'registered'
                        WHERE status = 'deployed'
                    """)

                # Update target model status
                await conn.execute("""
                    UPDATE model_registry
                    SET status = $1
                    WHERE model_id = $2
                """, target_stage, model_id)

                # Log promotion event
                await conn.execute("""
                    INSERT INTO model_audit_log
                    (action, from_model, to_model, reason, initiated_by, timestamp)
                    VALUES ('promote', $1, $2, $3, $4, NOW())
                """, current_stage, target_stage, request_body.reason, request_body.promoted_by)

            # Reload configs
            load_model_configs_from_db()

            # Notify team
            background_tasks.add_task(
                _notify_promotion,
                model_id,
                request_body.target_stage,
                request_body.promoted_by
            )

            logger.info(
                f"PROMOTE: {model_id} from {current_stage} to {target_stage} "
                f"by {request_body.promoted_by}: {request_body.reason}"
            )

            return PromoteRequestResponse(
                success=True,
                model_id=model_id,
                new_stage=target_stage,
                previous_stage=current_stage,
                message=f"Successfully promoted {model_id} to {request_body.target_stage}",
                metrics_passed=metrics_passed,
                promoted_at=datetime.now(timezone.utc).isoformat()
            )

        finally:
            await conn.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Promotion failed: {e}")
        raise HTTPException(500, f"Promotion failed: {str(e)}")


# =========================================================================
# Helper Functions for Notifications
# =========================================================================


async def _notify_rollback(from_model: str, to_model: str, reason: str, initiated_by: str):
    """Send rollback notification to team."""
    try:
        # Try to use SlackClient if available
        try:
            from src.shared.notifications.slack_client import get_slack_client
            client = get_slack_client()
            await client.notify_rollback(from_model, to_model, reason, initiated_by)
        except ImportError:
            logger.info(f"Rollback notification: {from_model} -> {to_model} ({reason})")
    except Exception as e:
        logger.warning(f"Failed to send rollback notification: {e}")


async def _notify_promotion(model_id: str, stage: str, promoted_by: str):
    """Send promotion notification to team."""
    try:
        # Try to use SlackClient if available
        try:
            from src.shared.notifications.slack_client import get_slack_client
            client = get_slack_client()
            await client.notify_model_promotion(model_id, "registered", stage, promoted_by)
        except ImportError:
            logger.info(f"Promotion notification: {model_id} to {stage} by {promoted_by}")
    except Exception as e:
        logger.warning(f"Failed to send promotion notification: {e}")
