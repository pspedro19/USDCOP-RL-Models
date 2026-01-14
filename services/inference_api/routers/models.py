"""
Models Router
=============
Endpoint for listing available models from file system and database registry.
Includes model promotion system (registered -> deployed).
"""

import logging
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["models"])


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


# Model configurations loaded dynamically from database
# This dict is populated at runtime from model_registry table
MODEL_CONFIGS = {}


def load_model_configs_from_db():
    """Load model configurations from database model_registry table"""
    global MODEL_CONFIGS

    try:
        import asyncpg
        import asyncio
        from ..config import get_settings

        settings = get_settings()

        async def fetch_models():
            conn = await asyncpg.connect(
                host=settings.postgres_host,
                port=settings.postgres_port,
                user=settings.postgres_user,
                password=settings.postgres_password,
                database=settings.postgres_db,
            )
            try:
                rows = await conn.fetch("""
                    SELECT model_id, model_version, model_path, observation_dim, status
                    FROM model_registry
                    WHERE status IN ('registered', 'deployed')
                    ORDER BY CASE status WHEN 'deployed' THEN 0 ELSE 1 END
                """)
                return rows
            finally:
                await conn.close()

        rows = asyncio.get_event_loop().run_until_complete(fetch_models())

        # Clear existing configs to get fresh data
        MODEL_CONFIGS.clear()

        for row in rows:
            model_id = row['model_id']
            version = row['model_version'] or 'v1'
            db_status = row['status']  # 'registered' or 'deployed'
            is_deployed = db_status == 'deployed'

            MODEL_CONFIGS[model_id] = {
                "display_name": f"PPO {version.upper()} {'(Production)' if is_deployed else '(Testing)'}",
                "version": version,
                "db_status": db_status,  # Store raw database status
                "observation_dim": row['observation_dim'] or 15,
                "description": f"{row['observation_dim'] or 15}-feature model{'  (Production)' if is_deployed else ''}",
                "patterns": [row['model_path']] if row['model_path'] else [f"{model_id}*.zip"],
            }

        logger.info(f"Loaded {len(MODEL_CONFIGS)} models from database: {list(MODEL_CONFIGS.keys())}")

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
    load_model_configs_from_db()

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

        # Determine runtime status
        if model_id in inference_engine.models:
            status = "loaded"
        elif model_file and model_file.exists():
            status = "available"
        else:
            status = "not_found"

        # Only include models that exist
        if status != "not_found":
            models.append(ModelInfo(
                model_id=model_id,
                display_name=config["display_name"],
                version=config.get("version"),
                observation_dim=config["observation_dim"],
                description=config.get("description"),
                status=status,
                db_status=config.get("db_status", "registered"),  # Include raw DB status
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
