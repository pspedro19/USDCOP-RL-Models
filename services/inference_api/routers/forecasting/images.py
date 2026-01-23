"""
Images Router
=============

Endpoints for serving forecast visualization images.

@version 1.0.0
"""

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import FileResponse
from typing import Optional, List
import os
import logging
from glob import glob

from services.inference_api.contracts.forecasting import (
    ImageMetadata,
    ImageListResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Image paths
IMAGES_PATH = os.environ.get('FORECASTING_IMAGES_PATH', 'public/forecasting')


def get_image_path(image_type: str, model: str, horizon: Optional[int] = None) -> Optional[str]:
    """
    Construct image path based on type and model.

    Naming convention:
    - Backtest: backtest_{model}_h{horizon}.png
    - Forecast: forward_forecast_{model}.png
    - Heatmap: heatmap_{model}.png
    """
    if image_type == "backtest" and horizon:
        filename = f"backtest_{model}_h{horizon}.png"
    elif image_type == "forecast":
        filename = f"forward_forecast_{model}.png"
    elif image_type == "heatmap":
        filename = f"heatmap_{model}.png"
    else:
        return None

    path = os.path.join(IMAGES_PATH, filename)
    return path if os.path.exists(path) else None


def list_available_images() -> List[ImageMetadata]:
    """List all available forecast images."""
    images = []

    if not os.path.exists(IMAGES_PATH):
        return images

    # Find all PNG files
    png_files = glob(os.path.join(IMAGES_PATH, "*.png"))

    for filepath in png_files:
        filename = os.path.basename(filepath)
        stat = os.stat(filepath)

        # Parse filename to extract metadata
        image_type = "unknown"
        model_id = "unknown"
        horizon_id = None

        if filename.startswith("backtest_"):
            image_type = "backtest"
            # Extract model and horizon: backtest_ridge_h5.png
            parts = filename.replace(".png", "").split("_")
            if len(parts) >= 3:
                model_id = parts[1]
                horizon_part = parts[-1]
                if horizon_part.startswith("h"):
                    try:
                        horizon_id = int(horizon_part[1:])
                    except ValueError:
                        pass
        elif filename.startswith("forward_forecast_"):
            image_type = "forecast"
            model_id = filename.replace("forward_forecast_", "").replace(".png", "")
        elif filename.startswith("heatmap_"):
            image_type = "heatmap"
            model_id = filename.replace("heatmap_", "").replace(".png", "")

        images.append(ImageMetadata(
            image_type=image_type,
            model_id=model_id,
            horizon_id=horizon_id,
            filename=filename,
            url=f"/api/v1/forecasting/images/{image_type}/{model_id}" + (f"/{horizon_id}" if horizon_id else ""),
            size=stat.st_size,
            last_modified=str(stat.st_mtime),
        ))

    return images


@router.get(
    "/",
    response_model=ImageListResponse,
    summary="List available images",
)
async def get_images_list():
    """List all available forecast visualization images."""
    images = list_available_images()
    return {"images": images, "count": len(images)}


@router.get(
    "/backtest/{model}/{horizon}",
    summary="Get backtest image for model and horizon",
    response_class=FileResponse,
)
async def get_backtest_image(
    model: str = Path(..., description="Model name"),
    horizon: int = Path(..., ge=1, le=60, description="Horizon"),
):
    """Get backtest visualization for specific model/horizon."""
    filepath = get_image_path("backtest", model, horizon)

    if not filepath:
        raise HTTPException(
            status_code=404,
            detail=f"Backtest image not found for {model} h{horizon}"
        )

    return FileResponse(filepath, media_type="image/png")


@router.get(
    "/forecast/{model}",
    summary="Get forecast image for model",
    response_class=FileResponse,
)
async def get_forecast_image(
    model: str = Path(..., description="Model name"),
):
    """Get forward forecast visualization for a model."""
    filepath = get_image_path("forecast", model)

    if not filepath:
        raise HTTPException(
            status_code=404,
            detail=f"Forecast image not found for {model}"
        )

    return FileResponse(filepath, media_type="image/png")


@router.get(
    "/heatmap/{model}",
    summary="Get heatmap image for model",
    response_class=FileResponse,
)
async def get_heatmap_image(
    model: str = Path(..., description="Model name"),
):
    """Get performance heatmap for a model."""
    filepath = get_image_path("heatmap", model)

    if not filepath:
        raise HTTPException(
            status_code=404,
            detail=f"Heatmap image not found for {model}"
        )

    return FileResponse(filepath, media_type="image/png")


@router.get(
    "/by-type/{image_type}",
    response_model=ImageListResponse,
    summary="Get images by type",
)
async def get_images_by_type(
    image_type: str = Path(..., description="Image type: backtest, forecast, heatmap"),
):
    """Get all images of a specific type."""
    all_images = list_available_images()
    filtered = [img for img in all_images if img.image_type == image_type]
    return {"images": filtered, "count": len(filtered)}
