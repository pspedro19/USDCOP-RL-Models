"""
Image serving endpoints - Serve forecast and backtest images from MinIO and local storage.

This module provides REST API endpoints for retrieving visualization images
including forecast plots, backtest results, heatmaps, and model comparisons
for the USD/COP forecasting pipeline.

All endpoints require authentication.
"""
from fastapi import APIRouter, HTTPException, Query, Depends, Path
from fastapi.responses import StreamingResponse, Response
from typing import Optional, List, Dict, Any
import os
import logging
import glob
from io import BytesIO
from enum import Enum

from ..auth.dependencies import get_current_user, User
from ..config import settings, get_minio_client

router = APIRouter()
logger = logging.getLogger(__name__)


class ImageTypeEnum(str, Enum):
    """Supported image types."""
    FORWARD_FORECAST = "forward_forecast"
    BACKTEST = "backtest"
    HEATMAP = "heatmap"
    RANKING = "ranking"
    COMPARISON = "comparison"


@router.get(
    "/forecast/{year}/{week}/{filename}",
    summary="Get forecast visualization image",
    description="""
    Retrieve a forecast visualization image from MinIO or local storage.

    Images are first looked up in MinIO under the path:
    `forecasts/{year}/week{week}/figures/{filename}`

    If not found, falls back to local file system search.

    **Requires authentication.**
    """,
    response_description="PNG image binary data",
    responses={
        200: {
            "description": "Image retrieved successfully",
            "content": {
                "image/png": {
                    "schema": {"type": "string", "format": "binary"}
                }
            }
        },
        401: {
            "description": "Not authenticated"
        },
        404: {
            "description": "Image not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Image not found: forward_forecast_ridge.png"}
                }
            }
        }
    }
)
async def get_forecast_image(
    year: int = Path(
        ...,
        description="Year of the forecast",
        example=2024,
        ge=2020,
        le=2030
    ),
    week: int = Path(
        ...,
        description="Week number (1-52)",
        example=3,
        ge=1,
        le=53
    ),
    filename: str = Path(
        ...,
        description="Image filename (e.g., 'forward_forecast_ridge.png')",
        example="forward_forecast_ridge.png"
    ),
    current_user: User = Depends(get_current_user)
):
    """
    Get a forecast image from MinIO or local storage.

    First attempts to retrieve from MinIO, then falls back to local
    file system if not found.

    Args:
        year: Year of the forecast
        week: Week number
        filename: Image filename
        current_user: Authenticated user (injected by dependency)

    Returns:
        Response with PNG image data

    Raises:
        HTTPException: 404 if image not found
    """
    client = get_minio_client()

    if client:
        try:
            bucket = "forecasts"
            object_name = f"{year}/week{week:02d}/figures/{filename}"

            # Get object from MinIO
            response = client.get_object(bucket, object_name)
            data = response.read()
            response.close()

            # Determine content-type
            content_type = "image/png" if filename.endswith(".png") else "application/octet-stream"

            return Response(
                content=data,
                media_type=content_type,
                headers={
                    "Cache-Control": "public, max-age=3600",
                    "X-Source": "minio"
                }
            )

        except Exception as e:
            logger.warning(f"MinIO error: {e}, trying local fallback")

    # Fallback: search in local file system
    local_paths = [
        f"{settings.OUTPUTS_PATH}/runs/*/figures/{filename}",
        f"{settings.OUTPUTS_PATH}/weekly/figures/{filename}",
        f"{settings.OUTPUTS_PATH}/bi/{filename}",
    ]

    for pattern in local_paths:
        matches = glob.glob(pattern)
        if matches:
            # Use most recent
            matches.sort(reverse=True)
            file_path = matches[0]

            with open(file_path, 'rb') as f:
                data = f.read()

            return Response(
                content=data,
                media_type="image/png",
                headers={
                    "Cache-Control": "public, max-age=3600",
                    "X-Source": "local"
                }
            )

    raise HTTPException(status_code=404, detail=f"Image not found: {filename}")


@router.get(
    "/backtest/{model}/{horizon}/{filename}",
    summary="Get backtest visualization image",
    description="""
    Retrieve a backtest visualization image for a specific model and horizon.

    First searches in MinIO under the `ml-models` bucket, then falls back
    to local file system.

    **Requires authentication.**
    """,
    response_description="PNG image binary data",
    responses={
        200: {
            "description": "Image retrieved successfully",
            "content": {
                "image/png": {
                    "schema": {"type": "string", "format": "binary"}
                }
            }
        },
        401: {
            "description": "Not authenticated"
        },
        404: {
            "description": "Backtest image not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Backtest image not found: backtest_ridge_h5.png"}
                }
            }
        }
    }
)
async def get_backtest_image(
    model: str = Path(
        ...,
        description="Model name",
        example="ridge"
    ),
    horizon: int = Path(
        ...,
        description="Forecast horizon in days",
        example=5,
        ge=1,
        le=60
    ),
    filename: str = Path(
        ...,
        description="Image filename",
        example="backtest_ridge_h5.png"
    ),
    current_user: User = Depends(get_current_user)
):
    """
    Get a backtest image from MinIO or local storage.

    Args:
        model: Model name
        horizon: Forecast horizon in days
        filename: Image filename
        current_user: Authenticated user (injected by dependency)

    Returns:
        Response with PNG image data

    Raises:
        HTTPException: 404 if image not found
    """
    client = get_minio_client()

    # Try MinIO first
    if client:
        try:
            bucket = "ml-models"
            # Search in the most recent run
            objects = list(client.list_objects(bucket, prefix="", recursive=True))

            for obj in objects:
                if filename in obj.object_name:
                    response = client.get_object(bucket, obj.object_name)
                    data = response.read()
                    response.close()

                    return Response(
                        content=data,
                        media_type="image/png",
                        headers={"X-Source": "minio"}
                    )

        except Exception as e:
            logger.warning(f"MinIO backtest error: {e}")

    # Fallback to local
    patterns = [
        f"{settings.OUTPUTS_PATH}/runs/*/figures/{filename}",
        f"{settings.OUTPUTS_PATH}/runs/*/figures/backtest_{model}_h{horizon}.png",
    ]

    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            matches.sort(reverse=True)
            with open(matches[0], 'rb') as f:
                data = f.read()

            return Response(
                content=data,
                media_type="image/png",
                headers={"X-Source": "local"}
            )

    raise HTTPException(status_code=404, detail=f"Backtest image not found: {filename}")


@router.get(
    "/latest/{image_type}",
    summary="Get the most recent image of a specific type",
    description="""
    Retrieve the most recent image of a specific visualization type.

    Supported image types:
    - **forward_forecast**: Forward forecast plots (optionally filtered by model)
    - **backtest**: Backtest result plots (optionally filtered by model and horizon)
    - **heatmap**: Direction accuracy heatmap
    - **ranking**: Model ranking visualization
    - **comparison**: Model comparison chart

    **Requires authentication.**
    """,
    response_description="PNG image binary data",
    responses={
        200: {
            "description": "Image retrieved successfully",
            "content": {
                "image/png": {
                    "schema": {"type": "string", "format": "binary"}
                }
            }
        },
        401: {
            "description": "Not authenticated"
        },
        404: {
            "description": "No images found for the specified type",
            "content": {
                "application/json": {
                    "example": {"detail": "No images found for type: forward_forecast"}
                }
            }
        }
    }
)
async def get_latest_image(
    image_type: str = Path(
        ...,
        description="Type of image to retrieve",
        example="forward_forecast"
    ),
    model: Optional[str] = Query(
        None,
        description="Filter by model name (for forward_forecast and backtest)",
        example="ridge"
    ),
    horizon: Optional[int] = Query(
        None,
        description="Filter by horizon (for backtest only)",
        example=5,
        ge=1,
        le=60
    ),
    current_user: User = Depends(get_current_user)
):
    """
    Get the most recent image of a specific type.

    Searches local file system for the most recently created image
    matching the specified type and optional filters.

    Args:
        image_type: Type of visualization (forward_forecast, backtest, heatmap, ranking, comparison)
        model: Optional model filter
        horizon: Optional horizon filter (for backtest)
        current_user: Authenticated user (injected by dependency)

    Returns:
        Response with PNG image data

    Raises:
        HTTPException: 404 if no matching images found
    """
    # Build pattern based on type
    if image_type == "forward_forecast":
        if model:
            pattern = f"{settings.OUTPUTS_PATH}/runs/*/figures/forward_forecast_{model.lower()}.png"
        else:
            pattern = f"{settings.OUTPUTS_PATH}/runs/*/figures/forward_forecast_all_models.png"

    elif image_type == "backtest":
        if model and horizon:
            pattern = f"{settings.OUTPUTS_PATH}/runs/*/figures/backtest_{model.lower()}_h{horizon}.png"
        else:
            pattern = f"{settings.OUTPUTS_PATH}/runs/*/figures/backtest_*.png"

    elif image_type == "heatmap":
        pattern = f"{settings.OUTPUTS_PATH}/runs/*/figures/metrics_heatmap_da.png"

    elif image_type == "ranking":
        pattern = f"{settings.OUTPUTS_PATH}/runs/*/figures/model_ranking_da.png"

    elif image_type == "comparison":
        pattern = f"{settings.OUTPUTS_PATH}/runs/*/figures/models_comparison.png"

    else:
        pattern = f"{settings.OUTPUTS_PATH}/runs/*/figures/{image_type}*.png"

    matches = glob.glob(pattern)

    if not matches:
        raise HTTPException(status_code=404, detail=f"No images found for type: {image_type}")

    # Use most recent
    matches.sort(reverse=True)
    file_path = matches[0]

    with open(file_path, 'rb') as f:
        data = f.read()

    return Response(
        content=data,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=300",
            "X-Source": "local",
            "X-File-Path": file_path
        }
    )


@router.get(
    "/list/{year}/{week}",
    summary="List all available images for a week",
    description="""
    List all visualization images available for a specific week.

    Searches both MinIO and local storage, returning metadata for each
    found image including name, path, size, source, and access URL.

    **Requires authentication.**
    """,
    response_description="List of available images with metadata",
    responses={
        200: {
            "description": "Image list retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "year": 2024,
                        "week": 3,
                        "count": 5,
                        "images": [
                            {
                                "name": "forward_forecast_ridge.png",
                                "path": "forecasts/2024/week03/figures/forward_forecast_ridge.png",
                                "size": 125430,
                                "source": "minio",
                                "url": "/api/images/forecast/2024/3/forward_forecast_ridge.png"
                            }
                        ]
                    }
                }
            }
        },
        401: {
            "description": "Not authenticated"
        }
    }
)
async def list_week_images(
    year: int = Path(
        ...,
        description="Year",
        example=2024,
        ge=2020,
        le=2030
    ),
    week: int = Path(
        ...,
        description="Week number (1-52)",
        example=3,
        ge=1,
        le=53
    ),
    current_user: User = Depends(get_current_user)
):
    """
    List all images available for a specific week.

    Combines images from both MinIO and local storage, avoiding duplicates.

    Args:
        year: Year
        week: Week number
        current_user: Authenticated user (injected by dependency)

    Returns:
        Dictionary with year, week, count, and list of images
    """
    images = []

    # Search in MinIO
    client = get_minio_client()
    if client:
        try:
            bucket = "forecasts"
            prefix = f"{year}/week{week:02d}/figures/"

            objects = client.list_objects(bucket, prefix=prefix, recursive=True)

            for obj in objects:
                if obj.object_name.endswith('.png'):
                    images.append({
                        "name": obj.object_name.split('/')[-1],
                        "path": obj.object_name,
                        "size": obj.size,
                        "source": "minio",
                        "url": f"/api/images/forecast/{year}/{week}/{obj.object_name.split('/')[-1]}"
                    })

        except Exception as e:
            logger.warning(f"MinIO list error: {e}")

    # Also search in local
    local_pattern = f"{settings.OUTPUTS_PATH}/runs/*/figures/*.png"

    for file_path in glob.glob(local_pattern):
        filename = os.path.basename(file_path)
        if not any(img['name'] == filename for img in images):
            images.append({
                "name": filename,
                "path": file_path,
                "size": os.path.getsize(file_path),
                "source": "local",
                "url": f"/api/images/latest/{filename.replace('.png', '')}"
            })

    return {
        "year": year,
        "week": week,
        "count": len(images),
        "images": images
    }


@router.get(
    "/local/{path:path}",
    summary="Get image from local storage with intelligent path resolution",
    description="""
    Retrieve an image from local storage using flexible path resolution.

    Accepts any path (e.g., `weekly_update/figures/foo.png` or `results/ml_pipeline/figures/bar.png`)
    and intelligently resolves it by:
    1. Extracting the filename from the path
    2. Searching in `outputs/weekly/figures/{filename}` (for forward forecasts)
    3. Searching in `outputs/runs/*/figures/{filename}` (for backtests, using most recent run)
    4. Searching in `outputs/bi/{filename}` (for BI dashboard images)

    This endpoint is designed to handle paths from CSV data sources that may not
    match the actual storage structure.

    **Requires authentication.**
    """,
    response_description="Image binary data",
    responses={
        200: {
            "description": "Image retrieved successfully",
            "content": {
                "image/png": {
                    "schema": {"type": "string", "format": "binary"}
                }
            }
        },
        401: {
            "description": "Not authenticated"
        },
        404: {
            "description": "Image not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Image not found: forward_forecast_ridge.png"}
                }
            }
        }
    }
)
async def get_local_image(
    path: str = Path(
        ...,
        description="Path to the image (e.g., 'weekly_update/figures/forward_forecast_ridge.png')",
        example="weekly_update/figures/forward_forecast_ridge.png"
    ),
    current_user: User = Depends(get_current_user)
):
    """
    Get an image from MinIO or local storage with intelligent path resolution.

    First searches in MinIO buckets, then falls back to local filesystem.
    Extracts the filename from the provided path and searches in multiple
    locations to find the actual file.

    Args:
        path: The path to the image (may be a CSV-style path)
        current_user: Authenticated user (injected by dependency)

    Returns:
        Response with image data

    Raises:
        HTTPException: 404 if image not found
    """
    # Extract just the filename from the path
    filename = os.path.basename(path)

    if not filename:
        raise HTTPException(status_code=400, detail="Invalid path: no filename found")

    # Determine content type based on extension
    ext = os.path.splitext(filename)[1].lower()
    content_type_map = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.svg': 'image/svg+xml',
        '.webp': 'image/webp'
    }
    content_type = content_type_map.get(ext, 'application/octet-stream')

    # 1. FIRST: Try exact path if it contains a date (e.g., forecasts/2025-12-29/forward_forecast_ridge.png)
    # This ensures week-specific forecast images are served correctly
    import re
    date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')
    if date_pattern.search(path):
        exact_path = f"{settings.OUTPUTS_PATH}/{path}"
        if os.path.isfile(exact_path):
            try:
                with open(exact_path, 'rb') as f:
                    data = f.read()
                logger.info(f"Serving date-specific image: {exact_path}")
                return Response(
                    content=data,
                    media_type=content_type,
                    headers={
                        "Cache-Control": "public, max-age=3600",
                        "X-Source": "local-exact",
                        "X-Resolved-Path": exact_path
                    }
                )
            except Exception as e:
                logger.warning(f"Error reading exact path {exact_path}: {e}")

    # 2. Try LOCAL filesystem FIRST (for backtest images which are updated locally)
    # This ensures locally updated images are served immediately without MinIO cache issues
    search_patterns = [
        # 1. Direct path under outputs (try exact match first)
        f"{settings.OUTPUTS_PATH}/{path}",
        # 2. Run-specific outputs (backtests) - most recent first
        f"{settings.OUTPUTS_PATH}/runs/*/figures/{filename}",
        # 3. Forecast outputs by date (for forward forecasts and consolidated images)
        f"{settings.OUTPUTS_PATH}/forecasts/*/{filename}",
        # 4. Weekly outputs (forward forecasts)
        f"{settings.OUTPUTS_PATH}/weekly/figures/{filename}",
        f"{settings.OUTPUTS_PATH}/weekly/figures/*/{filename}",
        f"{settings.OUTPUTS_PATH}/weekly_update/figures/{filename}",
        # 5. BI dashboard images
        f"{settings.OUTPUTS_PATH}/bi/{filename}",
        f"{settings.OUTPUTS_PATH}/bi/figures/{filename}",
        f"{settings.OUTPUTS_PATH}/figures/{filename}",
    ]

    for pattern in search_patterns:
        if '*' in pattern:
            # Glob pattern - find all matches and use most recent
            matches = glob.glob(pattern)
            if matches:
                # Sort by modification time (most recent first)
                matches.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                file_path = matches[0]

                try:
                    with open(file_path, 'rb') as f:
                        data = f.read()

                    logger.info(f"Serving image from: {file_path}")
                    return Response(
                        content=data,
                        media_type=content_type,
                        headers={
                            "Cache-Control": "public, max-age=3600",
                            "X-Source": "local",
                            "X-Resolved-Path": file_path
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {e}")
                    continue
        else:
            # Direct path - check if exists
            if os.path.isfile(pattern):
                try:
                    with open(pattern, 'rb') as f:
                        data = f.read()

                    logger.info(f"Serving image from: {pattern}")
                    return Response(
                        content=data,
                        media_type=content_type,
                        headers={
                            "Cache-Control": "public, max-age=3600",
                            "X-Source": "local",
                            "X-Resolved-Path": pattern
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error reading {pattern}: {e}")
                    continue

    # 2. Fallback to MinIO if not found locally
    client = get_minio_client()
    if client:
        try:
            buckets = ["forecasts", "ml-models"]
            for bucket in buckets:
                if bucket == "forecasts" and "backtest" in filename: continue
                if bucket == "ml-models" and "forward_forecast" in filename: continue
                objects = client.list_objects(bucket, recursive=True)
                for obj in objects:
                    if obj.object_name.endswith(filename):
                        response = client.get_object(bucket, obj.object_name)
                        data = response.read()
                        response.close()
                        logger.info(f"Serving image from MinIO: {bucket}/{obj.object_name}")
                        return Response(
                            content=data,
                            media_type=content_type,
                            headers={
                                "Cache-Control": "public, max-age=3600",
                                "X-Source": "minio",
                                "X-Bucket": bucket,
                                "X-Object": obj.object_name
                            }
                        )
        except Exception as e:
            logger.debug(f"MinIO lookup failed: {e}")

    logger.warning(f"Image not found: {filename} (original path: {path})")
    raise HTTPException(status_code=404, detail=f"Image not found: {filename}")


@router.get(
    "/types",
    summary="Get list of available image types",
    description="""
    Get a list of all supported image types that can be retrieved.

    **Requires authentication.**
    """,
    response_description="List of image types with descriptions",
    responses={
        200: {
            "description": "Image types list",
            "content": {
                "application/json": {
                    "example": {
                        "types": [
                            {"id": "forward_forecast", "name": "Forward Forecast", "description": "Future price predictions"},
                            {"id": "backtest", "name": "Backtest", "description": "Historical validation results"}
                        ]
                    }
                }
            }
        },
        401: {
            "description": "Not authenticated"
        }
    }
)
async def get_image_types(current_user: User = Depends(get_current_user)):
    """
    Get list of available image types.

    Args:
        current_user: Authenticated user (injected by dependency)

    Returns:
        Dictionary with list of image types
    """
    return {
        "types": [
            {
                "id": "forward_forecast",
                "name": "Forward Forecast",
                "description": "Visualization of future price predictions by model"
            },
            {
                "id": "backtest",
                "name": "Backtest",
                "description": "Historical validation results showing predicted vs actual values"
            },
            {
                "id": "heatmap",
                "name": "Direction Accuracy Heatmap",
                "description": "Heatmap showing direction accuracy by model and horizon"
            },
            {
                "id": "ranking",
                "name": "Model Ranking",
                "description": "Bar chart ranking models by performance"
            },
            {
                "id": "comparison",
                "name": "Model Comparison",
                "description": "Side-by-side comparison of model predictions"
            }
        ]
    }
