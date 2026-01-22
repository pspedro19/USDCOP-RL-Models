"""
Health check endpoints - Real connectivity verification for all services.

This module provides health check endpoints that verify actual connectivity
to PostgreSQL, MinIO, and MLflow services, returning detailed status information
including latency measurements.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import Dict, Any, Optional
import os
import time
import logging
import httpx

from ..config import settings, parse_database_url, get_minio_client

router = APIRouter()
logger = logging.getLogger(__name__)


async def check_postgresql() -> Dict[str, Any]:
    """
    Check PostgreSQL connectivity by executing SELECT 1.

    Returns:
        Dictionary with status, latency_ms, and optional error message
    """
    start_time = time.time()

    try:
        import psycopg2

        db_config = parse_database_url(settings.DATABASE_URL)

        conn = psycopg2.connect(
            host=db_config["host"],
            port=db_config["port"],
            database=db_config["database"],
            user=db_config["user"],
            password=db_config["password"],
            connect_timeout=5
        )

        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        latency_ms = round((time.time() - start_time) * 1000, 2)

        if result and result[0] == 1:
            return {
                "status": "healthy",
                "latency_ms": latency_ms,
                "details": {
                    "host": db_config["host"],
                    "port": db_config["port"],
                    "database": db_config["database"]
                }
            }
        else:
            return {
                "status": "unhealthy",
                "latency_ms": latency_ms,
                "error": "Unexpected query result"
            }

    except ImportError:
        return {
            "status": "unhealthy",
            "latency_ms": round((time.time() - start_time) * 1000, 2),
            "error": "psycopg2 package not installed"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "latency_ms": round((time.time() - start_time) * 1000, 2),
            "error": str(e)
        }


async def check_minio() -> Dict[str, Any]:
    """
    Check MinIO connectivity by listing buckets.

    Returns:
        Dictionary with status, latency_ms, bucket count, and optional error
    """
    start_time = time.time()

    try:
        from minio import Minio
        from minio.error import S3Error

        client = Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=False
        )

        # List buckets to verify connectivity
        buckets = client.list_buckets()
        bucket_names = [b.name for b in buckets]

        latency_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "status": "healthy",
            "latency_ms": latency_ms,
            "details": {
                "endpoint": settings.MINIO_ENDPOINT,
                "bucket_count": len(bucket_names),
                "buckets": bucket_names[:10]  # Limit to first 10 buckets
            }
        }

    except ImportError:
        return {
            "status": "unhealthy",
            "latency_ms": round((time.time() - start_time) * 1000, 2),
            "error": "minio package not installed"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "latency_ms": round((time.time() - start_time) * 1000, 2),
            "error": str(e)
        }


async def check_mlflow() -> Dict[str, Any]:
    """
    Check MLflow connectivity by calling GET /health or /api/2.0/mlflow/experiments/list.

    Returns:
        Dictionary with status, latency_ms, and optional error
    """
    start_time = time.time()

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Try health endpoint first
            try:
                response = await client.get(f"{settings.MLFLOW_TRACKING_URI}/health")
                if response.status_code == 200:
                    latency_ms = round((time.time() - start_time) * 1000, 2)
                    return {
                        "status": "healthy",
                        "latency_ms": latency_ms,
                        "details": {
                            "tracking_uri": settings.MLFLOW_TRACKING_URI,
                            "endpoint_checked": "/health"
                        }
                    }
            except httpx.HTTPError:
                pass

            # Fallback: try experiments list endpoint
            response = await client.get(f"{settings.MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/list")
            latency_ms = round((time.time() - start_time) * 1000, 2)

            if response.status_code == 200:
                data = response.json()
                experiment_count = len(data.get("experiments", []))
                return {
                    "status": "healthy",
                    "latency_ms": latency_ms,
                    "details": {
                        "tracking_uri": settings.MLFLOW_TRACKING_URI,
                        "endpoint_checked": "/api/2.0/mlflow/experiments/list",
                        "experiment_count": experiment_count
                    }
                }
            else:
                return {
                    "status": "unhealthy",
                    "latency_ms": latency_ms,
                    "error": f"MLflow returned status code {response.status_code}"
                }

    except httpx.TimeoutException:
        return {
            "status": "unhealthy",
            "latency_ms": round((time.time() - start_time) * 1000, 2),
            "error": "Connection timeout"
        }
    except httpx.ConnectError:
        return {
            "status": "unhealthy",
            "latency_ms": round((time.time() - start_time) * 1000, 2),
            "error": f"Cannot connect to MLflow at {settings.MLFLOW_TRACKING_URI}"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "latency_ms": round((time.time() - start_time) * 1000, 2),
            "error": str(e)
        }


@router.get(
    "/health",
    summary="Basic health check",
    description="Returns basic health status of the API service. Does not check external dependencies.",
    response_description="Service health status",
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "timestamp": "2024-01-15T10:30:00.000000",
                        "service": "usdcop-forecasting-api",
                        "version": "1.0.0"
                    }
                }
            }
        }
    }
)
async def health_check():
    """
    Basic health check endpoint.

    Returns the service status without checking external dependencies.
    Use /health/ready for a complete readiness check including all dependencies.

    Returns:
        JSON with status, timestamp, service name, and version
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "usdcop-forecasting-api",
        "version": "1.0.0",
    }


@router.get(
    "/health/ready",
    summary="Readiness check with dependency verification",
    description="""
    Comprehensive readiness check that verifies connectivity to all external dependencies:

    - **PostgreSQL**: Executes SELECT 1 to verify database connectivity
    - **MinIO**: Lists buckets to verify object storage connectivity
    - **MLflow**: Calls /health endpoint to verify ML tracking service

    Returns 503 Service Unavailable if any dependency check fails.
    Includes latency measurements for each service.
    """,
    response_description="Detailed readiness status with per-service latencies",
    responses={
        200: {
            "description": "All services are healthy and ready",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ready",
                        "timestamp": "2024-01-15T10:30:00.000000",
                        "total_latency_ms": 45.5,
                        "checks": {
                            "postgresql": {
                                "status": "healthy",
                                "latency_ms": 12.3,
                                "details": {
                                    "host": "localhost",
                                    "port": "5432",
                                    "database": "pipeline_db"
                                }
                            },
                            "minio": {
                                "status": "healthy",
                                "latency_ms": 8.7,
                                "details": {
                                    "endpoint": "localhost:9000",
                                    "bucket_count": 3,
                                    "buckets": ["forecasts", "ml-models", "data"]
                                }
                            },
                            "mlflow": {
                                "status": "healthy",
                                "latency_ms": 24.5,
                                "details": {
                                    "tracking_uri": "http://localhost:5000",
                                    "endpoint_checked": "/health"
                                }
                            }
                        }
                    }
                }
            }
        },
        503: {
            "description": "One or more services are unhealthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "not_ready",
                        "timestamp": "2024-01-15T10:30:00.000000",
                        "total_latency_ms": 5023.4,
                        "failed_services": ["postgresql"],
                        "checks": {
                            "postgresql": {
                                "status": "unhealthy",
                                "latency_ms": 5001.2,
                                "error": "Connection refused"
                            },
                            "minio": {
                                "status": "healthy",
                                "latency_ms": 10.1,
                                "details": {}
                            },
                            "mlflow": {
                                "status": "healthy",
                                "latency_ms": 12.1,
                                "details": {}
                            }
                        }
                    }
                }
            }
        }
    }
)
async def readiness_check():
    """
    Readiness check - verifies all dependencies are available.

    Performs real connectivity checks to:
    - PostgreSQL: Executes SELECT 1
    - MinIO: Lists buckets
    - MLflow: GET /health

    Returns 503 if any service is unhealthy.

    Returns:
        JSON with overall status, timestamp, latencies, and detailed checks
    """
    start_time = time.time()

    # Run all checks
    postgresql_check = await check_postgresql()
    minio_check = await check_minio()
    mlflow_check = await check_mlflow()

    total_latency_ms = round((time.time() - start_time) * 1000, 2)

    checks = {
        "postgresql": postgresql_check,
        "minio": minio_check,
        "mlflow": mlflow_check,
    }

    # Determine overall status
    failed_services = [
        name for name, check in checks.items()
        if check["status"] == "unhealthy"
    ]

    all_healthy = len(failed_services) == 0

    response_data = {
        "status": "ready" if all_healthy else "not_ready",
        "timestamp": datetime.utcnow().isoformat(),
        "total_latency_ms": total_latency_ms,
        "checks": checks,
    }

    if not all_healthy:
        response_data["failed_services"] = failed_services
        return JSONResponse(
            status_code=503,
            content=response_data
        )

    return response_data


@router.get(
    "/health/live",
    summary="Liveness probe",
    description="Simple liveness probe for Kubernetes. Returns 200 if the service is running.",
    response_description="Liveness status",
    responses={
        200: {
            "description": "Service is alive",
            "content": {
                "application/json": {
                    "example": {
                        "status": "alive",
                        "timestamp": "2024-01-15T10:30:00.000000"
                    }
                }
            }
        }
    }
)
async def liveness_check():
    """
    Liveness probe endpoint.

    Simple check that returns 200 if the service is running.
    Use this for Kubernetes liveness probes.

    Returns:
        JSON with status and timestamp
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get(
    "/health/{service}",
    summary="Individual service health check",
    description="Check health of a specific service: postgresql, minio, or mlflow",
    response_description="Individual service health status",
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "service": "postgresql",
                        "status": "healthy",
                        "timestamp": "2024-01-15T10:30:00.000000",
                        "latency_ms": 12.3,
                        "details": {
                            "host": "localhost",
                            "port": "5432",
                            "database": "pipeline_db"
                        }
                    }
                }
            }
        },
        400: {
            "description": "Invalid service name",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Invalid service. Must be one of: postgresql, minio, mlflow"
                    }
                }
            }
        },
        503: {
            "description": "Service is unhealthy",
            "content": {
                "application/json": {
                    "example": {
                        "service": "postgresql",
                        "status": "unhealthy",
                        "timestamp": "2024-01-15T10:30:00.000000",
                        "latency_ms": 5001.2,
                        "error": "Connection refused"
                    }
                }
            }
        }
    }
)
async def check_individual_service(service: str):
    """
    Check health of an individual service.

    Args:
        service: Service name (postgresql, minio, or mlflow)

    Returns:
        JSON with service health status

    Raises:
        HTTPException: 400 if invalid service name, 503 if service unhealthy
    """
    valid_services = ["postgresql", "minio", "mlflow"]

    if service.lower() not in valid_services:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid service. Must be one of: {', '.join(valid_services)}"
        )

    service = service.lower()

    if service == "postgresql":
        check_result = await check_postgresql()
    elif service == "minio":
        check_result = await check_minio()
    else:  # mlflow
        check_result = await check_mlflow()

    response_data = {
        "service": service,
        "timestamp": datetime.utcnow().isoformat(),
        **check_result
    }

    if check_result["status"] == "unhealthy":
        return JSONResponse(
            status_code=503,
            content=response_data
        )

    return response_data
