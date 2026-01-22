"""
Health check Pydantic schemas.

This module defines the schema models for health check endpoints,
including basic health, readiness, liveness, and individual service checks.
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class ServiceStatusEnum(str, Enum):
    """Possible service status values."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    READY = "ready"
    NOT_READY = "not_ready"
    ALIVE = "alive"


class ServiceDetails(BaseModel):
    """
    Detailed information about a service connection.

    Attributes:
        host: Service host address
        port: Service port number
        database: Database name (for PostgreSQL)
        endpoint: Service endpoint URL (for MinIO)
        bucket_count: Number of buckets (for MinIO)
        buckets: List of bucket names (for MinIO)
        tracking_uri: MLflow tracking URI
        endpoint_checked: Which endpoint was used for health check
        experiment_count: Number of experiments (for MLflow)
    """
    host: Optional[str] = Field(None, description="Service host address", example="localhost")
    port: Optional[str] = Field(None, description="Service port number", example="5432")
    database: Optional[str] = Field(None, description="Database name", example="pipeline_db")
    endpoint: Optional[str] = Field(None, description="Service endpoint URL", example="localhost:9000")
    bucket_count: Optional[int] = Field(None, description="Number of buckets in MinIO", example=3)
    buckets: Optional[List[str]] = Field(None, description="List of bucket names", example=["forecasts", "ml-models"])
    tracking_uri: Optional[str] = Field(None, description="MLflow tracking URI", example="http://localhost:5000")
    endpoint_checked: Optional[str] = Field(None, description="Endpoint used for health check", example="/health")
    experiment_count: Optional[int] = Field(None, description="Number of MLflow experiments", example=5)

    class Config:
        json_schema_extra = {
            "example": {
                "host": "localhost",
                "port": "5432",
                "database": "pipeline_db"
            }
        }


class ServiceStatus(BaseModel):
    """
    Health status of a single service.

    Attributes:
        status: Health status (healthy/unhealthy)
        latency_ms: Response latency in milliseconds
        error: Error message if unhealthy
        details: Additional service details
    """
    status: str = Field(
        ...,
        description="Service health status",
        example="healthy"
    )
    latency_ms: float = Field(
        ...,
        description="Response latency in milliseconds",
        example=12.5
    )
    error: Optional[str] = Field(
        None,
        description="Error message if service is unhealthy",
        example="Connection refused"
    )
    details: Optional[ServiceDetails] = Field(
        None,
        description="Additional service connection details"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "latency_ms": 12.5,
                "details": {
                    "host": "localhost",
                    "port": "5432",
                    "database": "pipeline_db"
                }
            }
        }


class HealthCheck(BaseModel):
    """
    Basic health check response.

    Attributes:
        status: Overall health status
        timestamp: ISO format timestamp
        service: Service name
        version: API version
    """
    status: str = Field(
        ...,
        description="Overall service health status",
        example="healthy"
    )
    timestamp: str = Field(
        ...,
        description="ISO format timestamp of the check",
        example="2024-01-15T10:30:00.000000"
    )
    service: str = Field(
        ...,
        description="Service identifier",
        example="usdcop-forecasting-api"
    )
    version: str = Field(
        ...,
        description="API version",
        example="1.0.0"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00.000000",
                "service": "usdcop-forecasting-api",
                "version": "1.0.0"
            }
        }


class ServiceChecks(BaseModel):
    """
    Collection of service health checks.

    Attributes:
        postgresql: PostgreSQL database health status
        minio: MinIO object storage health status
        mlflow: MLflow tracking server health status
    """
    postgresql: ServiceStatus = Field(..., description="PostgreSQL health status")
    minio: ServiceStatus = Field(..., description="MinIO health status")
    mlflow: ServiceStatus = Field(..., description="MLflow health status")


class ReadinessCheck(BaseModel):
    """
    Comprehensive readiness check response with all dependencies.

    Attributes:
        status: Overall readiness status (ready/not_ready)
        timestamp: ISO format timestamp
        total_latency_ms: Total time to complete all checks
        failed_services: List of failed service names (if any)
        checks: Detailed status for each service
    """
    status: str = Field(
        ...,
        description="Overall readiness status",
        example="ready"
    )
    timestamp: str = Field(
        ...,
        description="ISO format timestamp",
        example="2024-01-15T10:30:00.000000"
    )
    total_latency_ms: float = Field(
        ...,
        description="Total time to complete all health checks in milliseconds",
        example=45.5
    )
    failed_services: Optional[List[str]] = Field(
        None,
        description="List of failed service names",
        example=["postgresql"]
    )
    checks: Dict[str, ServiceStatus] = Field(
        ...,
        description="Detailed status for each service"
    )

    class Config:
        json_schema_extra = {
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
                            "bucket_count": 3
                        }
                    },
                    "mlflow": {
                        "status": "healthy",
                        "latency_ms": 24.5,
                        "details": {
                            "tracking_uri": "http://localhost:5000"
                        }
                    }
                }
            }
        }


class LivenessCheck(BaseModel):
    """
    Simple liveness probe response for Kubernetes.

    Attributes:
        status: Liveness status (always "alive" on success)
        timestamp: ISO format timestamp
    """
    status: str = Field(
        ...,
        description="Liveness status",
        example="alive"
    )
    timestamp: str = Field(
        ...,
        description="ISO format timestamp",
        example="2024-01-15T10:30:00.000000"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "alive",
                "timestamp": "2024-01-15T10:30:00.000000"
            }
        }


class IndividualServiceCheck(BaseModel):
    """
    Individual service health check response.

    Attributes:
        service: Name of the service checked
        status: Health status
        timestamp: ISO format timestamp
        latency_ms: Response latency in milliseconds
        error: Error message if unhealthy
        details: Additional service details
    """
    service: str = Field(
        ...,
        description="Service name",
        example="postgresql"
    )
    status: str = Field(
        ...,
        description="Service health status",
        example="healthy"
    )
    timestamp: str = Field(
        ...,
        description="ISO format timestamp",
        example="2024-01-15T10:30:00.000000"
    )
    latency_ms: float = Field(
        ...,
        description="Response latency in milliseconds",
        example=12.3
    )
    error: Optional[str] = Field(
        None,
        description="Error message if service is unhealthy"
    )
    details: Optional[ServiceDetails] = Field(
        None,
        description="Additional service connection details"
    )

    class Config:
        json_schema_extra = {
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
