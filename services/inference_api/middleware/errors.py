"""
Unified Error Handling for Inference API.

This module provides:
- Standard error codes and messages
- Exception classes for domain errors
- FastAPI exception handlers
- Correlation ID tracking in error responses

Contract: CTR-API-001

Usage:
    from middleware.errors import (
        APIException,
        ValidationException,
        NotFoundError,
        setup_exception_handlers,
    )

    # Raise structured errors
    raise ValidationException("start_date must be before end_date")

    # Setup handlers in main.py
    setup_exception_handlers(app)
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


# =============================================================================
# ERROR CODES
# =============================================================================

class ErrorCode(str, Enum):
    """
    Standard error codes for API responses.

    These codes are stable and can be used by clients
    for programmatic error handling.
    """

    # Validation errors (400)
    VALIDATION_ERROR = "validation_error"
    INVALID_DATE_RANGE = "invalid_date_range"
    INVALID_MODEL_ID = "invalid_model_id"
    INVALID_PARAMETER = "invalid_parameter"

    # Authentication/Authorization (401, 403)
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    INVALID_API_KEY = "invalid_api_key"

    # Not found errors (404)
    NOT_FOUND = "not_found"
    MODEL_NOT_FOUND = "model_not_found"
    TRADE_NOT_FOUND = "trade_not_found"
    DATA_NOT_FOUND = "data_not_found"

    # Rate limiting (429)
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"

    # Server errors (500)
    INTERNAL_ERROR = "internal_error"
    DATABASE_ERROR = "database_error"
    MODEL_ERROR = "model_error"
    INFERENCE_ERROR = "inference_error"

    # Service unavailable (503)
    SERVICE_UNAVAILABLE = "service_unavailable"


# =============================================================================
# ERROR RESPONSE MODEL
# =============================================================================

class ErrorDetail(BaseModel):
    """
    Standardized error response model.

    All API errors use this format for consistency.
    """

    error: str  # Error code (machine-readable)
    message: str  # Human-readable message
    details: Optional[Dict[str, Any]] = None  # Additional context
    request_id: Optional[str] = None  # Correlation ID for tracing
    timestamp: str = None  # ISO timestamp

    def __init__(self, **data):
        if "timestamp" not in data or data["timestamp"] is None:
            data["timestamp"] = datetime.utcnow().isoformat() + "Z"
        super().__init__(**data)

    class Config:
        json_schema_extra = {
            "example": {
                "error": "validation_error",
                "message": "start_date must be before end_date",
                "details": {"start_date": "2025-01-15", "end_date": "2025-01-10"},
                "request_id": "abc123-def456",
                "timestamp": "2025-01-14T15:30:00Z"
            }
        }


# =============================================================================
# EXCEPTION CLASSES
# =============================================================================

class APIException(Exception):
    """
    Base exception for all API errors.

    Attributes:
        status_code: HTTP status code
        error_code: Machine-readable error code
        message: Human-readable error message
        details: Additional error context
    """

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = ErrorCode.INTERNAL_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details
        super().__init__(message)


class ValidationException(APIException):
    """Raised when request validation fails."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        error_code: str = ErrorCode.VALIDATION_ERROR
    ):
        super().__init__(
            message=message,
            status_code=400,
            error_code=error_code,
            details=details
        )


class NotFoundError(APIException):
    """Raised when a requested resource is not found."""

    def __init__(
        self,
        message: str,
        resource_type: str = "resource",
        resource_id: Optional[str] = None
    ):
        details = {"resource_type": resource_type}
        if resource_id:
            details["resource_id"] = resource_id

        super().__init__(
            message=message,
            status_code=404,
            error_code=ErrorCode.NOT_FOUND,
            details=details
        )


class ModelNotFoundError(NotFoundError):
    """Raised when a model is not found."""

    def __init__(self, model_id: str):
        super().__init__(
            message=f"Model not found: {model_id}",
            resource_type="model",
            resource_id=model_id
        )
        self.error_code = ErrorCode.MODEL_NOT_FOUND


class DatabaseError(APIException):
    """Raised when database operations fail."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=500,
            error_code=ErrorCode.DATABASE_ERROR,
            details=details
        )


class InferenceError(APIException):
    """Raised when model inference fails."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=500,
            error_code=ErrorCode.INFERENCE_ERROR,
            details=details
        )


class RateLimitError(APIException):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        retry_after: int = 60,
        limit: int = 100,
        window: str = "minute"
    ):
        super().__init__(
            message=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            status_code=429,
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            details={
                "retry_after": retry_after,
                "limit": limit,
                "window": window
            }
        )
        self.retry_after = retry_after


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

def get_correlation_id(request: Request) -> Optional[str]:
    """Extract correlation ID from request state."""
    return getattr(request.state, "correlation_id", None)


async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
    """Handle APIException and subclasses."""
    correlation_id = get_correlation_id(request)

    logger.error(
        f"APIException: {exc.error_code} - {exc.message}",
        extra={
            "error_code": exc.error_code,
            "status_code": exc.status_code,
            "correlation_id": correlation_id,
            "details": exc.details
        }
    )

    error_detail = ErrorDetail(
        error=exc.error_code,
        message=exc.message,
        details=exc.details,
        request_id=correlation_id
    )

    headers = {"X-Request-ID": correlation_id} if correlation_id else {}

    if isinstance(exc, RateLimitError):
        headers["Retry-After"] = str(exc.retry_after)

    return JSONResponse(
        status_code=exc.status_code,
        content=error_detail.model_dump(),
        headers=headers
    )


async def validation_error_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    correlation_id = get_correlation_id(request)

    # Format validation errors
    errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"])
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"]
        })

    logger.warning(
        f"Validation error: {len(errors)} issues",
        extra={
            "correlation_id": correlation_id,
            "errors": errors
        }
    )

    error_detail = ErrorDetail(
        error=ErrorCode.VALIDATION_ERROR,
        message="Request validation failed",
        details={"validation_errors": errors},
        request_id=correlation_id
    )

    return JSONResponse(
        status_code=400,
        content=error_detail.model_dump(),
        headers={"X-Request-ID": correlation_id} if correlation_id else {}
    )


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException
) -> JSONResponse:
    """Handle Starlette HTTP exceptions."""
    correlation_id = get_correlation_id(request)

    # Map status codes to error codes
    error_code_map = {
        400: ErrorCode.VALIDATION_ERROR,
        401: ErrorCode.UNAUTHORIZED,
        403: ErrorCode.FORBIDDEN,
        404: ErrorCode.NOT_FOUND,
        429: ErrorCode.RATE_LIMIT_EXCEEDED,
        500: ErrorCode.INTERNAL_ERROR,
        503: ErrorCode.SERVICE_UNAVAILABLE,
    }

    error_code = error_code_map.get(exc.status_code, ErrorCode.INTERNAL_ERROR)

    error_detail = ErrorDetail(
        error=error_code,
        message=str(exc.detail),
        request_id=correlation_id
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_detail.model_dump(),
        headers={"X-Request-ID": correlation_id} if correlation_id else {}
    )


async def generic_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """Handle unexpected exceptions."""
    correlation_id = get_correlation_id(request)

    logger.exception(
        f"Unexpected error: {type(exc).__name__}: {exc}",
        extra={"correlation_id": correlation_id}
    )

    error_detail = ErrorDetail(
        error=ErrorCode.INTERNAL_ERROR,
        message="An unexpected error occurred. Please try again later.",
        request_id=correlation_id
    )

    return JSONResponse(
        status_code=500,
        content=error_detail.model_dump(),
        headers={"X-Request-ID": correlation_id} if correlation_id else {}
    )


# =============================================================================
# SETUP FUNCTION
# =============================================================================

def setup_exception_handlers(app: FastAPI) -> None:
    """
    Register all exception handlers with the FastAPI app.

    Call this in main.py after creating the app.

    Args:
        app: FastAPI application instance
    """
    app.add_exception_handler(APIException, api_exception_handler)
    app.add_exception_handler(ValidationException, api_exception_handler)
    app.add_exception_handler(NotFoundError, api_exception_handler)
    app.add_exception_handler(DatabaseError, api_exception_handler)
    app.add_exception_handler(InferenceError, api_exception_handler)
    app.add_exception_handler(RateLimitError, api_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    logger.info("Exception handlers registered")
